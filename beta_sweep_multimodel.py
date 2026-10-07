"""
Multi-model beta sweep — evaluate multiple models across fog densities.
Produces a tueplots-styled PSNR vs β curve for the NeurIPS paper.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from baselines.registry import get_baseline_model
from dataset import STFDehazeDataset

# ── metrics ────────────────────────────────────────────────────────────────

def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


# ── model loading ──────────────────────────────────────────────────────────

DISPLAY_NAMES = {
    "robust_ch64_fft+ctr@185": "PhysDNet-M",
    "robust_ch64_fft+ctr@185+nophy": "PhysDNet-M (NoPhy)",
    "robust_ch96_fft+ctr@185": "PhysDNet-L",
    "aod_net(rgb)": "AOD-Net",
    "ffa_net(rgb)": "FFA-Net",
    "dehaze_former(rgb)": "DehazeFormer-T",
    "dea_net(rgb)": "DEA-Net",
    "dark_channel_prior(rgb)": "DCP",
}


def _short_name(ckpt_path):
    """Extract a short model name from checkpoint path."""
    d = os.path.basename(os.path.dirname(ckpt_path))
    # baseline_stf_ffa_net_rgb_0.77M_... → ffa_net(rgb)
    if "baseline_stf_" in d:
        parts = d.replace("baseline_stf_", "").split("_wclear")[0]
        # e.g. "aod_net_rgb_0.00M" → extract arch and mode
        tokens = parts.split("_")
        # find size token like "0.00M", "0.77M"
        size_idx = None
        for i, t in enumerate(tokens):
            if t.endswith("M") and t[0].isdigit():
                size_idx = i
                break
        if size_idx is not None:
            mode = tokens[size_idx - 1]  # rgb, lidar, etc.
            arch = "_".join(tokens[:size_idx - 1])
            return f"{arch}({mode})"
        return parts
    # Our models: stf_robust_ch64_3.0M_..._v2_fft+ctr@185_...
    parts = d.replace("stf_", "").split("_wclear")[0]
    tokens = parts.split("_")
    size_idx = None
    for i, t in enumerate(tokens):
        if t.endswith("M") and t[0].isdigit():
            size_idx = i
            break
    arch = "_".join(tokens[:size_idx]) if size_idx else parts
    # Append loss tag if present (check longer tags first!)
    for tag in ["fft+ctr@185+nophy", "fft+ctr@185", "fft+ctr", "mse"]:
        if tag in d:
            arch += f"_{tag}"
            break
    return arch


def load_model(ckpt_path, device):
    """Load model from checkpoint, auto-detecting architecture."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mc = cfg.get("model", {})

    baseline = mc.get("baseline")
    if baseline:
        mode = mc.get("mode", "rgb")
        kw = {k: v for k, v in mc.items()
              if k not in ("baseline", "mode", "variant")}
        model = get_baseline_model(baseline, mode=mode, **kw).to(device)
    else:
        variant = mc.get("variant", "lite")
        ov = {"base_ch": mc.get("base_ch", 64)}
        for k in ("use_cbam", "use_residual", "use_physics_head", "lidar_drop_rate"):
            v = mc.get(k)
            if v is not None:
                ov[k] = v
        model = get_model(variant, **ov).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    name = _short_name(ckpt_path)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {name} ({n_params/1e6:.2f}M params)")
    return model, name


# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-model beta sweep")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10])
    parser.add_argument("--airlight", type=float, default=0.8)
    parser.add_argument("--crop_size", type=int, default=256,
                        help="Evaluation crop size (overrides checkpoint config)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", default="results/beta_sweep_multi")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # ── load all models ──
    print("Loading models...")
    models = []
    for ckpt_path in args.checkpoints:
        model, name = load_model(ckpt_path, device)
        models.append((model, name))

    # ── dataset config from first checkpoint ──
    ckpt0 = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    data_cfg = ckpt0.get("config", {}).get("data", {})
    stf_root = data_cfg.get("stf_root", "data/stf/SeeingThroughFog")
    meta_dir = data_cfg.get("meta_dir", "data/stf/meta")
    val_ts = os.path.join(meta_dir, "val_timestamps.txt")
    crop_size = (args.crop_size, args.crop_size)
    max_depth = data_cfg.get("max_depth", 120.0)
    del ckpt0

    # ── sweep ──
    # results[model_name][beta] = (psnr_mean, psnr_std)
    all_results = {name: {} for _, name in models}

    for beta in args.betas:
        print(f"\n{'='*60}")
        print(f"  Beta = {beta:.3f}")
        print(f"{'='*60}")

        ds = STFDehazeDataset(
            stf_root=stf_root,
            timestamps_file=val_ts,
            crop_size=crop_size,
            beta_range=(beta, beta),
            airlight_range=(args.airlight, args.airlight),
            max_depth=max_depth,
            augment=False,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

        # Pre-fetch all batches (same data for all models)
        batches = []
        for batch in loader:
            batches.append({
                "hazy": batch["hazy"].to(device),
                "clear": batch["clear"].to(device),
                "sparse_depth": batch["sparse_depth"].to(device),
                "mask": batch["mask"].to(device),
            })

        for model, name in models:
            psnr_vals = []
            with torch.no_grad():
                for b in tqdm(batches, desc=f"  {name}", unit="batch"):
                    out = model(b["hazy"], b["sparse_depth"], b["mask"])
                    restored = out["restored"]
                    for i in range(restored.shape[0]):
                        psnr_vals.append(psnr(restored[i:i+1], b["clear"][i:i+1]))

            avg_p = np.mean(psnr_vals)
            std_p = np.std(psnr_vals)
            all_results[name][beta] = (avg_p, std_p, len(psnr_vals))
            print(f"  {name}: PSNR {avg_p:.2f} ±{std_p:.2f} ({len(psnr_vals)} samples)")

        # Free batch memory before next beta
        del batches
        torch.cuda.empty_cache()

    # ── save CSV ──
    csv_path = os.path.join(args.out_dir, "beta_sweep_multi.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "beta", "psnr_mean", "psnr_std", "n_samples"])
        for name in all_results:
            for beta in sorted(all_results[name]):
                avg_p, std_p, n = all_results[name][beta]
                writer.writerow([name, beta, f"{avg_p:.4f}", f"{std_p:.4f}", n])
    print(f"\nCSV saved to {csv_path}")

    # ── generate tueplots figure ──
    generate_figure(all_results, args.betas, args.out_dir)
    print("\nDone!")


def generate_figure(all_results, betas, out_dir):
    """Generate PSNR vs beta curve using tueplots NeurIPS bundle."""
    from tueplots import bundles
    import matplotlib.patheffects as pe

    # ── Colour + style per model ──────────────────────────────────
    STYLE = {
        "AOD-Net":              dict(color="#E53935", marker="v",  ls="--", lw=1.0, zorder=4),
        "DehazeFormer-T":       dict(color="#43A047", marker="s",  ls="--", lw=1.0, zorder=4),
        "DEA-Net":              dict(color="#8E24AA", marker="P",  ls="--", lw=1.0, zorder=4),
        "FFA-Net":              dict(color="#FB8C00", marker="D",  ls="--", lw=1.0, zorder=4),
        "PhysDNet-M (NoPhy)":   dict(color="#90CAF9", marker="h",  ls="-.", lw=1.3, zorder=7),
        "PhysDNet-M":           dict(color="#1E88E5", marker="o",  ls="-",  lw=1.8, zorder=9),
        "PhysDNet-L":           dict(color="#0D47A1", marker="^",  ls="-",  lw=1.8, zorder=10),
    }
    ORDER = list(STYLE.keys())

    rc = bundles.neurips2024(usetex=False)
    rc["font.family"] = "serif"
    plt.rcParams.update(rc)

    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    # ── Training range ────────────────────────────────────────────
    ax.axvspan(0.005, 0.04, alpha=0.08, color="#4CAF50", zorder=1)
    ax.text(0.022, 39.5, "training range", fontsize=5.5, color="#2E7D32",
            ha="center", va="bottom", style="italic",
            path_effects=[pe.withStroke(linewidth=2, foreground="white")])

    # ── Sort models for drawing order ─────────────────────────────
    sorted_results = sorted(
        all_results.items(),
        key=lambda kv: ORDER.index(DISPLAY_NAMES.get(kv[0], kv[0]))
        if DISPLAY_NAMES.get(kv[0], kv[0]) in ORDER else 99
    )

    for raw_name, beta_data in sorted_results:
        display = DISPLAY_NAMES.get(raw_name, raw_name)
        st = STYLE.get(display, dict(color="gray", marker=".", ls="--", lw=0.8, zorder=2))
        xs = sorted(beta_data.keys())
        ys = [beta_data[b][0] for b in xs]
        yerr = [beta_data[b][1] for b in xs]

        is_ours = display.startswith("PhysDNet")
        ax.errorbar(xs, ys, yerr=yerr, label=display,
                    color=st["color"], marker=st["marker"],
                    ls=st["ls"], lw=st["lw"],
                    markersize=5, capsize=2, capthick=0.8,
                    markeredgecolor="white", markeredgewidth=0.4,
                    alpha=1.0 if is_ours else 0.85,
                    zorder=st["zorder"])

    # ── Axes ──────────────────────────────────────────────────────
    ax.set_xlabel(r"Scattering coefficient $\beta$", fontsize=9)
    ax.set_ylabel("PSNR (dB)", fontsize=9)
    ax.set_xlim(-0.003, max(betas) + 0.008)
    ax.set_ylim(8, 42)
    ax.grid(True, which="major", axis="y", ls="-", lw=0.25, alpha=0.35)
    ax.grid(True, which="major", axis="x", ls="-", lw=0.25, alpha=0.2)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    ax.legend(fontsize=5.5, loc="upper right", framealpha=0.92,
              edgecolor="0.85", fancybox=True, ncol=2,
              handlelength=1.8, handletextpad=0.3, columnspacing=0.8)

    pdf_path = os.path.join(out_dir, "psnr_vs_beta_multi.pdf")
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    print(f"Figure saved to {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
