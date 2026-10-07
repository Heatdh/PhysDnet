"""
Gradient amplification analysis: PhysDNet-M vs NoPhy baseline.

Empirically validates that the physics head induces density-adaptive
gradient amplification following a (t + eps)^{-2} curve, while the
NoPhy baseline exhibits spatially uniform gradients.

Produces a grouped bar chart saved to neurips_docs/figures/.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── project imports ──
from model import LiDARDehazeNet, get_model, MODEL_VARIANTS
from dataset import STFDehazeDataset
from torch.utils.data import DataLoader

try:
    import yaml
except ImportError:
    sys.exit("PyYAML required: pip install pyyaml")


# ────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────

def load_model_from_checkpoint(ckpt_path, device):
    """Load model + config from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    variant = model_cfg.get("variant", "robust")
    base_ch = model_cfg.get("base_ch", 64)
    use_physics = model_cfg.get("use_physics_head", True)

    model = get_model(variant, base_ch=base_ch, use_physics_head=use_physics)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model, cfg, use_physics


def build_val_loader(cfg, batch_size=8, num_workers=4):
    """Build STF validation dataloader from config."""
    data_cfg = cfg.get("data", {})
    stf_root = data_cfg.get("stf_root", "data/stf/SeeingThroughFog")
    meta_dir = data_cfg.get("meta_dir", "data/stf/meta")
    val_ts = os.path.join(meta_dir, "val_timestamps.txt")
    depth_dir = data_cfg.get("depth_dir", None)

    val_ds = STFDehazeDataset(
        stf_root=stf_root,
        depth_dir=depth_dir,
        timestamps_file=val_ts,
        crop_size=tuple(data_cfg.get("crop_size", [256, 256])),
        beta_range=tuple(data_cfg.get("beta_range", [0.005, 0.04])),
        airlight_range=tuple(data_cfg.get("airlight_range", [0.7, 1.0])),
        max_depth=data_cfg.get("max_depth", 120.0),
        weather_filter=data_cfg.get("val_weather_filter", ["clear", "overcast"]),
        augment=False,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=True)
    return loader


# ────────────────────────────────────────────────────────
# Gradient collection
# ────────────────────────────────────────────────────────

N_BINS = 10
BIN_EDGES = np.linspace(0.0, 1.0, N_BINS + 1)
BIN_CENTRES = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])


def collect_physdnet_gradients(model, loader, device, n_batches=50):
    """
    PhysDNet-M: hook on t_hat, backward from L1(j_phys, clean).
    Bin grad magnitudes by t_hat value.
    """
    bin_sums = np.zeros(N_BINS, dtype=np.float64)
    bin_counts = np.zeros(N_BINS, dtype=np.float64)

    grad_storage = {}

    def hook_fn(grad):
        grad_storage["t_hat_grad"] = grad.detach()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        hazy = batch["hazy"].to(device)
        clear = batch["clear"].to(device)
        sparse = batch["sparse_depth"].to(device)
        mask = batch["mask"].to(device)

        # Enable grad on t_hat by running forward manually
        model.zero_grad()

        # We need t_hat to have requires_grad for the hook
        with torch.enable_grad():
            out = model(hazy, sparse, mask)
            t_hat = out["transmission"]  # B x 1 x H x W

            # Re-derive physics_restored so t_hat is in the graph
            A_hat = out["airlight"]  # B x 3
            A_spatial = A_hat[:, :, None, None]
            eps = 1e-4
            t_hat_grad = t_hat.detach().requires_grad_(True)
            j_phys = (hazy - A_spatial * (1.0 - t_hat_grad)) / (t_hat_grad + eps)
            j_phys = torch.clamp(j_phys, 0.0, 1.0)

            handle = t_hat_grad.register_hook(hook_fn)

            loss = F.l1_loss(j_phys, clear)
            loss.backward()
            handle.remove()

        # Bin by t_hat value
        t_vals = t_hat.detach().cpu().numpy().ravel()
        grad_mag = grad_storage["t_hat_grad"].abs().cpu().numpy().ravel()

        bin_idx = np.digitize(t_vals, BIN_EDGES) - 1
        bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

        for b in range(N_BINS):
            m = bin_idx == b
            bin_sums[b] += grad_mag[m].sum()
            bin_counts[b] += m.sum()

        if (i + 1) % 10 == 0:
            print(f"  PhysDNet-M: {i+1}/{n_batches} batches")

    means = np.divide(bin_sums, bin_counts,
                      out=np.zeros_like(bin_sums), where=bin_counts > 0)
    return means, bin_counts


def collect_nophy_gradients(model, loader, device, n_batches=50):
    """
    NoPhy: hook on decoder last feature map (dec1 output / head_restore input).
    Backward from L1(j_dir, clean). Bin by GT transmission.
    """
    bin_sums = np.zeros(N_BINS, dtype=np.float64)
    bin_counts = np.zeros(N_BINS, dtype=np.float64)

    grad_storage = {}

    def hook_fn(module, grad_input, grad_output):
        # grad_output[0] is the gradient w.r.t. the output of dec1
        grad_storage["feat_grad"] = grad_output[0].detach()

    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)  # Need grads to flow through

    # Register hook on the last decoder block
    handle = model.dec1.register_full_backward_hook(hook_fn)

    for i, batch in enumerate(loader):
        if i >= n_batches:
            break

        hazy = batch["hazy"].to(device)
        clear = batch["clear"].to(device)
        sparse = batch["sparse_depth"].to(device)
        mask_inp = batch["mask"].to(device)
        trans_gt = batch["trans_gt"]  # B x 1 x H x W

        model.zero_grad()

        with torch.enable_grad():
            out = model(hazy, sparse, mask_inp)
            j_dir = out["restored"]  # B x 3 x H x W
            loss = F.l1_loss(j_dir, clear)
            loss.backward()

        feat_grad = grad_storage["feat_grad"]  # B x C x Hf x Wf
        grad_mag = feat_grad.abs().mean(dim=1, keepdim=True)  # B x 1 x Hf x Wf

        # Match GT transmission resolution to feature map
        _, _, Hf, Wf = grad_mag.shape
        t_gt_matched = F.interpolate(trans_gt, size=(Hf, Wf),
                                     mode="bilinear", align_corners=False)

        t_vals = t_gt_matched.cpu().numpy().ravel()
        g_vals = grad_mag.cpu().numpy().ravel()

        bin_idx = np.digitize(t_vals, BIN_EDGES) - 1
        bin_idx = np.clip(bin_idx, 0, N_BINS - 1)

        for b in range(N_BINS):
            m = bin_idx == b
            bin_sums[b] += g_vals[m].sum()
            bin_counts[b] += m.sum()

        if (i + 1) % 10 == 0:
            print(f"  NoPhy: {i+1}/{n_batches} batches")

    handle.remove()
    for p in model.parameters():
        p.requires_grad_(False)

    means = np.divide(bin_sums, bin_counts,
                      out=np.zeros_like(bin_sums), where=bin_counts > 0)
    return means, bin_counts


# ────────────────────────────────────────────────────────
# Plotting
# ────────────────────────────────────────────────────────

def plot_gradient_amplification(phys_means, nophy_means, out_path):
    """Two-panel figure: (a) log-scale absolute gradients, (b) normalised shape."""
    fig, (ax_abs, ax_norm) = plt.subplots(1, 2, figsize=(11, 4.2))

    bar_w = 0.35
    x = np.arange(N_BINS)
    bin_labels = [f"{BIN_EDGES[i]:.1f}–{BIN_EDGES[i+1]:.1f}"
                  for i in range(N_BINS)]

    # ── Panel (a): absolute magnitudes, log scale ──
    ax_abs.bar(x - bar_w / 2, phys_means, bar_w,
               label="PhysDNet-M", color="#2196F3",
               edgecolor="white", linewidth=0.5, zorder=3)
    ax_abs.bar(x + bar_w / 2, nophy_means, bar_w,
               label="NoPhy", color="#FF9800",
               edgecolor="white", linewidth=0.5, zorder=3)
    ax_abs.set_yscale("log")

    # Theoretical curve scaled to match PhysDNet-M first bin
    eps = 1e-4
    t_fine = np.linspace(BIN_CENTRES[0], BIN_CENTRES[-1], 200)
    theory = (t_fine + eps) ** (-2)
    theory_scaled = theory * (phys_means[0] / theory[0])
    ax_abs.plot(np.interp(t_fine, BIN_CENTRES, x), theory_scaled,
                "k--", linewidth=2.0, label=r"$(t+\epsilon)^{-2}$", zorder=4)

    # Ratio annotation
    ratio_max = phys_means[0] / (nophy_means[0] if nophy_means[0] > 0 else 1)
    ax_abs.annotate(f"~{ratio_max:.0f}×", xy=(0, phys_means[0]),
                    xytext=(2.0, phys_means[0] * 0.6),
                    fontsize=9, fontweight="bold", color="#333",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1.2))

    ax_abs.set_xticks(x)
    ax_abs.set_xticklabels(bin_labels, fontsize=7, rotation=35, ha="right")
    ax_abs.set_xlabel("Transmission $\\hat{t}$ (binned)", fontsize=10)
    ax_abs.set_ylabel("Mean $|\\nabla_{\\hat{t}}\\mathcal{L}|$", fontsize=10)
    ax_abs.set_title("(a) Absolute gradient magnitude", fontsize=11,
                     fontweight="bold")
    ax_abs.legend(fontsize=8, loc="upper right")
    ax_abs.grid(axis="y", alpha=0.3, zorder=0)

    # ── Panel (b): normalised shape comparison ──
    phys_max = phys_means.max() if phys_means.max() > 0 else 1.0
    nophy_max = nophy_means.max() if nophy_means.max() > 0 else 1.0

    phys_norm = phys_means / phys_max
    nophy_norm = nophy_means / nophy_max

    ax_norm.bar(x - bar_w / 2, phys_norm, bar_w,
                label="PhysDNet-M", color="#2196F3",
                edgecolor="white", linewidth=0.5, zorder=3)
    ax_norm.bar(x + bar_w / 2, nophy_norm, bar_w,
                label="NoPhy", color="#FF9800",
                edgecolor="white", linewidth=0.5, zorder=3)

    theory_norm = theory / theory.max()
    ax_norm.plot(np.interp(t_fine, BIN_CENTRES, x), theory_norm,
                 "k--", linewidth=2.0, label=r"$(t+\epsilon)^{-2}$", zorder=4)

    ax_norm.set_xticks(x)
    ax_norm.set_xticklabels(bin_labels, fontsize=7, rotation=35, ha="right")
    ax_norm.set_xlabel("Transmission $\\hat{t}$ (binned)", fontsize=10)
    ax_norm.set_ylabel("Normalised $|\\nabla_{\\hat{t}}\\mathcal{L}|$",
                       fontsize=10)
    ax_norm.set_title("(b) Normalised gradient profile", fontsize=11,
                      fontweight="bold")
    ax_norm.legend(fontsize=8, loc="upper right")
    ax_norm.grid(axis="y", alpha=0.3, zorder=0)
    ax_norm.set_ylim(0, 1.35)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")

    pdf_path = out_path.rsplit(".", 1)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {pdf_path}")
    plt.close(fig)


# ────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gradient amplification analysis: PhysDNet vs NoPhy")
    parser.add_argument("--physdnet_ckpt", type=str, default="",
                        help="PhysDNet-M checkpoint path")
    parser.add_argument("--nophy_ckpt", type=str, default="",
                        help="NoPhy baseline checkpoint path")
    parser.add_argument("--n_batches", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--out", type=str,
                        default="neurips_docs/figures/fig_gradient_amplification.png")
    parser.add_argument("--plot_only", action="store_true",
                        help="Skip gradient collection, re-plot from cached .npz")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_path = args.out.rsplit(".", 1)[0] + "_data.npz"

    if args.plot_only:
        print(f"\n─── Loading cached data from {cache_path} ───")
        data = np.load(cache_path)
        phys_means, nophy_means = data["phys_means"], data["nophy_means"]
    else:
        # ── Load PhysDNet-M ──
        print(f"\nLoading PhysDNet-M: {args.physdnet_ckpt}")
        phys_model, phys_cfg, has_physics = load_model_from_checkpoint(
            args.physdnet_ckpt, device)
        assert has_physics, "PhysDNet checkpoint must have physics head enabled"

        # ── Val loader (use PhysDNet config for data params) ──
        print("Building validation loader...")
        loader = build_val_loader(phys_cfg, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
        print(f"Val samples: {len(loader.dataset)}, "
              f"batches: {len(loader)}, using {args.n_batches}")

        # ── Collect gradients ──
        print("\n─── PhysDNet-M gradient collection ───")
        phys_means, phys_counts = collect_physdnet_gradients(
            phys_model, loader, device, n_batches=args.n_batches)
        print("Bin counts:", phys_counts.astype(int))
        print("Bin means: ", np.array2string(phys_means, precision=6))

        # Free PhysDNet-M GPU memory before loading NoPhy
        del phys_model
        torch.cuda.empty_cache()

        # ── Load NoPhy ──
        print(f"\nLoading NoPhy: {args.nophy_ckpt}")
        nophy_model, nophy_cfg, nophy_has_physics = load_model_from_checkpoint(
            args.nophy_ckpt, device)

        print("\n─── NoPhy gradient collection ───")
        nophy_means, nophy_counts = collect_nophy_gradients(
            nophy_model, loader, device, n_batches=args.n_batches)
        print("Bin counts:", nophy_counts.astype(int))
        print("Bin means: ", np.array2string(nophy_means, precision=6))

        # Save cache
        np.savez(cache_path, phys_means=phys_means, nophy_means=nophy_means,
                 phys_counts=phys_counts, nophy_counts=nophy_counts)
        print(f"Cached data to {cache_path}")

    # ── Plot ──
    print("\n─── Plotting ───")
    plot_gradient_amplification(phys_means, nophy_means, args.out)

    # ── Print summary table ──
    print("\n─── Summary ───")
    print(f"{'Bin':>12s}  {'PhysDNet-M':>12s}  {'NoPhy':>12s}  {'Ratio':>8s}")
    for i in range(N_BINS):
        label = f"{BIN_EDGES[i]:.1f}-{BIN_EDGES[i+1]:.1f}"
        ratio = (phys_means[i] / nophy_means[i]
                 if nophy_means[i] > 0 else float("inf"))
        print(f"{label:>12s}  {phys_means[i]:12.6f}  {nophy_means[i]:12.6f}  {ratio:8.2f}")


if __name__ == "__main__":
    main()
