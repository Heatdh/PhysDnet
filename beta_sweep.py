"""
Beta sweep evaluation — evaluate a trained model at fixed beta values
from light to aggressive fog.  Produces:
  1. CSV with PSNR/SSIM per beta
  2. Grid figure showing hazy → restored → clear at each beta level
"""

import argparse
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from model import get_model
from dataset import STFDehazeDataset

# ── metrics ────────────────────────────────────────────────────────────────

def psnr(pred, target, max_val=1.0):
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def ssim(pred, target, window_size=11, C1=0.01**2, C2=0.03**2):
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device)
    kernel /= kernel.sum()
    pad = window_size // 2
    vals = []
    for c in range(pred.shape[1]):
        p, t = pred[:, c:c+1], target[:, c:c+1]
        mu_p = torch.nn.functional.conv2d(p, kernel, padding=pad)
        mu_t = torch.nn.functional.conv2d(t, kernel, padding=pad)
        s_p = torch.nn.functional.conv2d(p*p, kernel, padding=pad) - mu_p**2
        s_t = torch.nn.functional.conv2d(t*t, kernel, padding=pad) - mu_t**2
        s_pt = torch.nn.functional.conv2d(p*t, kernel, padding=pad) - mu_p*mu_t
        num = (2*mu_p*mu_t + C1) * (2*s_pt + C2)
        den = (mu_p**2 + mu_t**2 + C1) * (s_p + s_t + C2)
        vals.append((num / (den + 1e-8)).mean().item())
    return float(np.mean(vals))

# ── main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Beta sweep evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--betas", type=float, nargs="+",
                        default=[0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15])
    parser.add_argument("--airlight", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=4,
                        help="Number of sample images to save per beta")
    parser.add_argument("--out_dir", default="results/beta_sweep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── load model ──
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    overrides = {"base_ch": model_cfg.get("base_ch", 64)}
    for k in ("use_cbam", "use_residual", "use_physics_head", "lidar_drop_rate"):
        v = model_cfg.get(k)
        if v is not None:
            overrides[k] = v
    model = get_model(model_cfg.get("variant", "robust"), **overrides).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model (epoch {ckpt.get('epoch', '?')}, {n_params/1e6:.2f}M params)")

    # ── dataset config (from checkpoint) ──
    data_cfg = cfg.get("data", {})
    stf_root = data_cfg.get("stf_root", "data/stf/SeeingThroughFog")
    meta_dir = data_cfg.get("meta_dir", "data/stf/meta")
    val_ts = os.path.join(meta_dir, "val_timestamps.txt")
    crop_size = tuple(data_cfg.get("crop_size", [256, 256]))
    max_depth = data_cfg.get("max_depth", 120.0)

    os.makedirs(args.out_dir, exist_ok=True)

    # ── sweep ──
    results = []
    # Collect sample images for the figure: {beta: [(hazy, restored, clear), ...]}
    sample_images = {}

    for beta in args.betas:
        print(f"\n{'='*60}")
        print(f"  Beta = {beta:.3f}")
        print(f"{'='*60}")

        # Fixed beta → set beta_range = [beta, beta]
        # Fixed airlight → same
        A = args.airlight
        ds = STFDehazeDataset(
            stf_root=stf_root,
            timestamps_file=val_ts,
            crop_size=crop_size,
            beta_range=(beta, beta),
            airlight_range=(A, A),
            max_depth=max_depth,
            augment=False,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4)

        all_p, all_s = [], []
        collected = 0
        samples_for_beta = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"β={beta:.3f}", unit="batch"):
                hazy = batch["hazy"].to(device)
                clear = batch["clear"].to(device)
                sparse = batch["sparse_depth"].to(device)
                mask = batch["mask"].to(device)

                out = model(hazy, sparse, mask)
                restored = out["restored"]

                B = hazy.shape[0]
                for i in range(B):
                    all_p.append(psnr(restored[i:i+1], clear[i:i+1]))
                    all_s.append(ssim(restored[i:i+1], clear[i:i+1]))

                    if collected < args.num_samples:
                        def _to_np(t):
                            return (t.cpu().numpy().transpose(1, 2, 0) * 255
                                    ).clip(0, 255).astype(np.uint8)
                        samples_for_beta.append((
                            _to_np(hazy[i]),
                            _to_np(restored[i]),
                            _to_np(clear[i]),
                        ))
                        collected += 1

        avg_p, avg_s = np.mean(all_p), np.mean(all_s)
        std_p, std_s = np.std(all_p), np.std(all_s)
        print(f"  PSNR  {avg_p:.2f} ±{std_p:.2f}  |  SSIM  {avg_s:.4f} ±{std_s:.4f}")
        results.append((beta, avg_p, std_p, avg_s, std_s, len(all_p)))
        sample_images[beta] = samples_for_beta

    # ── save CSV ──
    csv_path = os.path.join(args.out_dir, "beta_sweep.csv")
    with open(csv_path, "w") as f:
        f.write("beta,psnr_mean,psnr_std,ssim_mean,ssim_std,n_samples\n")
        for row in results:
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"\nCSV saved to {csv_path}")

    # ── save per-beta sample images ──
    img_dir = os.path.join(args.out_dir, "samples")
    os.makedirs(img_dir, exist_ok=True)
    for beta, imgs in sample_images.items():
        for j, (h, r, c) in enumerate(imgs):
            Image.fromarray(h).save(os.path.join(img_dir, f"beta{beta:.3f}_s{j}_hazy.png"))
            Image.fromarray(r).save(os.path.join(img_dir, f"beta{beta:.3f}_s{j}_restored.png"))
            Image.fromarray(c).save(os.path.join(img_dir, f"beta{beta:.3f}_s{j}_clear.png"))

    # ── PSNR vs beta curve ──
    betas_arr = [r[0] for r in results]
    psnr_arr = [r[1] for r in results]
    psnr_std_arr = [r[2] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(betas_arr, psnr_arr, yerr=psnr_std_arr, marker="o",
                capsize=3, linewidth=2, color="#2563eb")
    ax.set_xlabel(r"Scattering coefficient $\beta$", fontsize=12)
    ax.set_ylabel("PSNR (dB)", fontsize=12)
    ax.set_title("Restoration quality vs. fog density", fontsize=13)
    ax.grid(True, alpha=0.3)
    # Mark training range
    ax.axvspan(0.005, 0.04, alpha=0.08, color="green", label="Training range")
    ax.legend(fontsize=10)
    fig.tight_layout()
    curve_path = os.path.join(args.out_dir, "psnr_vs_beta.pdf")
    fig.savefig(curve_path, dpi=150)
    fig.savefig(curve_path.replace(".pdf", ".png"), dpi=150)
    print(f"Curve saved to {curve_path}")
    plt.close(fig)

    # ── Grid figure for appendix: rows = beta, cols = hazy | restored | clear ──
    n_betas = len(args.betas)
    # Use first sample for each beta
    fig, axes = plt.subplots(n_betas, 3, figsize=(9, 2.8 * n_betas))
    if n_betas == 1:
        axes = axes[np.newaxis, :]
    col_titles = ["Hazy input", "Restored (ours)", "Ground truth"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")
    for row, beta in enumerate(args.betas):
        imgs = sample_images[beta]
        if len(imgs) == 0:
            continue
        h, r, c = imgs[0]
        axes[row, 0].imshow(h)
        axes[row, 1].imshow(r)
        axes[row, 2].imshow(c)
        # beta label on left
        psnr_val = [x[1] for x in results if x[0] == beta][0]
        axes[row, 0].set_ylabel(f"β={beta:.3f}\n{psnr_val:.1f} dB",
                                fontsize=10, rotation=0, labelpad=55, va="center")
        for col in range(3):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    fig.suptitle("DeL-PUNet restoration across fog densities", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    grid_path = os.path.join(args.out_dir, "beta_sweep_grid.pdf")
    fig.savefig(grid_path, dpi=150, bbox_inches="tight")
    fig.savefig(grid_path.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"Grid figure saved to {grid_path}")
    plt.close(fig)

    print("\nDone!")


if __name__ == "__main__":
    main()
