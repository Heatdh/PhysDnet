#!/usr/bin/env python3
"""
LiDAR sparsity ablation: sweep keep_ratio from 100% to 0% and measure
PSNR/SSIM degradation.  Demonstrates graceful failure when LiDAR density
drops.

At each keep_ratio, valid mask pixels are randomly subsampled (the
underlying depth values are unchanged).  keep_ratio=0.0 is equivalent
to --strip_lidar (pure RGB fallback).

Usage:
    python eval_lidar_sparsity.py
    python eval_lidar_sparsity.py --models S M --batch_size 4
"""

import argparse
import csv
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from dataset import STFDehazeDataset


# ── Metrics ───────────────────────────────────────────────────────────────

def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(1.0 / mse)


def ssim(pred: torch.Tensor, target: torch.Tensor,
         window_size: int = 11) -> float:
    C1, C2 = 0.01 ** 2, 0.03 ** 2
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


# ── Model registry ────────────────────────────────────────────────────────

MODELS = {
    "S": {
        "display": "PhysDNet-S",
        "ckpt": "runs/stf_robust_ch32_0.8M_wclear+overcast_crop256_"
                "b0.005-0.04_bs32_lr2e-04_ep500_v2_fft+ctr@185_0422_2234"
                "/best_psnr.pth",
    },
    "M": {
        "display": "PhysDNet-M",
        "ckpt": "runs/stf_robust_ch64_3.0M_wclear+overcast_crop256_"
                "b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185_0406_2105"
                "/best_psnr.pth",
    },
    "L": {
        "display": "PhysDNet-L",
        "ckpt": "runs/stf_robust_ch96_6.6M_wclear+overcast_crop256_"
                "b0.005-0.04_bs8_lr2e-04_ep500_v2_fft+ctr@185_0407_2326"
                "/best_psnr.pth",
    },
    "NoPhy": {
        "display": "NoPhy",
        "ckpt": "runs/stf_robust_ch64_2.9M_wclear+overcast_crop256_"
                "b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185+nophy_"
                "0413_1008/best_psnr.pth",
    },
}

KEEP_RATIOS = [1.0, 0.75, 0.50, 0.25, 0.10, 0.05, 0.01, 0.0]


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    variant = model_cfg.get("variant", "lite")
    overrides = {"base_ch": model_cfg.get("base_ch", 32)}
    for k in ("use_cbam", "use_residual", "use_attention",
              "use_physics_head", "lidar_drop_rate"):
        v = model_cfg.get(k)
        if v is not None:
            overrides[k] = v
    model = get_model(variant, **overrides).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {variant} ({n/1e6:.2f}M) from {ckpt_path}")
    return model, cfg


def subsample_mask(mask: torch.Tensor, keep_ratio: float,
                   generator: torch.Generator) -> torch.Tensor:
    """Randomly drop valid pixels from mask. Returns sub_mask (same shape)."""
    if keep_ratio >= 1.0:
        return mask
    if keep_ratio <= 0.0:
        return torch.zeros_like(mask)
    sub = mask.clone()
    B = sub.shape[0]
    for b in range(B):
        valid_idx = torch.nonzero(sub[b, 0] > 0.5, as_tuple=False)
        n_valid = valid_idx.shape[0]
        if n_valid == 0:
            continue
        n_drop = int(n_valid * (1.0 - keep_ratio))
        if n_drop <= 0:
            continue
        perm = torch.randperm(n_valid, generator=generator)[:n_drop]
        drop_pts = valid_idx[perm]
        sub[b, 0, drop_pts[:, 0], drop_pts[:, 1]] = 0.0
    return sub


def main():
    parser = argparse.ArgumentParser(
        description="LiDAR sparsity robustness ablation")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default="results/lidar_sparsity")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    # Build dataset once (same for all models)
    ref_ckpt = torch.load(MODELS["M"]["ckpt"], map_location="cpu",
                          weights_only=False)
    data_cfg = ref_ckpt.get("config", {}).get("data", {})
    val_ds = STFDehazeDataset(
        stf_root=data_cfg.get("stf_root", "data/stf/SeeingThroughFog"),
        timestamps_file=os.path.join(
            data_cfg.get("meta_dir", "data/stf/meta"), "val_timestamps.txt"),
        crop_size=tuple(data_cfg.get("crop_size", [256, 256])),
        beta_range=tuple(data_cfg.get("beta_range", [0.005, 0.04])),
        airlight_range=tuple(data_cfg.get("airlight_range", [0.7, 1.0])),
        max_depth=data_cfg.get("max_depth", 120.0),
        augment=False,
    )

    def _worker_init_fn(wid):
        np.random.seed(args.seed + wid)
        random.seed(args.seed + wid)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            worker_init_fn=_worker_init_fn)
    print(f"Dataset: {len(val_ds)} val samples\n")

    csv_path = os.path.join(args.out_dir, "sparsity_sweep.csv")
    rows = []

    for tag in args.models:
        info = MODELS[tag]
        print(f"{'='*60}")
        print(f"  {info['display']}")
        print(f"{'='*60}")
        model, _ = load_model(info["ckpt"], device)

        for kr in KEEP_RATIOS:
            gen = torch.Generator()
            gen.manual_seed(args.seed)

            all_psnr, all_ssim = [], []
            desc = f"  keep={kr:.0%}"
            with torch.no_grad():
                for batch in tqdm(val_loader, desc=desc, leave=False):
                    hazy = batch["hazy"].to(device)
                    clear = batch["clear"].to(device)
                    sparse = batch["sparse_depth"].to(device)
                    mask = batch["mask"].to(device)

                    sub_mask = subsample_mask(mask, kr, gen)
                    out = model(hazy, sparse * sub_mask, sub_mask)
                    restored = out["restored"]

                    for i in range(hazy.shape[0]):
                        all_psnr.append(psnr(restored[i:i+1], clear[i:i+1]))
                        all_ssim.append(ssim(restored[i:i+1], clear[i:i+1]))

            p_mean, p_std = np.mean(all_psnr), np.std(all_psnr)
            s_mean, s_std = np.mean(all_ssim), np.std(all_ssim)
            print(f"  keep={kr:.0%}:  PSNR {p_mean:.2f}±{p_std:.2f}  "
                  f"SSIM {s_mean:.4f}±{s_std:.4f}")
            rows.append({
                "model": info["display"],
                "keep_ratio": kr,
                "psnr_mean": round(p_mean, 2),
                "psnr_std": round(p_std, 2),
                "ssim_mean": round(s_mean, 4),
                "ssim_std": round(s_std, 4),
            })

        del model
        torch.cuda.empty_cache()

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
