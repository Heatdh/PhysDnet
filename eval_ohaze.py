"""
Zero-shot O-HAZE / I-HAZE / NH-HAZE evaluation for DeL-PUNet and baselines.

O-HAZE  (Ancuti et al., NTIRE 2018): 45 real outdoor hazy/clean pairs.
I-HAZE  (Ancuti et al., NTIRE 2018): 30 real indoor hazy/clean pairs.
NH-HAZE (Ancuti et al., NTIRE 2020): 55 non-homogeneous hazy/clean pairs.

Since none provide LiDAR data, we feed zero depth and zero mask — this tests
the model's RGB-only dehazing capability.

All images are resized to 256×256 for a fair resolution-controlled comparison.

Usage:
    python eval_ohaze.py                                  # O-HAZE (default)
    python eval_ohaze.py --dataset ihaze                  # I-HAZE
    python eval_ohaze.py --dataset nhhaze                 # NH-HAZE
    python eval_ohaze.py --dataset ohaze ihaze nhhaze     # all three
    python eval_ohaze.py --models ours_v3b                # single model
    python eval_ohaze.py --save_images                    # save restored PNGs
"""

import argparse
import csv
import os
import random
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from model import get_model
from baselines import get_baseline_model


# ── Metrics ───────────────────────────────────────────────────────────────

def rgb_to_y(img: torch.Tensor) -> torch.Tensor:
    """Convert RGB (B,3,H,W) to Y channel of YCbCr (B,1,H,W).
    ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B"""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def rgb_to_ycbcr(img: torch.Tensor) -> torch.Tensor:
    """RGB [0,1] (B,3,H,W) -> YCbCr [0,1] (B,3,H,W). ITU-R BT.601."""
    r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    y  =  0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.169 * r - 0.331 * g + 0.500 * b + 0.5
    cr =  0.500 * r - 0.419 * g - 0.081 * b + 0.5
    return torch.cat([y, cb, cr], dim=1)


def ycbcr_to_rgb(img: torch.Tensor) -> torch.Tensor:
    """YCbCr [0,1] (B,3,H,W) -> RGB [0,1] (B,3,H,W)."""
    y, cb, cr = img[:, 0:1], img[:, 1:2], img[:, 2:3]
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344 * (cb - 0.5) - 0.714 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    return torch.clamp(torch.cat([r, g, b], dim=1), 0.0, 1.0)


def luminance_swap(restored: torch.Tensor, hazy: torch.Tensor) -> torch.Tensor:
    """Keep Y (luminance) from restored, Cb/Cr (chrominance) from hazy."""
    ycbcr_res = rgb_to_ycbcr(restored)
    ycbcr_haz = rgb_to_ycbcr(hazy)
    merged = torch.cat([ycbcr_res[:, 0:1], ycbcr_haz[:, 1:2], ycbcr_haz[:, 2:3]], dim=1)
    return ycbcr_to_rgb(merged)


def calc_psnr(pred: torch.Tensor, target: torch.Tensor,
              max_val: float = 1.0) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def calc_ssim(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11,
              C1: float = 0.01 ** 2, C2: float = 0.03 ** 2) -> float:
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device)
    kernel /= kernel.sum()
    pad = window_size // 2
    channels = pred.shape[1]
    ssim_vals = []
    for c in range(channels):
        p = pred[:, c:c+1, :, :]
        t = target[:, c:c+1, :, :]
        mu_p = torch.nn.functional.conv2d(p, kernel, padding=pad)
        mu_t = torch.nn.functional.conv2d(t, kernel, padding=pad)
        mu_p_sq = mu_p ** 2
        mu_t_sq = mu_t ** 2
        mu_pt = mu_p * mu_t
        sigma_p_sq = torch.nn.functional.conv2d(p * p, kernel, padding=pad) - mu_p_sq
        sigma_t_sq = torch.nn.functional.conv2d(t * t, kernel, padding=pad) - mu_t_sq
        sigma_pt = torch.nn.functional.conv2d(p * t, kernel, padding=pad) - mu_pt
        num = (2 * mu_pt + C1) * (2 * sigma_pt + C2)
        den = (mu_p_sq + mu_t_sq + C1) * (sigma_p_sq + sigma_t_sq + C2)
        ssim_map = num / (den + 1e-8)
        ssim_vals.append(ssim_map.mean().item())
    return float(np.mean(ssim_vals))


# ── Model registry ────────────────────────────────────────────────────────

# model_tag → (loader_fn, display_name)
# loader_fn returns (model, device)

CKPT_DIR = "runs"

MODEL_DEFS = {
    # ── Our models ──
    "ours_v3s": {
        "display": "PhysDNet-S (ours)",
        "ckpt": os.path.join(CKPT_DIR,
            "stf_robust_ch32_0.8M_wclear+overcast_crop256_b0.005-0.04_bs32_lr2e-04_ep500_v2_fft+ctr@185_0422_2234",
            "best_psnr.pth"),
    },
    "ours_v3b": {
        "display": "PhysDNet-M (ours)",
        "ckpt": os.path.join(CKPT_DIR,
            "stf_robust_ch64_3.0M_wclear+overcast_crop256_b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185_0406_2105",
            "best_psnr.pth"),
    },
    "ours_v3c": {
        "display": "PhysDNet-L (ours)",
        "ckpt": os.path.join(CKPT_DIR,
            "stf_robust_ch96_6.6M_wclear+overcast_crop256_b0.005-0.04_bs8_lr2e-04_ep500_v2_fft+ctr@185_0407_2326",
            "best_psnr.pth"),
    },
    # ── RGB-only baselines (fair: no depth info either) ──
    "aodnet_rgb": {
        "display": "AOD-Net (RGB)",
        "ckpt": os.path.join(CKPT_DIR,
            "baseline_stf_aod_net_rgb_0.00M_wclear+overcast_crop256_bs16_lr2e-04_ep200_0410_0520",
            "best_psnr.pth"),
    },
    "ffanet_rgb": {
        "display": "FFA-Net (RGB)",
        "ckpt": os.path.join(CKPT_DIR,
            "baseline_stf_ffa_net_rgb_0.77M_wclear+overcast_crop256_bs8_lr2e-04_ep200_0411_1402",
            "best_psnr.pth"),
    },
    "dehazeformer_rgb": {
        "display": "DehazeFormer-T (RGB)",
        "ckpt": os.path.join(CKPT_DIR,
            "baseline_stf_dehaze_former_rgb_1.36M_wclear+overcast_crop256_bs8_lr2e-04_ep200_0412_2235",
            "best_psnr.pth"),
    },
    "deanet_rgb": {
        "display": "DEA-Net (RGB)",
        "ckpt": os.path.join(CKPT_DIR,
            "baseline_stf_dea_net_rgb_3.65M_wclear+overcast_crop256_bs8_lr2e-04_ep200_0416_0715",
            "best_psnr.pth"),
    },
    # ── Ablation ──
    "nophy": {
        "display": "NoPhy (ablation)",
        "ckpt": os.path.join(CKPT_DIR,
            "stf_robust_ch64_2.9M_wclear+overcast_crop256_b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185+nophy_0413_1008",
            "best_psnr.pth"),
    },
}


def load_model(tag: str, device: torch.device):
    """Load a model from its checkpoint, return (model, display_name)."""
    info = MODEL_DEFS[tag]
    ckpt_path = info["ckpt"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    baseline_name = model_cfg.get("baseline")
    if baseline_name is not None:
        mode = model_cfg.get("mode", "rgb")
        model_kwargs = {k: v for k, v in model_cfg.items()
                        if k not in ("baseline", "mode", "variant")}
        model = get_baseline_model(baseline_name, mode=mode, **model_kwargs)
    else:
        variant = model_cfg.get("variant", "lite")
        overrides = {"base_ch": model_cfg.get("base_ch", 32)}
        for k in ("use_cbam", "use_residual", "use_physics_head",
                  "lidar_drop_rate"):
            v = model_cfg.get(k)
            if v is not None:
                overrides[k] = v
        model = get_model(variant, **overrides)

    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Loaded {info['display']} — {n_params/1e6:.2f}M params "
          f"(epoch {ckpt.get('epoch', '?')})")
    return model, info["display"]


# ── O-HAZE data loading ──────────────────────────────────────────────────

def load_ohaze(data_dir: str, size: int = 256):
    """Load O-HAZE pairs as list of (hazy_tensor, gt_tensor, filename)."""
    hazy_dir = os.path.join(data_dir, "hazy")
    gt_dir = os.path.join(data_dir, "GT")

    # Build mapping: prefix → (hazy_path, gt_path)
    hazy_files = sorted(os.listdir(hazy_dir))
    pairs = []
    for hf in hazy_files:
        prefix = hf.split("_outdoor_")[0]  # e.g. "01"
        # Find matching GT (case-insensitive extension)
        gt_candidates = [f for f in os.listdir(gt_dir)
                         if f.lower().startswith(prefix + "_outdoor_gt")]
        if not gt_candidates:
            print(f"  Warning: no GT for {hf}, skipping")
            continue
        gf = gt_candidates[0]

        hazy_img = Image.open(os.path.join(hazy_dir, hf)).convert("RGB")
        gt_img = Image.open(os.path.join(gt_dir, gf)).convert("RGB")

        # Resize to target size
        hazy_img = hazy_img.resize((size, size), Image.LANCZOS)
        gt_img = gt_img.resize((size, size), Image.LANCZOS)

        hazy_t = torch.from_numpy(
            np.array(hazy_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)  # 3×H×W
        gt_t = torch.from_numpy(
            np.array(gt_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        pairs.append((hazy_t, gt_t, hf))

    print(f"Loaded {len(pairs)} O-HAZE image pairs (resized to {size}×{size})")
    return pairs


def load_ihaze(data_dir: str, size: int = 256):
    """Load I-HAZE pairs as list of (hazy_tensor, gt_tensor, filename)."""
    hazy_dir = os.path.join(data_dir, "hazy")
    gt_dir = os.path.join(data_dir, "GT")

    hazy_files = sorted([f for f in os.listdir(hazy_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    pairs = []
    for hf in hazy_files:
        prefix = hf.split("_indoor_")[0]  # e.g. "01"
        gt_candidates = [f for f in os.listdir(gt_dir)
                         if f.lower().startswith(prefix + "_indoor_gt")]
        if not gt_candidates:
            print(f"  Warning: no GT for {hf}, skipping")
            continue
        gf = gt_candidates[0]

        hazy_img = Image.open(os.path.join(hazy_dir, hf)).convert("RGB")
        gt_img = Image.open(os.path.join(gt_dir, gf)).convert("RGB")
        hazy_img = hazy_img.resize((size, size), Image.LANCZOS)
        gt_img = gt_img.resize((size, size), Image.LANCZOS)

        hazy_t = torch.from_numpy(
            np.array(hazy_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        gt_t = torch.from_numpy(
            np.array(gt_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        pairs.append((hazy_t, gt_t, hf))

    print(f"Loaded {len(pairs)} I-HAZE image pairs (resized to {size}×{size})")
    return pairs


def load_nhhaze(data_dir: str, size: int = 256):
    """Load NH-HAZE pairs as list of (hazy_tensor, gt_tensor, filename).
    NH-HAZE has a flat directory with NN_GT.png and NN_hazy.png."""
    all_files = sorted(os.listdir(data_dir))
    hazy_files = [f for f in all_files if "_hazy" in f.lower()
                  and f.lower().endswith((".jpg", ".jpeg", ".png"))]
    pairs = []
    for hf in hazy_files:
        prefix = hf.split("_hazy")[0]  # e.g. "01"
        gt_candidates = [f for f in all_files
                         if f.startswith(prefix + "_GT")
                         and f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not gt_candidates:
            print(f"  Warning: no GT for {hf}, skipping")
            continue
        gf = gt_candidates[0]

        hazy_img = Image.open(os.path.join(data_dir, hf)).convert("RGB")
        gt_img = Image.open(os.path.join(data_dir, gf)).convert("RGB")
        hazy_img = hazy_img.resize((size, size), Image.LANCZOS)
        gt_img = gt_img.resize((size, size), Image.LANCZOS)

        hazy_t = torch.from_numpy(
            np.array(hazy_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        gt_t = torch.from_numpy(
            np.array(gt_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        pairs.append((hazy_t, gt_t, hf))

    print(f"Loaded {len(pairs)} NH-HAZE image pairs (resized to {size}×{size})")
    return pairs


def load_densehaze(data_dir: str, size: int = 256):
    """Load Dense-Haze pairs as list of (hazy_tensor, gt_tensor, filename).
    Dense-Haze (Ancuti et al., NTIRE 2019): 55 dense outdoor hazy/clean pairs.
    Layout: hazy/NN_hazy.png, GT/NN_GT.png"""
    hazy_dir = os.path.join(data_dir, "hazy")
    gt_dir = os.path.join(data_dir, "GT")

    hazy_files = sorted([f for f in os.listdir(hazy_dir)
                         if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    pairs = []
    for hf in hazy_files:
        prefix = hf.split("_hazy")[0]  # e.g. "01"
        gt_candidates = [f for f in os.listdir(gt_dir)
                         if f.startswith(prefix + "_GT")
                         and f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not gt_candidates:
            print(f"  Warning: no GT for {hf}, skipping")
            continue
        gf = gt_candidates[0]

        hazy_img = Image.open(os.path.join(hazy_dir, hf)).convert("RGB")
        gt_img = Image.open(os.path.join(gt_dir, gf)).convert("RGB")
        hazy_img = hazy_img.resize((size, size), Image.LANCZOS)
        gt_img = gt_img.resize((size, size), Image.LANCZOS)

        hazy_t = torch.from_numpy(
            np.array(hazy_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        gt_t = torch.from_numpy(
            np.array(gt_img).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        pairs.append((hazy_t, gt_t, hf))

    print(f"Loaded {len(pairs)} Dense-Haze image pairs (resized to {size}×{size})")
    return pairs


# Dataset registry
DATASET_LOADERS = {
    "ohaze": {
        "loader": load_ohaze,
        "default_dir": "O-HAZY-NTIRE-2018",
        "label": "O-HAZE",
        "cite": "Ancuti et al. 2018",
    },
    "ihaze": {
        "loader": load_ihaze,
        "default_dir": "I-HAZE/# I-HAZY NTIRE 2018",
        "label": "I-HAZE",
        "cite": "Ancuti et al. 2018",
    },
    "nhhaze": {
        "loader": load_nhhaze,
        "default_dir": "NH-HAZE/NH-HAZE",
        "label": "NH-HAZE",
        "cite": "Ancuti et al. 2020",
    },
    "densehaze": {
        "loader": load_densehaze,
        "default_dir": "Dense_Haze_NTIRE19",
        "label": "Dense-Haze",
        "cite": "Ancuti et al. 2019",
    },
}


# ── Main evaluation ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot haze benchmark evaluation (no LiDAR)")
    parser.add_argument("--dataset", type=str, nargs="+",
                        default=["ohaze"],
                        choices=["ohaze", "ihaze", "nhhaze", "densehaze"],
                        help="Which dataset(s) to evaluate on")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Override data directory (auto-detected per dataset)")
    parser.add_argument("--size", type=int, default=256,
                        help="Resize images to this square resolution")
    parser.add_argument("--models", type=str, nargs="+",
                        default=list(MODEL_DEFS.keys()),
                        choices=list(MODEL_DEFS.keys()),
                        help="Which models to evaluate")
    parser.add_argument("--save_images", action="store_true",
                        help="Save restored images")
    parser.add_argument("--color_transfer", action="store_true",
                        help="Luminance swap: Y from restored, Cb/Cr from hazy")
    parser.add_argument("--out_dir", type=str,
                        default="results",
                        help="Root output directory for results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for ds_name in args.dataset:
        ds_info = DATASET_LOADERS[ds_name]
        data_dir = args.data_dir or ds_info["default_dir"]
        out_dir = os.path.join(args.out_dir, f"{ds_name}_eval")
        os.makedirs(out_dir, exist_ok=True)

        print(f"\n{'#'*70}")
        print(f"  Dataset: {ds_info['label']} ({ds_info['cite']})")
        print(f"  Data dir: {data_dir}")
        print(f"{'#'*70}")

        pairs = ds_info["loader"](data_dir, size=args.size)
        if not pairs:
            print(f"ERROR: No image pairs found for {ds_name}!")
            continue

        summary_rows = []

        for tag in args.models:
            print(f"\n{'='*60}")
            print(f"Evaluating: {tag}")
            model, display_name = load_model(tag, device)

            psnrs, ssims = [], []
            psnrs_y, ssims_y = [], []
            per_image = []

            if args.save_images:
                img_dir = os.path.join(out_dir, tag, "images")
                os.makedirs(img_dir, exist_ok=True)

            with torch.no_grad():
                for hazy_t, gt_t, fname in tqdm(pairs, desc=display_name):
                    hazy = hazy_t.unsqueeze(0).to(device)
                    gt = gt_t.unsqueeze(0).to(device)
                    zeros = torch.zeros(1, 1, args.size, args.size, device=device)

                    out = model(hazy, zeros, zeros)
                    restored = out["restored"]

                    if args.color_transfer:
                        restored = luminance_swap(restored, hazy)

                    p = calc_psnr(restored, gt)
                    s = calc_ssim(restored, gt)
                    psnrs.append(p)
                    ssims.append(s)
                    y_pred = rgb_to_y(restored)
                    y_gt = rgb_to_y(gt)
                    py = calc_psnr(y_pred, y_gt)
                    sy = calc_ssim(y_pred, y_gt)
                    psnrs_y.append(py)
                    ssims_y.append(sy)
                    per_image.append((fname, p, s, py, sy))

                    if args.save_images:
                        arr = (restored[0].cpu().numpy().transpose(1, 2, 0) * 255
                               ).clip(0, 255).astype(np.uint8)
                        out_name = os.path.splitext(fname)[0] + "_restored.png"
                        Image.fromarray(arr).save(os.path.join(img_dir, out_name))

            avg_p = np.mean(psnrs)
            avg_s = np.mean(ssims)
            std_p = np.std(psnrs)
            std_s = np.std(ssims)
            avg_py = np.mean(psnrs_y)
            avg_sy = np.mean(ssims_y)
            std_py = np.std(psnrs_y)
            std_sy = np.std(ssims_y)

            print(f"  RGB  — PSNR: {avg_p:.2f} ±{std_p:.2f} dB  |  SSIM: {avg_s:.4f} ±{std_s:.4f}")
            print(f"  Y-ch — PSNR: {avg_py:.2f} ±{std_py:.2f} dB  |  SSIM: {avg_sy:.4f} ±{std_sy:.4f}")

            summary_rows.append({
                "model": display_name, "tag": tag,
                "psnr_mean": avg_p, "psnr_std": std_p,
                "ssim_mean": avg_s, "ssim_std": std_s,
                "psnr_y_mean": avg_py, "psnr_y_std": std_py,
                "ssim_y_mean": avg_sy, "ssim_y_std": std_sy,
            })

            csv_path = os.path.join(out_dir, f"{tag}_per_image.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["filename", "psnr_rgb", "ssim_rgb", "psnr_y", "ssim_y"])
                for fname, p, s, py, sy in per_image:
                    w.writerow([fname, f"{p:.4f}", f"{s:.6f}",
                                f"{py:.4f}", f"{sy:.6f}"])

            del model
            torch.cuda.empty_cache()

        # ── Summary table ──
        ds_label = ds_info["label"]
        print(f"\n{'='*76}")
        print(f"{ds_label} Zero-Shot Results @ {args.size}×{args.size}")
        print(f"{'='*76}")
        print(f"{'Method':<28} {'PSNR-Y (dB)':>14} {'SSIM-Y':>12} {'PSNR-RGB':>12} {'SSIM-RGB':>10}")
        print(f"{'-'*76}")
        for row in summary_rows:
            print(f"{row['model']:<28} "
                  f"{row['psnr_y_mean']:>8.2f} ±{row['psnr_y_std']:.2f} "
                  f"{row['ssim_y_mean']:>8.4f} ±{row['ssim_y_std']:.4f} "
                  f"{row['psnr_mean']:>7.2f}  "
                  f"{row['ssim_mean']:>7.4f}")
        print(f"{'='*76}")

        # Summary CSV
        csv_path = os.path.join(out_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "tag",
                         "psnr_y_mean", "psnr_y_std", "ssim_y_mean", "ssim_y_std",
                         "psnr_rgb_mean", "psnr_rgb_std", "ssim_rgb_mean", "ssim_rgb_std"])
            for row in summary_rows:
                w.writerow([row["model"], row["tag"],
                            f"{row['psnr_y_mean']:.4f}", f"{row['psnr_y_std']:.4f}",
                            f"{row['ssim_y_mean']:.6f}", f"{row['ssim_y_std']:.6f}",
                            f"{row['psnr_mean']:.4f}", f"{row['psnr_std']:.4f}",
                            f"{row['ssim_mean']:.6f}", f"{row['ssim_std']:.6f}"])
        print(f"\nSummary saved to {csv_path}")


if __name__ == "__main__":
    main()
