#!/usr/bin/env python3
"""
No-reference IQA (NIQE + BRISQUE) on real fog and cross-dataset images.

Evaluates perceptual quality of restored images without ground truth.
Runs inference on:
  1. STF real fog frames (light_fog + dense_fog)
  2. O-HAZE / I-HAZE / NH-HAZE benchmarks

Lower NIQE / BRISQUE = better perceived quality.

Requires: pip install pyiqa

Usage:
    python eval_niqe_brisque.py
    python eval_niqe_brisque.py --datasets stf_fog ohaze
    python eval_niqe_brisque.py --models S M
"""

import argparse
import csv
import os
import sys
import random

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from model import get_model
from baselines import get_baseline_model
from dataset import STFDehazeDataset


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
    "AOD": {
        "display": "AOD-Net",
        "ckpt": "runs/baseline_stf_aod_net_rgb_0.00M_wclear+overcast_"
                "crop256_bs16_lr2e-04_ep200_0410_0520/best_psnr.pth",
    },
    "FFA": {
        "display": "FFA-Net",
        "ckpt": "runs/baseline_stf_ffa_net_rgb_0.77M_wclear+overcast_"
                "crop256_bs8_lr2e-04_ep200_0411_1402/best_psnr.pth",
    },
    "DHF": {
        "display": "DehazeFormer-T",
        "ckpt": "runs/baseline_stf_dehaze_former_rgb_1.36M_wclear+"
                "overcast_crop256_bs8_lr2e-04_ep200_0412_2235"
                "/best_psnr.pth",
    },
    "DEA": {
        "display": "DEA-Net",
        "ckpt": "runs/baseline_stf_dea_net_rgb_3.65M_wclear+overcast_"
                "crop256_bs8_lr2e-04_ep200_0416_0715/best_psnr.pth",
    },
}


def load_model(ckpt_path, device):
    """Load model from checkpoint, auto-detecting variant/baseline."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    baseline_name = model_cfg.get("baseline")

    if baseline_name is not None:
        mode = model_cfg.get("mode", "rgb")
        kwargs = {k: v for k, v in model_cfg.items()
                  if k not in ("baseline", "mode", "variant")}
        model = get_baseline_model(baseline_name, mode=mode, **kwargs)
    else:
        variant = model_cfg.get("variant", "lite")
        overrides = {"base_ch": model_cfg.get("base_ch", 32)}
        for k in ("use_cbam", "use_residual", "use_attention",
                  "use_physics_head", "lidar_drop_rate"):
            v = model_cfg.get(k)
            if v is not None:
                overrides[k] = v
        model = get_model(variant, **overrides)

    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    return model


# ── Dataset loaders ───────────────────────────────────────────────────────

def load_haze_pairs(data_dir, hazy_sub, gt_sub, prefix_fn, size=256):
    """Generic loader for *-HAZE datasets. Returns [(hazy_t, fname), ...]"""
    hazy_dir = os.path.join(data_dir, hazy_sub)
    pairs = []
    if not os.path.isdir(hazy_dir):
        return pairs
    for hf in sorted(os.listdir(hazy_dir)):
        if not hf.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(hazy_dir, hf)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0
                             ).permute(2, 0, 1)
        pairs.append((t, hf))
    return pairs


def load_ohaze(size=256):
    return load_haze_pairs("O-HAZY-NTIRE-2018", "hazy", "GT", None, size)

def load_ihaze(size=256):
    return load_haze_pairs("I-HAZE/# I-HAZY NTIRE 2018", "hazy", "GT",
                           None, size)

def load_nhhaze(size=256):
    return load_haze_pairs("NH-HAZE/NH-HAZE", "hazy", "GT", None, size)

def load_densehaze(size=256):
    return load_haze_pairs("Dense_Haze_NTIRE19", "hazy", "GT", None, size)


def load_stf_fog(stf_root="data/stf/SeeingThroughFog", n_samples=100,
                 crop_size=(256, 256), max_depth=120.0):
    """Load real fog frames from STF (no GT — raw foggy camera image + depth).
    Uses the raw camera image directly (not synthesised haze)."""
    # Use dataset just for sample discovery with weather filtering
    ds = STFDehazeDataset(
        stf_root=stf_root,
        timestamps_file="data/stf/meta/all_timestamps.txt",
        crop_size=None,
        beta_range=(0.01, 0.04),
        airlight_range=(0.7, 1.0),
        max_depth=max_depth,
        augment=False,
        weather_filter=["light_fog", "dense_fog"],
    )
    n = min(n_samples, len(ds.samples))
    print(f"  STF fog frames: {len(ds.samples)} total, using {n}")
    ch, cw = crop_size
    samples = []
    for i in range(n):
        sample = ds.samples[i]
        # Load raw camera image (this IS the real foggy image)
        raw_np = ds._load_stf_image(sample["cam_path"])
        # Load sparse depth
        depth_raw = np.load(sample["depth_path"])
        if isinstance(depth_raw, np.lib.npyio.NpzFile):
            depth_np = depth_raw["arr_0"].astype(np.float32)
        else:
            depth_np = depth_raw.astype(np.float32)
        # Centre crop
        h, w = raw_np.shape[:2]
        top = max(0, (h - ch) // 2)
        left = max(0, (w - cw) // 2)
        raw_np = raw_np[top:top+ch, left:left+cw]
        depth_np = depth_np[top:top+ch, left:left+cw]
        # Build tensors
        sparse = np.clip(depth_np / max_depth, 0.0, 1.0).astype(np.float32)
        mask = (depth_np > 0).astype(np.float32)
        hazy_t = torch.from_numpy(raw_np.transpose(2, 0, 1)).float()
        sparse_t = torch.from_numpy(sparse[None]).float()
        mask_t = torch.from_numpy(mask[None]).float()
        samples.append({
            "hazy": hazy_t,
            "sparse_depth": sparse_t,
            "mask": mask_t,
            "fname": f"stf_fog_{i:04d}",
        })
    return samples


# ── Main evaluation ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="No-reference IQA (NIQE + BRISQUE)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()))
    parser.add_argument("--datasets", nargs="+",
                        default=["stf_fog", "ohaze", "ihaze", "nhhaze", "densehaze"],
                        choices=["stf_fog", "ohaze", "ihaze", "nhhaze", "densehaze"])
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--n_fog_samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str,
                        default="results/no_ref_iqa")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_device = torch.device("cpu")  # pyiqa metrics use CPU to avoid CUDA runtime issues
    os.makedirs(args.out_dir, exist_ok=True)

    # Load pyiqa metrics (on CPU to avoid CUDA compilation errors)
    import pyiqa
    niqe_metric = pyiqa.create_metric("niqe", device=metric_device)
    brisque_metric = pyiqa.create_metric("brisque", device=metric_device)
    print("Loaded NIQE + BRISQUE metrics")

    all_rows = []

    for ds_name in args.datasets:
        print(f"\n{'#'*60}")
        print(f"  Dataset: {ds_name}")
        print(f"{'#'*60}")

        is_stf_fog = (ds_name == "stf_fog")

        # Load data
        if ds_name == "ohaze":
            pairs = load_ohaze(args.size)
        elif ds_name == "ihaze":
            pairs = load_ihaze(args.size)
        elif ds_name == "nhhaze":
            pairs = load_nhhaze(args.size)
        elif ds_name == "densehaze":
            pairs = load_densehaze(args.size)
        elif ds_name == "stf_fog":
            fog_samples = load_stf_fog(n_samples=args.n_fog_samples,
                                       crop_size=(args.size, args.size))
        else:
            continue

        # Evaluate hazy input first
        print(f"\n  Hazy input (baseline):")
        hazy_niqe, hazy_brisque = [], []
        if is_stf_fog:
            for s in fog_samples:
                inp = s["hazy"].unsqueeze(0).to(metric_device)
                hazy_niqe.append(niqe_metric(inp).item())
                hazy_brisque.append(brisque_metric(inp).item())
        else:
            for hazy_t, fname in pairs:
                inp = hazy_t.unsqueeze(0).to(metric_device)
                hazy_niqe.append(niqe_metric(inp).item())
                hazy_brisque.append(brisque_metric(inp).item())

        hn, hb = np.mean(hazy_niqe), np.mean(hazy_brisque)
        print(f"    NIQE: {hn:.2f}±{np.std(hazy_niqe):.2f}  "
              f"BRISQUE: {hb:.2f}±{np.std(hazy_brisque):.2f}")
        all_rows.append({
            "dataset": ds_name, "model": "Hazy input",
            "niqe_mean": round(hn, 2), "niqe_std": round(np.std(hazy_niqe), 2),
            "brisque_mean": round(hb, 2),
            "brisque_std": round(np.std(hazy_brisque), 2),
        })

        # Evaluate each model
        for tag in args.models:
            info = MODELS[tag]
            print(f"\n  {info['display']}:")
            model = load_model(info["ckpt"], device)

            model_niqe, model_brisque = [], []

            with torch.no_grad():
                if is_stf_fog:
                    for s in tqdm(fog_samples, desc=f"    {info['display']}",
                                  leave=False):
                        hazy = s["hazy"].unsqueeze(0).to(device)
                        sparse = s["sparse_depth"].unsqueeze(0).to(device)
                        mask = s["mask"].unsqueeze(0).to(device)
                        out = model(hazy, sparse, mask)
                        restored = out["restored"].clamp(0, 1).to(metric_device)
                        model_niqe.append(niqe_metric(restored).item())
                        model_brisque.append(brisque_metric(restored).item())
                else:
                    sz = args.size
                    zeros = torch.zeros(1, 1, sz, sz, device=device)
                    for hazy_t, fname in tqdm(pairs,
                                              desc=f"    {info['display']}",
                                              leave=False):
                        hazy = hazy_t.unsqueeze(0).to(device)
                        out = model(hazy, zeros, zeros)
                        restored = out["restored"].clamp(0, 1).to(metric_device)
                        model_niqe.append(niqe_metric(restored).item())
                        model_brisque.append(brisque_metric(restored).item())

            mn, mb = np.mean(model_niqe), np.mean(model_brisque)
            print(f"    NIQE: {mn:.2f}±{np.std(model_niqe):.2f}  "
                  f"BRISQUE: {mb:.2f}±{np.std(model_brisque):.2f}")
            all_rows.append({
                "dataset": ds_name, "model": info["display"],
                "niqe_mean": round(mn, 2),
                "niqe_std": round(np.std(model_niqe), 2),
                "brisque_mean": round(mb, 2),
                "brisque_std": round(np.std(model_brisque), 2),
            })

            del model
            torch.cuda.empty_cache()

    # Save CSV
    csv_path = os.path.join(args.out_dir, "niqe_brisque.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {csv_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Dataset':<12} {'Model':<18} {'NIQE↓':>10} {'BRISQUE↓':>12}")
    print(f"{'-'*70}")
    for r in all_rows:
        print(f"{r['dataset']:<12} {r['model']:<18} "
              f"{r['niqe_mean']:>6.2f}±{r['niqe_std']:<4.2f}"
              f"{r['brisque_mean']:>8.2f}±{r['brisque_std']:<4.2f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
