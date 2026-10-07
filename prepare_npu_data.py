#!/usr/bin/env python3
"""
Prepare Hailo calibration set and NPU eval set from real STF data.

Outputs (NCHW float32 numpy arrays):
  npu/calib/
      rgb_hazy.npy       (N_calib, 3, 256, 256)
      sparse_depth.npy   (N_calib, 1, 256, 256)
      mask.npy           (N_calib, 1, 256, 256)
  npu/eval/
      hazy.npy           (N_eval, 3, 256, 256)
      sparse_depth.npy   (N_eval, 1, 256, 256)
      mask.npy           (N_eval, 1, 256, 256)
      clear.npy          (N_eval, 3, 256, 256)   ← ground truth
      meta.npy           structured array with beta per sample

Run from dehaze_model/:
    conda run -n modelzoo python prepare_npu_data.py

Hailo optimizer usage (after generation):
    hailo optimize physdnet_s_ch32.har \\
        --calib-path ../shared_with_docker/edge_deh/npu/calib \\
        --model-script modelscript.alls
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset import STFDehazeDataset

# ── defaults ───────────────────────────────────────────────────────────────
STF_ROOT   = "data/stf/SeeingThroughFog"
META_DIR   = "data/stf/meta"
VAL_TS     = "data/stf/meta/val_timestamps.txt"
OUT_DIR    = "../shared_with_docker/edge_deh/npu"
CROP       = 256
N_CALIB    = 64   # Hailo modelscript calibset_size=64
N_EVAL     = 200
SEED       = 42


def collect_samples(ds, indices, seed, beta_value=0.0):
    """Return NCHW numpy arrays for given dataset indices (same crop per call)."""
    hazy_list, depth_list, mask_list, clear_list = [], [], [], []
    for idx in indices:
        # Pin RNG so crop is deterministic per idx
        torch.manual_seed(seed + idx)
        np.random.seed(seed + idx)
        sample = ds[idx]
        hazy_list.append(sample["hazy"].numpy())           # (3,H,W)
        depth_list.append(sample["sparse_depth"].numpy())  # (1,H,W)
        mask_list.append(sample["mask"].numpy())            # (1,H,W)
        clear_list.append(sample["clear"].numpy())          # (3,H,W)

    n = len(indices)
    return (
        np.stack(hazy_list).astype(np.float32),            # (N,3,H,W)
        np.stack(depth_list).astype(np.float32),           # (N,1,H,W)
        np.stack(mask_list).astype(np.float32),            # (N,1,H,W)
        np.stack(clear_list).astype(np.float32),           # (N,3,H,W)
        np.full(n, beta_value, dtype=np.float32),          # (N,)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",  default=OUT_DIR)
    parser.add_argument("--n_calib",  type=int, default=N_CALIB)
    parser.add_argument("--n_eval",   type=int, default=N_EVAL)
    parser.add_argument("--crop",     type=int, default=CROP)
    parser.add_argument("--seed",     type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    calib_dir = Path(args.out_dir) / "calib"
    eval_dir  = Path(args.out_dir) / "eval"
    calib_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # ── calibration dataset: uniform β in training range ──────────────────
    # Use fixed β=0.02 (mid-range) to keep calibration data stable.
    # The Hailo quantizer just needs representative activations — not a sweep.
    print(f"\n[calib] Building dataset (β=0.02 fixed, {args.n_calib} samples)...")
    ds_calib = STFDehazeDataset(
        stf_root=STF_ROOT,
        timestamps_file=VAL_TS,
        crop_size=(args.crop, args.crop),
        beta_range=(0.02, 0.02),
        airlight_range=(0.8, 0.8),
        max_depth=120.0,
        augment=False,
    )
    n_total = len(ds_calib)
    # Evenly spaced indices across the val set → good scene diversity
    calib_indices = np.linspace(0, n_total - 1, args.n_calib, dtype=int)
    hazy, depth, mask, clear, betas = collect_samples(ds_calib, calib_indices, args.seed, beta_value=0.02)

    np.save(calib_dir / "rgb_hazy.npy",     hazy)
    np.save(calib_dir / "sparse_depth.npy", depth)
    np.save(calib_dir / "mask.npy",         mask)
    print(f"[calib] Saved {args.n_calib} samples → {calib_dir}")
    print(f"        rgb_hazy:     {hazy.shape}  [{hazy.min():.3f}, {hazy.max():.3f}]")
    print(f"        sparse_depth: {depth.shape} [{depth.min():.3f}, {depth.max():.3f}]")
    print(f"        mask:         {mask.shape}  [{mask.min():.3f}, {mask.max():.3f}]")

    # ── eval dataset: sample across β range including OOD ─────────────────
    # β values: 0.005 0.01 0.02 0.04 0.06 0.08 0.10 (same as beta sweep)
    eval_betas = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.10]
    n_per_beta = args.n_eval // len(eval_betas)
    remainder  = args.n_eval - n_per_beta * len(eval_betas)

    print(f"\n[eval] Building eval set ({args.n_eval} samples across β={eval_betas})...")
    all_hazy, all_depth, all_mask, all_clear, all_betas = [], [], [], [], []

    for i, beta in enumerate(eval_betas):
        n = n_per_beta + (1 if i < remainder else 0)
        ds_eval = STFDehazeDataset(
            stf_root=STF_ROOT,
            timestamps_file=VAL_TS,
            crop_size=(args.crop, args.crop),
            beta_range=(beta, beta),
            airlight_range=(0.8, 0.8),
            max_depth=120.0,
            augment=False,
        )
        # Pick different indices than calib, spread across val set
        offset = 17 + i * 37  # prime offsets for non-overlapping coverage
        indices = (np.linspace(offset, offset + n_total - 1, n, dtype=int)) % n_total
        h, d, m, c, b = collect_samples(ds_eval, indices, args.seed + i * 1000, beta_value=beta)
        all_hazy.append(h); all_depth.append(d); all_mask.append(m)
        all_clear.append(c); all_betas.append(b)
        print(f"  β={beta:.3f}: {n} samples")

    hazy_eval  = np.concatenate(all_hazy,  axis=0)
    depth_eval = np.concatenate(all_depth, axis=0)
    mask_eval  = np.concatenate(all_mask,  axis=0)
    clear_eval = np.concatenate(all_clear, axis=0)
    betas_eval = np.concatenate(all_betas, axis=0)

    np.save(eval_dir / "hazy.npy",         hazy_eval)
    np.save(eval_dir / "sparse_depth.npy", depth_eval)
    np.save(eval_dir / "mask.npy",         mask_eval)
    np.save(eval_dir / "clear.npy",        clear_eval)
    np.save(eval_dir / "betas.npy",        betas_eval)
    print(f"\n[eval] Saved {len(hazy_eval)} samples → {eval_dir}")
    print(f"       hazy:  {hazy_eval.shape}")
    print(f"       clear: {clear_eval.shape}")

    print("\nDone.")
    print(f"\nHailo optimizer command:")
    print(f"  hailo optimize physdnet_s_ch32.har \\")
    print(f"      --calib-path npu/calib \\")
    print(f"      --model-script modelscript.alls")


if __name__ == "__main__":
    main()
