#!/usr/bin/env python3
"""
Batch inference on real foggy images (unsupervised — no ground truth).

Loads the trained model and runs it on STF frames filtered to foggy weather
conditions (light_fog, dense_fog).  Outputs side-by-side comparison figures
and optionally individual dehazed images.

Usage
-----
    # Default: use robust checkpoint, infer on fog frames
    python infer_real_fog.py \
        --checkpoint runs/<robust_run>/best.pth \
        --stf_root data/stf/SeeingThroughFog \
        --n_samples 20

    # Include all weather categories (not just fog)
    python infer_real_fog.py \
        --checkpoint runs/<robust_run>/best.pth \
        --stf_root data/stf/SeeingThroughFog \
        --weather all

    # Save individual dehazed images as PNG
    python infer_real_fog.py \
        --checkpoint runs/<robust_run>/best.pth \
        --save_individual

    # Custom crop size & output directory
    python infer_real_fog.py \
        --checkpoint runs/<robust_run>/best.pth \
        --crop_size 768 768 --out_dir results/fog_eval
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import get_model, MODEL_VARIANTS
from dataset import STFDehazeDataset
from baselines import get_baseline_model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(ckpt_path: str, device: torch.device):
    """Load trained model from checkpoint, auto-detecting variant."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", {})
    variant = model_cfg.get("variant", "lite")

    # Check if this is a baseline checkpoint
    baseline_name = model_cfg.get("baseline")
    if baseline_name is not None:
        mode = model_cfg.get("mode", "rgb")
        model_kwargs = {k: v for k, v in model_cfg.items()
                        if k not in ("baseline", "mode", "variant")}
        model = get_baseline_model(baseline_name, mode=mode, **model_kwargs).to(device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded baseline {baseline_name} ({mode}) from {ckpt_path}  "
              f"({n_params/1e6:.2f}M params)")
        return model, f"baseline_{baseline_name}"

    overrides = {"base_ch": model_cfg.get("base_ch", 32)}
    for k in ("use_cbam", "use_residual"):
        v = model_cfg.get(k)
        if v is not None:
            overrides[k] = v

    model = get_model(variant, **overrides).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {variant} model from {ckpt_path}  ({n_params/1e6:.2f}M params)")
    return model, variant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numpy(t: torch.Tensor) -> np.ndarray:
    """C×H×W tensor → H×W×C or H×W numpy clipped to [0,1]."""
    img = t.detach().cpu().numpy()
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = img.transpose(1, 2, 0)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img.squeeze(-1)
    return np.clip(img, 0.0, 1.0)


def colorize_depth(depth_t: torch.Tensor, mask_t: torch.Tensor) -> np.ndarray:
    """Sparse depth (1×H×W) → H×W×3 turbo colormap."""
    d = depth_t.squeeze(0).numpy()
    m = mask_t.squeeze(0).numpy().astype(bool)
    valid = d[m]
    if len(valid) == 0:
        return np.zeros((*d.shape, 3), dtype=np.float32)
    d_vis = np.zeros_like(d)
    d_vis[m] = (d[m] - valid.min()) / (valid.max() - valid.min() + 1e-8)
    rgba = plt.cm.turbo(d_vis)[:, :, :3].astype(np.float32)
    rgba[~m] = 0.0
    return rgba


def colorize_transmission(t: torch.Tensor) -> np.ndarray:
    """1×H×W transmission → H×W×3 inferno colormap."""
    return plt.cm.inferno(t.squeeze(0).numpy())[:, :, :3].astype(np.float32)


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_comparison_figure(
    raw: np.ndarray,
    dehazed: np.ndarray,
    physics: np.ndarray,
    depth_rgb: np.ndarray,
    trans_pred: np.ndarray,
    airlight: np.ndarray,
    title: str = "",
) -> plt.Figure:
    """
    2×3 comparison figure for unsupervised inference.

    Row 1: Raw input  |  Dehazed (direct)  |  Physics restored
    Row 2: LiDAR depth |  Transmission map  |  Estimated airlight
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(title, fontsize=14, y=0.99)

    axes[0, 0].imshow(raw);         axes[0, 0].set_title("Raw Input (foggy)", fontsize=11)
    axes[0, 1].imshow(dehazed);     axes[0, 1].set_title("Dehazed (direct head)", fontsize=11)
    axes[0, 2].imshow(physics);     axes[0, 2].set_title("Physics Restored", fontsize=11)

    axes[1, 0].imshow(depth_rgb);   axes[1, 0].set_title("LiDAR Depth (sparse)", fontsize=11)
    axes[1, 1].imshow(trans_pred);  axes[1, 1].set_title("Transmission (predicted)", fontsize=11)

    # Airlight info panel
    a = airlight
    info = (
        f"Estimated Airlight\n"
        f"───────────────────\n"
        f"R: {a[0]:.3f}\n"
        f"G: {a[1]:.3f}\n"
        f"B: {a[2]:.3f}\n\n"
        f"Mean: {a.mean():.3f}\n\n"
        f"Higher = more atmospheric\n"
        f"light (thicker fog)"
    )
    color_patch = np.ones((50, 50, 3), dtype=np.float32) * a[None, None, :]
    axes[1, 2].imshow(np.clip(color_patch, 0, 1), extent=[0.3, 0.7, 0.15, 0.35])
    axes[1, 2].text(0.5, 0.72, info, transform=axes[1, 2].transAxes,
                    ha="center", va="center", fontsize=10, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.4", fc="#f5f5f5", ec="#aaa"))
    axes[1, 2].set_title("Airlight Estimate", fontsize=11)
    axes[1, 2].set_xlim(0, 1); axes[1, 2].set_ylim(0, 1)

    for row in axes:
        for ax in row:
            ax.axis("off")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer_sample(model, raw_np, depth_np, crop_size, max_depth, device):
    """Run model on a single raw image + depth pair."""
    # Centre crop
    if crop_size is not None:
        ch, cw = crop_size
        h, w = raw_np.shape[:2]
        top = max(0, (h - ch) // 2)
        left = max(0, (w - cw) // 2)
        raw_np = raw_np[top:top+ch, left:left+cw]
        depth_np = depth_np[top:top+ch, left:left+cw]

    # Normalise depth, build mask
    sparse = np.clip(depth_np / max_depth, 0.0, 1.0).astype(np.float32)
    mask = (depth_np > 0).astype(np.float32)

    # To tensors (add batch dim)
    raw_t = torch.from_numpy(raw_np.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    sparse_t = torch.from_numpy(sparse[None]).float().unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask[None]).float().unsqueeze(0).to(device)

    out = model(raw_t, sparse_t, mask_t)

    # Back to CPU, remove batch dim
    result = {k: v.squeeze(0).cpu() for k, v in out.items()}
    result["raw_np"] = raw_np
    result["sparse_t"] = torch.from_numpy(sparse[None]).float()
    result["mask_t"] = torch.from_numpy(mask[None]).float()
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Inference on real foggy STF images (unsupervised)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--stf_root", type=str,
                   default="data/stf/SeeingThroughFog",
                   help="Root of the STF dataset")
    p.add_argument("--out_dir", type=str, default=None,
                   help="Output directory (default: <checkpoint_run_dir>/real_fog_inference/)")
    p.add_argument("--n_samples", type=int, default=20,
                   help="Number of samples to process")
    p.add_argument("--weather", type=str, nargs="+",
                   default=["light_fog", "dense_fog"],
                   help="Weather categories to select "
                        "(e.g. light_fog dense_fog rain snow; "
                        "use 'all' for everything)")
    p.add_argument("--crop_size", type=int, nargs=2, default=[512, 512],
                   help="Centre crop size (H W)")
    p.add_argument("--max_depth", type=float, default=120.0)
    p.add_argument("--save_individual", action="store_true",
                   help="Also save dehazed & physics images as individual PNGs")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    model, variant = load_model(args.checkpoint, device)

    # ---- Resolve output directory (default: inside the run dir) ----
    if args.out_dir:
        out_path = Path(args.out_dir)
    else:
        # Derive from checkpoint: runs/<run_name>/best.pth → runs/<run_name>/real_fog_inference/
        ckpt_dir = Path(args.checkpoint).resolve().parent
        weather_tag = "_".join(args.weather)
        out_path = ckpt_dir / f"real_fog_inference_{weather_tag}"
    
    # ---- Build sample list with weather filtering ----
    weather_filter = None if "all" in args.weather else args.weather

    # Use STFDehazeDataset just for sample discovery
    ds = STFDehazeDataset(
        stf_root=args.stf_root,
        crop_size=None,
        augment=False,
        weather_filter=weather_filter,
    )
    samples = ds.samples
    print(f"Found {len(samples)} frames matching weather={args.weather}")

    if len(samples) == 0:
        print("No samples found! Check --stf_root and --weather filter.")
        return

    # ---- Select subset ----
    np.random.seed(args.seed)
    n = min(args.n_samples, len(samples))
    indices = np.random.choice(len(samples), size=n, replace=False)
    indices.sort()

    # ---- Output base dir (no subdirs yet — will create per weather category) ----
    out_path.mkdir(parents=True, exist_ok=True)

    crop = tuple(args.crop_size)

    print(f"\nRunning inference on {n} foggy frames → {out_path}/")
    print(f"  Model: {variant} | Crop: {crop} | Max depth: {args.max_depth}")
    print(f"  Weather filter: {args.weather}")
    print(f"  Results will be organized by weather category:")
    print()

    # ---- Inference loop (organize by weather category) ----
    for i, idx in enumerate(indices):
        sample = samples[int(idx)]
        ts = sample["timestamp"]

        # Load raw image
        raw_np = STFDehazeDataset._load_stf_image(sample["cam_path"])

        # Load depth
        depth_raw = np.load(sample["depth_path"])
        if isinstance(depth_raw, np.lib.npyio.NpzFile):
            depth_np = depth_raw["arr_0"].astype(np.float32)
        else:
            depth_np = depth_raw.astype(np.float32)

        # Run inference
        result = infer_sample(model, raw_np, depth_np, crop, args.max_depth, device)

        # Extract outputs
        raw_vis = result["raw_np"]
        dehazed = to_numpy(result["restored"])
        depth_vis = colorize_depth(result["sparse_t"], result["mask_t"])

        has_physics = "physics_restored" in result
        if has_physics:
            physics = to_numpy(result["physics_restored"])
            trans = colorize_transmission(result["transmission"])
            airlight = result["airlight"].numpy()
        else:
            physics = dehazed.copy()
            trans = np.zeros_like(dehazed)
            airlight = np.array([0.0, 0.0, 0.0])

        # Weather info — determine category and create weather-specific dirs
        w_data = ds._weather_data.get(ts, {})
        w_cat = STFDehazeDataset.classify_weather(w_data) if w_data else "unknown"
        humidity = w_data.get("outHumidity", "?")
        temp_f = w_data.get("outTemp", "?")

        # Create weather-specific subdirectories
        weather_fig_dir = out_path / w_cat / "figures"
        weather_img_dir = out_path / w_cat / "images"
        weather_fig_dir.mkdir(parents=True, exist_ok=True)
        if args.save_individual:
            weather_img_dir.mkdir(parents=True, exist_ok=True)

        title = (f"[{i+1}/{n}]  ts={ts}  |  weather={w_cat}  |  "
                 f"humidity={humidity}%  temp={temp_f}°F")

        # Build and save figure
        fig = make_comparison_figure(
            raw=raw_vis, dehazed=dehazed, physics=physics,
            depth_rgb=depth_vis, trans_pred=trans,
            airlight=airlight, title=title,
        )
        fig_path = weather_fig_dir / f"{i:03d}_{ts}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Save individual images
        if args.save_individual:
            _save_img(weather_img_dir / f"{i:03d}_{ts}_input.png", raw_vis)
            _save_img(weather_img_dir / f"{i:03d}_{ts}_dehazed.png", dehazed)
            _save_img(weather_img_dir / f"{i:03d}_{ts}_physics.png", physics)
            _save_img(weather_img_dir / f"{i:03d}_{ts}_transmission.png", trans)

        print(f"  [{i+1:3d}/{n}] {ts}  weather={w_cat:>10s}  "
              f"airlight={airlight.mean():.3f}  → {w_cat}/{fig_path.name}")

    # ---- Summary ----
    print(f"\nDone. Results organized by weather category in {out_path}/")
    print(f"  Each category has:")
    print(f"    <weather_category>/figures/  — comparison PNGs")
    if args.save_individual:
        print(f"    <weather_category>/images/   — individual dehazed/input/physics PNGs")


def _save_img(path: Path, arr: np.ndarray):
    """Save H×W×3 float32 array as uint8 PNG."""
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    img.save(str(path))


if __name__ == "__main__":
    main()
