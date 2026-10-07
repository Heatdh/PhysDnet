"""
Visualization utilities for LiDAR-guided dehazing.

Creates comparison grids: hazy | restored | physics_restored | clear_gt |
                          transmission | sparse_depth_overlay

Used by:
  - train.py (per-epoch validation visualization)
  - Standalone: python visualize.py --checkpoint ... --test_dir ...
"""

import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert a CxHxW or 1xHxW float tensor to HxWx3 uint8 numpy array."""
    if t.dim() == 3 and t.shape[0] == 1:
        # Single channel -> colormap
        arr = t[0].numpy()
        return apply_colormap(arr)
    elif t.dim() == 3 and t.shape[0] == 3:
        arr = t.permute(1, 2, 0).numpy()
        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        return arr
    else:
        raise ValueError(f"Unexpected tensor shape: {t.shape}")


def apply_colormap(arr: np.ndarray, cmap: str = "turbo") -> np.ndarray:
    """Apply a colormap to a 2D float array, returns HxWx3 uint8."""
    # Normalize to [0, 1]
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin > 1e-6:
        arr_norm = (arr - vmin) / (vmax - vmin)
    else:
        arr_norm = np.zeros_like(arr)

    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap(cmap)
        colored = colormap(arr_norm)[:, :, :3]  # drop alpha
        return (colored * 255).astype(np.uint8)
    except ImportError:
        # Fallback: simple grayscale -> RGB
        gray = (arr_norm * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)


def depth_overlay(rgb: torch.Tensor, sparse_depth: torch.Tensor,
                  mask: torch.Tensor) -> np.ndarray:
    """Overlay sparse depth points on RGB image.

    Args:
        rgb:          3xHxW float tensor [0,1]
        sparse_depth: 1xHxW float tensor
        mask:         1xHxW float tensor (1=valid)

    Returns:
        HxWx3 uint8 numpy array
    """
    img = tensor_to_uint8(rgb).copy()
    depth = sparse_depth[0].numpy()
    m = mask[0].numpy() > 0.5

    if m.sum() == 0:
        return img

    # Colormap the valid depth values
    valid_depths = depth[m]
    vmin, vmax = valid_depths.min(), valid_depths.max()
    if vmax - vmin < 1e-6:
        vmax = vmin + 1.0

    # Get colored depth values
    ys, xs = np.where(m)
    d_vals = depth[ys, xs]
    d_norm = (d_vals - vmin) / (vmax - vmin)

    try:
        import matplotlib.cm as cm
        colormap = cm.get_cmap("turbo")
        colors = (colormap(d_norm)[:, :3] * 255).astype(np.uint8)
    except ImportError:
        gray = (d_norm * 255).astype(np.uint8)
        colors = np.stack([gray, gray, gray], axis=-1)

    # Draw points (2px radius for visibility)
    for i in range(len(ys)):
        y, x = ys[i], xs[i]
        r = 1
        y0, y1 = max(0, y - r), min(img.shape[0], y + r + 1)
        x0, x1 = max(0, x - r), min(img.shape[1], x + r + 1)
        img[y0:y1, x0:x1] = colors[i]

    return img


def make_vis_grid(
    samples: list[dict],
    columns: list[str] | None = None,
    pad: int = 4,
    label_height: int = 20,
) -> Image.Image:
    """
    Create a visualization grid from a list of sample dicts.

    Each row is one sample. Columns:
        hazy | restored | physics_restored | clear | transmission | depth_overlay

    Args:
        samples: list of dicts with tensor values (CxHxW, float, CPU)
        columns: column names to include (default: all standard)
        pad: padding between cells
        label_height: height of column labels

    Returns:
        PIL Image of the grid
    """
    if columns is None:
        columns = ["hazy", "restored", "physics_restored", "clear",
                    "transmission", "depth_overlay"]

    if len(samples) == 0:
        return Image.new("RGB", (200, 100), (0, 0, 0))

    # Build cell images
    rows = []
    for s in samples:
        row = []
        for col in columns:
            if col == "depth_overlay":
                cell = depth_overlay(s["hazy"], s["sparse_depth"], s["mask"])
            elif col == "transmission":
                cell = tensor_to_uint8(s["transmission"])
            elif col in s:
                cell = tensor_to_uint8(s[col])
            else:
                # Placeholder
                h = s["hazy"].shape[1] if "hazy" in s else 256
                w = s["hazy"].shape[2] if "hazy" in s else 256
                cell = np.zeros((h, w, 3), dtype=np.uint8)
            row.append(cell)
        rows.append(row)

    # All cells should have the same size (from same crop)
    cell_h, cell_w = rows[0][0].shape[:2]
    n_rows = len(rows)
    n_cols = len(columns)

    # Grid dimensions
    grid_w = n_cols * cell_w + (n_cols + 1) * pad
    grid_h = n_rows * cell_h + (n_rows + 1) * pad + label_height

    grid = Image.new("RGB", (grid_w, grid_h), (40, 40, 40))
    draw = ImageDraw.Draw(grid)

    # Column labels
    col_labels = {
        "hazy": "Hazy Input",
        "restored": "Restored",
        "physics_restored": "Physics",
        "clear": "Ground Truth",
        "transmission": "Transmission",
        "depth_overlay": "LiDAR Depth",
    }

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for j, col in enumerate(columns):
        x = pad + j * (cell_w + pad)
        label = col_labels.get(col, col)
        draw.text((x + 2, 2), label, fill=(220, 220, 220), font=font)

    # Paste cells
    for i, row in enumerate(rows):
        for j, cell in enumerate(row):
            x = pad + j * (cell_w + pad)
            y = label_height + pad + i * (cell_h + pad)
            cell_img = Image.fromarray(cell)
            grid.paste(cell_img, (x, y))

    return grid


def save_vis_grid(grid: Image.Image, path: str):
    """Save a visualization grid to disk."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    grid.save(path)


# ---------------------------------------------------------------------------
# Standalone visualization
# ---------------------------------------------------------------------------

def main():
    """Standalone: visualize model predictions on a dataset."""
    import argparse
    from model import LiDARDehazeNet
    from dataset import STFDehazeDataset, DummyDehazeDataset

    parser = argparse.ArgumentParser(description="Visualize dehazing results")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pth")
    parser.add_argument("--stf_root", type=str,
                        default="data/stf/SeeingThroughFog")
    parser.add_argument("--depth_dir", type=str,
                        default=None,
                        help="Depth map dir (default: stf_root/lidar_hdl64_strongest_stereo_left)")
    parser.add_argument("--timestamps_file", type=str, default=None)
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--save_path", type=str, default="results/visualization.png")
    parser.add_argument("--base_ch", type=int, default=32)
    parser.add_argument("--dummy", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LiDARDehazeNet(base_ch=args.base_ch).to(device)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        print(f"[WARN] No checkpoint at {args.checkpoint}, using random weights")

    model.eval()

    # Dataset
    if args.dummy:
        dataset = DummyDehazeDataset(length=args.n_samples, img_size=(256, 256))
    else:
        dataset = STFDehazeDataset(
            stf_root=args.stf_root,
            depth_dir=args.depth_dir,
            timestamps_file=args.timestamps_file,
            crop_size=(512, 512),
            augment=False,
        )

    n = min(args.n_samples, len(dataset))
    samples = []

    with torch.no_grad():
        for i in range(n):
            batch = dataset[i]
            hazy = batch["hazy"].unsqueeze(0).to(device)
            sparse = batch["sparse_depth"].unsqueeze(0).to(device)
            mask = batch["mask"].unsqueeze(0).to(device)

            out = model(hazy, sparse, mask)

            samples.append({
                "hazy": hazy[0].cpu(),
                "clear": batch["clear"],
                "restored": out["restored"][0].cpu(),
                "physics_restored": out["physics_restored"][0].cpu(),
                "transmission": out["transmission"][0].cpu(),
                "sparse_depth": sparse[0].cpu(),
                "mask": mask[0].cpu(),
            })

    grid = make_vis_grid(samples)
    save_vis_grid(grid, args.save_path)
    print(f"Saved visualization ({n} samples) -> {args.save_path}")


if __name__ == "__main__":
    main()
