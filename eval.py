"""
Evaluation script for LiDAR-guided dehazing model.

Computes PSNR and SSIM for both direct and physics heads on the
validation split (paired synthetic-haze vs clear GT).  All results
are saved inside the checkpoint's run directory for traceability.

Usage:
    python eval.py --checkpoint runs/<run_dir>/best.pth

    # With image saving:
    python eval.py --checkpoint runs/<run_dir>/best.pth --save_images
"""

import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

from model import get_model, MODEL_VARIANTS
from dataset import STFDehazeDataset
from baselines import get_baseline_model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio (higher is better)."""
    mse = torch.mean((pred - target) ** 2).item()
    if mse < 1e-10:
        return 100.0
    return 10.0 * np.log10(max_val ** 2 / mse)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01 ** 2,
    C2: float = 0.03 ** 2,
) -> float:
    """
    Structural Similarity Index (higher is better).
    Simplified channel-averaged version.
    """
    # Use average pooling as a crude window
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


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate LiDAR Dehaze Net")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint (best.pth inside a run dir)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--save_images", action="store_true",
                        help="Save per-sample images (hazy/restored/physics/clear)")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for deterministic fog synthesis (default: 42)")
    parser.add_argument("--strip_lidar", action="store_true",
                        help="Zero out depth + mask inputs (NIR-only, no LiDAR)")
    parser.add_argument("--pseudo_depth", action="store_true",
                        help="Replace LiDAR with Depth Anything V2 monocular depth")
    args = parser.parse_args()

    if args.strip_lidar and args.pseudo_depth:
        raise ValueError("Cannot use both --strip_lidar and --pseudo_depth")

    # ---- Deterministic evaluation ----
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Eval seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    # ---- Resolve run directory from checkpoint path ----
    ckpt_path = os.path.abspath(args.checkpoint)
    run_dir = os.path.dirname(ckpt_path)
    if args.strip_lidar:
        suffix = "_NIR_lidar_stripped"
    elif args.pseudo_depth:
        suffix = "_pseudo_depth"
    else:
        suffix = ""
    save_dir = os.path.join(run_dir, f"eval_paired{suffix}")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")
    if args.strip_lidar:
        print(f"MODE: NIR-only (depth + mask zeroed out)")
    elif args.pseudo_depth:
        print(f"MODE: Pseudo-depth (Depth Anything V2 ViT-S)")
    print(f"Results will be saved to: {save_dir}")

    # ---- Load model (auto-detect everything from checkpoint) ----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})
    variant = model_cfg.get("variant", "lite")

    # Check if this is a baseline checkpoint
    baseline_name = model_cfg.get("baseline")
    if baseline_name is not None:
        mode = model_cfg.get("mode", "rgb")
        model_kwargs = {k: v for k, v in model_cfg.items()
                        if k not in ("baseline", "mode", "variant")}
        model = get_baseline_model(baseline_name, mode=mode, **model_kwargs).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded baseline {baseline_name} ({mode}) "
              f"(epoch {ckpt.get('epoch', '?')}, {n_params/1e6:.2f}M params)")
    else:
        overrides = {"base_ch": model_cfg.get("base_ch", 32)}
        # Pass arch flags so the model matches the checkpoint
        for k in ("use_cbam", "use_residual", "use_attention",
                  "use_physics_head", "lidar_drop_rate"):
            v = model_cfg.get(k)
            if v is not None:
                overrides[k] = v

        model = get_model(variant, **overrides).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded {variant} model (epoch {ckpt.get('epoch', '?')}, "
              f"{n_params/1e6:.2f}M params)")

    # ---- Build validation dataset from checkpoint config ----
    data_cfg = cfg.get("data", {})
    stf_root = data_cfg.get("stf_root", "data/stf/SeeingThroughFog")
    meta_dir = data_cfg.get("meta_dir", "data/stf/meta")
    val_ts = os.path.join(meta_dir, "val_timestamps.txt")
    crop_size = tuple(data_cfg.get("crop_size", [512, 512]))
    beta_range = tuple(data_cfg.get("beta_range", [0.005, 0.04]))
    airlight_range = tuple(data_cfg.get("airlight_range", [0.7, 1.0]))
    max_depth = data_cfg.get("max_depth", 120.0)

    val_ds = STFDehazeDataset(
        stf_root=stf_root,
        timestamps_file=val_ts,
        crop_size=crop_size,
        beta_range=beta_range,
        airlight_range=airlight_range,
        max_depth=max_depth,
        augment=False,
    )
    def _worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)

    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=_worker_init_fn,
    )
    print(f"Validation samples: {len(val_ds)}")

    # ---- Load Depth Anything V2 if pseudo_depth ----
    da_model = None
    if args.pseudo_depth:
        import sys
        sys.path.insert(0, "third_party/Depth-Anything-V2")
        from depth_anything_v2.dpt import DepthAnythingV2
        da_model = DepthAnythingV2(
            encoder="vits", features=64,
            out_channels=[48, 96, 192, 384])
        da_model.load_state_dict(torch.load(
            "third_party/Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth",
            map_location="cpu"))
        da_model = da_model.to(device).eval()
        print(f"Loaded Depth Anything V2 ViT-S for pseudo-depth")

    if args.save_images:
        img_dir = os.path.join(save_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

    # ---- Evaluate ----
    all_psnr, all_ssim = [], []
    all_psnr_phys, all_ssim_phys = [], []
    total_time = 0.0
    img_idx = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            sparse = batch["sparse_depth"].to(device)
            mask = batch["mask"].to(device)

            if args.strip_lidar:
                sparse = torch.zeros_like(sparse)
                mask = torch.zeros_like(mask)
            elif args.pseudo_depth and da_model is not None:
                # Run DA-V2 per image, get relative inverse depth
                import cv2
                with torch.no_grad():
                    B = hazy.shape[0]
                    pseudo_depths = []
                    for bi in range(B):
                        # Convert tensor to cv2 BGR uint8
                        img_np = (hazy[bi].cpu().numpy().transpose(1, 2, 0) * 255
                                  ).clip(0, 255).astype(np.uint8)
                        img_bgr = img_np[:, :, ::-1].copy()
                        rel_inv = da_model.infer_image(img_bgr, input_size=518)
                        # Normalize to [0,1] and invert (far=0, near=1)
                        rmin, rmax = rel_inv.min(), rel_inv.max()
                        if rmax - rmin > 1e-6:
                            norm = (rel_inv - rmin) / (rmax - rmin)
                        else:
                            norm = np.zeros_like(rel_inv)
                        pseudo_depths.append(torch.from_numpy(
                            norm.astype(np.float32)).unsqueeze(0))
                    sparse = torch.stack(pseudo_depths).to(device)
                    mask = torch.ones_like(sparse)

            t0 = time.time()
            out = model(hazy, sparse, mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - t0

            restored = out["restored"]
            physics_restored = out.get("physics_restored")
            has_physics = physics_restored is not None

            B = hazy.shape[0]
            for i in range(B):
                p = psnr(restored[i:i+1], clear[i:i+1])
                s = ssim(restored[i:i+1], clear[i:i+1])
                all_psnr.append(p)
                all_ssim.append(s)

                if has_physics:
                    pp = psnr(physics_restored[i:i+1], clear[i:i+1])
                    sp = ssim(physics_restored[i:i+1], clear[i:i+1])
                    all_psnr_phys.append(pp)
                    all_ssim_phys.append(sp)

                if args.save_images:
                    save_pairs = [
                        ("hazy", hazy[i]),
                        ("restored", restored[i]),
                        ("clear", clear[i]),
                    ]
                    if has_physics:
                        save_pairs.insert(2, ("physics_restored", physics_restored[i]))
                    for name, tensor in save_pairs:
                        arr = (tensor.cpu().numpy().transpose(1, 2, 0) * 255
                               ).clip(0, 255).astype(np.uint8)
                        Image.fromarray(arr).save(
                            os.path.join(img_dir, f"{img_idx:04d}_{name}.png"))
                    if "transmission" in out:
                        t_map = out["transmission"][i, 0].cpu().numpy()
                        t_vis = (t_map * 255).clip(0, 255).astype(np.uint8)
                        Image.fromarray(t_vis).save(
                            os.path.join(img_dir, f"{img_idx:04d}_transmission.png"))
                img_idx += 1

    # ---- Report ----
    n = len(all_psnr)
    avg_psnr = np.mean(all_psnr)
    avg_ssim = np.mean(all_ssim)
    avg_time = total_time / max(n, 1) * 1000

    has_physics_metrics = len(all_psnr_phys) > 0

    report = []
    report.append(f"{'='*60}")
    report.append(f"Results on {n} validation images:")
    report.append(f"  {'Head':<20} {'PSNR (dB)':>12} {'SSIM':>12}")
    report.append(f"  {'-'*44}")
    report.append(f"  {'Direct':<20} {avg_psnr:>8.2f} ±{np.std(all_psnr):.2f}"
                  f" {avg_ssim:>8.4f} ±{np.std(all_ssim):.4f}")
    if has_physics_metrics:
        avg_psnr_phys = np.mean(all_psnr_phys)
        avg_ssim_phys = np.mean(all_ssim_phys)
        report.append(f"  {'Physics':<20} {avg_psnr_phys:>8.2f} ±{np.std(all_psnr_phys):.2f}"
                      f" {avg_ssim_phys:>8.4f} ±{np.std(all_ssim_phys):.4f}")
    report.append(f"  {'-'*44}")
    report.append(f"  Time:  {avg_time:.1f} ms/image  ({1000/max(avg_time,0.1):.1f} FPS)")
    report.append(f"{'='*60}")

    for line in report:
        print(line)

    # Save metrics CSV
    csv_path = os.path.join(save_dir, "metrics.csv")
    with open(csv_path, "w") as f:
        if has_physics_metrics:
            f.write("idx,psnr_direct,ssim_direct,psnr_physics,ssim_physics\n")
            for i in range(n):
                f.write(f"{i},{all_psnr[i]:.4f},{all_ssim[i]:.4f},"
                        f"{all_psnr_phys[i]:.4f},{all_ssim_phys[i]:.4f}\n")
        else:
            f.write("idx,psnr_direct,ssim_direct\n")
            for i in range(n):
                f.write(f"{i},{all_psnr[i]:.4f},{all_ssim[i]:.4f}\n")

    # Save summary text
    summary_path = os.path.join(save_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(report) + "\n")

    print(f"\nMetrics CSV: {csv_path}")
    print(f"Summary:     {summary_path}")
    if args.save_images:
        print(f"Images:      {img_dir}/")


if __name__ == "__main__":
    evaluate()
