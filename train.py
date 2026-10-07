"""
Training script for LiDAR-guided dehazing model.

Supports:
  - YAML config file (--config) with CLI overrides
  - Seeing-Through-Fog (STF) dataset or generic on-the-fly / pregenerated data
  - Weights & Biases (wandb) experiment tracking
  - Per-epoch sample visualization grids
  - Linear warmup + cosine annealing LR schedule

Usage:
    # Quick test with dummy data (no files needed):
    python train.py --dummy --epochs 5 --batch_size 4

    # Train on STF with config:
    python train.py --config configs/stf_train.yaml

    # Override individual params:
    python train.py --config configs/stf_train.yaml --epochs 50 --lr 1e-4

    # Disable wandb:
    python train.py --config configs/stf_train.yaml --no_wandb

    # Resume from checkpoint:
    python train.py --config configs/stf_train.yaml --resume checkpoints/best.pth
"""

import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from torch.amp import autocast, GradScaler

from model import LiDARDehazeNet, get_model, MODEL_VARIANTS
from losses import DehazeLoss
from dataset import DehazeDataset, DummyDehazeDataset, STFDehazeDataset
from visualize import make_vis_grid, save_vis_grid


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    if not HAS_YAML:
        raise ImportError("PyYAML required: pip install pyyaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def merge_config_with_args(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge flat CLI args into nested config dict (CLI takes priority)."""
    cli_map = {
        "epochs": ("training", "epochs"),
        "batch_size": ("training", "batch_size"),
        "lr": ("training", "lr"),
        "weight_decay": ("training", "weight_decay"),
        "base_ch": ("model", "base_ch"),
        "num_workers": ("training", "num_workers"),
        "seed": ("training", "seed"),
        "w_rec": ("training", "w_rec"),
        "w_perc": ("training", "w_perc"),
        "w_grad": ("training", "w_grad"),
        "w_phys": ("training", "w_phys"),
        "w_smooth": ("training", "w_smooth"),
        "save_dir": ("training", "save_dir"),
        "log_every": ("training", "log_every"),
        "val_every": ("training", "val_every"),
    }

    for arg_name, cfg_path in cli_map.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            section, key = cfg_path
            cfg.setdefault(section, {})[key] = val

    return cfg


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_datasets(cfg: dict, args: argparse.Namespace):
    """Create train and val datasets from config."""
    if args.dummy:
        img_size = tuple(cfg.get("data", {}).get("crop_size", [256, 256]))
        train_ds = DummyDehazeDataset(length=200, img_size=img_size)
        val_ds = DummyDehazeDataset(length=40, img_size=img_size)
        print("Using DummyDehazeDataset for quick pipeline test.")
        return train_ds, val_ds

    data_cfg = cfg.get("data", {})

    if args.dataset == "stf":
        stf_root = data_cfg.get("stf_root", "data/stf/SeeingThroughFog")
        depth_dir = data_cfg.get("depth_dir", None)  # None -> auto-detect
        meta_dir = data_cfg.get("meta_dir", "data/stf/meta")

        train_ts = data_cfg.get("train_timestamps") or \
                   os.path.join(meta_dir, "train_timestamps.txt")
        val_ts = data_cfg.get("val_timestamps") or \
                 os.path.join(meta_dir, "val_timestamps.txt")

        # Auto-split if train/val files don't exist
        if not os.path.exists(train_ts) or not os.path.exists(val_ts):
            # Discover all timestamps from depth directory
            if depth_dir is None:
                _depth_dir = os.path.join(
                    stf_root, "lidar_hdl64_strongest_stereo_left")
            else:
                _depth_dir = depth_dir
            all_ts_file = os.path.join(meta_dir, "all_timestamps.txt")
            if not os.path.exists(all_ts_file):
                print("[INFO] Building timestamp list from depth maps...")
                depth_files = sorted(
                    list(Path(_depth_dir).glob("*.npz"))
                    + list(Path(_depth_dir).glob("*.npy"))
                )
                timestamps = [p.stem for p in depth_files]
                os.makedirs(meta_dir, exist_ok=True)
                with open(all_ts_file, "w") as f:
                    f.write("\n".join(timestamps) + "\n")
                print(f"  Found {len(timestamps)} timestamps")

            print(f"Creating train/val split from {all_ts_file}...")
            STFDehazeDataset.make_train_val_split(
                all_ts_file, meta_dir,
                val_ratio=data_cfg.get("val_ratio", 0.1),
                seed=data_cfg.get("split_seed", 42),
            )

        crop = tuple(data_cfg.get("crop_size", [512, 512]))
        beta = tuple(data_cfg.get("beta_range", [0.005, 0.04]))
        airlight = tuple(data_cfg.get("airlight_range", [0.7, 1.0]))
        max_depth = data_cfg.get("max_depth", 120.0)

        # Weather-based filtering:
        #   train_weather: conditions used for training (synthetic haze on
        #                  clear/overcast frames only)
        #   val_weather:   conditions for validation (same as train by default,
        #                  or can include fog for real-world eval)
        train_weather = data_cfg.get("train_weather_filter", None)
        val_weather = data_cfg.get("val_weather_filter", None)

        train_ds = STFDehazeDataset(
            stf_root=stf_root, depth_dir=depth_dir,
            timestamps_file=train_ts, crop_size=crop,
            beta_range=beta, airlight_range=airlight,
            max_depth=max_depth,
            augment=data_cfg.get("augment", True),
            weather_filter=train_weather,
        )
        val_ds = STFDehazeDataset(
            stf_root=stf_root, depth_dir=depth_dir,
            timestamps_file=val_ts, crop_size=crop,
            beta_range=beta, airlight_range=airlight,
            max_depth=max_depth,
            augment=False,
            weather_filter=val_weather,
        )
    else:
        # Generic dataset (on_the_fly / pregenerated)
        img_size = tuple(data_cfg.get("crop_size", [256, 256]))
        mode = args.mode or "on_the_fly"
        train_ds = DehazeDataset(
            root=args.train_dir, mode=mode,
            img_size=img_size, augment=True,
        )
        val_ds = DehazeDataset(
            root=args.val_dir, mode=mode,
            img_size=img_size, augment=False,
        )

    print(f"Train: {len(train_ds)} samples, Val: {len(val_ds)} samples")
    return train_ds, val_ds


def build_model_from_cfg(model_cfg: dict, args: argparse.Namespace | None = None) -> tuple[nn.Module, str]:
    """Build a model from config using the same variant resolution as train/eval."""
    variant = (getattr(args, "model", None) if args is not None else None) or model_cfg.get("variant", "lite")
    default_ch = MODEL_VARIANTS[variant].get("base_ch", 32)
    if args is not None and getattr(args, "base_ch", None) is not None:
        base_ch = args.base_ch
    elif "base_ch" in model_cfg:
        base_ch = model_cfg["base_ch"]
    else:
        base_ch = default_ch
    extra_kwargs = {}
    for key in ("use_cbam", "use_residual", "use_attention", "lidar_drop_rate", "use_physics_head"):
        if key in model_cfg:
            extra_kwargs[key] = model_cfg[key]
    model = get_model(variant, base_ch=base_ch, **extra_kwargs)
    return model, variant


def load_checkpoint_with_overlap(model: nn.Module, checkpoint_path: str, device: torch.device) -> dict[str, int]:
    """Load a checkpoint with exact or overlapping tensor copies for width-changed models."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    src_state = ckpt["model"]
    dst_state = model.state_dict()
    loaded = 0
    partial = 0
    skipped = 0

    for key, dst_tensor in dst_state.items():
        src_tensor = src_state.get(key)
        if src_tensor is None:
            skipped += 1
            continue
        if src_tensor.shape == dst_tensor.shape:
            dst_state[key] = src_tensor
            loaded += 1
            continue
        if src_tensor.ndim != dst_tensor.ndim or src_tensor.ndim == 0:
            skipped += 1
            continue
        slices = tuple(slice(0, min(src_dim, dst_dim))
                       for src_dim, dst_dim in zip(src_tensor.shape, dst_tensor.shape))
        patched = dst_tensor.clone()
        patched[slices] = src_tensor[slices].to(dtype=patched.dtype)
        dst_state[key] = patched
        partial += 1

    model.load_state_dict(dst_state, strict=False)
    return {"loaded": loaded, "partial": partial, "skipped": skipped}


def load_teacher_model(teacher_ckpt: str, device: torch.device) -> tuple[nn.Module, dict, str]:
    """Load a frozen teacher model in fp16 on GPU (inference only, no grads)."""
    ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
    teacher_cfg = ckpt.get("config", {})
    teacher_model_cfg = teacher_cfg.get("model", {})
    teacher_model, teacher_variant = build_model_from_cfg(teacher_model_cfg)
    teacher_model.load_state_dict(ckpt["model"])
    teacher_model = teacher_model.to(device).half().eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    return teacher_model, teacher_cfg, teacher_variant


# ---------------------------------------------------------------------------
# wandb helpers
# ---------------------------------------------------------------------------

def _build_run_name(cfg: dict, args: argparse.Namespace, model: nn.Module) -> str:
    """Build a descriptive run name from config (used for both wandb and directory)."""
    from datetime import datetime
    data_cfg = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    parts = []
    # dataset
    ds = "stf" if args.dataset == "stf" else ("dummy" if args.dummy else "custom")
    parts.append(ds)
    # model variant
    variant = args.model or model_cfg.get("variant", "lite")
    parts.append(variant)
    n_params = sum(p.numel() for p in model.parameters())
    base_ch = model_cfg.get("base_ch", MODEL_VARIANTS[variant]["base_ch"])
    if args.base_ch is not None:
        base_ch = args.base_ch
    parts.append(f"ch{base_ch}_{n_params/1e6:.1f}M")
    # weather filter hint
    wf = data_cfg.get("train_weather_filter")
    if wf:
        parts.append("w" + "+".join(sorted(wf)))
    # key hparams
    crop = data_cfg.get("crop_size", [512, 512])
    parts.append(f"crop{crop[0]}")
    beta = data_cfg.get("beta_range", [0.005, 0.04])
    parts.append(f"b{beta[0]}-{beta[1]}")
    parts.append(f"bs{train_cfg.get('batch_size', 8)}")
    parts.append(f"lr{train_cfg.get('lr', 2e-4):.0e}")
    parts.append(f"ep{train_cfg.get('epochs', 100)}")
    # v2 feature tags
    v2_tags = []
    if model_cfg.get("use_cbam", False):
        v2_tags.append("cbam")
    if model_cfg.get("use_attention", False) and not model_cfg.get("use_cbam", False):
        v2_tags.append("attn")
    if model_cfg.get("use_residual", False):
        v2_tags.append("res")
    if train_cfg.get("w_fft", 0) > 0:
        v2_tags.append("fft")
    if train_cfg.get("w_contrast", 0) > 0:
        start_ep = train_cfg.get("w_contrast_start_epoch", 0)
        v2_tags.append(f"ctr@{start_ep}" if start_ep > 0 else "ctr")
    if model_cfg.get("lidar_drop_rate", 0) > 0:
        pct = int(model_cfg["lidar_drop_rate"] * 100)
        v2_tags.append(f"ldrop{pct}")
    if not model_cfg.get("use_physics_head", True):
        v2_tags.append("nophy")
    init_cfg = cfg.get("initialization", {})
    if init_cfg.get("checkpoint"):
        v2_tags.append("init")
    distill_cfg = cfg.get("distillation", {})
    if distill_cfg.get("teacher_checkpoint"):
        v2_tags.append("kd")
    if v2_tags:
        parts.append("v2_" + "+".join(v2_tags))
    # training seed tag
    seed = train_cfg.get("seed")
    if seed is not None:
        parts.append(f"seed{seed}")
    # timestamp
    parts.append(datetime.now().strftime("%m%d_%H%M"))
    return "_".join(parts)


def get_run_name(cfg: dict, args: argparse.Namespace, model: nn.Module) -> str:
    """Determine the run name (shared by wandb + run directory)."""
    wandb_cfg = cfg.get("wandb", {})
    return (wandb_cfg.get("run_name")
            or args.wandb_run
            or _build_run_name(cfg, args, model))


def init_wandb(cfg: dict, args: argparse.Namespace, model: nn.Module,
               run_name: str):
    """Initialize wandb run (no-op if disabled)."""
    wandb_cfg = cfg.get("wandb", {})

    if args.no_wandb or not wandb_cfg.get("enabled", True) or not HAS_WANDB:
        if not HAS_WANDB and wandb_cfg.get("enabled", True):
            print("[WARN] wandb not installed: pip install wandb")
        return None

    run = wandb.init(
        project=wandb_cfg.get("project", "lidar-dehazing"),
        name=run_name,
        tags=wandb_cfg.get("tags", []),
        config={
            **cfg,
            "cli_args": vars(args),
            "n_params": sum(p.numel() for p in model.parameters()),
        },
    )
    wandb.watch(model, log="gradients", log_freq=200)
    print(f"wandb run: {run_name}")
    return run


def log_wandb(metrics: dict, step: int | None = None):
    """Log to wandb if it's active."""
    if HAS_WANDB and wandb.run is not None:
        wandb.log(metrics, step=step)


# ---------------------------------------------------------------------------
# Learning rate schedule with warmup
# ---------------------------------------------------------------------------

class WarmupCosineScheduler:
    """Linear warmup + cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 eta_min: float = 1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            factor = self.current_epoch / max(1, self.warmup_epochs)
        else:
            progress = (self.current_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + np.cos(np.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = self.eta_min + (base_lr - self.eta_min) * factor

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
# prior runs val_seed was not fixed for reference
# meaning eval on the fly on new synth haze each step.
# affect only the savings
VAL_SEED = 42  # fixed seed for deterministic validation (matches eval.py)


def validate(model, val_loader, criterion, device):
    """Run one pass over the validation set and return average losses + PSNR/SSIM."""
    # Save RNG state so we don't corrupt training randomness
    rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.random.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all()

    # Seed RNGs so fog synthesis is identical every epoch (and matches eval.py --seed 42)
    random.seed(VAL_SEED)
    np.random.seed(VAL_SEED)
    torch.manual_seed(VAL_SEED)
    torch.cuda.manual_seed_all(VAL_SEED)

    model.eval()
    running = {}
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            sparse = batch["sparse_depth"].to(device)
            mask = batch["mask"].to(device)
            trans_gt = batch.get("trans_gt")
            if trans_gt is not None:
                trans_gt = trans_gt.to(device)

            out = model(hazy, sparse, mask)
            _, loss_dict = criterion(out, hazy, clear, trans_gt=trans_gt)

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v

            # Per-batch PSNR/SSIM on direct head
            restored = out["restored"]
            mse = (restored - clear).pow(2).mean(dim=(1, 2, 3))  # per-image
            psnr_batch = (-10.0 * torch.log10(mse.clamp(min=1e-10))).sum().item()
            psnr_sum += psnr_batch

            # Structural similarity (per-image, simplified)
            for i in range(restored.size(0)):
                ssim_sum += _ssim(restored[i], clear[i])

            n += restored.size(0)

    n_batches = len(val_loader)
    avg = {k: v / max(n_batches, 1) for k, v in running.items()}
    avg["psnr"] = psnr_sum / max(n, 1)
    avg["ssim"] = ssim_sum / max(n, 1)

    # Restore RNG state so training DataLoader shuffle/augmentation isn't corrupted
    random.setstate(rng_state)
    np.random.set_state(np_rng_state)
    torch.random.set_rng_state(torch_rng_state)
    torch.cuda.set_rng_state_all(cuda_rng_state)

    return avg


def _ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Compute SSIM for a single CxHxW image pair (simplified, no window)."""
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    num = (2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)
    den = (mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2)
    return (num / den).item()


# ---------------------------------------------------------------------------
# Visualization callback
# ---------------------------------------------------------------------------

def visualize_epoch(
    model, val_dataset, device, epoch, cfg,
    fixed_indices=None, save_dir="results",
):
    """Generate and save/log a visualization grid from validation samples."""
    n_vis = cfg.get("training", {}).get("n_vis_samples", 8)
    n_vis = min(n_vis, len(val_dataset))

    if fixed_indices is None:
        fixed_indices = list(range(n_vis))

    model.eval()
    samples = []

    with torch.no_grad():
        for idx in fixed_indices:
            batch = val_dataset[idx]
            hazy = batch["hazy"].unsqueeze(0).to(device)
            clear = batch["clear"].unsqueeze(0).to(device)
            sparse = batch["sparse_depth"].unsqueeze(0).to(device)
            mask = batch["mask"].unsqueeze(0).to(device)

            out = model(hazy, sparse, mask)

            samples.append({
                "hazy": hazy[0].cpu(),
                "clear": clear[0].cpu(),
                "restored": out["restored"][0].cpu(),
                "physics_restored": out["physics_restored"][0].cpu(),
                "transmission": out["transmission"][0].cpu(),
                "sparse_depth": sparse[0].cpu(),
                "mask": mask[0].cpu(),
            })

    grid = make_vis_grid(samples)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"vis_epoch_{epoch:03d}.png")
    save_vis_grid(grid, save_path)
    print(f"  Visualization saved -> {save_path}")

    wandb_cfg = cfg.get("wandb", {})
    if (HAS_WANDB and wandb.run is not None
            and wandb_cfg.get("log_images", True)):
        wandb.log({
            "val/samples": wandb.Image(grid, caption=f"Epoch {epoch}"),
        })


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Train LiDAR Dehaze Net")

    p.add_argument("--config", type=str, default=None,
                   help="YAML config file (e.g., configs/stf_train.yaml)")

    p.add_argument("--dataset", type=str, default="stf",
                   choices=["stf", "generic"])
    p.add_argument("--model", type=str, default=None,
                   choices=list(MODEL_VARIANTS),
                   help="Model variant: lite (~0.72M) or robust (~2.9M)")
    p.add_argument("--train_dir", type=str, default="data/train")
    p.add_argument("--val_dir", type=str, default="data/val")
    p.add_argument("--dummy", action="store_true",
                   help="Use DummyDehazeDataset (no files needed)")
    p.add_argument("--mode", type=str, default=None,
                   choices=["on_the_fly", "pregenerated"])

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--base_ch", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--img_size", type=int, nargs=2, default=None)

    p.add_argument("--w_rec", type=float, default=None)
    p.add_argument("--w_perc", type=float, default=None)
    p.add_argument("--w_grad", type=float, default=None)
    p.add_argument("--w_phys", type=float, default=None)
    p.add_argument("--w_smooth", type=float, default=None)
    p.add_argument("--no_perceptual", action="store_true",
                   help="Disable VGG perceptual loss")

    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="Path to checkpoint to resume from (e.g. runs/<dir>/latest.pth)")
    p.add_argument("--auto_eval", action="store_true", default=True,
                   help="Auto-run eval.py after training (default: on)")
    p.add_argument("--no_auto_eval", action="store_true",
                   help="Disable auto eval after training")
    p.add_argument("--log_every", type=int, default=None)
    p.add_argument("--val_every", type=int, default=None)

    p.add_argument("--seed", type=int, default=None,
                   help="Training seed for reproducibility (val seed is always 42)")

    p.add_argument("--no_wandb", action="store_true",
                   help="Disable wandb logging")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_run", type=str, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load config ----
    if args.config:
        cfg = load_config(args.config)
        print(f"Loaded config: {args.config}")
    else:
        cfg = {
            "data": {"crop_size": args.img_size or [256, 256]},
            "model": {},
            "training": {
                "epochs": args.epochs or 100,
                "batch_size": args.batch_size or 8,
                "lr": args.lr or 2e-4,
                "weight_decay": args.weight_decay or 1e-4,
                "num_workers": args.num_workers or 4,
                "w_rec": 1.0, "w_perc": 0.05, "w_grad": 0.5,
                "w_phys": 0.5, "w_smooth": 0.1,
                "save_dir": "checkpoints",
                "log_every": 20, "val_every": 1,
                "vis_every": 5, "n_vis_samples": 8,
                "grad_clip": 1.0, "eta_min": 1e-6,
                "warmup_epochs": 3,
                "use_perceptual": True,
            },
            "wandb": {"enabled": False},
        }

    cfg = merge_config_with_args(cfg, args)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})

    # ---- Training seed (val seed is always VAL_SEED=42) ----
    train_seed = args.seed if args.seed is not None else train_cfg.get("seed", None)
    if train_seed is not None:
        random.seed(train_seed)
        np.random.seed(train_seed)
        torch.manual_seed(train_seed)
        torch.cuda.manual_seed_all(train_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print(f"Training seed: {train_seed}")
    else:
        print("Training seed: None (non-deterministic)")

    # ---- Data ----
    train_ds, val_ds = build_datasets(cfg, args)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    def _val_worker_init_fn(worker_id):
        np.random.seed(VAL_SEED + worker_id)
        random.seed(VAL_SEED + worker_id)

    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 8),
        shuffle=False,
        num_workers=train_cfg.get("num_workers", 4),
        pin_memory=True,
        worker_init_fn=_val_worker_init_fn,
    )

    # ---- Model ----
    model, variant = build_model_from_cfg(model_cfg, args=args)
    model = model.to(device)

    init_cfg = cfg.get("initialization", {})
    init_checkpoint = init_cfg.get("checkpoint")
    if init_checkpoint:
        init_stats = load_checkpoint_with_overlap(model, init_checkpoint, device)
        print(f"Initialization checkpoint: {init_checkpoint}")
        print(f"  loaded={init_stats['loaded']} partial={init_stats['partial']} skipped={init_stats['skipped']}")

    distill_cfg = cfg.get("distillation", {})
    teacher_model = None
    teacher_checkpoint = distill_cfg.get("teacher_checkpoint")
    w_distill = float(distill_cfg.get("w_distill", 0.0))
    w_distill_phys = float(distill_cfg.get("w_distill_phys", 0.0))
    distill_start_epoch = int(distill_cfg.get("start_epoch", 0))
    if teacher_checkpoint:
        teacher_model, _, teacher_variant = load_teacher_model(teacher_checkpoint, device)
        print(f"Teacher: {teacher_variant} from {teacher_checkpoint}")
        print(f"  distill direct={w_distill} phys={w_distill_phys} start_epoch={distill_start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {variant} | total: {n_params:,} ({n_params/1e6:.2f}M) | "
          f"trainable: {n_trainable:,} ({n_trainable/1e6:.2f}M)")

    # ---- Loss ----
    use_perc = train_cfg.get("use_perceptual", True) and not args.no_perceptual
    criterion = DehazeLoss(
        w_rec=train_cfg.get("w_rec", 1.0),
        w_perc=train_cfg.get("w_perc", 0.05),
        w_grad=train_cfg.get("w_grad", 0.5),
        w_phys=train_cfg.get("w_phys", 0.2),
        w_phys_rec=train_cfg.get("w_phys_rec", 1.0),
        w_smooth=train_cfg.get("w_smooth", 0.1),
        w_trans=train_cfg.get("w_trans", 2.0),
        w_fft=train_cfg.get("w_fft", 0.0),
        w_contrast=train_cfg.get("w_contrast", 0.0),
        use_perceptual=use_perc,
    ).to(device)

    # Late-start for contrastive loss (0 until w_contrast_start_epoch)
    w_contrast_target = train_cfg.get("w_contrast", 0.0)
    w_contrast_start_epoch = train_cfg.get("w_contrast_start_epoch", 0)
    if w_contrast_start_epoch > 0 and w_contrast_target > 0:
        criterion.w_contrast = 0.0  # disable initially
        print(f"  Contrastive loss: delayed to epoch {w_contrast_start_epoch} "
              f"(w={w_contrast_target})")

    # ---- Optimizer + Scheduler ----
    lr = train_cfg.get("lr", 2e-4)
    wd = train_cfg.get("weight_decay", 1e-4)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd,
    )
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=train_cfg.get("warmup_epochs", 3),
        total_epochs=train_cfg.get("epochs", 100),
        eta_min=train_cfg.get("eta_min", 1e-6),
    )

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_psnr = 0.0
    resume_run_dir = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_val_psnr = ckpt.get("best_val_psnr", 0.0)
        # Fast-forward scheduler to resume epoch
        for _ in range(start_epoch):
            scheduler.step()
        print(f"Resumed from epoch {start_epoch}, "
              f"LR={scheduler.get_last_lr()[0]:.2e}")
        # Reuse existing run directory from checkpoint path
        resume_run_dir = os.path.dirname(os.path.abspath(args.resume))

    # ---- Run directory: runs/<experiment_name>/ ----
    if resume_run_dir and os.path.isdir(resume_run_dir):
        run_dir = resume_run_dir
        run_name = os.path.basename(run_dir)
    else:
        run_name = get_run_name(cfg, args, model)
        runs_root = train_cfg.get("runs_dir", "runs")
        run_dir = os.path.join(runs_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Save config snapshot into the run directory
    import shutil, yaml as _yaml
    config_snap = os.path.join(run_dir, "config.yaml")
    with open(config_snap, "w") as f:
        _yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    if args.config and os.path.exists(args.config):
        shutil.copy2(args.config, os.path.join(run_dir, "config_original.yaml"))
    print(f"Run directory: {run_dir}/")

    # ---- wandb ----
    init_wandb(cfg, args, model, run_name=run_name)

    # ---- AMP (mixed precision) ----
    use_amp = train_cfg.get("use_amp", False)
    scaler = GradScaler(enabled=use_amp)
    print(f"Mixed precision (AMP): {'ON' if use_amp else 'OFF'}")

    # ---- Fixed val indices for visualization ----
    n_vis = train_cfg.get("n_vis_samples", 8)
    vis_indices = list(range(min(n_vis, len(val_ds))))

    # ---- Training loop ----
    epochs = train_cfg.get("epochs", 100)
    log_every = train_cfg.get("log_every", 20)
    val_every = train_cfg.get("val_every", 1)
    vis_every = train_cfg.get("vis_every", 5)
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    global_step = 0

    print(f"\n{'='*60}")
    print(f"Starting training: {epochs} epochs, "
          f"bs={train_cfg.get('batch_size', 8)}, "
          f"accum={grad_accum_steps}, "
          f"eff_bs={train_cfg.get('batch_size', 8) * grad_accum_steps}, "
          f"lr={train_cfg.get('lr', 2e-4):.1e}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, epochs):
        # Toggle contrastive loss at the scheduled epoch
        if (w_contrast_start_epoch > 0 and w_contrast_target > 0
                and epoch == w_contrast_start_epoch):
            criterion.w_contrast = w_contrast_target
            print(f"  >>> Enabling contrastive loss (w={w_contrast_target}) "
                  f"at epoch {epoch}")

        model.train()
        epoch_losses = {}
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            hazy = batch["hazy"].to(device)
            clear = batch["clear"].to(device)
            sparse = batch["sparse_depth"].to(device)
            mask = batch["mask"].to(device)
            trans_gt = batch.get("trans_gt")
            if trans_gt is not None:
                trans_gt = trans_gt.to(device)

            with autocast("cuda", enabled=use_amp):
                out = model(hazy, sparse, mask)
                total_loss, loss_dict = criterion(out, hazy, clear, trans_gt=trans_gt)
                if teacher_model is not None and epoch >= distill_start_epoch:
                    with torch.no_grad(), autocast("cuda", enabled=True):
                        teacher_out = teacher_model(hazy, sparse, mask)
                    l_distill = torch.nn.functional.l1_loss(
                        out["restored"], teacher_out["restored"]
                    )
                    total_loss = total_loss + w_distill * l_distill
                    loss_dict["distill"] = l_distill.item()
                    if (w_distill_phys > 0
                            and "physics_restored" in out
                            and "physics_restored" in teacher_out):
                        l_distill_phys = torch.nn.functional.l1_loss(
                            out["physics_restored"], teacher_out["physics_restored"]
                        )
                        total_loss = total_loss + w_distill_phys * l_distill_phys
                        loss_dict["distill_phys"] = l_distill_phys.item()
                    else:
                        loss_dict["distill_phys"] = 0.0
                elif teacher_model is not None:
                    loss_dict["distill"] = 0.0
                    loss_dict["distill_phys"] = 0.0
                loss_for_backward = total_loss / grad_accum_steps

            scaler.scale(loss_for_backward).backward()

            do_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == len(train_loader))
            if do_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=train_cfg.get("grad_clip", 1.0),
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1

            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v

            if (step + 1) % log_every == 0:
                info = " | ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                print(f"  [E{epoch:03d} S{step+1:04d}] {info}")

                log_wandb({
                    f"train/{k}": v for k, v in loss_dict.items()
                }, step=global_step)
                log_wandb({
                    "train/lr": scheduler.get_last_lr()[0],
                }, step=global_step)

        scheduler.step()
        n_steps = len(train_loader)
        avg_train = {k: v / max(n_steps, 1) for k, v in epoch_losses.items()}
        elapsed = time.time() - t0
        lr_now = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch:03d} | train total={avg_train.get('total', 0):.4f} | "
              f"lr={lr_now:.2e} | {elapsed:.1f}s")

        log_wandb({
            **{f"epoch/train_{k}": v for k, v in avg_train.items()},
            "epoch/lr": lr_now,
            "epoch/epoch": epoch,
            "epoch/time_s": elapsed,
        }, step=global_step)

        # ---- Validation ----
        if (epoch + 1) % val_every == 0:
            val_losses = validate(model, val_loader, criterion, device)
            info = " | ".join(f"{k}={v:.4f}" for k, v in val_losses.items())
            print(f"  val: {info}")

            log_wandb({
                **{f"epoch/val_{k}": v for k, v in val_losses.items()},
            }, step=global_step)

            val_total = val_losses.get("total", float("inf"))
            val_psnr = val_losses.get("psnr", 0.0)
            is_best_loss = val_total < best_val_loss
            is_best_psnr = val_psnr > best_val_psnr
            if is_best_loss:
                best_val_loss = val_total
            if is_best_psnr:
                best_val_psnr = val_psnr

            ckpt = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_psnr": best_val_psnr,
                "config": cfg,
                "variant": variant,
                "run_name": run_name,
            }
            torch.save(ckpt, os.path.join(run_dir, "latest.pth"))
            if is_best_loss:
                torch.save(ckpt, os.path.join(run_dir, "best.pth"))
                print(f"  -> New best val loss: {best_val_loss:.4f}")
            if is_best_psnr:
                torch.save(ckpt, os.path.join(run_dir, "best_psnr.pth"))
                print(f"  -> New best val PSNR: {best_val_psnr:.2f} dB")

        # ---- Visualization ----
        if (epoch + 1) % vis_every == 0 or epoch == 0:
            try:
                vis_dir = os.path.join(run_dir, "visualizations")
                visualize_epoch(
                    model, val_ds, device, epoch, cfg,
                    fixed_indices=vis_indices,
                    save_dir=vis_dir,
                )
            except Exception as e:
                print(f"  [WARN] Visualization failed: {e}")

    # ---- Finish ----
    if HAS_WANDB and wandb.run is not None:
        wandb.finish()

    print(f"\nTraining complete.")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val PSNR: {best_val_psnr:.2f} dB")
    print(f"All outputs saved to {run_dir}/")

    # ---- Auto eval ----
    if not args.no_auto_eval:
        import subprocess
        for ckpt_name in ["best_psnr.pth", "best.pth"]:
            ckpt_file = os.path.join(run_dir, ckpt_name)
            if os.path.isfile(ckpt_file):
                print(f"\n{'='*60}")
                print(f"  Auto-eval: {ckpt_name}")
                print(f"{'='*60}")
                subprocess.run(
                    ["python", "eval.py", "--checkpoint", ckpt_file],
                    check=False,
                )


if __name__ == "__main__":
    train()
