#!/usr/bin/env python3
"""
Self-supervised adaptation of PhysDNet to real haze (URHI / RTTS domain).

No clear ground truth exists for real haze, so supervision comes from the
model's own physics structure plus priors:

  L_real = w_recomp * Charbonnier( J_dir*t + A*(1-t),  I_hazy )   (Koschmieder
           re-composition: predictions must re-synthesize the input)
         + w_con    * Charbonnier( J_dir, stopgrad(J_phys) )      (dual-head
           consistency: the parameter-free inversion of the *colored* input
           acts as a chrominance teacher for the NIR-trained direct head)
         + w_dcp    * DarkChannel( J_dir )                        (prior: haze-free
           outdoor images have near-zero dark channel; guards the identity trap)
         + w_smooth * EdgeAwareSmoothness( t, I_hazy )

  L_total = L_real + w_replay * L_STF   (supervised replay batches from the
           original synthetic-STF recipe anchor t/A estimation and prevent
           collapse to J=I, t=1)

Inputs on the real branch use sparse_depth = mask = 0 (no LiDAR at test time,
same as the paper's O-HAZE/I-HAZE zero-shot protocol).

Usage:
    python finetune_real_haze.py \
        --checkpoint "runs/stf_robust_ch64_3.0M_wclear+overcast_crop256_b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185_0406_2105/best_psnr.pth" \
        --urhi-dir "datasets/URHI/images" \
        --stf-root "datasets/stf/SeeingThroughFog" \
        --epochs 5
"""

import argparse
import itertools
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import get_model
from losses import CharbonnierLoss, TransmissionSmoothnessLoss, DehazeLoss
from dataset import STFDehazeDataset


# ── real-haze dataset (no GT) ───────────────────────────────────────────────

class RealHazyDataset(Dataset):
    """Unpaired real hazy images (URHI). Returns hazy crops only."""

    EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

    def __init__(self, root: str, crop: int = 256, augment: bool = True):
        self.paths = sorted(p for p in Path(root).iterdir()
                            if p.suffix in self.EXTS)
        if not self.paths:
            raise FileNotFoundError(f"no images under {root}")
        self.crop = crop
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        for attempt in range(3):  # skip rare corrupt files
            try:
                img = Image.open(self.paths[(idx + attempt) % len(self)])
                img = img.convert("RGB")
                break
            except Exception:
                continue
        else:
            raise RuntimeError("too many corrupt images")

        w, h = img.size
        s = self.crop
        if min(w, h) < s:  # upscale shorter side to crop size
            scale = s / min(w, h)
            img = img.resize((max(s, int(w * scale + 0.5)),
                              max(s, int(h * scale + 0.5))), Image.BILINEAR)
            w, h = img.size
        x0 = random.randint(0, w - s)
        y0 = random.randint(0, h - s)
        img = img.crop((x0, y0, x0 + s, y0 + s))
        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        t = torch.from_numpy(np.asarray(img, np.float32) / 255.0)
        return t.permute(2, 0, 1)  # 3 x s x s


# ── priors / metrics ────────────────────────────────────────────────────────

def dark_channel(img: torch.Tensor, patch: int = 15) -> torch.Tensor:
    """Per-pixel dark channel: min over RGB then min-filter (B,1,H,W)."""
    dc = img.min(dim=1, keepdim=True).values
    return -F.max_pool2d(-dc, patch, stride=1, padding=patch // 2)


def colorfulness(img: torch.Tensor) -> float:
    """Hasler-Süsstrunk colorfulness metric, batch mean (img in [0,1])."""
    r, g, b = img[:, 0] * 255, img[:, 1] * 255, img[:, 2] * 255
    rg = r - g
    yb = 0.5 * (r + g) - b
    std = torch.sqrt(rg.var(dim=(1, 2)) + yb.var(dim=(1, 2)))
    mean = torch.sqrt(rg.mean(dim=(1, 2)) ** 2 + yb.mean(dim=(1, 2)) ** 2)
    return (std + 0.3 * mean).mean().item()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--urhi-dir", default="datasets/URHI/images")
    ap.add_argument("--stf-root", default="datasets/stf/SeeingThroughFog")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="decoder + heads learning rate")
    ap.add_argument("--enc-lr", type=float, default=5e-6,
                    help="encoder learning rate (0 = freeze encoders)")
    ap.add_argument("--w-recomp", type=float, default=1.0)
    ap.add_argument("--w-con", type=float, default=0.2)
    ap.add_argument("--w-dcp", type=float, default=0.05)
    ap.add_argument("--w-smooth", type=float, default=0.1)
    ap.add_argument("--w-replay", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=0,
                    help="stop after N optimizer steps (0 = full epochs); "
                         "for smoke tests")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out_dir or
                   f"runs/finetune_realhaze_{time.strftime('%m%d_%H%M')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(exist_ok=True)

    # ---- model ----
    model = get_model("robust")  # PhysDNet-M (ch64 + SE)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    model = model.to(device).train()
    print(f"[INIT] loaded {args.checkpoint}")

    enc_params, dec_params = [], []
    for n, p in model.named_parameters():
        if n.startswith(("rgb_enc", "dep_enc", "fuse")):
            enc_params.append(p)
        else:
            dec_params.append(p)
    groups = [{"params": dec_params, "lr": args.lr}]
    if args.enc_lr > 0:
        groups.append({"params": enc_params, "lr": args.enc_lr})
    else:
        for p in enc_params:
            p.requires_grad_(False)
        print("[INIT] encoders frozen")
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)

    # ---- data ----
    real_ds = RealHazyDataset(args.urhi_dir, crop=args.crop)
    real_dl = DataLoader(real_ds, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.workers, drop_last=True,
                         persistent_workers=args.workers > 0)
    print(f"[DATA] URHI: {len(real_ds)} real hazy images")

    # official STF clear-weather train split (devkit train_clear_day+night);
    # the weather_station JSONs used for the paper's clear+overcast filter
    # are not on this machine, so this is the reproducible equivalent
    ts_file = "data/stf/meta/train_clear_official.txt"
    if not os.path.exists(ts_file):
        print(f"[DATA] {ts_file} missing — falling back to all frames")
        ts_file = None
    stf_ds = STFDehazeDataset(
        stf_root=args.stf_root, timestamps_file=ts_file,
        crop_size=(args.crop, args.crop),
        beta_range=(0.005, 0.04), airlight_range=(0.7, 1.0), augment=True)
    stf_dl = DataLoader(stf_ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, drop_last=True,
                        persistent_workers=args.workers > 0)
    stf_iter = itertools.cycle(stf_dl)
    print(f"[DATA] STF replay: {len(stf_ds)} frames")

    # ---- losses ----
    charb = CharbonnierLoss().to(device)
    smooth = TransmissionSmoothnessLoss().to(device)
    replay_loss = DehazeLoss(w_perc=0.0, use_perceptual=False,
                             w_fft=0.1).to(device)

    # fixed batch for qualitative tracking
    fixed = torch.stack([real_ds[i] for i in range(0, 64, 8)]).to(device)

    step = 0
    for epoch in range(1, args.epochs + 1):
        for hazy in real_dl:
            if args.max_steps and step >= args.max_steps:
                break
            hazy = hazy.to(device, non_blocking=True)
            B = hazy.shape[0]
            zeros = torch.zeros(B, 1, args.crop, args.crop, device=device)

            # ---- real branch (self-supervised) ----
            out = model(hazy, zeros, zeros)
            J, t, A = out["restored"], out["transmission"], out["airlight"]
            A_sp = A[:, :, None, None]

            recomp = J * t + A_sp * (1.0 - t)
            l_recomp = charb(recomp, hazy)
            l_con = charb(J, out["physics_restored"].detach())
            l_dcp = dark_channel(J).mean()
            l_smooth = smooth(t, hazy)
            loss_real = (args.w_recomp * l_recomp + args.w_con * l_con +
                         args.w_dcp * l_dcp + args.w_smooth * l_smooth)

            # backward each branch separately: gradients accumulate the same,
            # but the real-branch graph is freed before the replay forward —
            # holding both graphs spills past 16 GB VRAM and thrashes
            opt.zero_grad(set_to_none=True)
            loss_real.backward()

            # ---- replay branch (supervised synthetic STF) ----
            stf = next(stf_iter)
            s_out = model(stf["hazy"].to(device),
                          stf["sparse_depth"].to(device),
                          stf["mask"].to(device))
            loss_stf, _ = replay_loss(s_out, stf["hazy"].to(device),
                                      stf["clear"].to(device),
                                      stf["trans_gt"].to(device))
            (args.w_replay * loss_stf).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in groups for p in g["params"]], 1.0)
            opt.step()

            if step % 50 == 0:
                with torch.no_grad():
                    cf_in = colorfulness(hazy)
                    cf_out = colorfulness(J)
                print(f"E{epoch} s{step:05d} | real {loss_real.item():.4f} "
                      f"(rc {l_recomp.item():.4f} con {l_con.item():.4f} "
                      f"dcp {l_dcp.item():.4f} sm {l_smooth.item():.4f}) | "
                      f"stf {loss_stf.item():.4f} | "
                      f"CF {cf_in:.1f}->{cf_out:.1f} | "
                      f"t̄ {t.mean().item():.3f}", flush=True)
            step += 1

        # ---- end of epoch: checkpoint + sample grid ----
        torch.save({"model": model.state_dict(), "epoch": epoch,
                    "args": vars(args)}, out_dir / f"adapted_ep{epoch}.pth")
        model.eval()
        with torch.no_grad():
            z = torch.zeros(fixed.shape[0], 1, args.crop, args.crop,
                            device=device)
            fo = model(fixed, z, z)
            grid = torch.cat([fixed, fo["restored"],
                              fo["physics_restored"]], dim=3)
            grid = (grid.clamp(0, 1) * 255).byte().cpu()
            for i in range(grid.shape[0]):
                Image.fromarray(
                    grid[i].permute(1, 2, 0).numpy()).save(
                    out_dir / "samples" / f"ep{epoch}_{i}.png")
        model.train()
        print(f"[EPOCH {epoch}] saved adapted_ep{epoch}.pth + samples")

    print(f"[DONE] checkpoints in {out_dir}")


if __name__ == "__main__":
    main()
