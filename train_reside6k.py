#!/usr/bin/env python3
"""
Train PhysDNet on RESIDE-6K (standard RGB-only dehazing benchmark).

Purpose: reviewer experiment — does physics-guided gradient amplification
help in a pure single-modality RGB setting with no LiDAR and no
transmission supervision? The depth branch receives zeros (the paper's
mask-aware gating handles absent LiDAR); t-hat is anchored only through the
physics reconstruction loss.

Recommended run matrix (600 epochs; 300-ep runs were still converging):

  # baseline pair, unanchored (replicates the original comparison, longer)
  python train_reside6k.py --tag physics600 --epochs 600
  python train_reside6k.py --tag nophy600 --nophy --epochs 600

  # anchored + density-augmented pair (the mechanism-relevant setting:
  # analytic t supervision restores the anchor; t'=t^k widens the dense
  # regime with exact physics; identical data for both variants)
  python train_reside6k.py --tag physics600_anc --epochs 600 \
      --w-trans 2.0 --haze-aug 0.5 3.0
  python train_reside6k.py --tag nophy600_anc --nophy --epochs 600 \
      --haze-aug 0.5 3.0

Report both pairs; the anchored pair changes the training distribution for
both variants equally, and w_trans mirrors the paper's STF recipe (Table 9).
Test evaluation is always on the untouched standard test set.

Data layout (DehazeFormer release):
    datasets/RESIDE-6K/train/{GT,hazy}   6000 pairs
    datasets/RESIDE-6K/test/{GT,hazy}    1000 pairs (SOTS-mix)
"""

import argparse
import math
import random
import time
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()  # Norton TLS interception (VGG download)
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import get_model
from losses import DehazeLoss


class PairedHazeDataset(Dataset):
    """RESIDE-style paired hazy/GT folders; returns zero depth + mask.

    RESIDE filenames encode the synthesis parameters (`id_A_beta.ext`), so
    the true transmission is analytically recoverable per pixel:
        t(x) = (I(x) - A) / (J(x) - A)        (valid where |J - A| > eps)
    This enables (a) anchoring supervision for the physics head on a pure
    RGB benchmark (`--w-trans`), and (b) an exact-physics density
    augmentation: t' = t^k re-renders the same scene at beta' = k*beta on
    the same depth (`--haze-aug p k_max`). Both are applied identically to
    physics and NoPhy runs; NoPhy simply has no transmission head, matching
    the paper's ablation definition.
    """

    def __init__(self, root, crop=256, train=True,
                 haze_aug=None, want_trans=False):
        self.hazy = sorted((Path(root) / "hazy").glob("*"))
        self.gt_dir = Path(root) / "GT"
        self.crop = crop
        self.train = train
        self.haze_aug = haze_aug          # (p, k_max) or None
        self.want_trans = want_trans or bool(haze_aug)
        assert self.hazy, f"no images in {root}/hazy"

    @staticmethod
    def _parse_ab(path):
        try:
            parts = path.stem.split("_")
            return float(parts[1]), float(parts[2])
        except (IndexError, ValueError):
            return None, None

    @staticmethod
    def _estimate_A(I, J):
        """Estimate the scalar airlight when filenames don't carry it
        (RESIDE-6K train split). Physical constraint: the true t is shared
        across channels, so the correct A minimises the cross-channel
        variance of t_A = (I - A)/(J - A) over well-conditioned pixels."""
        best_A, best_v = None, None
        for stage in (np.arange(0.60, 1.001, 0.02),):
            for A in stage:
                denom = J - A
                ok = (np.abs(denom) > 0.05).all(axis=2)
                if ok.mean() < 0.05:
                    continue
                t = np.clip((I - A) / np.where(np.abs(denom) > 0.05,
                                               denom, 1.0), 0.02, 1.2)
                v = t.var(axis=2)[ok].mean()
                if best_v is None or v < best_v:
                    best_v, best_A = v, float(A)
        if best_A is None:
            return None
        # refine
        for A in np.arange(max(0.55, best_A - 0.02),
                           min(1.0, best_A + 0.02) + 1e-9, 0.005):
            denom = J - A
            ok = (np.abs(denom) > 0.05).all(axis=2)
            if ok.mean() < 0.05:
                continue
            t = np.clip((I - A) / np.where(np.abs(denom) > 0.05,
                                           denom, 1.0), 0.02, 1.2)
            v = t.var(axis=2)[ok].mean()
            if v < best_v:
                best_v, best_A = v, float(A)
        return best_A

    def __len__(self):
        return len(self.hazy)

    def _gt_path(self, hazy_path):
        # RESIDE naming: hazy '0001_0.8_0.2.png' -> GT '0001.png'; 6K uses
        # matching stems; try exact, then base-id prefix
        p = self.gt_dir / hazy_path.name
        if p.exists():
            return p
        base = hazy_path.stem.split("_")[0]
        for ext in (".png", ".jpg", ".jpeg"):
            q = self.gt_dir / f"{base}{ext}"
            if q.exists():
                return q
        raise FileNotFoundError(f"no GT for {hazy_path.name}")

    def __getitem__(self, idx):
        hp = self.hazy[idx]
        hazy = Image.open(hp).convert("RGB")
        gt = Image.open(self._gt_path(hp)).convert("RGB")
        if gt.size != hazy.size:
            gt = gt.resize(hazy.size, Image.BILINEAR)

        if self.train:
            s = self.crop
            w, h = hazy.size
            if w < s or h < s:
                sc = s / min(w, h)
                hazy = hazy.resize((math.ceil(w * sc), math.ceil(h * sc)),
                                   Image.BILINEAR)
                gt = gt.resize(hazy.size, Image.BILINEAR)
                w, h = hazy.size
            x0, y0 = random.randint(0, w - s), random.randint(0, h - s)
            hazy = hazy.crop((x0, y0, x0 + s, y0 + s))
            gt = gt.crop((x0, y0, x0 + s, y0 + s))
            if random.random() < 0.5:
                hazy = hazy.transpose(Image.FLIP_LEFT_RIGHT)
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)

        I = np.asarray(hazy, np.float32) / 255.0
        J = np.asarray(gt, np.float32) / 255.0
        H, W = I.shape[:2]
        t_map = np.zeros((H, W), np.float32)
        t_valid = np.zeros((H, W), np.float32)

        if self.want_trans:
            A, _beta = self._parse_ab(hp)
            if A is None:  # train split: filenames carry no parameters
                A = self._estimate_A(I, J)
            if A is not None:
                denom = J - A
                ok = np.abs(denom) > 0.03            # ill-conditioned near A
                t_ch = np.clip((I - A) / np.where(ok, denom, 1.0), 0.02, 1.0)
                n_ok = ok.sum(axis=2)
                t_map = np.where(
                    n_ok > 0,
                    (t_ch * ok).sum(axis=2) / np.maximum(n_ok, 1), 0.0
                ).astype(np.float32)
                t_valid = (n_ok >= 2).astype(np.float32)

                # exact-physics density augmentation: t' = t^k  (beta' = k*beta)
                if (self.train and self.haze_aug is not None
                        and random.random() < self.haze_aug[0]):
                    k = random.uniform(1.0, self.haze_aug[1])
                    t_map = np.where(t_valid > 0, t_map ** k, t_map)
                    t3 = t_map[..., None]
                    I = np.where(t_valid[..., None] > 0,
                                 J * t3 + A * (1.0 - t3), I)
                    I = np.clip(
                        I + np.random.randn(*I.shape).astype(np.float32)
                        * 0.005, 0.0, 1.0).astype(np.float32)

        to_t = lambda a: torch.from_numpy(
            np.ascontiguousarray(a)).float()
        h_t = to_t(I.transpose(2, 0, 1))
        g_t = to_t(J.transpose(2, 0, 1))
        z = torch.zeros(1, H, W)
        return {"hazy": h_t, "clear": g_t, "sparse_depth": z, "mask": z,
                "trans_gt": to_t(t_map[None]),
                "trans_valid": to_t(t_valid[None])}


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))


@torch.no_grad()
def evaluate(model, root, device, limit=None):
    """Full-resolution test-set PSNR/SSIM-free quick eval (PSNR only)."""
    ds = PairedHazeDataset(root, train=False)
    n = len(ds) if limit is None else min(limit, len(ds))
    tot = 0.0
    for i in range(n):
        s = ds[i]
        x = s["hazy"][None].to(device)
        g = s["clear"][None].to(device)
        _, _, H, W = x.shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        z = torch.zeros(1, 1, x.shape[2], x.shape[3], device=device)
        out = model(x, z, z)["restored"][:, :, :H, :W]
        tot += psnr(out.clamp(0, 1), g)
    return tot / n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default="RESIDE-6K/")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nophy", action="store_true",
                    help="remove physics head + physics losses (ablation)")
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--val-limit", type=int, default=200)
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--haze-aug", type=float, nargs=2, default=None,
                    metavar=("P", "K_MAX"),
                    help="density augmentation: with prob P re-render the "
                         "pair at beta'=k*beta, k~U(1,K_MAX) (exact physics "
                         "via t'=t^k). Applied identically to both variants.")
    ap.add_argument("--w-trans", type=float, default=0.0,
                    help="masked L1 on analytically-recovered transmission "
                         "(anchoring supervision; ignored for --nophy)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    out_dir = Path(f"runs/reside6k_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project="physdnet-reside6k",
                            name=f"reside6k_{args.tag}",
                            config=vars(args))
        except Exception as e:
            print(f"[WANDB] disabled ({e})", flush=True)

    model = get_model("robust", use_physics_head=not args.nophy).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] PhysDNet-M {'NoPhy' if args.nophy else 'physics'} "
          f"({n_params/1e6:.2f}M params)", flush=True)

    # paper recipe minus LiDAR-dependent terms (w_trans=0: no t GT in RESIDE)
    if args.nophy:
        loss_fn = DehazeLoss(w_rec=1.0, w_perc=0.05, w_grad=0.5,
                             w_phys=0.0, w_phys_rec=0.0, w_smooth=0.0,
                             w_trans=0.0, w_fft=0.1).to(device)
    else:
        loss_fn = DehazeLoss(w_rec=1.0, w_perc=0.05, w_grad=0.5,
                             w_phys=0.2, w_phys_rec=1.0, w_smooth=0.1,
                             w_trans=0.0, w_fft=0.1).to(device)

    train_ds = PairedHazeDataset(
        Path(args.data_root) / "train", crop=args.crop, train=True,
        haze_aug=tuple(args.haze_aug) if args.haze_aug else None,
        want_trans=args.w_trans > 0)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=True, persistent_workers=args.workers > 0)
    print(f"[DATA] train {len(train_ds)} pairs", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_dl), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for b in train_dl:
            hazy = b["hazy"].to(device, non_blocking=True)
            clear = b["clear"].to(device, non_blocking=True)
            z = b["sparse_depth"].to(device, non_blocking=True)
            m = b["mask"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                out = model(hazy, z, m)
                loss, _ = loss_fn(out, hazy, clear, None)
                # anchoring supervision from analytically-recovered t
                if args.w_trans > 0 and not args.nophy:
                    tg = b["trans_gt"].to(device, non_blocking=True)
                    tv = b["trans_valid"].to(device, non_blocking=True)
                    denom = tv.sum().clamp(min=1.0)
                    l_t = ((out["transmission"] - tg).abs() * tv).sum() / denom
                    loss = loss + args.w_trans * l_t

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += loss.item()

        msg = (f"E{epoch:03d} | loss {run/len(train_dl):.4f} | "
               f"{time.time()-t0:.0f}s | lr {sched.get_last_lr()[0]:.2e}")
        log = {"epoch": epoch, "train_loss": run / len(train_dl),
               "lr": sched.get_last_lr()[0]}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            v = evaluate(model, Path(args.data_root) / "test", device,
                         limit=args.val_limit)
            msg += f" | val PSNR {v:.2f}"
            log["val_psnr"] = v
            if v > best:
                best = v
                torch.save({"model": model.state_dict(), "epoch": epoch,
                            "val_psnr": v,
                            "config": {"model": {
                                "variant": "robust", "base_ch": 64,
                                "use_physics_head": not args.nophy}}},
                           out_dir / "best_psnr.pth")
                msg += " *best*"
        if wb:
            wb.log(log)
        print(msg, flush=True)

    # final: full test set at full resolution with best checkpoint
    ck = torch.load(out_dir / "best_psnr.pth", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    final = evaluate(model, Path(args.data_root) / "test", device, limit=None)
    print(f"[FINAL] {args.tag}: full test PSNR {final:.2f} dB "
          f"(best-val epoch {ck['epoch']})", flush=True)
    with open(out_dir / "final_psnr.txt", "w") as f:
        f.write(f"{final:.4f}\n")
    if wb:
        wb.summary["final_test_psnr"] = final
        wb.finish()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
