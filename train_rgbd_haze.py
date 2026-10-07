#!/usr/bin/env python3
"""
RGBD Koschmieder dehazing on SUN-RGBD kv2 (Kinect-v2 scenes): the anchored
RGB counterpart of the RESIDE-6K experiment.

Evidence-matrix cell this fills: dehazing + RGB + depth-anchored t. STF vs
RESIDE confounds modality and anchoring; this isolates the anchor within
the paper's own task, using the UNMODIFIED LiDARDehazeNet and the paper's
loss recipe (Table 9, incl. w_trans=2.0 supervised by depth-derived t).

Synthesis (on the fly, exact Koschmieder): t = exp(-beta*d) with
beta ~ U(0.1, 0.8), chosen so GT transmission spans light-to-dense at
Kinect depth statistics (~45% of pixels dense t<0.2, ~26% light t>0.5);
per-channel airlight A_c ~ U(0.7, 1.0); sparse-depth input at 3% of pixels
with validity mask (LiDAR protocol analog). Test synthesis is seeded per
index for reproducible evaluation.

Runs (identical budget/seed per pair):
    python train_rgbd_haze.py --tag kv2_physics
    python train_rgbd_haze.py --tag kv2_nophy --nophy

End of each run, automatically (all pushed to wandb, project
physdnet-rgbdhaze): full-test PSNR + GT-t-binned PSNR (dense/mid/light) +
SSIM + CIEDE2000, qualitative panels (hazy | direct | physics | GT), and a
test-time depth-density sweep (keep ratio 0.10/0.03/0.01/0.003/0 — the
App. N analog). The physics run also saves grad_bins.npz (Fig. 3 analog).

Depends on train_underwater.py (shared eval utilities) — pull both files.
"""

import argparse
import math
import random
import time
from pathlib import Path

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import get_model
from losses import DehazeLoss
from train_underwater import (ssim, delta_e00, GradBinner, save_qualitative,
                              psnr, MAX_DEPTH)


class RGBDHazeDataset(Dataset):
    """SUN-RGBD scenes -> on-the-fly Koschmieder haze with sparse depth."""

    DEFAULT_ROOT = "datasets/SUNRGBD/kv2/kinect2data"
    # chosen so GT-t spans light-to-dense on Kinect depth statistics
    # (~45% of pixels below t=0.2, ~26% above 0.5; ITS's [0.6,1.8] range
    # saturates at kv2's closer depths, STF's [0.005,0.04] never densifies)
    BETA = (0.1, 0.8)
    ALIGHT = (0.7, 1.0)

    def __init__(self, root=None, crop=256, train=True, keep_ratio=0.03):
        root = root or self.DEFAULT_ROOT
        dirs = [d for d in sorted(Path(root).iterdir())
                if (d / "image").exists() and (d / "depth_bfx").exists()]
        assert dirs, f"no RGBD scene dirs under {root}"
        n_test = int(0.15 * len(dirs))
        self.dirs = dirs[:-n_test] if train else dirs[-n_test:]
        self.crop = crop
        self.train = train
        self.keep_ratio = keep_ratio

    def __len__(self):
        return len(self.dirs)

    def _load(self, d):
        img = Image.open(next((d / "image").iterdir())).convert("RGB")
        dep = Image.open(next((d / "depth_bfx").iterdir()))
        depth = (np.asarray(dep).astype(np.uint16) >> 3).astype(
            np.float32) / 1000.0
        rgb = np.asarray(img, np.float32) / 255.0
        if rgb.shape[:2] != depth.shape:
            img = img.resize((depth.shape[1], depth.shape[0]),
                             Image.BILINEAR)
            rgb = np.asarray(img, np.float32) / 255.0
        return rgb, depth

    def __getitem__(self, idx):
        rng = np.random if self.train else np.random.RandomState(2000 + idx)
        rgb, depth = self._load(self.dirs[idx])
        H, W = depth.shape

        s = self.crop
        if self.train:
            if H < s or W < s:
                sc = s / min(H, W)
                rgb = np.asarray(Image.fromarray(
                    (rgb * 255).astype(np.uint8)).resize(
                    (math.ceil(W * sc), math.ceil(H * sc)))) / 255.0
                depth = np.asarray(Image.fromarray(depth).resize(
                    (math.ceil(W * sc), math.ceil(H * sc)),
                    Image.BILINEAR))
                H, W = depth.shape
            y0 = rng.randint(0, H - s + 1)
            x0 = rng.randint(0, W - s + 1)
            rgb = rgb[y0:y0 + s, x0:x0 + s]
            depth = depth[y0:y0 + s, x0:x0 + s]
            if rng.rand() < 0.5:
                rgb = rgb[:, ::-1].copy()
                depth = depth[:, ::-1].copy()
        H, W = depth.shape

        beta = rng.uniform(*self.BETA)
        A = np.array([rng.uniform(*self.ALIGHT) for _ in range(3)],
                     np.float32)
        t = np.exp(-beta * depth).astype(np.float32)          # H,W (1-ch)
        t3 = t[..., None]
        hazy = rgb * t3 + A[None, None, :] * (1.0 - t3)
        hazy += rng.randn(*hazy.shape).astype(np.float32) * 0.005
        hazy = np.clip(hazy, 0.0, 1.0).astype(np.float32)

        mask = np.zeros(H * W, np.float32)
        if self.keep_ratio > 0:
            mask[rng.choice(H * W, max(1, int(H * W * self.keep_ratio)),
                            replace=False)] = 1.0
        mask = mask.reshape(H, W)
        sparse = (depth / MAX_DEPTH) * mask

        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).float()
        return {
            "hazy": to(hazy.transpose(2, 0, 1)),
            "clear": to(rgb.transpose(2, 0, 1)),
            "sparse_depth": to(sparse[None]),
            "mask": to(mask[None]),
            "trans_gt": to(t[None]),                          # 1-ch GT t
        }


@torch.no_grad()
def evaluate(model, ds, device, limit=None, detailed=False):
    """Overall PSNR (+ GT-t-binned PSNR, SSIM, CIEDE2000 when detailed)."""
    n = len(ds) if limit is None else min(limit, len(ds))
    tot = 0.0
    sq = {"dense": [0.0, 0], "mid": [0.0, 0], "light": [0.0, 0]}
    extra = {"ssim": [], "de00": []}
    for i in range(n):
        s = ds[i]
        x = s["hazy"][None].to(device)
        g = s["clear"][None].to(device)
        _, _, H, W = x.shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        sd = F.pad(s["sparse_depth"][None].to(device), (0, pw, 0, ph))
        m = F.pad(s["mask"][None].to(device), (0, pw, 0, ph))
        out = model(x, sd, m)["restored"][:, :, :H, :W].clamp(0, 1)
        tot += psnr(out, g)
        if detailed:
            extra["ssim"].append(ssim(out, g))
            de = delta_e00(out, g)
            if de is not None:
                extra["de00"].append(de)
            err = (out - g) ** 2
            tgt = s["trans_gt"][None].to(device).expand_as(err)
            for name, lo, hi in (("dense", 0.0, 0.2), ("mid", 0.2, 0.5),
                                 ("light", 0.5, 1.01)):
                msk = (tgt >= lo) & (tgt < hi)
                if msk.any():
                    sq[name][0] += err[msk].sum().item()
                    sq[name][1] += int(msk.sum().item())
    if not detailed:
        return tot / n
    bins = {k: (10 * math.log10(1.0 / max(v[0] / v[1], 1e-10))
                if v[1] else float("nan")) for k, v in sq.items()}
    bins["ssim"] = float(np.mean(extra["ssim"]))
    if extra["de00"]:
        bins["de00"] = float(np.mean(extra["de00"]))
    return tot / n, bins


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nophy", action="store_true")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--keep-ratio", type=float, default=0.03)
    ap.add_argument("--grad-bin-every", type=int, default=100)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    out_dir = Path(f"runs/rgbdhaze_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project="physdnet-rgbdhaze",
                            name=f"rgbdhaze_{args.tag}", config=vars(args))
        except Exception as e:
            print(f"[WANDB] disabled ({e})", flush=True)

    # UNMODIFIED paper model + recipe
    model = get_model("robust", use_physics_head=not args.nophy).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] LiDARDehazeNet robust "
          f"{'NoPhy' if args.nophy else 'physics'} ({n_params/1e6:.2f}M)",
          flush=True)

    if args.nophy:
        loss_fn = DehazeLoss(w_rec=1.0, w_perc=0.05, w_grad=0.5, w_phys=0.0,
                             w_phys_rec=0.0, w_smooth=0.0, w_trans=0.0,
                             w_fft=0.1).to(device)
    else:
        loss_fn = DehazeLoss(w_rec=1.0, w_perc=0.05, w_grad=0.5, w_phys=0.2,
                             w_phys_rec=1.0, w_smooth=0.1, w_trans=2.0,
                             w_fft=0.1).to(device)

    train_ds = RGBDHazeDataset(root=args.data_root, crop=args.crop,
                               train=True, keep_ratio=args.keep_ratio)
    test_ds = RGBDHazeDataset(root=args.data_root, train=False,
                              keep_ratio=args.keep_ratio)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.workers, drop_last=True,
                          pin_memory=True,
                          persistent_workers=args.workers > 0)
    print(f"[DATA] train {len(train_ds)} | test {len(test_ds)}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(train_dl), eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda")
    binner = GradBinner()

    best, step = 0.0, 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for b in train_dl:
            hazy = b["hazy"].to(device, non_blocking=True)
            clear = b["clear"].to(device, non_blocking=True)
            sd = b["sparse_depth"].to(device, non_blocking=True)
            m = b["mask"].to(device, non_blocking=True)
            tgt = b["trans_gt"].to(device, non_blocking=True)

            bin_step = (not args.nophy) and step % args.grad_bin_every == 0
            with torch.amp.autocast("cuda", enabled=not bin_step):
                out = model(hazy, sd, m)
                if bin_step:
                    out["transmission"].retain_grad()
                loss, _ = loss_fn(out, hazy, clear, tgt)

            opt.zero_grad(set_to_none=True)
            if bin_step:
                loss.backward()
                if out["transmission"].grad is not None:
                    binner.update(out["transmission"],
                                  out["transmission"].grad)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            sched.step()
            run += loss.item()
            step += 1

        msg = (f"E{epoch:03d} | loss {run/len(train_dl):.4f} | "
               f"{time.time()-t0:.0f}s | lr {sched.get_last_lr()[0]:.2e}")
        log = {"epoch": epoch, "train_loss": run / len(train_dl),
               "lr": sched.get_last_lr()[0]}

        if epoch % args.val_every == 0 or epoch == args.epochs:
            model.eval()
            v = evaluate(model, test_ds, device, limit=60)
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
            if not args.nophy:
                binner.save(out_dir / "grad_bins.npz")
        if wb:
            wb.log(log)
        print(msg, flush=True)

    # ---- final: full test, bins, panels, depth sweep (all -> wandb) ----
    ck = torch.load(out_dir / "best_psnr.pth", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    final, bins = evaluate(model, test_ds, device, limit=None, detailed=True)
    print(f"[FINAL] {args.tag}: full test PSNR {final:.2f} dB "
          f"(best@{ck['epoch']}) | " +
          " | ".join(f"{k} {v:.3f}" for k, v in bins.items()), flush=True)

    import json
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump({"psnr": final, **bins, "seed": args.seed}, f, indent=2)

    qdir = save_qualitative(model, test_ds, device, out_dir,
                            physics=not args.nophy)
    print(f"[QUAL] panels saved to {qdir}", flush=True)

    sweep = {}
    for kr in (0.10, 0.03, 0.01, 0.003, 0.0):
        ds_k = RGBDHazeDataset(root=args.data_root, train=False,
                               keep_ratio=kr)
        sweep[str(kr)] = round(evaluate(model, ds_k, device, limit=None), 3)
        print(f"[SWEEP] keep_ratio {kr}: PSNR {sweep[str(kr)]}", flush=True)
    with open(out_dir / "depth_sweep.json", "w") as f:
        json.dump(sweep, f, indent=2)

    if wb:
        wb.summary["final_test_psnr"] = final
        for k, v in bins.items():
            wb.summary[f"final_{k}"] = v
        for k, v in sweep.items():
            wb.summary[f"sweep_psnr_keep{k}"] = v
        try:
            import wandb as _wandb
            wb.log({"qualitative": [_wandb.Image(str(p))
                                    for p in sorted(qdir.glob("*.png"))]})
        except Exception:
            pass
        wb.finish()
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
