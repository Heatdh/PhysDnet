#!/usr/bin/env python3
"""
Underwater-restoration inverse-problem experiment (NYU-v2 RGBD synthesis).

Purpose: test whether physics-driven gradient amplification generalizes to a
second depth-anchored inverse problem. Underwater image formation (UWCNN-style
synthesis from in-air RGBD):

    I_c(x) = J_c(x) * t_c(x) + B_c * (1 - t_c(x)),   t_c(x) = exp(-beta_c d(x))

with per-channel attenuation beta_r > beta_g > beta_b (red dies first — the
dense-degradation regime lives in the red channel) and greenish-blue veiling
light B. The physics head predicts a per-channel transmission t_c (3ch) and
B (3ch), inverts in closed form, and — as on STF — t is anchored by
depth-derived ground truth (w_trans) with a sparse depth input (3% of pixels,
mirroring the LiDAR protocol).

    python train_underwater.py --tag physics
    python train_underwater.py --tag nophy --nophy

Data: SUNRGBD kv1/NYUdata (1,449 NYU-v2 RGBD pairs, already on disk).
Depth encoding: metres = (uint16 >> 3) / 1000.

v2 (pre-registered comparison axes, no post-hoc selection): harder
degradation (higher beta, signal-dependent noise sigma = s0 + s1*(1-t)) so
the dense regime is information-poor; evaluation reports overall PSNR plus
per-GT-transmission-bin PSNR (dense/mid/light), red-channel PSNR, and
convergence curves via wandb; --train-frac for data-efficiency runs. The
mechanism's predictions: physics > nophy in the dense bin, red channel, and
early-budget PSNR; overall converged PSNR may equalize (cf. App. E
saturation). Run >=3 seeds before quoting any delta.
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
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from model import LiDARDehazeNet
from losses import DehazeLoss

MAX_DEPTH = 10.0  # metres, for input normalisation


# ── data ────────────────────────────────────────────────────────────────────

class NYUUnderwaterDataset(Dataset):
    """NYU-v2 RGBD -> synthetic underwater pairs (per-channel Koschmieder).

    Split: deterministic 85/15 by sorted scene id. Test synthesis is seeded
    per index so evaluation is reproducible.
    """

    # v2: harder, physically-motivated degradation so the dense regime is
    # genuinely information-poor (red-channel t reaches ~0.03 at NYU depths)
    BETA = {"r": (0.60, 1.50), "g": (0.15, 0.60), "b": (0.05, 0.35)}
    BLIGHT = {"r": (0.00, 0.15), "g": (0.25, 0.55), "b": (0.35, 0.75)}
    NOISE0 = 0.005   # base read-noise
    NOISE1 = 0.020   # extra noise where signal is attenuated: sigma(x) =
                     # NOISE0 + NOISE1*(1-t(x)) — constant sensor noise
                     # relative to an attenuated signal

    DEFAULT_ROOT = "datasets/SUNRGBD/kv1/NYUdata"

    def __init__(self, root=None, crop=256, train=True, keep_ratio=0.03):
        root = root or self.DEFAULT_ROOT
        # any SUN-RGBD subset: scene dirs are those with image/ + depth_bfx/
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
        depth = (np.asarray(dep).astype(np.uint16) >> 3).astype(np.float32) / 1000.0
        rgb = np.asarray(img, np.float32) / 255.0
        return rgb, depth

    def __getitem__(self, idx):
        rng = np.random if self.train else np.random.RandomState(1000 + idx)
        rgb, depth = self._load(self.dirs[idx])
        H, W = depth.shape

        s = self.crop
        if self.train:
            if H < s or W < s:  # upscale rare small frames
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

        # per-channel underwater synthesis
        beta = np.array([rng.uniform(*self.BETA["r"]),
                         rng.uniform(*self.BETA["g"]),
                         rng.uniform(*self.BETA["b"])], np.float32)
        B = np.array([rng.uniform(*self.BLIGHT["r"]),
                      rng.uniform(*self.BLIGHT["g"]),
                      rng.uniform(*self.BLIGHT["b"])], np.float32)
        t = np.exp(-beta[None, None, :] * depth[..., None])  # H,W,3
        hazy = rgb * t + B[None, None, :] * (1.0 - t)
        sigma = self.NOISE0 + self.NOISE1 * (1.0 - t)
        hazy += rng.randn(*hazy.shape).astype(np.float32) * sigma
        hazy = np.clip(hazy, 0.0, 1.0).astype(np.float32)

        # sparse depth prior (LiDAR-protocol analog)
        mask = np.zeros(H * W, np.float32)
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
            "trans_gt": to(t.transpose(2, 0, 1)),  # 3ch per-channel t
        }


# ── model: per-channel transmission head ───────────────────────────────────

class UnderwaterPhysDNet(LiDARDehazeNet):
    """PhysDNet-M with a 3-channel transmission head (per-channel beta)."""

    def __init__(self, use_physics_head=True):
        super().__init__(base_ch=64, use_attention=True,
                         use_physics_head=use_physics_head)
        if use_physics_head:
            c = 64
            self.head_trans[-1] = nn.Conv2d(c, 3, 1)  # 1ch -> 3ch

    def forward(self, rgb_hazy, sparse_depth, mask):
        depth_in = torch.cat([sparse_depth, mask], dim=1)
        r1 = self.rgb_enc1(rgb_hazy); d1 = self.dep_enc1(depth_in)
        f1 = self.fuse1(r1, d1)
        r2 = self.rgb_enc2(self.pool(r1)); d2 = self.dep_enc2(self.pool(d1))
        f2 = self.fuse2(r2, d2)
        r3 = self.rgb_enc3(self.pool(r2)); d3 = self.dep_enc3(self.pool(d2))
        f3 = self.fuse3(r3, d3)
        r4 = self.rgb_enc4(self.pool(r3)); d4 = self.dep_enc4(self.pool(d3))
        f4 = self.bottleneck_attn(self.fuse4(r4, d4))

        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)

        restored = self.head_restore(x)

        if self.use_physics_head:
            t_raw = self.head_trans(x)
            t_hat = self.t_min + (self.t_max - self.t_min) * torch.sigmoid(t_raw)
            B_hat = self.head_airlight(f4)                    # veiling light
            eps = 1e-4
            B_sp = B_hat[:, :, None, None]
            physics_restored = (rgb_hazy - B_sp * (1.0 - t_hat)) / (t_hat + eps)
            physics_restored = torch.clamp(physics_restored, 0.0, 1.0)
        else:
            b, _, H, W = rgb_hazy.shape
            t_hat = torch.zeros(b, 3, H, W, device=rgb_hazy.device)
            B_hat = torch.zeros(b, 3, device=rgb_hazy.device)
            physics_restored = torch.zeros_like(restored)

        return {"restored": restored, "transmission": t_hat,
                "airlight": B_hat, "physics_restored": physics_restored}


# ── evaluation / gradient binning ──────────────────────────────────────────

def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def ssim(pred, target, window=11, C1=0.01 ** 2, C2=0.03 ** 2):
    """Conv-based SSIM, mean over channels (no external deps)."""
    k = torch.ones(1, 1, window, window, device=pred.device) / window ** 2
    pad = window // 2
    vals = []
    for c in range(pred.shape[1]):
        p, t = pred[:, c:c + 1], target[:, c:c + 1]
        mp, mt = F.conv2d(p, k, padding=pad), F.conv2d(t, k, padding=pad)
        sp = F.conv2d(p * p, k, padding=pad) - mp ** 2
        st = F.conv2d(t * t, k, padding=pad) - mt ** 2
        spt = F.conv2d(p * t, k, padding=pad) - mp * mt
        num = (2 * mp * mt + C1) * (2 * spt + C2)
        den = (mp ** 2 + mt ** 2 + C1) * (sp + st + C2)
        vals.append((num / (den + 1e-8)).mean().item())
    return float(np.mean(vals))


def delta_e00(pred, target):
    """Mean CIEDE2000 vs GT (colour fidelity); None if skimage missing."""
    try:
        from skimage.color import rgb2lab, deltaE_ciede2000
    except ImportError:
        return None
    p = pred[0].permute(1, 2, 0).cpu().numpy()
    t = target[0].permute(1, 2, 0).cpu().numpy()
    return float(deltaE_ciede2000(rgb2lab(t), rgb2lab(p)).mean())


@torch.no_grad()
def evaluate(model, ds, device, limit=None, detailed=False):
    """Overall PSNR; with detailed=True also per-GT-transmission-bin PSNR
    (dense t<0.2 / mid 0.2-0.5 / light t>0.5) and red-channel PSNR — the
    axes where the amplification mechanism predicts its effect."""
    n = len(ds) if limit is None else min(limit, len(ds))
    tot = 0.0
    sq = {"dense": [0.0, 0], "mid": [0.0, 0], "light": [0.0, 0],
          "red": [0.0, 0]}
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
            sq.setdefault("_ssim", []).append(ssim(out, g))
            de = delta_e00(out, g)
            if de is not None:
                sq.setdefault("_de00", []).append(de)
            err = (out - g) ** 2
            tgt = s["trans_gt"][None].to(device)
            for name, lo, hi in (("dense", 0.0, 0.2), ("mid", 0.2, 0.5),
                                 ("light", 0.5, 1.01)):
                msk = (tgt >= lo) & (tgt < hi)
                if msk.any():
                    sq[name][0] += err[msk].sum().item()
                    sq[name][1] += int(msk.sum().item())
            sq["red"][0] += err[:, 0].sum().item()
            sq["red"][1] += err[:, 0].numel()
    if not detailed:
        return tot / n
    extra = {}
    for key, name in (("_ssim", "ssim"), ("_de00", "de00")):
        if key in sq:
            extra[name] = float(np.mean(sq.pop(key)))
    bins = {k: (10 * math.log10(1.0 / max(v[0] / v[1], 1e-10))
                if v[1] else float("nan")) for k, v in sq.items()}
    bins.update(extra)
    return tot / n, bins


@torch.no_grad()
def save_qualitative(model, ds, device, out_dir, n=8, physics=True):
    """Panels: underwater input | direct restored | (physics restored) | GT."""
    from PIL import ImageDraw
    qdir = Path(out_dir) / "qualitative"
    qdir.mkdir(parents=True, exist_ok=True)
    idxs = np.linspace(0, len(ds) - 1, n).astype(int)
    for k, i in enumerate(idxs):
        s = ds[int(i)]
        x = s["hazy"][None].to(device)
        _, _, H, W = x.shape
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        sd = F.pad(s["sparse_depth"][None].to(device), (0, pw, 0, ph))
        m = F.pad(s["mask"][None].to(device), (0, pw, 0, ph))
        out = model(xp, sd, m)
        to_img = lambda t: Image.fromarray(
            (t[0, :, :H, :W].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
             * 255).astype(np.uint8))
        cols = [(to_img(x), "underwater input"),
                (to_img(out["restored"]), "direct restored")]
        if physics:
            cols.append((to_img(out["physics_restored"]),
                         "physics inversion"))
        cols.append((to_img(s["clear"][None]), "ground truth"))
        w, h = cols[0][0].size
        panel = Image.new("RGB", (len(cols) * (w + 4), h + 24), "white")
        d = ImageDraw.Draw(panel)
        for j, (im, lab) in enumerate(cols):
            panel.paste(im, (j * (w + 4), 24))
            d.text((j * (w + 4) + 4, 5), lab, fill="black")
        panel.save(qdir / f"sample_{k}.png")
    return qdir


class GradBinner:
    """Accumulate mean |dL/dt| binned by predicted t (Fig. 3 analog)."""

    def __init__(self, n_bins=10):
        self.edges = np.linspace(0.0, 1.0, n_bins + 1)
        self.sums = np.zeros(n_bins)
        self.counts = np.zeros(n_bins)

    def update(self, t_val, t_grad):
        t = t_val.detach().flatten().cpu().numpy()
        g = t_grad.detach().abs().flatten().cpu().numpy()
        idx = np.clip(np.digitize(t, self.edges) - 1, 0, len(self.sums) - 1)
        np.add.at(self.sums, idx, g)
        np.add.at(self.counts, idx, 1)

    def save(self, path):
        np.savez(path, edges=self.edges, sums=self.sums, counts=self.counts,
                 mean=self.sums / np.maximum(self.counts, 1))


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nophy", action="store_true")
    ap.add_argument("--data-root", default=None,
                    help="SUN-RGBD scene directory (e.g. kv1/NYUdata or "
                         "kv2/kinect2data)")
    ap.add_argument("--train-frac", type=float, default=1.0,
                    help="fraction of training scenes (data-efficiency runs)")
    ap.add_argument("--keep-ratio", type=float, default=0.03,
                    help="training sparse-depth density")
    ap.add_argument("--depth-sweep", action="store_true",
                    help="after final eval, sweep test-time depth density "
                         "(App. N analog): PSNR at keep ratios "
                         "0.10/0.03/0.01/0.003/0")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--grad-bin-every", type=int, default=100)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"

    out_dir = Path(f"runs/underwater_{args.tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    wb = None
    if not args.no_wandb:
        try:
            import wandb
            wb = wandb.init(project="physdnet-underwater",
                            name=f"underwater_{args.tag}", config=vars(args))
        except Exception as e:
            print(f"[WANDB] disabled ({e})", flush=True)

    model = UnderwaterPhysDNet(use_physics_head=not args.nophy).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INIT] Underwater PhysDNet-M "
          f"{'NoPhy' if args.nophy else 'physics'} ({n_params/1e6:.2f}M)",
          flush=True)

    if args.nophy:
        loss_fn = DehazeLoss(w_rec=1.0, w_perc=0.05, w_grad=0.5, w_phys=0.0,
                             w_phys_rec=0.0, w_smooth=0.0, w_trans=0.0,
                             w_fft=0.1).to(device)
    else:  # STF recipe incl. depth-anchored transmission supervision
        loss_fn = DehazeLoss(w_rec=1.0, w_perc=0.05, w_grad=0.5, w_phys=0.2,
                             w_phys_rec=1.0, w_smooth=0.1, w_trans=2.0,
                             w_fft=0.1).to(device)

    train_ds = NYUUnderwaterDataset(root=args.data_root, crop=args.crop,
                                    train=True, keep_ratio=args.keep_ratio)
    if args.train_frac < 1.0:  # deterministic subsample for data-efficiency
        k = max(1, int(len(train_ds.dirs) * args.train_frac))
        train_ds.dirs = train_ds.dirs[:k]
    test_ds = NYUUnderwaterDataset(root=args.data_root, train=False)
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
            if bin_step:  # fp32 step for clean gradient statistics
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
                            "val_psnr": v}, out_dir / "best_psnr.pth")
                msg += " *best*"
            if not args.nophy:
                binner.save(out_dir / "grad_bins.npz")
        if wb:
            wb.log(log)
        print(msg, flush=True)

    ck = torch.load(out_dir / "best_psnr.pth", map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    final, bins = evaluate(model, test_ds, device, limit=None, detailed=True)
    print(f"[FINAL] {args.tag}: full test PSNR {final:.2f} dB "
          f"(best-val epoch {ck['epoch']}) | "
          f"dense(t<0.2) {bins['dense']:.2f} | mid {bins['mid']:.2f} | "
          f"light {bins['light']:.2f} | red-ch {bins['red']:.2f}",
          flush=True)
    import json
    with open(out_dir / "final_metrics.json", "w") as f:
        json.dump({"psnr": final, **bins, "seed": args.seed,
                   "train_frac": args.train_frac}, f, indent=2)
    if not args.nophy:
        binner.save(out_dir / "grad_bins.npz")
    qdir = save_qualitative(model, test_ds, device, out_dir,
                            physics=not args.nophy)
    print(f"[QUAL] panels saved to {qdir}", flush=True)

    if args.depth_sweep:
        # test-time depth-density robustness (App. N analog); scene
        # parameters are seeded per index, so only the mask density varies
        import json as _json
        sweep = {}
        for kr in (0.10, 0.03, 0.01, 0.003, 0.0):
            ds_k = NYUUnderwaterDataset(root=args.data_root, train=False,
                                        keep_ratio=kr)
            sweep[str(kr)] = round(evaluate(model, ds_k, device,
                                            limit=None), 3)
            print(f"[SWEEP] keep_ratio {kr}: PSNR {sweep[str(kr)]}",
                  flush=True)
        with open(out_dir / "depth_sweep.json", "w") as f:
            _json.dump(sweep, f, indent=2)
        if wb:
            for k, v in sweep.items():
                wb.summary[f"sweep_psnr_keep{k}"] = v
    if wb:
        wb.summary["final_test_psnr"] = final
        for k, v in bins.items():
            wb.summary[f"final_psnr_{k}"] = v
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
