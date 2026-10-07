#!/usr/bin/env python3
"""
Extended paired metrics for the RESIDE-6K physics-vs-NoPhy comparison:
PSNR, SSIM, LPIPS (AlexNet), and CIEDE2000 color difference vs GT —
overall and by scene-severity quartile. All metrics reported regardless
of direction; battery fixed before looking at results.

    python eval_reside_ext.py --out results/reside6k_compare
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # pyiqa/skimage clash

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import get_model
from train_reside6k import PairedHazeDataset


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}).get("model", {})
    m = get_model("robust", use_physics_head=cfg.get("use_physics_head", True))
    m.load_state_dict(ck["model"])
    return m.to(device).eval()


@torch.no_grad()
def restore(model, hazy, device):
    x = hazy[None].to(device)
    _, _, H, W = x.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    z = torch.zeros(1, 1, x.shape[2], x.shape[3], device=device)
    return model(x, z, z)["restored"][:, :, :H, :W].clamp(0, 1)


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/RESIDE-6K/test")
    ap.add_argument("--physics", default="runs/reside6k_physics/best_psnr.pth")
    ap.add_argument("--nophy", default="runs/reside6k_nophy/best_psnr.pth")
    ap.add_argument("--out", default="results/reside6k_compare")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import pyiqa
    from skimage.color import rgb2lab, deltaE_ciede2000
    ssim_m = pyiqa.create_metric("ssim", device=device)
    lpips_m = pyiqa.create_metric("lpips", device=device)

    models = {"physics": load(args.physics, device),
              "nophy": load(args.nophy, device)}
    ds = PairedHazeDataset(args.data_root, train=False)
    print(f"[DATA] {len(ds)} test pairs", flush=True)

    rows = []
    for i in range(len(ds)):
        s = ds[i]
        hazy, gt = s["hazy"], s["clear"]
        g = gt[None].to(device)
        gt_lab = rgb2lab(gt.permute(1, 2, 0).numpy())
        row = {"input_psnr": psnr(hazy, gt)}
        for name, m in models.items():
            r = restore(m, hazy, device)
            row[f"{name}_psnr"] = psnr(r.cpu()[0], gt)
            row[f"{name}_ssim"] = float(ssim_m(r, g).item())
            row[f"{name}_lpips"] = float(lpips_m(r, g).item())
            r_lab = rgb2lab(r[0].permute(1, 2, 0).cpu().numpy())
            row[f"{name}_de00"] = float(
                deltaE_ciede2000(gt_lab, r_lab).mean())
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ds)}", flush=True)

    METRICS = ["psnr", "ssim", "lpips", "de00"]
    res = {"overall": {}, "quartiles": {}}
    for mname in METRICS:
        res["overall"][mname] = {
            k: float(np.mean([r[f"{k}_{mname}"] for r in rows]))
            for k in models}

    order = np.argsort([r["input_psnr"] for r in rows])
    for qi, idxs in enumerate(np.array_split(order, 4)):
        q = {"input_psnr": float(np.mean(
            [rows[j]["input_psnr"] for j in idxs]))}
        for mname in METRICS:
            for k in models:
                q[f"{k}_{mname}"] = float(np.mean(
                    [rows[j][f"{k}_{mname}"] for j in idxs]))
        res["quartiles"][f"Q{qi+1}"] = q

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "extended_metrics.json", "w") as f:
        json.dump(res, f, indent=2)

    hdr = ("metric   |  physics |    nophy |   delta  (better)")
    print("\n===== RESIDE-6K extended metrics (full test) =====")
    print(hdr)
    better = {"psnr": +1, "ssim": +1, "lpips": -1, "de00": -1}
    for mname in METRICS:
        p = res["overall"][mname]["physics"]
        n = res["overall"][mname]["nophy"]
        d = p - n
        win = "physics" if d * better[mname] > 0 else "nophy"
        print(f"{mname:8} | {p:8.4f} | {n:8.4f} | {d:+8.4f}  ({win})")
    print("\nby severity quartile (Q1 = densest):")
    for q, d in res["quartiles"].items():
        print(f" {q}: " + " | ".join(
            f"{mn} p{d[f'physics_{mn}']:.3f}/n{d[f'nophy_{mn}']:.3f}"
            for mn in METRICS))


if __name__ == "__main__":
    main()
