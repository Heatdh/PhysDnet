#!/usr/bin/env python3
"""
RESIDE-6K physics vs NoPhy: haze-density-binned comparison + qualitative
panels, from the trained best checkpoints.

RESIDE provides no GT transmission, so density is measured from the pair
itself: per-pixel input degradation e(x) = mean_c |hazy - GT| (a monotone
proxy for 1 - t at fixed airlight). Reported:
  - overall full-test PSNR per model;
  - pixel-binned PSNR: light e<0.1 / mid 0.1-0.3 / dense e>0.3;
  - image-quartile PSNR (images ranked by input PSNR = scene severity);
  - qualitative panels hazy | physics | nophy | GT across the severity range.

Usage:
    python eval_reside_bins.py --out results/reside6k_compare
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from model import get_model
from train_reside6k import PairedHazeDataset


def load(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck.get("config", {}).get("model", {})
    model = get_model("robust",
                      use_physics_head=cfg.get("use_physics_head", True))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck.get("epoch", "?")


@torch.no_grad()
def restore(model, hazy, device):
    x = hazy[None].to(device)
    _, _, H, W = x.shape
    ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    z = torch.zeros(1, 1, x.shape[2], x.shape[3], device=device)
    return model(x, z, z)["restored"][:, :, :H, :W].clamp(0, 1)[0].cpu()


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="datasets/RESIDE-6K/test")
    ap.add_argument("--physics", default="runs/reside6k_physics/best_psnr.pth")
    ap.add_argument("--nophy", default="runs/reside6k_nophy/best_psnr.pth")
    ap.add_argument("--out", default="results/reside6k_compare")
    ap.add_argument("--n-qual", type=int, default=8)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    (out / "panels").mkdir(parents=True, exist_ok=True)

    mp, ep_p = load(args.physics, device)
    mn, ep_n = load(args.nophy, device)
    print(f"[LOAD] physics best@{ep_p} | nophy best@{ep_n}", flush=True)

    ds = PairedHazeDataset(args.data_root, train=False)
    print(f"[DATA] {len(ds)} test pairs", flush=True)

    BINS = (("light", 0.0, 0.1), ("mid", 0.1, 0.3), ("dense", 0.3, 10.0))
    sq = {m: {b[0]: [0.0, 0] for b in BINS} for m in ("physics", "nophy")}
    rows = []

    for i in range(len(ds)):
        s = ds[i]
        hazy, gt = s["hazy"], s["clear"]
        rp = restore(mp, hazy, device)
        rn = restore(mn, hazy, device)
        e_in = (hazy - gt).abs().mean(0)  # H,W degradation proxy
        for name, r in (("physics", rp), ("nophy", rn)):
            err = ((r - gt) ** 2).mean(0)
            for bname, lo, hi in BINS:
                msk = (e_in >= lo) & (e_in < hi)
                if msk.any():
                    sq[name][bname][0] += float(err[msk].sum())
                    sq[name][bname][1] += int(msk.sum())
        rows.append({"idx": i, "input_psnr": psnr(hazy, gt),
                     "physics": psnr(rp, gt), "nophy": psnr(rn, gt)})
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(ds)}", flush=True)

    # overall + pixel bins
    res = {"checkpoints": {"physics_epoch": ep_p, "nophy_epoch": ep_n},
           "overall": {m: float(np.mean([r[m] for r in rows]))
                       for m in ("physics", "nophy")},
           "pixel_bins": {}}
    for m in ("physics", "nophy"):
        res["pixel_bins"][m] = {
            b: 10 * math.log10(1.0 / max(v[0] / v[1], 1e-10))
            for b, (v) in ((k, sq[m][k]) for k in sq[m]) if v[1] > 0}

    # image severity quartiles (by input PSNR: Q1 = densest scenes)
    order = np.argsort([r["input_psnr"] for r in rows])
    qs = np.array_split(order, 4)
    res["image_quartiles"] = {}
    for qi, idxs in enumerate(qs):
        res["image_quartiles"][f"Q{qi+1}"] = {
            "input_psnr": float(np.mean([rows[j]["input_psnr"] for j in idxs])),
            "physics": float(np.mean([rows[j]["physics"] for j in idxs])),
            "nophy": float(np.mean([rows[j]["nophy"] for j in idxs])),
            "n": len(idxs)}

    with open(out / "bins_summary.json", "w") as f:
        json.dump(res, f, indent=2)

    print("\n===== RESIDE-6K physics vs NoPhy =====")
    print(f"overall: physics {res['overall']['physics']:.2f} | "
          f"nophy {res['overall']['nophy']:.2f} | "
          f"delta {res['overall']['physics']-res['overall']['nophy']:+.2f}")
    for b, _, _ in BINS:
        p, n = res["pixel_bins"]["physics"][b], res["pixel_bins"]["nophy"][b]
        print(f"pixel bin {b:>5}: physics {p:.2f} | nophy {n:.2f} | "
              f"delta {p-n:+.2f}")
    for q, d in res["image_quartiles"].items():
        print(f"severity {q} (input {d['input_psnr']:.1f} dB): "
              f"physics {d['physics']:.2f} | nophy {d['nophy']:.2f} | "
              f"delta {d['physics']-d['nophy']:+.2f}")

    # qualitative panels across severity range
    picks = [order[int(p * (len(order) - 1))]
             for p in np.linspace(0.0, 0.95, args.n_qual)]
    for k, j in enumerate(picks):
        s = ds[int(j)]
        hazy, gt = s["hazy"], s["clear"]
        rp = restore(mp, hazy, device)
        rn = restore(mn, hazy, device)
        to = lambda t: Image.fromarray(
            (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        cols = [(to(hazy), f"hazy ({rows[j]['input_psnr']:.1f} dB)"),
                (to(rp), f"physics ({rows[j]['physics']:.1f} dB)"),
                (to(rn), f"nophy ({rows[j]['nophy']:.1f} dB)"),
                (to(gt), "GT")]
        w, h = cols[0][0].size
        panel = Image.new("RGB", (4 * (w + 4), h + 24), "white")
        d = ImageDraw.Draw(panel)
        for c, (im, lab) in enumerate(cols):
            panel.paste(im, (c * (w + 4), 24))
            d.text((c * (w + 4) + 4, 5), lab, fill="black")
        panel.save(out / "panels" / f"sev{k}_img{j}.png")
    print(f"[QUAL] {args.n_qual} panels in {out/'panels'}")


if __name__ == "__main__":
    main()
