#!/usr/bin/env python3
"""
Weather-robustness matrix on real STF captures (no synthesis, no GT):
extends the paper's real-fog protocol (Table 5: dual-head consistency +
App. O: no-reference IQA) across ALL adverse conditions and day/night,
using the released PhysDNet-M checkpoint, real camera frames, and real
(weather-attenuated) LiDAR.

Per official devkit split (dense_fog/light_fog/rain/snow x day/night +
clear test): dual-head agreement PSNR and Pearson r between the deployed
direct head and the parameter-free physics inversion, NIQE/BRISQUE of the
restored output vs the input, mean predicted transmission (haze-severity
proxy), and qualitative panels.

    python eval_stf_weather.py --out results_stf_weather
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import get_model

STF = Path("datasets/stf/SeeingThroughFog")
META = Path("data/stf/meta")
CKPT = ("runs/stf_robust_ch64_3.0M_wclear+overcast_crop256_"
        "b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185_0406_2105/"
        "best_psnr.pth")
SPLITS = ["test_clear_day", "test_clear_night", "light_fog_day",
          "light_fog_night", "dense_fog_day", "dense_fog_night",
          "rain", "snow_day", "snow_night"]
MAX_DEPTH = 120.0


def load_frame(stem):
    import cv2
    raw = cv2.imread(str(STF / "cam_stereo_left" / f"{stem}.tiff"),
                     cv2.IMREAD_UNCHANGED).astype(np.float32)
    scale = 4095.0 if raw.max() <= 4095 else 65535.0
    grey = np.clip(raw / scale, 0.0, 1.0)
    img = np.stack([grey] * 3, axis=-1)
    depth = np.load(STF / "lidar_hdl64_strongest_stereo_left" /
                    f"{stem}.npz")["arr_0"].astype(np.float32)
    return img, depth


def psnr(a, b):
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / max(mse, 1e-10))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default="results_stf_weather")
    ap.add_argument("--max-frames", type=int, default=250,
                    help="cap per split (seeded subsample for large splits)")
    ap.add_argument("--iqa-frames", type=int, default=100)
    ap.add_argument("--n-qual", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out)
    (out_root / "panels").mkdir(parents=True, exist_ok=True)

    model = get_model("robust")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    import pyiqa
    niqe = pyiqa.create_metric("niqe", device=device)
    brisque = pyiqa.create_metric("brisque", device=device)

    rows = []
    for split in SPLITS:
        stems = [l.strip().replace(",", "_")
                 for l in open(META / f"{split}.txt") if l.strip()]
        rng = np.random.RandomState(42)
        if len(stems) > args.max_frames:
            stems = list(rng.choice(stems, args.max_frames, replace=False))
        iqa_idx = set(rng.choice(len(stems),
                                 min(args.iqa_frames, len(stems)),
                                 replace=False).tolist())

        cons_psnr, cons_r, t_means = [], [], []
        niqe_in, niqe_out, bris_in, bris_out = [], [], [], []
        with torch.no_grad():
            for k, stem in enumerate(stems):
                try:
                    img, depth = load_frame(stem)
                except Exception:
                    continue
                x = torch.from_numpy(img.transpose(2, 0, 1))[None].to(device)
                sd = torch.from_numpy(
                    (np.clip(depth, 0, MAX_DEPTH) / MAX_DEPTH)[None, None]
                ).to(device)
                m = (sd > 0).float()
                out = model(x, sd, m)
                J, P = out["restored"].clamp(0, 1), out["physics_restored"]
                cons_psnr.append(psnr(J, P))
                jf = J.flatten().cpu().numpy()
                pf = P.flatten().cpu().numpy()
                cons_r.append(float(np.corrcoef(jf, pf)[0, 1]))
                t_means.append(float(out["transmission"].mean()))
                if k in iqa_idx:
                    try:
                        niqe_in.append(niqe(x).item())
                        niqe_out.append(niqe(J).item())
                        bris_in.append(brisque(x).item())
                        bris_out.append(brisque(J).item())
                    except Exception:
                        pass
                if k < args.n_qual:
                    to = lambda t: Image.fromarray(
                        (t[0].permute(1, 2, 0).cpu().numpy() * 255
                         ).astype(np.uint8))
                    cols = [(to(x), "input"), (to(J), "direct head"),
                            (to(P.clamp(0, 1)), "physics inversion")]
                    w, h = cols[0][0].size
                    sc = 360 / h
                    cols = [(im.resize((int(w * sc), 360)), lab)
                            for im, lab in cols]
                    W = sum(im.width for im, _ in cols) + 2 * 6
                    panel = Image.new("RGB", (W, 390), "white")
                    d = ImageDraw.Draw(panel)
                    xx = 0
                    for im, lab in cols:
                        panel.paste(im, (xx, 30))
                        d.text((xx + 6, 8), lab, fill="black")
                        xx += im.width + 6
                    panel.save(out_root / "panels" / f"{split}_{k}.png")

        row = {
            "split": split, "n": len(cons_psnr),
            "consistency_psnr": round(float(np.mean(cons_psnr)), 2),
            "pearson_r": round(float(np.mean(cons_r)), 4),
            "t_mean": round(float(np.mean(t_means)), 3),
            "niqe_in": round(float(np.mean(niqe_in)), 2) if niqe_in else None,
            "niqe_out": round(float(np.mean(niqe_out)), 2)
            if niqe_out else None,
            "brisque_in": round(float(np.mean(bris_in)), 2)
            if bris_in else None,
            "brisque_out": round(float(np.mean(bris_out)), 2)
            if bris_out else None,
        }
        rows.append(row)
        print(" | ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        with open(out_root / "weather_matrix.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print("WEATHER_DONE", flush=True)


if __name__ == "__main__":
    main()
