#!/usr/bin/env python3
"""
Haze-density-stratified RTTS detection analysis.

Pre-registered hypothesis (paper Sec. 3.7 / beta=0.06 study): dehazing's
detection value concentrates in severe degradation. RTTS is predominantly
light haze, so global mAP parity is expected; the informative comparison
is within density strata.

Density proxy: mean dark channel (He et al.) of the HAZY input, 15x15
erosion, computed before any dehazing and identical for every condition,
so the stratification is model-independent and outcome-blind. Images are
split into quartiles Q1 (lightest) .. Q4 (densest); AP@0.5 is computed
within each quartile for every cached condition. All bins are reported.

    python eval_rtts_density_bins.py --out results_rtts_bins
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from eval_rtts import RTTS_NAMES, COCO_TO_RTTS, yolo_label, average_precision

CONDITIONS = {
    "hazy": "results_rtts_physhead/hazy",
    "direct_stf": "results_rtts/base",
    "direct_reside": "results_rtts_reside/reside_physics",
    "phys_stf": "results_rtts_physhead/phys_stf",
    "phys_urhi": "results_rtts_physhead/phys_urhi",
    "phys_reside": "results_rtts_physhead/phys_reside",
}
HAZY_DIR = "results_rtts_physhead/hazy/images"


def dark_channel_density(path, ksize=15):
    img = cv2.imread(str(path))
    img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    dc = img.min(axis=2)
    dc = cv2.erode(dc, np.ones((ksize, ksize), np.uint8))
    return float(dc.mean()) / 255.0


def detect_condition(yolo, img_dir, cache_path, conf=0.25):
    """Run YOLO over a condition folder once; cache detections as JSON."""
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    dets = []  # [img_id, cls_name, conf, x1, y1, x2, y2]
    imgs = sorted(Path(img_dir).glob("*.jpg")) + \
        sorted(Path(img_dir).glob("*.png"))
    for p in tqdm(imgs, desc=f"  yolo {Path(img_dir).parent.name}",
                  leave=False):
        for r in yolo(str(p), verbose=False, conf=conf):
            for b in r.boxes:
                name = COCO_TO_RTTS.get(int(b.cls.item()))
                if name is None:
                    continue
                dets.append([p.stem, name, float(b.conf.item()),
                             *[float(v) for v in b.xyxy[0].tolist()]])
    cache_path.write_text(json.dumps(dets))
    return dets


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--out", default="results_rtts_bins")
    ap_.add_argument("--labels", default="results_rtts_physhead/hazy/labels")
    ap_.add_argument("--conf", type=float, default=0.25)
    ap_.add_argument("--n-bins", type=int, default=4)
    args = ap_.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1) density per image from the hazy input only
    dens_path = out / "density.json"
    if dens_path.exists():
        density = json.loads(dens_path.read_text())
    else:
        density = {}
        for p in tqdm(sorted(Path(HAZY_DIR).glob("*.jpg")), desc="density"):
            density[p.stem] = dark_channel_density(p)
        dens_path.write_text(json.dumps(density))
    ids = sorted(density)
    vals = np.array([density[i] for i in ids])
    edges = np.quantile(vals, np.linspace(0, 1, args.n_bins + 1))
    edges[0] -= 1e-9
    bin_of = {i: int(np.searchsorted(edges, density[i], side="left") - 1)
              for i in ids}
    print("[BINS] density quartile edges:",
          [round(float(e), 4) for e in edges])

    # 2) ground truth per image
    gts = defaultdict(dict)  # class -> img_id -> [boxes]
    sizes = {}
    for p in sorted(Path(HAZY_DIR).glob("*.jpg")):
        with Image.open(p) as im:
            sizes[p.stem] = im.size
        w, h = sizes[p.stem]
        for name, x1, y1, x2, y2 in yolo_label(
                str(Path(args.labels) / f"{p.stem}.txt"), w, h):
            gts[name].setdefault(p.stem, []).append((x1, y1, x2, y2))

    # 3) detections per condition (cached)
    from ultralytics import YOLO
    yolo = YOLO("yolov8m.pt")
    all_dets = {}
    for cond, root in CONDITIONS.items():
        all_dets[cond] = detect_condition(
            yolo, Path(root) / "images", out / f"dets_{cond}.json",
            conf=args.conf)

    # 4) AP per bin
    rows = []
    for cond, dets in all_dets.items():
        by_class = defaultdict(list)
        for img_id, name, conf, x1, y1, x2, y2 in dets:
            by_class[name].append((img_id, conf, (x1, y1, x2, y2)))
        for b in range(args.n_bins):
            in_bin = {i for i in ids if bin_of[i] == b}
            ap = {}
            for name in RTTS_NAMES:
                d = [t for t in by_class[name] if t[0] in in_bin]
                g = {i: v for i, v in gts[name].items() if i in in_bin}
                ap[name] = average_precision(d, g)
            row = {"condition": cond, "bin": f"Q{b+1}",
                   "n_imgs": len(in_bin),
                   "mAP@0.5": round(100 * float(
                       np.nanmean([ap[n] for n in RTTS_NAMES])), 2),
                   **{f"AP_{n}": round(100 * ap[n], 2)
                      for n in RTTS_NAMES}}
            rows.append(row)

    csv_path = out / "density_bins.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # print pivot: bins as columns
    print(f"\n[CSV] {csv_path}\n")
    print(f"{'condition':<16}" + "".join(
        f"{'Q'+str(b+1):>10}" for b in range(args.n_bins)) + "   (mAP@0.5)")
    for cond in CONDITIONS:
        vals_ = [r["mAP@0.5"] for r in rows if r["condition"] == cond]
        print(f"{cond:<16}" + "".join(f"{v:>10}" for v in vals_))
    print(f"\n{'condition':<16}" + "".join(
        f"{'Q'+str(b+1):>10}" for b in range(args.n_bins)) + "   (AP_person)")
    for cond in CONDITIONS:
        vals_ = [r["AP_person"] for r in rows if r["condition"] == cond]
        print(f"{cond:<16}" + "".join(f"{v:>10}" for v in vals_))


if __name__ == "__main__":
    main()
