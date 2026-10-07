#!/usr/bin/env python3
"""
Faster R-CNN (ResNet50-FPN, COCO) over the cached RTTS zero-shot condition
folders: third detector family for the robustness check (two-stage, and the
family AOD-Net originally tuned).

    python eval_rtts_frcnn.py --cache results_rtts_zeroshot --out results_rtts_zeroshot/frcnn_summary.csv
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
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from eval_rtts import RTTS_NAMES, yolo_label, average_precision

# torchvision COCO category ids -> RTTS names
TV_TO_RTTS = {1: "person", 2: "bicycle", 3: "car", 4: "motorbike", 6: "bus"}


@torch.no_grad()
def evaluate_condition(model, idir, ldir, device, conf=0.25):
    dets = defaultdict(list)
    gts = defaultdict(dict)
    imgs = sorted(Path(idir).glob("*.jpg")) + sorted(Path(idir).glob("*.png"))
    for p in tqdm(imgs, desc="  frcnn", leave=False):
        img_id = p.stem
        im = Image.open(p).convert("RGB")
        w, h = im.size
        for name, x1, y1, x2, y2 in yolo_label(
                str(Path(ldir) / f"{p.stem}.txt"), w, h):
            gts[name].setdefault(img_id, []).append((x1, y1, x2, y2))
        x = torch.from_numpy(
            np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        )[None].to(device)
        out = model(x)[0]
        keep = out["scores"] >= conf
        for box, lab, sc in zip(out["boxes"][keep].cpu().numpy(),
                                out["labels"][keep].cpu().numpy(),
                                out["scores"][keep].cpu().numpy()):
            name = TV_TO_RTTS.get(int(lab))
            if name is None:
                continue
            dets[name].append((img_id, float(sc), tuple(box.tolist())))
    ap = {n: average_precision(dets[n], gts[n]) for n in RTTS_NAMES}
    ap["mAP"] = float(np.nanmean([ap[n] for n in RTTS_NAMES]))
    return ap


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--cache", default="results_rtts_zeroshot")
    ap_.add_argument("--out",
                     default="results_rtts_zeroshot/frcnn_summary.csv")
    ap_.add_argument("--conf", type=float, default=0.25)
    args = ap_.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from torchvision.models.detection import (
        fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).to(device).eval()

    cache = Path(args.cache)
    conditions = sorted(d.name for d in cache.iterdir()
                        if (d / "images").exists())
    rows = []
    for cond in conditions:
        print(f"[COND] {cond}", flush=True)
        ap = evaluate_condition(model, cache / cond / "images",
                                cache / cond / "labels", device, args.conf)
        row = {"condition": cond, "mAP@0.5": round(ap["mAP"] * 100, 2),
               **{f"AP_{n}": round(ap[n] * 100, 2) for n in RTTS_NAMES}}
        rows.append(row)
        print("  " + " | ".join(f"{k}={v}" for k, v in row.items()),
              flush=True)
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print("FRCNN_DONE", flush=True)


if __name__ == "__main__":
    main()
