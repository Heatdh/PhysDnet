#!/usr/bin/env python3
"""
Detector-adapted RTTS evaluation (the AOD-Net-style protocol).

The literature's positive dehazing-for-detection results come from adapting
the detector to the dehazed domain (AOD-Net ICCV'17 joint tuning; RESIDE
TPAMI'19 shows frozen-detector preprocessing is inconsistent). This script
runs the fair adapted-vs-adapted comparison:

  1. dehaze the full RTTS dataset (train/valid/test) at full resolution
     with PhysDNet-M (zero-shot STF checkpoint, no LiDAR);
  2. fine-tune YOLOv8-M once on hazy train images, once on dehazed train
     images (identical budget);
  3. evaluate each detector on its matching test condition (mAP@0.5).

    python rtts_detector_adapt.py --stage all
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

BASE_CKPT = ("runs/stf_robust_ch64_3.0M_wclear+overcast_crop256_"
             "b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185_0406_2105/"
             "best_psnr.pth")
RTTS = Path("datasets/RTTS.v1i.yolov8")
OUT = Path("results_rtts_detadapt")
SPLITS = ["train", "valid", "test"]


def stage_dehaze(device):
    from eval_rtts import load_model, dehaze_image
    model = load_model(BASE_CKPT, device)
    for split in SPLITS:
        idir = RTTS / split / "images"
        odir = OUT / "dehazed" / split / "images"
        ldir = OUT / "dehazed" / split / "labels"
        odir.mkdir(parents=True, exist_ok=True)
        if not ldir.exists():
            shutil.copytree(RTTS / split / "labels", ldir)
        srcs = [p for p in sorted(idir.iterdir())
                if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        for p in tqdm(srcs, desc=f"dehaze {split}"):
            tgt = odir / p.name
            if tgt.exists():
                continue
            img = dehaze_image(model, Image.open(p).convert("RGB"), device)
            img.save(tgt)
    # data yamls
    names = "['bicycle', 'bus', 'car', 'motorbike', 'person']"
    (OUT / "dehazed.yaml").write_text(
        f"path: {(OUT / 'dehazed').resolve().as_posix()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: 5\nnames: {names}\n")
    (OUT / "hazy.yaml").write_text(
        f"path: {RTTS.resolve().as_posix()}\n"
        f"train: train/images\nval: valid/images\ntest: test/images\n"
        f"nc: 5\nnames: {names}\n")
    print("[DEHAZE] done", flush=True)


def stage_train(condition, epochs, device):
    from ultralytics import YOLO
    yaml = OUT / f"{condition}.yaml"
    yolo = YOLO("yolov8m.pt")
    yolo.train(data=str(yaml), epochs=epochs, imgsz=640, batch=16,
               project=str(OUT / "yolo"), name=condition, exist_ok=True,
               device=0 if device == "cuda" else "cpu",
               patience=0, verbose=False, plots=False)
    print(f"[TRAIN {condition}] done", flush=True)


def stage_eval(device):
    from ultralytics import YOLO
    print("\n===== detector-adapted RTTS (test split, mAP@0.5) =====",
          flush=True)
    results = {}
    for condition in ("hazy", "dehazed"):
        candidates = [
            OUT / "yolo" / condition / "weights" / "best.pt",
            Path("runs/detect") / OUT / "yolo" / condition / "weights"
            / "best.pt",
        ]
        best = next(p for p in candidates if p.exists())
        yolo = YOLO(str(best))
        m = yolo.val(data=str(OUT / f"{condition}.yaml"), split="test",
                     verbose=False, plots=False,
                     device=0 if device == "cuda" else "cpu")
        results[condition] = {"map50": float(m.box.map50),
                              "map": float(m.box.map)}
        print(f"{condition:8}: mAP@0.5 {m.box.map50*100:.2f} | "
              f"mAP@0.5:0.95 {m.box.map*100:.2f}", flush=True)
    d = (results["dehazed"]["map50"] - results["hazy"]["map50"]) * 100
    print(f"delta (dehazed-adapted - hazy-adapted): {d:+.2f} pp mAP@0.5",
          flush=True)
    import json
    with open(OUT / "detadapt_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print("DETADAPT_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="all",
                    choices=["dehaze", "train", "eval", "all"])
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(exist_ok=True)

    if args.stage in ("dehaze", "all"):
        stage_dehaze(device)
    if args.stage in ("train", "all"):
        stage_train("hazy", args.epochs, device)
        stage_train("dehazed", args.epochs, device)
    if args.stage in ("eval", "all"):
        stage_eval(device)


if __name__ == "__main__":
    main()
