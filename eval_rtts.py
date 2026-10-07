#!/usr/bin/env python3
"""
RTTS evaluation battery for PhysDNet (real haze, no clean GT).

RTTS ships detection boxes only, so evaluation is:
  1. downstream detection — COCO-pretrained YOLOv8 on the dehazed images,
     AP@0.5 per RTTS class (bicycle/bus/car/motorbike/person), protocol
     mirroring downstream_detection/detect_and_eval.py (conf>=0.25);
  2. no-reference IQA — NIQE + BRISQUE (pyiqa), same metrics as App. O;
  3. colorfulness (Hasler-Süsstrunk) — quantifies chrominance recovery of
     the NIR-trained checkpoint before/after real-haze adaptation.

Conditions compared: 'hazy' (no dehazing) + one per --checkpoints entry.

Usage:
    python eval_rtts.py \
        --rtts-dir "datasets/RTTS.v1i.yolov8" --splits valid test \
        --checkpoints base="runs/stf_robust_ch64_.../best_psnr.pth" \
                      adapted="runs/finetune_realhaze_.../adapted_ep5.pth" \
        --out results_rtts
"""

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

try:  # Norton TLS interception breaks python HTTPS; use Windows cert store
    import truststore
    truststore.inject_into_ssl()
except ImportError:
    pass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from model import get_model

# COCO class id -> RTTS class name (RTTS yolo ids: see data.yaml)
COCO_TO_RTTS = {0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 5: "bus"}
RTTS_NAMES = ["bicycle", "bus", "car", "motorbike", "person"]


# ── dehazing ────────────────────────────────────────────────────────────────

def load_model(ckpt_path, device):
    model = get_model("robust")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    return model.to(device).eval()


@torch.no_grad()
def dehaze_image(model, img: Image.Image, device,
                 key="restored") -> Image.Image:
    x = torch.from_numpy(np.asarray(img, np.float32) / 255.0)
    x = x.permute(2, 0, 1)[None].to(device)
    _, _, h, w = x.shape
    ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    z = torch.zeros(1, 1, x.shape[2], x.shape[3], device=device)
    out = model(x, z, z)[key][:, :, :h, :w]
    out = (out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255)
    return Image.fromarray(out.astype(np.uint8))


# ── detection AP@0.5 ────────────────────────────────────────────────────────

def yolo_label(path, w, h):
    """Read YOLO txt -> list of (cls_name, x1, y1, x2, y2)."""
    boxes = []
    if not os.path.exists(path):
        return boxes
    for line in open(path):
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        boxes.append((RTTS_NAMES[c],
                      (cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) +
          (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def average_precision(dets, gts, thr=0.5):
    """dets: [(img_id, conf, box)], gts: {img_id: [box]} -> AP@thr."""
    npos = sum(len(v) for v in gts.values())
    if npos == 0:
        return float("nan")
    dets = sorted(dets, key=lambda d: -d[1])
    matched = defaultdict(set)
    tp = np.zeros(len(dets))
    fp = np.zeros(len(dets))
    for i, (img, _, box) in enumerate(dets):
        best, bj = 0.0, -1
        for j, g in enumerate(gts.get(img, [])):
            if j in matched[img]:
                continue
            v = iou(box, g)
            if v > best:
                best, bj = v, j
        if best >= thr:
            tp[i] = 1
            matched[img].add(bj)
        else:
            fp[i] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / npos
    prec = tp / np.maximum(tp + fp, 1e-9)
    # continuous-envelope VOC AP
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[0], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())


def evaluate_detection(yolo, img_dir, label_dir, conf=0.25):
    dets = defaultdict(list)   # class -> [(img_id, conf, box)]
    gts = defaultdict(dict)    # class -> {img_id: [box]}
    imgs = sorted(Path(img_dir).glob("*.jpg")) + \
        sorted(Path(img_dir).glob("*.png"))
    for p in tqdm(imgs, desc="  yolo", leave=False):
        img_id = p.stem
        with Image.open(p) as im:
            w, h = im.size
        for name, x1, y1, x2, y2 in yolo_label(
                str(Path(label_dir) / f"{p.stem}.txt"), w, h):
            gts[name].setdefault(img_id, []).append((x1, y1, x2, y2))
        for r in yolo(str(p), verbose=False, conf=conf):
            for b in r.boxes:
                name = COCO_TO_RTTS.get(int(b.cls.item()))
                if name is None:
                    continue
                dets[name].append(
                    (img_id, float(b.conf.item()),
                     tuple(b.xyxy[0].tolist())))
    ap = {}
    for name in RTTS_NAMES:
        ap[name] = average_precision(dets[name], gts[name])
    ap["mAP"] = float(np.nanmean([ap[n] for n in RTTS_NAMES]))
    return ap


# ── colorfulness ────────────────────────────────────────────────────────────

def colorfulness_np(arr):
    r, g, b = (arr[..., 0].astype(np.float64), arr[..., 1].astype(np.float64),
               arr[..., 2].astype(np.float64))
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) +
                 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def main():
    ap_ = argparse.ArgumentParser(description=__doc__)
    ap_.add_argument("--rtts-dir", default="datasets/RTTS.v1i.yolov8")
    ap_.add_argument("--splits", nargs="*", default=["valid", "test"])
    ap_.add_argument("--checkpoints", nargs="*", default=[],
                     help="name=path entries; 'hazy' baseline is automatic")
    ap_.add_argument("--out", default="results_rtts")
    ap_.add_argument("--yolo-model", default="yolov8m.pt")
    ap_.add_argument("--conf", type=float, default=0.25)
    ap_.add_argument("--iqa-limit", type=int, default=300,
                     help="images sampled for NIQE/BRISQUE (speed)")
    ap_.add_argument("--skip-iqa", action="store_true")
    args = ap_.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # gather source images (splits share one flat namespace after Roboflow)
    sources = []  # (img_path, label_path)
    for split in args.splits:
        idir = Path(args.rtts_dir) / split / "images"
        ldir = Path(args.rtts_dir) / split / "labels"
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                sources.append((p, ldir / f"{p.stem}.txt"))
    print(f"[DATA] {len(sources)} RTTS images from splits {args.splits}")

    # "name=path" uses the direct head; "name=PHYS:path" evaluates the
    # physics-head inversion output instead
    conditions = {"hazy": (None, "restored")}
    for entry in args.checkpoints:
        name, path = entry.split("=", 1)
        if path.startswith("PHYS:"):
            conditions[name] = (path[5:], "physics_restored")
        else:
            conditions[name] = (path, "restored")

    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    iqa = None
    if not args.skip_iqa:
        import pyiqa
        iqa = {
            "niqe": pyiqa.create_metric("niqe", device=device),
            "brisque": pyiqa.create_metric("brisque", device=device),
        }

    rng = np.random.RandomState(42)
    iqa_idx = set(rng.choice(len(sources),
                             min(args.iqa_limit, len(sources)),
                             replace=False).tolist())

    rows = []
    for cond, (ckpt, out_key) in conditions.items():
        print(f"\n[COND] {cond} (head: {out_key})")
        cdir = out_root / cond
        idir = cdir / "images"
        ldir = cdir / "labels"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)

        model = load_model(ckpt, device) if ckpt else None

        cf, iqa_vals = [], defaultdict(list)
        for k, (ip, lp) in enumerate(tqdm(sources, desc="  dehaze")):
            tgt = idir / ip.name
            if not tgt.exists():
                if model is None:
                    img = Image.open(ip).convert("RGB")
                    img.save(tgt)
                else:
                    img = dehaze_image(model, Image.open(ip).convert("RGB"),
                                       device, key=out_key)
                    img.save(tgt)
            else:
                img = Image.open(tgt).convert("RGB")
            # copy label next to it once
            ltgt = ldir / lp.name
            if lp.exists() and not ltgt.exists():
                ltgt.write_text(lp.read_text())

            arr = np.asarray(img)
            cf.append(colorfulness_np(arr))
            if iqa is not None and k in iqa_idx:
                t = torch.from_numpy(arr.astype(np.float32) / 255.0)
                t = t.permute(2, 0, 1)[None].to(device)
                for mname, metric in iqa.items():
                    try:
                        iqa_vals[mname].append(metric(t).item())
                    except Exception:
                        pass

        print("  running detection...")
        ap = evaluate_detection(yolo, idir, ldir, conf=args.conf)

        row = {"condition": cond,
               "mAP@0.5": round(ap["mAP"] * 100, 2),
               **{f"AP_{n}": round(ap[n] * 100, 2) for n in RTTS_NAMES},
               "colorfulness": round(float(np.mean(cf)), 2)}
        if iqa_vals:
            row["NIQE"] = round(float(np.mean(iqa_vals["niqe"])), 3)
            row["BRISQUE"] = round(float(np.mean(iqa_vals["brisque"])), 2)
        rows.append(row)
        print("  " + " | ".join(f"{k}={v}" for k, v in row.items()))

        del model
        torch.cuda.empty_cache()

    csv_path = out_root / "rtts_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[CSV] {csv_path}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
