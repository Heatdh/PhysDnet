#!/usr/bin/env python3
"""
Zero-shot RTTS evaluation of ALL available trained models (Reviewer 7bqR Q1).

Uses the edge_onnx/ exports (weights baked in, fixed 1x3x256x256, zero
depth/mask = the paper's no-LiDAR zero-shot protocol) so every model runs
under an identical pipeline: resize to 256 -> dehaze -> resize back to the
original resolution -> YOLOv8-M detection AP@0.5 vs RTTS boxes + NIQE /
BRISQUE (300-image sample) + colorfulness. The resize round-trip penalty is
identical for all models; the 'hazy' reference row is unprocessed.

    python eval_rtts_zeroshot_baselines.py --out results_rtts_zeroshot
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from eval_rtts import (RTTS_NAMES, evaluate_detection, colorfulness_np)

ONNX_MODELS = {
    # control: same 256 resize round-trip, no dehazing — isolates the
    # resolution penalty the fixed-shape ONNX pipeline imposes
    "hazy_256": "RESIZE_ONLY",
    "aodnet_lidar": "edge_onnx/aodnet_lidar.onnx",
    "ffanet_lidar": "edge_onnx/ffanet_lidar.onnx",
    "dehazeformer_lidar": "edge_onnx/dehazeformer_lidar.onnx",
    "deanet_lidar": "edge_onnx/deanet_lidar.onnx",
    "deanet_rgb": "edge_onnx/deanet_rgb.onnx",
    "physdnet_s": "edge_onnx/physdnet_s_ch32.onnx",
    "physdnet_m": "edge_onnx/physdnet_m_ch64.onnx",
    "physdnet_l": "edge_onnx/physdnet_l_ch96.onnx",
    # NOT zero-shot: URHI self-supervised adaptation (report separately)
    "physdnet_m_urhi": "TORCH:runs/finetune_realhaze_urhi/adapted_ep5.pth",
}


def load_torch_model(ckpt_path, device):
    import torch
    from model import get_model
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = get_model("robust")
    m.load_state_dict(ck["model"] if "model" in ck else ck)
    return m.to(device).eval()


def dehaze_256_torch(model, img: Image.Image, device) -> Image.Image:
    import torch
    w, h = img.size
    x = torch.from_numpy(
        np.asarray(img.resize((256, 256), Image.BILINEAR),
                   np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
    z = torch.zeros(1, 1, 256, 256, device=device)
    with torch.no_grad():
        out = model(x, z, z)["restored"].clamp(0, 1)
    out = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(out).resize((w, h), Image.BILINEAR)


def make_session(path, cpu_threads, device="cpu"):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = cpu_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if device == "cuda" else ["CPUExecutionProvider"])
    return ort.InferenceSession(path, so, providers=providers)


def dehaze_256(sess, img: Image.Image) -> Image.Image:
    w, h = img.size
    x = np.asarray(img.resize((256, 256), Image.BILINEAR),
                   np.float32).transpose(2, 0, 1)[None] / 255.0
    z = np.zeros((1, 1, 256, 256), np.float32)
    names = [i.name for i in sess.get_inputs()]
    feed = {names[0]: x}
    for n in names[1:]:
        feed[n] = z
    out = sess.run(None, feed)[0]  # first output = restored
    out = np.clip(out[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(out).resize((w, h), Image.BILINEAR)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rtts-dir", default="datasets/RTTS.v1i.yolov8")
    ap.add_argument("--splits", nargs="*", default=["valid", "test"])
    ap.add_argument("--models", nargs="*", default=list(ONNX_MODELS))
    ap.add_argument("--out", default="results_rtts_zeroshot")
    ap.add_argument("--yolo-model", default="yolov8m.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iqa-limit", type=int, default=300)
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    sources = []
    for split in args.splits:
        idir = Path(args.rtts_dir) / split / "images"
        ldir = Path(args.rtts_dir) / split / "labels"
        for p in sorted(idir.iterdir()):
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                sources.append((p, ldir / f"{p.stem}.txt"))
    print(f"[DATA] {len(sources)} RTTS images", flush=True)

    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)

    import torch
    torch.set_num_threads(args.cpu_threads)
    import pyiqa
    iqa = {"niqe": pyiqa.create_metric("niqe", device=args.device),
           "brisque": pyiqa.create_metric("brisque", device=args.device)}
    rng = np.random.RandomState(42)
    iqa_idx = set(rng.choice(len(sources),
                             min(args.iqa_limit, len(sources)),
                             replace=False).tolist())

    conditions = {"hazy": None}
    for m in args.models:
        conditions[m] = ONNX_MODELS[m]

    rows = []
    csv_path = out_root / "rtts_zeroshot_summary.csv"
    for cond, onnx_path in conditions.items():
        print(f"\n[COND] {cond}", flush=True)
        idir = out_root / cond / "images"
        ldir = out_root / cond / "labels"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        resize_only = onnx_path == "RESIZE_ONLY"
        tmodel = None
        if onnx_path and onnx_path.startswith("TORCH:"):
            tmodel = load_torch_model(onnx_path[6:], args.device)
            sess = None
        else:
            sess = (make_session(onnx_path, args.cpu_threads, args.device)
                    if onnx_path and not resize_only else None)

        cf, iqa_vals = [], defaultdict(list)
        for k, (ip, lp) in enumerate(tqdm(sources, desc="  dehaze")):
            tgt = idir / ip.name
            if not tgt.exists():
                img = Image.open(ip).convert("RGB")
                if resize_only:
                    w, h = img.size
                    img = img.resize((256, 256), Image.BILINEAR).resize(
                        (w, h), Image.BILINEAR)
                elif tmodel is not None:
                    img = dehaze_256_torch(tmodel, img, args.device)
                elif sess is not None:
                    img = dehaze_256(sess, img)
                img.save(tgt)
            else:
                img = Image.open(tgt).convert("RGB")
            ltgt = ldir / lp.name
            if lp.exists() and not ltgt.exists():
                ltgt.write_text(lp.read_text())

            arr = np.asarray(img)
            cf.append(colorfulness_np(arr))
            if k in iqa_idx:
                t = (np.asarray(arr, np.float32) / 255.0)
                tt = torch.from_numpy(t).permute(2, 0, 1)[None].to(args.device)
                for mname, metric in iqa.items():
                    try:
                        iqa_vals[mname].append(metric(tt).item())
                    except Exception:
                        pass

        print("  detection...", flush=True)
        ap_ = evaluate_detection(yolo, idir, ldir, conf=args.conf)
        row = {"condition": cond,
               "mAP@0.5": round(ap_["mAP"] * 100, 2),
               **{f"AP_{n}": round(ap_[n] * 100, 2) for n in RTTS_NAMES},
               "colorfulness": round(float(np.mean(cf)), 2),
               "NIQE": round(float(np.mean(iqa_vals["niqe"])), 3)
               if iqa_vals["niqe"] else None,
               "BRISQUE": round(float(np.mean(iqa_vals["brisque"])), 2)
               if iqa_vals["brisque"] else None}
        rows.append(row)
        print("  " + " | ".join(f"{k}={v}" for k, v in row.items()),
              flush=True)
        # incremental CSV so partial results survive interruption
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    print(f"\n[CSV] {csv_path}", flush=True)
    for r in rows:
        print(r)
    print("ZEROSHOT_RTTS_DONE", flush=True)


if __name__ == "__main__":
    main()
