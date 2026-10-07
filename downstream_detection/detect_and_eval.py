#!/usr/bin/env python3
"""
Multi-model downstream detection evaluation with synthetic fog.

Takes clear/overcast VAL-split frames, synthesizes heavy fog, then evaluates
multiple dehazing models side-by-side:
  - Clean   (upper bound — no fog)
  - Foggy   (lower bound — synthetic haze)
  - Model_1 … Model_N  (each dehazer restores foggy → detect)

Produces:
  - summary.txt           combined mAP table
  - results.csv           machine-readable results
  - samples/              N+2-column comparison grids (Clean | Foggy | models…)

Uses only val_timestamps.txt — zero overlap with training data.

Usage:
    # Compare multiple models:
    python downstream_detection/detect_and_eval.py \
        --checkpoints runs/ch64_run/best_psnr.pth \
                      runs/baseline_ffa_net_lidar/best.pth \
                      runs/baseline_aod_net_rgb/best.pth

    # Quick test:
    python downstream_detection/detect_and_eval.py \
        --checkpoints runs/ch64_run/best_psnr.pth --max_frames 50
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Force unbuffered output
sys.stdout = open(sys.stdout.fileno(), "w", buffering=1)

# Add parent to path for model/dataset imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from dataset import STFDehazeDataset
from model import get_model, MODEL_VARIANTS
from baselines import get_baseline_model

# ---------------------------------------------------------------------------
# Class family mapping
# ---------------------------------------------------------------------------
EVAL_FAMILIES = {
    0: {"name": "vehicle",
        "stf": ["PassengerCar", "PassengerCar_is_group",
                "Vehicle", "Vehicle_is_group",
                "LargeVehicle", "LargeVehicle_is_group"],
        "coco": {2, 5, 7}},
    1: {"name": "person",
        "stf": ["Pedestrian", "Pedestrian_is_group", "person"],
        "coco": {0}},
    2: {"name": "cyclist",
        "stf": ["RidableVehicle", "RidableVehicle_is_group"],
        "coco": {1, 3}},
}

STF_TO_FAMILY, COCO_TO_FAMILY = {}, {}
for _fid, _fam in EVAL_FAMILIES.items():
    for _cls in _fam["stf"]:
        STF_TO_FAMILY[_cls] = _fid
    for _cid in _fam["coco"]:
        COCO_TO_FAMILY[_cid] = _fid
FAMILY_NAMES = {fid: f["name"] for fid, f in EVAL_FAMILIES.items()}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_stf_label(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) < 8:
                continue
            fid = STF_TO_FAMILY.get(p[0])
            if fid is None or p[0] in ("DontCare", "Obstacle"):
                continue
            x1, y1, x2, y2 = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            x1, y1 = max(0, min(x1, img_w)), max(0, min(y1, img_h))
            x2, y2 = max(0, min(x2, img_w)), max(0, min(y2, img_h))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append({"class_id": fid, "bbox": [x1, y1, x2, y2]})
    return boxes


def load_stf_image(path):
    pil = Image.open(path)
    if pil.mode in ("I;16", "I"):
        raw = np.array(pil).astype(np.float32)
        mx = raw.max()
        scale = 4095.0 if mx <= 4095 else (65535.0 if mx > 255 else max(mx, 1.0))
        g = np.clip(raw / scale, 0.0, 1.0)
        return np.stack([g, g, g], axis=-1)
    return np.array(pil.convert("RGB")).astype(np.float32) / 255.0


def to_uint8(img):
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Fog synthesis
# ---------------------------------------------------------------------------

def _fill_depth_nn(depth, default=50.0):
    if np.count_nonzero(depth) == 0:
        return np.full_like(depth, default)
    from scipy.ndimage import distance_transform_edt
    mask = depth > 0
    _, idx = distance_transform_edt(~mask, return_distances=True,
                                     return_indices=True)
    return depth[tuple(idx)]


def synthesize_fog(clear, depth, beta, airlight=0.8):
    df = _fill_depth_nn(depth)
    t = np.exp(-beta * df).astype(np.float32)
    foggy = clear * t[..., None] + np.float32(airlight) * (1.0 - t[..., None])
    return np.clip(foggy, 0, 1).astype(np.float32), t


# ---------------------------------------------------------------------------
# Tiled dehazing
# ---------------------------------------------------------------------------

def _dehaze_tile(model, img, depth, max_depth, device):
    sp = np.clip(depth / max_depth, 0, 1).astype(np.float32)
    mk = (depth > 0).astype(np.float32)
    it = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    st = torch.from_numpy(sp[None]).float().unsqueeze(0).to(device)
    mt = torch.from_numpy(mk[None]).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(it, st, mt)
    return np.clip(out["restored"].squeeze(0).cpu().numpy().transpose(1, 2, 0), 0, 1)


def dehaze_image(model, img, depth, max_depth, device, ts=512, ov=64):
    h, w = img.shape[:2]
    if h <= ts and w <= ts:
        return _dehaze_tile(model, img, depth, max_depth, device)
    stride = ts - ov
    out = np.zeros_like(img)
    wt = np.zeros((h, w, 1), dtype=np.float32)
    for y0 in range(0, h, stride):
        for x0 in range(0, w, stride):
            ye, xe = min(y0 + ts, h), min(x0 + ts, w)
            ys, xs = max(0, ye - ts), max(0, xe - ts)
            out[ys:ye, xs:xe] += _dehaze_tile(
                model, img[ys:ye, xs:xe], depth[ys:ye, xs:xe], max_depth, device)
            wt[ys:ye, xs:xe] += 1.0
    return np.clip(out / np.maximum(wt, 1.0), 0, 1)


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

def run_yolo_batch(yolo, images, conf=0.25):
    results = yolo(images, verbose=False, conf=conf)
    all_d = []
    for r in results:
        ds = []
        for b in r.boxes:
            fid = COCO_TO_FAMILY.get(int(b.cls.item()))
            if fid is None:
                continue
            ds.append({"class_id": fid,
                       "confidence": float(b.conf.item()),
                       "bbox": b.xyxy[0].cpu().numpy().tolist()})
        all_d.append(ds)
    return all_d


# ---------------------------------------------------------------------------
# mAP
# ---------------------------------------------------------------------------

def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x2 - x1) * max(0, y2 - y1)
    return i / max((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i, 1e-8)


def _ap(prec, rec):
    mr = np.concatenate(([0.0], rec, [1.0]))
    mp = np.concatenate(([1.0], prec, [0.0]))
    for i in range(len(mp)-2, -1, -1):
        mp[i] = max(mp[i], mp[i+1])
    return sum(
        mp[np.searchsorted(mr, rp, side="left")]
        if np.searchsorted(mr, rp, side="left") < len(mp) else 0.0
        for rp in np.linspace(0, 1, 101)
    ) / 101.0


def compute_map(all_dets, all_gt, iou_thresh=0.5):
    pc = {}
    for cid in FAMILY_NAMES:
        dets, n_gt, gm = [], 0, {}
        for i in range(len(all_gt)):
            gc = [g for g in all_gt[i] if g["class_id"] == cid]
            gm[i] = {"b": [g["bbox"] for g in gc], "m": [False]*len(gc)}
            n_gt += len(gc)
            for d in all_dets[i]:
                if d["class_id"] == cid:
                    dets.append((i, d["confidence"], d["bbox"]))
        if n_gt == 0:
            pc[cid] = 0.0; continue
        dets.sort(key=lambda x: x[1], reverse=True)
        tp, fp = np.zeros(len(dets)), np.zeros(len(dets))
        for di, (ii, _, db) in enumerate(dets):
            gi = gm[ii]; bi, bg = 0.0, -1
            for gj, gb in enumerate(gi["b"]):
                v = _iou(db, gb)
                if v > bi:
                    bi, bg = v, gj
            if bi >= iou_thresh and not gi["m"][bg]:
                tp[di] = 1; gi["m"][bg] = True
            else:
                fp[di] = 1
        tc, fc = np.cumsum(tp), np.cumsum(fp)
        pc[cid] = _ap(tc / (tc + fc + 1e-8), tc / n_gt)
    vals = list(pc.values())
    return pc, float(np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _short_name(ckpt_path: str) -> str:
    """Extract a human-readable short name from a checkpoint path."""
    run_dir = Path(ckpt_path).parent.name
    # Baselines: baseline_stf_<name>_<mode>_...
    if run_dir.startswith("baseline_stf_"):
        parts = run_dir.split("_")
        # e.g. baseline_stf_ffa_net_lidar_physics_0.79M_...
        after = parts[2:]  # drop "baseline", "stf"
        # find the Params field (contains "M")
        mi = next((i for i, p in enumerate(after) if "M" in p and p[0].isdigit()), len(after))
        name_parts = after[:mi]
        # last part before M is the mode if it's a known mode
        modes = {"rgb", "lidar", "lidar_physics", "lidar_gated"}
        mode = None
        for m in modes:
            mtokens = m.split("_")
            if name_parts[-len(mtokens):] == mtokens:
                mode = m
                name_parts = name_parts[:-len(mtokens)]
                break
        base = "_".join(name_parts)
        return f"{base}({mode})" if mode else base
    # Custom: stf_<variant>_ch<N>_...
    if run_dir.startswith("stf_"):
        parts = run_dir.split("_")
        # variant + channel
        variant = parts[1]  # lite, robust
        ch = next((p for p in parts if p.startswith("ch")), "")
        # loss tag
        v2_idx = next((i for i, p in enumerate(parts) if p == "v2"), None)
        if v2_idx is not None and v2_idx + 1 < len(parts):
            loss_tag = parts[v2_idx + 1]
        else:
            loss_tag = "v1"
        return f"{variant}_{ch}_{loss_tag}" if ch else f"{variant}_{loss_tag}"
    return run_dir[:30]


def load_dehaze_model(ckpt_path, device):
    """Load a dehazing model from checkpoint. Returns (model, short_name, n_params)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    mc = cfg.get("model", {})
    variant = mc.get("variant", "lite")

    baseline = mc.get("baseline")
    if baseline:
        mode = mc.get("mode", "rgb")
        kw = {k: v for k, v in mc.items()
              if k not in ("baseline", "mode", "variant")}
        model = get_baseline_model(baseline, mode=mode, **kw).to(device)
    else:
        ov = {"base_ch": mc.get("base_ch", 32)}
        for k in ("use_cbam", "use_residual", "use_physics_head", "lidar_drop_rate"):
            v = mc.get(k)
            if v is not None:
                ov[k] = v
        model = get_model(variant, **ov).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    name = _short_name(ckpt_path)
    return model, name, n_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model downstream detection: "
                    "clean vs synthetic-fog vs dehazed (N models)")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="One or more dehazing checkpoint paths")
    parser.add_argument("--stf_root", default="data/stf/SeeingThroughFog")
    parser.add_argument("--meta_dir", default="data/stf/meta")
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--beta", type=float, default=0.06)
    parser.add_argument("--airlight", type=float, default=0.8)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--yolo_model", default="yolov8m.pt")
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--iou_thresh", type=float, default=0.5)
    parser.add_argument("--max_depth", type=float, default=120.0)
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_overlap", type=int, default=64)
    parser.add_argument("--yolo_batch_size", type=int, default=8)
    parser.add_argument("--out_dir", default="results/downstream_detection")
    parser.add_argument("--save_samples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _res(p):
        pp = Path(p)
        return pp if pp.is_absolute() else PROJECT_ROOT / pp

    stf_root = _res(args.stf_root)
    meta_dir = _res(args.meta_dir)
    label_dir = _res(args.label_dir) if args.label_dir else \
        stf_root / "gt_labels" / "cam_left_labels_TMP"
    checkpoints = [str(_res(c)) for c in args.checkpoints]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Discover VAL clear+overcast frames ----
    val_ts = str(meta_dir / "val_timestamps.txt")
    print(f"[1] Loading val clear+overcast frames...", flush=True)
    ds = STFDehazeDataset(
        stf_root=str(stf_root), timestamps_file=val_ts,
        crop_size=None, augment=False,
        weather_filter=["clear", "overcast"], use_all_if_no_labels=True)
    samples = ds.samples
    if args.max_frames:
        samples = samples[:args.max_frames]
    samples = [s for s in samples
               if (label_dir / f"{s['timestamp']}.txt").exists()]
    print(f"  {len(samples)} annotated val frames", flush=True)
    if not samples:
        print("ERROR: no frames"); return

    # ---- 2. Load YOLO ----
    print(f"[2] Loading YOLO {args.yolo_model}...", flush=True)
    from ultralytics import YOLO
    yolo = YOLO(args.yolo_model)
    print(f"  YOLO loaded", flush=True)

    # ---- 3. Load all dehazing models ----
    models = []
    for ci, ckpt in enumerate(checkpoints):
        print(f"[3] Loading model {ci+1}/{len(checkpoints)}: {Path(ckpt).parent.name}",
              flush=True)
        m, name, np_ = load_dehaze_model(ckpt, device)
        models.append({"model": m, "name": name, "params": np_, "ckpt": ckpt})
        print(f"  → {name}  ({np_/1e6:.2f}M)", flush=True)

    n_models = len(models)
    model_names = [m["name"] for m in models]

    # ---- 4. Process frames ----
    BS = args.yolo_batch_size
    print(f"\n[4] Evaluating {len(samples)} frames × {n_models+2} conditions "
          f"(β={args.beta}, tile={args.tile_size}, bs={BS})...", flush=True)

    # Per-condition detection lists: clean, foggy, model_0, model_1, …
    all_dets = {"clean": [], "foggy": []}
    for m in models:
        all_dets[m["name"]] = []
    gt_all = []

    # Batch accumulators
    batch_imgs = {"clean": [], "foggy": []}
    for m in models:
        batch_imgs[m["name"]] = []
    batch_meta = []

    def _flush():
        nonlocal batch_meta
        if not batch_meta:
            return
        # Run batched YOLO for each condition
        cond_results = {}
        for cond, imgs in batch_imgs.items():
            cond_results[cond] = run_yolo_batch(yolo, imgs, args.conf_thresh)

        for bi, meta in enumerate(batch_meta):
            idx, ts, gt, imgs_u8 = meta
            gt_all.append(gt)
            for cond in all_dets:
                all_dets[cond].append(cond_results[cond][bi])

            if idx < args.save_samples:
                _save_grid(out_dir / "samples", idx, ts,
                           {c: imgs_u8[c] for c in imgs_u8},
                           {c: cond_results[c][bi] for c in cond_results},
                           gt, model_names)

        for k in batch_imgs:
            batch_imgs[k].clear()
        batch_meta.clear()

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        ts = sample["timestamp"]
        clean_np = load_stf_image(sample["cam_path"])
        h, w = clean_np.shape[:2]

        depth_raw = np.load(sample["depth_path"])
        depth_np = (depth_raw["arr_0"] if isinstance(depth_raw, np.lib.npyio.NpzFile)
                    else depth_raw).astype(np.float32)

        foggy_np, _ = synthesize_fog(clean_np, depth_np, args.beta, args.airlight)
        gt_boxes = parse_stf_label(str(label_dir / f"{ts}.txt"), w, h)

        imgs_u8 = {"clean": to_uint8(clean_np), "foggy": to_uint8(foggy_np)}
        batch_imgs["clean"].append(imgs_u8["clean"])
        batch_imgs["foggy"].append(imgs_u8["foggy"])

        # Dehaze with each model
        for m in models:
            deh = dehaze_image(m["model"], foggy_np, depth_np,
                               args.max_depth, device,
                               ts=args.tile_size, ov=args.tile_overlap)
            deh_u8 = to_uint8(deh)
            imgs_u8[m["name"]] = deh_u8
            batch_imgs[m["name"]].append(deh_u8)

        batch_meta.append((idx, ts, gt_boxes, imgs_u8))

        if len(batch_imgs["clean"]) >= BS:
            _flush()
        if (idx + 1) % 50 == 0:
            print(f"  [{idx+1}/{len(samples)}]", flush=True)

    _flush()

    # ---- 5. Compute mAP for all conditions ----
    print(f"\n[5] Computing mAP...", flush=True)
    results = {}
    for cond in all_dets:
        ap, mAP = compute_map(all_dets[cond], gt_all, args.iou_thresh)
        results[cond] = {"ap": ap, "mAP": mAP}

    clean_map = results["clean"]["mAP"]
    foggy_map = results["foggy"]["mAP"]
    fog_drop = foggy_map - clean_map

    # ---- Print table ----
    sep = "=" * 80
    print(f"\n{sep}", flush=True)
    print(f"  Downstream Detection — Synthetic Fog (β={args.beta}, A={args.airlight})", flush=True)
    print(f"  {len(samples)} val frames | YOLO {args.yolo_model} | "
          f"conf≥{args.conf_thresh} | IoU≥{args.iou_thresh}", flush=True)
    print(sep, flush=True)

    # Header
    fam_hdrs = [f"AP({n})" for n in FAMILY_NAMES.values()]
    hdr = f"  {'Method':<28s}" + "".join(f"{h:>12s}" for h in fam_hdrs) + f"{'mAP':>8s}  {'Δ':>7s}"
    print(hdr, flush=True)
    print(f"  {'-'*76}", flush=True)

    def _row(label, r, delta=None):
        s = f"  {label:<28s}"
        for fid in FAMILY_NAMES:
            s += f"{r['ap'].get(fid,0)*100:>11.1f}%"
        s += f"{r['mAP']*100:>7.1f}%"
        if delta is not None:
            s += f"  {delta*100:>+6.1f}%"
        else:
            s += f"  {'—':>6s}"
        print(s, flush=True)
        return s

    rows = []
    rows.append(_row("Clean (upper bound)", results["clean"]))
    rows.append(_row(f"Foggy (β={args.beta})", results["foggy"], fog_drop))

    print(f"  {'-'*76}", flush=True)
    for m in models:
        n = m["name"]
        delta = results[n]["mAP"] - foggy_map
        rows.append(_row(f"{n}", results[n], delta))

    print(sep, flush=True)
    print(f"\n  Fog degradation (clean→foggy): {fog_drop*100:+.1f}% mAP", flush=True)
    for m in models:
        n = m["name"]
        recovery = results[n]["mAP"] - foggy_map
        rate = (recovery / abs(fog_drop) * 100) if abs(fog_drop) > 1e-6 else 0.0
        print(f"  {n}: recovery {recovery*100:+.1f}% mAP "
              f"({rate:.0f}% of lost mAP)", flush=True)
    print(sep, flush=True)

    # ---- Save summary.txt ----
    summary_lines = [
        "Downstream Detection — Multi-Model Comparison",
        sep,
        f"Frames: {len(samples)} val clear+overcast",
        f"Fog: β={args.beta}, airlight={args.airlight}",
        f"Detector: {args.yolo_model} (conf≥{args.conf_thresh}, IoU≥{args.iou_thresh})",
        f"Tile: {args.tile_size} (overlap {args.tile_overlap})",
        "",
    ]
    for r in rows:
        summary_lines.append(r.rstrip())
    summary_lines.append(sep)
    for m in models:
        summary_lines.append(f"\n{m['name']}: {m['ckpt']}  ({m['params']/1e6:.2f}M)")

    (out_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")

    # ---- Save results.csv ----
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["method"] + [f"AP_{FAMILY_NAMES[fid]}" for fid in FAMILY_NAMES] + \
                 ["mAP", "delta_mAP", "params_M", "checkpoint"]
        w.writerow(header)
        for cond in ["clean", "foggy"] + [m["name"] for m in models]:
            r = results[cond]
            delta = r["mAP"] - foggy_map if cond not in ("clean", "foggy") else \
                    (fog_drop if cond == "foggy" else 0.0)
            params = next((m["params"]/1e6 for m in models if m["name"] == cond), 0)
            ckpt = next((m["ckpt"] for m in models if m["name"] == cond), "")
            row = [cond] + [f"{r['ap'].get(fid,0)*100:.2f}" for fid in FAMILY_NAMES] + \
                  [f"{r['mAP']*100:.2f}", f"{delta*100:.2f}", f"{params:.2f}", ckpt]
            w.writerow(row)

    print(f"\nSummary: {out_dir / 'summary.txt'}", flush=True)
    print(f"CSV:     {csv_path}", flush=True)
    print(f"Samples: {out_dir / 'samples'}/", flush=True)


# ---------------------------------------------------------------------------
# N+2 column sample grid
# ---------------------------------------------------------------------------

def _save_grid(save_dir, idx, timestamp, imgs_u8, dets, gt, model_names):
    """Save a comparison grid: Clean | Foggy | Model_1 | … | Model_N."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["clean", "foggy"] + model_names
    n_cols = len(conditions)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    colors = {0: "cyan", 1: "lime", 2: "orange"}
    titles = {"clean": "Clean", "foggy": "Foggy"}
    for mn in model_names:
        titles[mn] = mn

    for ci, cond in enumerate(conditions):
        ax = axes[ci]
        img = imgs_u8.get(cond)
        if img is None:
            ax.axis("off"); continue
        ax.imshow(img)
        d = dets.get(cond, [])
        ax.set_title(f"{titles[cond]} ({len(d)} det)", fontsize=10)
        ax.axis("off")

        # GT (dashed white)
        for g in gt:
            x1, y1, x2, y2 = g["bbox"]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, lw=1.2,
                edgecolor="white", facecolor="none", ls="--"))

        # Detections
        for dd in d:
            x1, y1, x2, y2 = dd["bbox"]
            fid = dd["class_id"]
            c = colors.get(fid, "yellow")
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, lw=1.5,
                edgecolor=c, facecolor="none"))
            ax.text(x1, y1-3,
                    f"{FAMILY_NAMES.get(fid,'?')} {dd['confidence']:.2f}",
                    color=c, fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="black", alpha=0.6))

    plt.suptitle(timestamp, fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(save_dir / f"{idx:04d}_{timestamp}.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
