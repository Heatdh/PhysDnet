#!/usr/bin/env python3
"""
Overnight orchestrator: RTTS real-haze adaptation pipeline.

  1. eval 'hazy' + 'base' conditions on RTTS (detection mAP, NIQE/BRISQUE,
     colorfulness)
  2. wait until the URHI download is complete (file count stable)
  3. self-supervised fine-tune of PhysDNet-M on URHI (finetune_real_haze.py)
  4. eval 'adapted' condition
  5. merge summaries -> results_rtts/rtts_final.csv

Run detached; all output to rtts_pipeline.log.
"""

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent
BASE_CKPT = ("runs/stf_robust_ch64_3.0M_wclear+overcast_crop256_"
             "b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185_0406_2105/"
             "best_psnr.pth")
URHI_DIR = Path("datasets/URHI/images")
URHI_EXPECTED = 4809
FT_OUT = REPO / "runs" / "finetune_realhaze_urhi"
EPOCHS = 5


def run(cmd, tag):
    print(f"\n===== [{tag}] {' '.join(cmd)}", flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO)
    print(f"===== [{tag}] exit {r.returncode} after "
          f"{(time.time()-t0)/60:.1f} min", flush=True)
    return r.returncode


def main():
    py = sys.executable

    # ---- step 1: hazy + base eval ----
    rc = run([py, "eval_rtts.py",
              "--checkpoints", f"base={BASE_CKPT}",
              "--out", "results_rtts"], "EVAL hazy+base")
    if rc != 0:
        print("WARNING: base eval failed; continuing", flush=True)

    # ---- step 2: wait for URHI completeness (count stable or full) ----
    print("\n===== [URHI] waiting for download to finish", flush=True)
    prev = -1
    while True:
        n = len(list(URHI_DIR.glob("*")))
        print(f"[URHI] {n}/{URHI_EXPECTED}", flush=True)
        if n >= URHI_EXPECTED * 0.98 or (n == prev and n > 3000):
            break  # complete, or downloader finished/stalled with enough data
        prev = n
        time.sleep(120)
    print(f"[URHI] proceeding with {n} images", flush=True)

    # ---- step 3: fine-tune ----
    rc = run([py, "finetune_real_haze.py",
              "--checkpoint", BASE_CKPT,
              "--urhi-dir", str(URHI_DIR),
              "--epochs", str(EPOCHS),
              "--out-dir", str(FT_OUT)], "FINETUNE")
    if rc != 0:
        print("FATAL: fine-tune failed", flush=True)
        sys.exit(1)
    adapted = FT_OUT / f"adapted_ep{EPOCHS}.pth"

    # ---- step 4: adapted eval ----
    rc = run([py, "eval_rtts.py",
              "--checkpoints", f"adapted={adapted}",
              "--out", "results_rtts_adapted"], "EVAL adapted")
    if rc != 0:
        print("FATAL: adapted eval failed", flush=True)
        sys.exit(1)

    # ---- step 5: merge ----
    rows = []
    for p in ["results_rtts/rtts_summary.csv",
              "results_rtts_adapted/rtts_summary.csv"]:
        if os.path.exists(p):
            for r in csv.DictReader(open(p)):
                if not any(x["condition"] == r["condition"] for x in rows):
                    rows.append(r)
    out = REPO / "results_rtts" / "rtts_final.csv"
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"\n===== FINAL SUMMARY ({out})", flush=True)
    for r in rows:
        print(r, flush=True)
    print("PIPELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
