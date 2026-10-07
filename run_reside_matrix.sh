#!/bin/bash
# ==========================================================================
# RESIDE-6K full experiment matrix: physics vs NoPhy at 600 epochs,
# unanchored and anchored(+density-aug) pairs, then all comparisons.
#
# Usage:
#   bash run_reside_matrix.sh /path/to/RESIDE-6K [gpu_id]
#
# Per-run artifacts land in runs/reside6k_<tag>/ (best_psnr.pth,
# final_psnr.txt, wandb: project physdnet-reside6k). Comparisons land in
# results/reside6k_compare600{,_anc}/ (binned table, panels, SSIM/LPIPS/
# CIEDE2000). Total: 4 trainings (~15 h each on an RTX 5080-class GPU,
# sequential) + 2 comparison passes.
# ==========================================================================
set -euo pipefail

DATA_ROOT="${1:?usage: run_reside_matrix.sh RESIDE-6K [gpu_id]}"
export CUDA_VISIBLE_DEVICES="${2:-0}"

EPOCHS=600
WORKERS=8

log() { printf '\n===== [%s] %s =====\n' "$(date +%H:%M:%S)" "$*"; }

run_train () {  # tag, extra args...
  local tag="$1"; shift
  if [ -f "runs/reside6k_${tag}/final_psnr.txt" ]; then
    log "skip ${tag} (final_psnr.txt exists)"
    return 0
  fi
  log "train ${tag}"
  python -u train_reside6k.py --tag "${tag}" --epochs "${EPOCHS}" \
      --workers "${WORKERS}" --data-root "${DATA_ROOT}" "$@"
}

run_compare () {  # physics_tag, nophy_tag, out_dir
  log "compare $1 vs $2 -> $3"
  python -u eval_reside_bins.py \
      --physics "runs/reside6k_$1/best_psnr.pth" \
      --nophy   "runs/reside6k_$2/best_psnr.pth" \
      --data-root "${DATA_ROOT}/test" --out "$3"
  python -u eval_reside_ext.py \
      --physics "runs/reside6k_$1/best_psnr.pth" \
      --nophy   "runs/reside6k_$2/best_psnr.pth" \
      --data-root "${DATA_ROOT}/test" --out "$3"
}

# ---- pair 1: unanchored (replicates the original comparison, longer) ----
run_train physics600
run_train nophy600 --nophy

# ---- pair 2: anchored + density augmentation (mechanism-relevant) -------
# w_trans mirrors the paper's STF recipe (Table 9); haze-aug is exact
# physics (t'=t^k) applied identically to both variants.
run_train physics600_anc --w-trans 2.0 --haze-aug 0.5 3.0
run_train nophy600_anc --nophy --haze-aug 0.5 3.0

# ---- comparisons (binned PSNR + panels + SSIM/LPIPS/CIEDE2000) ----------
run_compare physics600 nophy600 results/reside6k_compare600
run_compare physics600_anc nophy600_anc results/reside6k_compare600_anc

log "MATRIX_DONE"
grep -H . runs/reside6k_*/final_psnr.txt || true
