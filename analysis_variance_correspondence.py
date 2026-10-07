#!/usr/bin/env python3
"""
Empirical validation of Proposition 1 (transmission-scaled noise variance):
Var[J_phys] = sigma^2 / t^2 + O(delta).

Using the released PhysDNet-M checkpoint on synthetic STF validation haze
with KNOWN injected sensor noise (sigma = 0.01, the training synthesis
value), we bin the per-pixel squared error of the parameter-free physics
inversion by ground-truth transmission and overlay the analytic
sigma^2/t^2 curve. The dark/dense-region noise visible in the qualitative
panels is this law made visible.

    python analysis_variance_correspondence.py --out results/variance_correspondence
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import torch

from model import get_model
from dataset import STFDehazeDataset

CKPT = ("runs/stf_robust_ch64_3.0M_wclear+overcast_crop256_"
        "b0.005-0.04_bs16_lr2e-04_ep500_v2_fft+ctr@185_0406_2105/"
        "best_psnr.pth")
SIGMA = 0.01  # synthesis noise (dataset._synthesize_haze)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--stf-root", default="datasets/stf/SeeingThroughFog")
    ap.add_argument("--n-samples", type=int, default=300)
    ap.add_argument("--n-bins", type=int, default=24)
    ap.add_argument("--out", default="results/variance_correspondence")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = get_model("robust")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    # combine official clear val splits (day+night)
    meta = Path("data/stf/meta")
    stems = []
    for f in ("val_clear_day.txt", "val_clear_night.txt"):
        stems += [l.strip().replace(",", "_") for l in open(meta / f)
                  if l.strip()]
    ts_file = out / "_val_stems.txt"
    ts_file.write_text("\n".join(stems))

    np.random.seed(42)
    ds = STFDehazeDataset(
        stf_root=args.stf_root, timestamps_file=str(ts_file),
        crop_size=(256, 256), beta_range=(0.005, 0.04),
        airlight_range=(0.7, 1.0), augment=False)
    print(f"[DATA] {len(ds)} val frames", flush=True)

    edges = np.linspace(0.05, 1.0, args.n_bins + 1)
    sq_sum = np.zeros(args.n_bins)
    counts = np.zeros(args.n_bins)

    with torch.no_grad():
        idxs = np.random.RandomState(0).choice(
            len(ds), min(args.n_samples, len(ds)), replace=False)
        for j, i in enumerate(idxs):
            s = ds[int(i)]
            hazy = s["hazy"][None].to(device)
            out_d = model(hazy, s["sparse_depth"][None].to(device),
                          s["mask"][None].to(device))
            # per-pixel squared error of the physics inversion vs GT
            err = ((out_d["physics_restored"].cpu()[0] - s["clear"]) ** 2
                   ).mean(0).numpy()                      # mean over RGB
            t = s["trans_gt"][0].numpy()
            # exclude clamp-saturated pixels (inversion clipped to [0,1])
            p = out_d["physics_restored"].cpu()[0].numpy()
            interior = (p.min(0) > 0.005) & (p.max(0) < 0.995)
            b = np.clip(np.digitize(t, edges) - 1, 0, args.n_bins - 1)
            for k in range(args.n_bins):
                m = (b == k) & interior
                if m.any():
                    sq_sum[k] += err[m].sum()
                    counts[k] += m.sum()
            if (j + 1) % 100 == 0:
                print(f"  {j+1}/{len(idxs)}", flush=True)

    centers = 0.5 * (edges[:-1] + edges[1:])
    emp = sq_sum / np.maximum(counts, 1)
    theory = SIGMA ** 2 / centers ** 2

    np.savez(out / "variance_bins.npz", centers=centers, empirical=emp,
             theory=theory, counts=counts, sigma=SIGMA)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ok = counts > 1000
    ax.loglog(centers[ok], emp[ok], "o-", ms=4, lw=1.2,
              label=r"empirical MSE of $\hat J_{\mathrm{phys}}$")
    ax.loglog(centers[ok], theory[ok], "--", lw=1.4, color="k",
              label=r"$\sigma^2/t^2$ (Prop. 1)")
    ax.set_xlabel(r"ground-truth transmission $t$")
    ax.set_ylabel("per-pixel squared error")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Transmission-scaled noise variance (real STF val, "
                 r"$\sigma{=}0.01$)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "variance_correspondence.png", dpi=220)
    fig.savefig(out / "variance_correspondence.pdf")

    # correlation in log space over well-populated bins
    r = np.corrcoef(np.log(emp[ok]), np.log(theory[ok]))[0, 1]
    print(f"[RESULT] log-log correlation empirical vs sigma^2/t^2: "
          f"r = {r:.4f} over {ok.sum()} bins", flush=True)
    print(f"[RESULT] figure: {out/'variance_correspondence.png'}",
          flush=True)


if __name__ == "__main__":
    main()
