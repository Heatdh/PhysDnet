# PhysDNet: Physics-Driven Gradient Amplification for Real-Time Image Dehazing

Reference implementation accompanying the NeurIPS 2026 submission
*"PhysDNet: Physics-Driven Gradient Amplification for Real-Time Image Dehazing"*.
# Disclaimer
This code is provided exclusively for the purpose of double-blind peer review. All rights reserved. No license is granted for commercial use, modification, redistribution, or use in derivative works. By accessing this repository, you agree to adhere to the confidentiality guidelines of the peer review process.


PhysDNet is a lightweight multimodal dehazing network that fuses gated NIR
imagery with sparse LiDAR depth and is trained with a physics-constrained
dual-head objective. Differentiating through a fixed Koschmieder inversion
induces *density-adaptive gradient amplification* — a `(t+eps)^-2`
gradient scaling that concentrates learning on heavily degraded regions at
**zero inference cost**.

## Model variants

| Variant | base_ch | Params | GMACs | PSNR (STF) |
|---------|---------|--------|-------|------------|
| PhysDNet-S | 32 | 0.72 M | 2.24 | 32.80 dB |
| PhysDNet-M | 64 | 2.95 M | 8.56 | 35.55 dB |
| PhysDNet-L | 96 | 6.59 M | 18.98 | 36.46 dB |

Pretrained weights (`checkpoints/`) and pre-exported ONNX files (`onnx/`) are
not tracked in git; place them in those folders locally to run the commands below.

## Setup

```bash
conda create -n physdnet python=3.10 -y
conda activate physdnet
pip install -r requirements.txt
```

## Data

We train on the **Seeing Through Fog (STF)** dataset, available from the
authors at:

> https://light.princeton.edu/datasets/automated_driving_dataset/

Request access on the page above. Once granted, you will receive a set of
signed download URLs. Save them as `data_download.json` in the repository
root using the schema below, then run the bundled downloader:

```json
{
  "urls": [
    { "key": "SeeingThroughFog/<file1>.zip", "url": "<signed-url-1>" },
    { "key": "SeeingThroughFog/<file2>.zip", "url": "<signed-url-2>" }
  ]
}
```

```bash
python download_stf.py --out data/stf
bash scripts/extract_stf.sh data/stf

# Project LiDAR points into camera frame to obtain sparse depth + mask
python project_lidar_to_depth.py --root data/stf/SeeingThroughFog
```

For cross-domain evaluation, place O-HAZE / I-HAZE / NH-HAZE under
`data/ohaze`, `data/ihaze`, `data/nhhaze` (folder layout: `hazy/`, `gt/`).
We do **not** redistribute these datasets; download them from their
official sources.

## Training

```bash
# PhysDNet-S / M / L (the paper's three variants)
python train.py --config configs/physdnet_s.yaml
python train.py --config configs/physdnet_m.yaml
python train.py --config configs/physdnet_l.yaml

# NoPhy ablation (Table 2)
python train.py --config configs/physdnet_m_nophy.yaml

# Multi-seed runs for the mean ± SE in Table 1
python train.py --config configs/seeds/physdnet_m_seed1.yaml
python train.py --config configs/seeds/physdnet_m_seed2.yaml

# Baselines (AOD-Net, FFA-Net, DehazeFormer-T, DEA-Net, DCP)
python train_baseline.py --config configs/baselines/ffa_net_256.yaml
python train_baseline.py --config configs/baselines/dehaze_former_256.yaml
python train_baseline.py --config configs/baselines/dea_net_256.yaml
python train_baseline.py --config configs/baselines/aod_net_256.yaml
python train_baseline.py --config configs/baselines/dark_channel_prior.yaml
```

Loss components (Charbonnier reconstruction, perceptual VGG, gradient,
physics reconstruction, transmission smoothness, FFT, contrastive) live in
`losses.py`. The contrastive term activates at epoch
`w_contrast_start_epoch` (default 185), matching the paper's late-activation
schedule.

## Evaluation

```bash
# In-domain STF evaluation (Table 1)
python eval.py --checkpoint checkpoints/physdnet_m.pth --save_samples 8

# Out-of-distribution beta sweep (Section 3.4 / Fig. beta_sweep)
python beta_sweep_multimodel.py \
  --models checkpoints/physdnet_s.pth checkpoints/physdnet_m.pth checkpoints/physdnet_l.pth \
  --betas 0.005 0.01 0.02 0.04 0.06 0.08 0.10 \
  --output results/beta_sweep.csv

# Zero-shot cross-domain on O-HAZE / I-HAZE / NH-HAZE (Section 3.6)
python eval_ohaze.py --checkpoint checkpoints/physdnet_m.pth --dataset ohaze
python eval_ohaze.py --checkpoint checkpoints/physdnet_m.pth --dataset ihaze
python eval_ohaze.py --checkpoint checkpoints/physdnet_m.pth --dataset nhhaze

# LiDAR-sparsity robustness (Appendix A.lidar_sparsity)
python eval_lidar_sparsity.py --checkpoint checkpoints/physdnet_m.pth \
                              --drop_rates 0.0 0.25 0.5 0.75 1.0

# No-reference perceptual quality on real haze (Appendix A.niqe)
python eval_niqe_brisque.py --checkpoint checkpoints/physdnet_m.pth \
                            --dataset ohaze

# Real STF fog: dual-head consistency (Section 3.7)
python infer_real_fog.py --checkpoint checkpoints/physdnet_m.pth \
                         --root data/stf/SeeingThroughFog --weather dense_fog
```

## Density-adaptive gradient amplification (Fig. grad_amp)

```bash
python gradient_analysis.py \
  --physics_ckpt checkpoints/physdnet_m.pth \
  --nophy_ckpt   runs/<NoPhy_run>/best_psnr.pth \
  --output figures/fig_gradient_amplification.pdf
```

This script reproduces the empirical `(t+eps)^-2` curve and the per-pixel
gradient ratio reported in the paper.

## Downstream detection (Section 3.5)

```bash
# Restore aggressive-fog frames with each model and run YOLOv8-M
python downstream_detection/detect_and_eval.py \
  --models checkpoints/physdnet_s.pth checkpoints/physdnet_m.pth checkpoints/physdnet_l.pth \
  --beta 0.06 --iou 0.5 --conf 0.25
```

## ONNX export and edge deployment (Section 3.8)

```bash
# Export all three variants
python export_onnx.py --checkpoint checkpoints/physdnet_s.pth \
                      --output onnx/physdnet_s.onnx --img_h 256 --img_w 256
python export_onnx.py --checkpoint checkpoints/physdnet_m.pth --output onnx/physdnet_m.onnx
python export_onnx.py --checkpoint checkpoints/physdnet_l.pth --output onnx/physdnet_l.onnx

# Pre-built ONNX files are also shipped under onnx/
```

For Jetson Orin Nano benchmarking:

```bash
python jetson_deploy.py --onnx onnx/physdnet_m.onnx --precision fp16
```

For fixed-function NPU INT8 calibration:

```bash
python prepare_npu_data.py --root data/stf/SeeingThroughFog --n 64 \
                           --out npu_calibration/
# Then convert with the vendor toolchain (e.g., Hailo Dataflow Compiler)
```

## FLOPs / parameters

```bash
python measure_flops.py --resolution 256 256
```

## Checkpoints

| File | Variant | base_ch | Notes |
|------|---------|---------|-------|
| `checkpoints/physdnet_s.pth` | robust | 32 | seed 0 (paper Table 1) |
| `checkpoints/physdnet_m.pth` | robust | 64 | seed 0 (paper Table 1) |
| `checkpoints/physdnet_l.pth` | robust | 96 | seed 0 (paper Table 1) |

All three were trained for 500 epochs on STF clear+overcast with
`beta in [0.005, 0.04]`, AdamW (lr 2e-4, wd 1e-4), cosine schedule with
10-epoch warm-up, contrastive loss activated at epoch 185.

## Repository layout

```
configs/                       Training configs (S/M/L + NoPhy + seeds + baselines)
checkpoints/                   Pretrained PhysDNet S/M/L
onnx/                          Pre-exported ONNX files
baselines/                     AOD-Net, FFA-Net, DehazeFormer-T, DEA-Net, DCP
downstream_detection/          YOLOv8-M evaluation pipeline
model.py                       PhysDNet backbone + dual heads
dataset.py                     STF dataloader + synthetic Koschmieder haze
losses.py                      Multi-term physics-anchored loss
train.py / train_baseline.py   Training entry points
eval*.py                       Quantitative evaluation scripts
beta_sweep*.py                 OOD fog robustness
gradient_analysis.py           Density-adaptive gradient amplification figure
infer_real_fog.py              Dual-head consistency on real STF fog
export_onnx.py / jetson_deploy.py / prepare_npu_data.py   Edge pipeline
measure_flops.py               FLOPs / params accounting
project_lidar_to_depth.py      LiDAR -> camera-frame sparse depth
parse_stf_labels.py            STF detection labels for downstream task
```
