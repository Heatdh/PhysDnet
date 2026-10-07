#!/usr/bin/env python3
"""
Measure FLOPs, parameters, and latency for DeL-PUNET variants and baselines.

Uses Meta's fvcore for FLOPs counting.
Install: pip install fvcore

Usage:
    python measure_flops.py                    # all models, default resolution
    python measure_flops.py --res 256 256      # custom resolution
    python measure_flops.py --models robust_64 ffa_net_lidar  # specific models
    python measure_flops.py --trt              # include TensorRT / torch.compile FPS
"""

import argparse
import time
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
except ImportError:
    print("ERROR: fvcore not installed. Run: pip install fvcore")
    sys.exit(1)

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False


def _setup_tensorrt_libs():
    """Auto-discover TensorRT libs and add to LD_LIBRARY_PATH if needed."""
    import os, ctypes, glob
    # Already loadable?
    try:
        ctypes.CDLL("libnvinfer.so.10")
        ctypes.CDLL("libnvonnxparser.so.10")
        return  # nothing to do
    except OSError:
        pass

    # Search common locations (prefer current env first)
    search_paths = []
    # Current conda env
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        search_paths += glob.glob(os.path.join(conda_prefix, "lib/python*/site-packages/tensorrt_libs"))
    # All conda envs
    search_paths += glob.glob(
        os.path.expanduser("~/miniconda3/envs/*/lib/python*/site-packages/tensorrt_libs"))
    search_paths += glob.glob(
        os.path.expanduser("~/miniconda3/lib/python*/site-packages/tensorrt_libs"))
    search_paths += ["/usr/lib/x86_64-linux-gnu", "/usr/local/tensorrt/lib"]

    for p in search_paths:
        candidate = os.path.join(p, "libnvinfer.so.10")
        if os.path.exists(candidate):
            ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if p not in ld_path:
                os.environ["LD_LIBRARY_PATH"] = p + (":" + ld_path if ld_path else "")
            # Preload ALL TRT + parser libs so ORT's dlopen finds them
            for pattern in ["libnvinfer*.so*", "libnvonnxparser*.so*"]:
                for lib in sorted(glob.glob(os.path.join(p, pattern))):
                    try:
                        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                    except OSError:
                        pass
            print(f"[INIT] Loaded TensorRT libs from {p}", flush=True)
            return
    print("[INIT] WARNING: TensorRT libs not found — TRT EP will fall back to CUDA EP", flush=True)


_setup_tensorrt_libs()

# ── Project imports ──────────────────────────────────────────────────────────
from model import get_model, LiDARDehazeNet
from baselines.registry import get_baseline_model


def build_models() -> OrderedDict:
    """Build all models to measure. Returns {name: (model, description)}."""
    models = OrderedDict()

    # ── DeL-PUNET variants ───────────────────────────────────────────────
    models["del_punet_lite_ch32"] = (
        get_model("lite"),
        "DeL-PUNET-S (ch32, no attention)",
    )
    models["del_punet_robust_ch64"] = (
        get_model("robust"),
        "DeL-PUNET-M (ch64, SE attention)",
    )
    models["del_punet_large_ch96"] = (
        get_model("robust", base_ch=96),
        "DeL-PUNET-L (ch96, SE attention)",
    )
    models["del_punet_robust_cbam"] = (
        get_model("robust", use_cbam=True),
        "DeL-PUNET-M + CBAM",
    )
    models["del_punet_robust_cbam_res"] = (
        get_model("robust", use_cbam=True, use_residual=True),
        "DeL-PUNET-M + CBAM + Residual",
    )

    # ── Baselines ────────────────────────────────────────────────────────
    models["aod_net_rgb"] = (
        get_baseline_model("aod_net", mode="rgb"),
        "AOD-Net (RGB only)",
    )
    models["aod_net_lidar"] = (
        get_baseline_model("aod_net", mode="lidar"),
        "AOD-Net (LiDAR)",
    )
    models["ffa_net_rgb"] = (
        get_baseline_model("ffa_net", mode="rgb"),
        "FFA-Net (RGB only)",
    )
    models["ffa_net_lidar"] = (
        get_baseline_model("ffa_net", mode="lidar"),
        "FFA-Net (LiDAR)",
    )
    models["dehaze_former_rgb"] = (
        get_baseline_model("dehaze_former", mode="rgb"),
        "DehazeFormer-T (RGB only)",
    )
    models["dehaze_former_lidar"] = (
        get_baseline_model("dehaze_former", mode="lidar"),
        "DehazeFormer-T (LiDAR)",
    )
    models["dea_net_rgb"] = (
        get_baseline_model("dea_net", mode="rgb"),
        "DEA-Net (RGB only)",
    )
    models["dea_net_lidar"] = (
        get_baseline_model("dea_net", mode="lidar"),
        "DEA-Net (LiDAR)",
    )
    models["plain_unet_rgb"] = (
        get_baseline_model("plain_unet", mode="rgb"),
        "Plain UNet (RGB only)",
    )

    return models


def measure_latency(model: nn.Module, inputs: tuple, n_warmup: int = 20,
                    n_runs: int = 100, device: str = "cuda") -> float:
    """Measure average inference latency in ms."""
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000.0  # ms


def _export_to_onnx(model: nn.Module, inputs: tuple, name: str,
                    tmp_dir: str = "/tmp/flops_onnx") -> str:
    """Export PyTorch model to ONNX, return path."""
    import os
    os.makedirs(tmp_dir, exist_ok=True)
    onnx_path = os.path.join(tmp_dir, f"{name}.onnx")

    model.eval()
    print(f"    [ONNX] Wrapping model...", flush=True)
    # Wrap model to return only the 'restored' tensor (dict output breaks ONNX)
    class Wrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, rgb, depth, mask):
            out = self.m(rgb, depth, mask)
            if isinstance(out, dict):
                return out["restored"]
            return out

    wrapped = Wrapper(model)
    wrapped.eval()

    print(f"    [ONNX] Exporting to {onnx_path}...", flush=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            inputs,
            onnx_path,
            input_names=["rgb", "depth", "mask"],
            output_names=["restored"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,  # force legacy exporter (avoids onnx_ir.schemas bug)
        )
    print(f"    [ONNX] Export complete", flush=True)
    return onnx_path


def measure_ort_latency(onnx_path: str, inputs: tuple, provider: str,
                        n_warmup: int = 30, n_runs: int = 200) -> float:
    """Measure ORT inference latency in ms for a given EP."""
    import numpy as np

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"    [ORT] Creating session with {provider}...", flush=True)
    try:
        sess = ort.InferenceSession(onnx_path, sess_options=sess_opts,
                                    providers=[provider])
    except Exception as e:
        print(f"    [ORT] {provider} failed: {e}", flush=True)
        raise

    # Verify the EP is actually being used (ORT silently falls back to CPU)
    active_ep = sess.get_providers()
    print(f"    [ORT] Active providers: {active_ep}", flush=True)
    if provider not in active_ep:
        del sess
        raise RuntimeError(f"{provider} not active — fell back to {active_ep}")

    feed = {
        "rgb": inputs[0].cpu().numpy(),
        "depth": inputs[1].cpu().numpy(),
        "mask": inputs[2].cpu().numpy(),
    }

    # Warmup
    print(f"    [ORT] Warmup ({n_warmup} runs)...", flush=True)
    for _ in range(n_warmup):
        sess.run(None, feed)

    # Timed runs
    print(f"    [ORT] Measuring ({n_runs} runs)...", flush=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_runs):
        sess.run(None, feed)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    del sess
    ms = (t1 - t0) / n_runs * 1000.0
    print(f"    [ORT] Latency: {ms:.1f}ms", flush=True)
    return ms


def main():
    parser = argparse.ArgumentParser(description="Measure FLOPs and params")
    parser.add_argument("--res", type=int, nargs=2, default=[512, 512],
                        help="Input resolution H W (default: 512 512)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Specific models to measure (default: all)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-latency", action="store_true",
                        help="Skip latency measurement")
    parser.add_argument("--trt", action="store_true",
                        help="Also measure ONNX Runtime TensorRT EP optimized FPS")
    parser.add_argument("--csv", type=str, default=None,
                        help="Save results to CSV file")
    args = parser.parse_args()

    print("[INIT] Starting benchmark script...", flush=True)
    H, W = args.res
    device = args.device
    print(f"[INIT] Resolution: {H}×{W} | Device: {device} | TRT: {args.trt}", flush=True)
    print(f"[INIT] HAS_ORT: {HAS_ORT}", flush=True)
    print("=" * 90, flush=True)

    # Build dummy inputs
    rgb = torch.randn(1, 3, H, W, device=device)
    depth = torch.randn(1, 1, H, W, device=device)
    mask = torch.ones(1, 1, H, W, device=device)
    inputs = (rgb, depth, mask)
    print(f"[INIT] Dummy inputs created: rgb={rgb.shape}, depth={depth.shape}, mask={mask.shape}", flush=True)

    all_models = build_models()
    print(f"[INIT] Built {len(all_models)} models", flush=True)
    if args.models:
        filtered = OrderedDict()
        for name in args.models:
            if name in all_models:
                filtered[name] = all_models[name]
            else:
                print(f"WARNING: Unknown model '{name}', skipping.")
                print(f"  Available: {list(all_models.keys())}")
        all_models = filtered

    results = []

    header = f"{'Model':<35} {'Params':>10} {'GFLOPs':>10} {'GMACs':>10}"
    if not args.no_latency:
        header += f" {'ms/img':>10} {'FPS':>10}"
    if args.trt:
        header += f" {'opt_ms':>10} {'opt_FPS':>10}"
    print(header)
    print("-" * len(header), flush=True)

    for name, (model, desc) in all_models.items():
        print(f"\n[MODEL] {name}", flush=True)
        model = model.to(device).eval()
        print(f"  [MOVE] Model on {device}", flush=True)

        # Parameter count
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [PARAMS] {params/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)", flush=True)

        # FLOPs via fvcore
        print(f"  [FLOPS] Computing...", flush=True)
        try:
            flops_analysis = FlopCountAnalysis(model, inputs)
            flops_analysis.unsupported_ops_warnings(False)
            flops_analysis.uncalled_modules_warnings(False)
            total_flops = flops_analysis.total()
            gflops = total_flops / 1e9
            gmacs = gflops / 2  # MACs ≈ FLOPs / 2
        except Exception as e:
            print(f"  {name}: FLOPs measurement failed: {e}", flush=True)
            gflops = float("nan")
            gmacs = float("nan")
        print(f"  [FLOPS] {gflops:.2f} GFLOPs, {gmacs:.2f} GMACs", flush=True)

        # Latency
        latency_ms = 0.0
        fps = 0.0
        if not args.no_latency:
            print(f"  [LATENCY] Measuring PyTorch...", flush=True)
            try:
                latency_ms = measure_latency(model, inputs, device=device)
                fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            except Exception as e:
                print(f"  [LATENCY] PyTorch failed: {e}", flush=True)
        if not args.no_latency and latency_ms > 0:
            print(f"  [LATENCY] PyTorch: {latency_ms:.1f}ms, {fps:.1f} FPS", flush=True)

        # Optimized (ORT TensorRT EP) latency
        opt_latency_ms = 0.0
        opt_fps = 0.0
        if args.trt:
            if not HAS_ORT:
                print(f"  {name}: onnxruntime not installed, skipping TRT benchmark", flush=True)
            else:
                try:
                    onnx_path = _export_to_onnx(model, inputs, name)
                    print(f"  [TRT] ONNX saved: {onnx_path}", flush=True)
                    providers = ort.get_available_providers()
                    print(f"  [TRT] Available providers: {providers}", flush=True)

                    # Try TRT first, then CUDA EP, then CPU
                    ep_order = []
                    if "TensorrtExecutionProvider" in providers:
                        ep_order.append("TensorrtExecutionProvider")
                    if "CUDAExecutionProvider" in providers:
                        ep_order.append("CUDAExecutionProvider")
                    ep_order.append("CPUExecutionProvider")

                    for ep in ep_order:
                        try:
                            print(f"  [TRT] Trying {ep}...", flush=True)
                            opt_latency_ms = measure_ort_latency(
                                onnx_path, inputs, ep)
                            opt_fps = 1000.0 / opt_latency_ms if opt_latency_ms > 0 else 0.0
                            print(f"  [TRT] Result ({ep}): {opt_latency_ms:.1f}ms, {opt_fps:.1f} FPS", flush=True)
                            break  # success
                        except Exception as e:
                            print(f"  [TRT] {ep} failed, trying next...", flush=True)
                            continue
                except Exception as e:
                    print(f"  [TRT] ONNX export failed: {e}", flush=True)

        row = f"{name:<35} {params/1e6:>9.2f}M {gflops:>10.2f} {gmacs:>10.2f}"
        if not args.no_latency:
            row += f" {latency_ms:>9.1f}ms {fps:>9.1f}"
        if args.trt:
            row += f" {opt_latency_ms:>9.1f}ms {opt_fps:>9.1f}"
        print(row, flush=True)
        print(f"  [DONE]\n", flush=True)

        results.append({
            "model": name,
            "description": desc,
            "params_M": round(params / 1e6, 3),
            "trainable_M": round(trainable / 1e6, 3),
            "gflops": round(gflops, 2),
            "gmacs": round(gmacs, 2),
            "latency_ms": round(latency_ms, 1),
            "fps": round(fps, 1),
            "opt_latency_ms": round(opt_latency_ms, 1),
            "opt_fps": round(opt_fps, 1),
        })

        # Free memory
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("=" * 90, flush=True)
    print(f"Note: FLOPs counted at {H}×{W} resolution, batch=1.", flush=True)
    print(f"      GMACs ≈ GFLOPs / 2. Latency on {device} with {100} runs.", flush=True)

    # Print LaTeX table
    print("\n% ---- LaTeX table ----")
    print("\\begin{tabular}{@{}lrrrrr@{}}")
    print("\\toprule")
    cols = "\\textbf{Model} & \\textbf{Params (M)} & \\textbf{GMACs} & \\textbf{ms/img} & \\textbf{FPS} & \\textbf{Opt.~FPS} \\\\"
    print(cols)
    print("\\midrule")
    for r in results:
        lat = f"{r['latency_ms']:.1f}" if r['latency_ms'] > 0 else "---"
        f = f"{r['fps']:.1f}" if r['fps'] > 0 else "---"
        of = f"{r['opt_fps']:.1f}" if r.get('opt_fps', 0) > 0 else "---"
        print(f"{r['description']:<35} & {r['params_M']:.2f} & {r['gmacs']:.1f} & {lat} & {f} & {of} \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")

    # CSV output
    if args.csv:
        print(f"\n[CSV] Writing to {args.csv}...", flush=True)
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print(f"[CSV] Done! Results saved to {args.csv}", flush=True)
    
    print("\n[COMPLETE] Benchmark finished!", flush=True)


if __name__ == "__main__":
    main()
