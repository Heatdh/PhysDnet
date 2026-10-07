"""
Jetson deployment script using ONNX Runtime with TensorRT EP (FP16).

Runs the LiDAR-guided dehazing model on NVIDIA Jetson using the
TensorrtExecutionProvider for FP16 inference, with fallback to
CUDAExecutionProvider or CPU.

Usage:
    # Benchmark mode (random inputs):
    python jetson_deploy.py --onnx dehaze_model.onnx --benchmark

    # Single image inference:
    python jetson_deploy.py --onnx dehaze_model.onnx \
        --image input_hazy.png --depth sparse_depth.npy --mask mask.npy \
        --output restored.png

    # Benchmark with custom iterations:
    python jetson_deploy.py --onnx dehaze_model.onnx --benchmark \
        --warmup 10 --iterations 100 --img_h 256 --img_w 256
"""

import argparse
import time
import os

import numpy as np


def get_providers(force_cpu: bool = False):
    """
    Build the ONNX Runtime provider list with TensorRT FP16 preferred.
    Falls back to CUDA, then CPU.
    """
    if force_cpu:
        return ["CPUExecutionProvider"]

    providers = []

    # --- TensorRT EP with FP16 ---
    trt_options = {
        "device_id": 0,
        "trt_fp16_enable": True,
        "trt_max_workspace_size": 1 << 30,           # 1 GB
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": "./trt_cache",
        "trt_builder_optimization_level": 3,
        "trt_max_partition_iterations": 1000,
        "trt_min_subgraph_size": 1,
    }
    providers.append(("TensorrtExecutionProvider", trt_options))

    # --- CUDA fallback ---
    cuda_options = {
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",
        "cudnn_conv_algo_search": "DEFAULT",
    }
    providers.append(("CUDAExecutionProvider", cuda_options))

    # --- CPU fallback ---
    providers.append("CPUExecutionProvider")

    return providers


def create_session(onnx_path: str, force_cpu: bool = False):
    """Create an ONNX Runtime InferenceSession with best available provider."""
    import onnxruntime as ort

    os.makedirs("./trt_cache", exist_ok=True)

    providers = get_providers(force_cpu)
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"Loading model: {onnx_path}")
    print(f"Requested providers: {[p if isinstance(p, str) else p[0] for p in providers]}")

    sess = ort.InferenceSession(onnx_path, sess_options, providers=providers)

    active = sess.get_providers()
    print(f"Active providers:    {active}")
    if "TensorrtExecutionProvider" in active:
        print("  -> TensorRT FP16 enabled")
    elif "CUDAExecutionProvider" in active:
        print("  -> CUDA (no TensorRT, will use CUDA EP)")
    else:
        print("  -> CPU only")

    return sess


def preprocess_image(path: str, h: int, w: int) -> np.ndarray:
    """Load and preprocess an image to float32 NCHW [0,1]."""
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[np.newaxis]  # 1 x 3 x H x W


def run_inference(sess, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray):
    """Run a single forward pass, return output dict."""
    outputs = sess.run(None, {
        "rgb_hazy": rgb,
        "sparse_depth": depth,
        "mask": mask,
    })
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, outputs))


def benchmark(sess, h: int, w: int, warmup: int, iterations: int):
    """Run benchmark with random inputs and report latency/FPS."""
    rgb = np.random.rand(1, 3, h, w).astype(np.float32)
    depth = np.random.rand(1, 1, h, w).astype(np.float32)
    mask = (np.random.rand(1, 1, h, w) > 0.95).astype(np.float32)

    # Warmup (includes TensorRT engine build on first run)
    print(f"\nWarmup ({warmup} iterations)...")
    for i in range(warmup):
        _ = run_inference(sess, rgb, depth, mask)
        if i == 0:
            print("  First inference done (TRT engine built if applicable)")

    # Timed iterations
    print(f"Benchmarking ({iterations} iterations)...")
    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = run_inference(sess, rgb, depth, mask)
        latencies.append((time.perf_counter() - t0) * 1000)  # ms

    latencies = np.array(latencies)
    print(f"\n{'='*50}")
    print(f"Input:   1 x 3 x {h} x {w}")
    print(f"Runs:    {iterations}")
    print(f"Latency: {latencies.mean():.2f} ms  (std {latencies.std():.2f})")
    print(f"  p50:   {np.percentile(latencies, 50):.2f} ms")
    print(f"  p95:   {np.percentile(latencies, 95):.2f} ms")
    print(f"  p99:   {np.percentile(latencies, 99):.2f} ms")
    print(f"  min:   {latencies.min():.2f} ms")
    print(f"  max:   {latencies.max():.2f} ms")
    print(f"FPS:     {1000.0 / latencies.mean():.1f}")
    print(f"{'='*50}")


def single_inference(sess, args):
    """Run on a single image and save the result."""
    from PIL import Image

    h, w = args.img_h, args.img_w
    rgb = preprocess_image(args.image, h, w)

    if args.depth:
        depth = np.load(args.depth).astype(np.float32)
        if depth.ndim == 2:
            depth = depth[np.newaxis, np.newaxis]
        elif depth.ndim == 3:
            depth = depth[np.newaxis]
    else:
        print("No depth provided, using zeros (model will rely on RGB only)")
        depth = np.zeros((1, 1, h, w), dtype=np.float32)

    if args.mask:
        mask = np.load(args.mask).astype(np.float32)
        if mask.ndim == 2:
            mask = mask[np.newaxis, np.newaxis]
        elif mask.ndim == 3:
            mask = mask[np.newaxis]
    else:
        mask = (depth > 0).astype(np.float32)

    t0 = time.perf_counter()
    out = run_inference(sess, rgb, depth, mask)
    elapsed = (time.perf_counter() - t0) * 1000
    print(f"Inference: {elapsed:.2f} ms")

    # Save restored image
    restored = out["restored"][0]  # 3 x H x W
    restored = np.clip(restored.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(restored).save(args.output)
    print(f"Saved: {args.output}")

    # Save transmission map
    if "transmission" in out:
        t_map = out["transmission"][0, 0]
        t_vis = np.clip(t_map * 255, 0, 255).astype(np.uint8)
        t_path = args.output.replace(".png", "_transmission.png")
        Image.fromarray(t_vis).save(t_path)
        print(f"Saved: {t_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Jetson deployment: ONNX Runtime + TensorRT FP16"
    )
    parser.add_argument("--onnx", type=str, required=True,
                        help="Path to ONNX model")
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU execution (for testing)")

    # Benchmark mode
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)

    # Single image mode
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--depth", type=str, default=None)
    parser.add_argument("--mask", type=str, default=None)
    parser.add_argument("--output", type=str, default="restored.png")

    args = parser.parse_args()

    sess = create_session(args.onnx, force_cpu=args.cpu)

    if args.benchmark:
        benchmark(sess, args.img_h, args.img_w, args.warmup, args.iterations)
    elif args.image:
        single_inference(sess, args)
    else:
        print("Specify --benchmark or --image <path>. Use -h for help.")


if __name__ == "__main__":
    main()
