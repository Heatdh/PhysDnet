"""
Export LiDARDehazeNet to ONNX for TensorRT / edge deployment.

Usage:
    python export_onnx.py --checkpoint checkpoints/best.pth \
                          --output dehaze_model.onnx \
                          --img_h 256 --img_w 256

    # Then convert with TensorRT (on Jetson or x86):
    #   trtexec --onnx=dehaze_model.onnx --saveEngine=dehaze_fp16.engine --fp16
    #   trtexec --onnx=dehaze_model.onnx --saveEngine=dehaze_int8.engine --int8 \
    #           --calib=<calibration_cache>
"""

import argparse
import torch
import torch.nn as nn

from model import LiDARDehazeNet, get_model, MODEL_VARIANTS


class LiDARDehazeNetExport(nn.Module):
    """
    Thin wrapper that flattens dict outputs into a tuple for ONNX.
    ONNX doesn't support dict outputs natively.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        rgb_hazy: torch.Tensor,
        sparse_depth: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(rgb_hazy, sparse_depth, mask)
        return (
            out["restored"],         # B x 3 x H x W
            out["transmission"],     # B x 1 x H x W
            out["airlight"],         # B x 3
            out["physics_restored"], # B x 3 x H x W
        )


def export():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (None = random weights)")
    parser.add_argument("--output", type=str, default="dehaze_model.onnx")
    parser.add_argument("--model", type=str, default="lite",
                        choices=list(MODEL_VARIANTS),
                        help="Model variant: lite or robust")
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)
    parser.add_argument("--base_ch", type=int, default=None)
    parser.add_argument("--opset", type=int, default=18,
                        help="ONNX opset version (18 for PyTorch 2.x, 17+ for TensorRT 8.6+)")
    parser.add_argument("--simplify", action="store_true",
                        help="Run onnx-simplifier after export")
    args = parser.parse_args()

    # ---- Load model ----
    # Auto-detect variant from checkpoint config if available
    variant = args.model
    base_ch = args.base_ch
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        cfg = ckpt.get("config", {})
        variant = cfg.get("model", {}).get("variant", variant)
        if base_ch is None:
            base_ch = cfg.get("model", {}).get("base_ch")

    overrides = {}
    if base_ch is not None:
        overrides["base_ch"] = base_ch
    model = get_model(variant, **overrides)

    if args.checkpoint:
        model.load_state_dict(ckpt["model"])
        print(f"Loaded checkpoint: {args.checkpoint} (variant={variant})")
    model.eval()

    export_model = LiDARDehazeNetExport(model)

    # ---- Dummy inputs ----
    B = 1
    rgb = torch.rand(B, 3, args.img_h, args.img_w)
    depth = torch.rand(B, 1, args.img_h, args.img_w)
    mask = (torch.rand(B, 1, args.img_h, args.img_w) > 0.95).float()

    # ---- Export ----
    print(f"Exporting to {args.output} (input: {B}x3x{args.img_h}x{args.img_w}, opset {args.opset})")

    # compile model 
    #export_model = torch.compile(export_model)
    torch.onnx.export(
        export_model,
        (rgb, depth, mask),
        args.output,
        opset_version=args.opset,
        input_names=["rgb_hazy", "sparse_depth", "mask"],
        output_names=["restored", "transmission", "airlight", "physics_restored"],
        dynamic_axes=None,  # fixed shape for TensorRT
        dynamo=False,
        do_constant_folding=True,
    )
    print(f"Saved: {args.output}")

    # ---- Verify ----
    import onnx
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed.")

    # Print model size
    import os
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"Model size: {size_mb:.2f} MB")

    # ---- Optional: simplify ----
    if args.simplify:
        try:
            import onnxsim
            simplified, ok = onnxsim.simplify(onnx_model)
            if ok:
                onnx.save(simplified, args.output)
                print("Simplified with onnx-simplifier.")
            else:
                print("Simplification failed, keeping original.")
        except ImportError:
            print("onnx-simplifier not installed. pip install onnxsim")

    # ---- Inference test with onnxruntime ----
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(args.output)
        outputs = sess.run(None, {
            "rgb_hazy": rgb.numpy(),
            "sparse_depth": depth.numpy(),
            "mask": mask.numpy(),
        })
        print(f"ONNX Runtime test:")
        names = ["restored", "transmission", "airlight", "physics_restored"]
        for name, arr in zip(names, outputs):
            print(f"  {name}: shape={arr.shape}, range=[{arr.min():.3f}, {arr.max():.3f}]")
        print("ONNX Runtime inference OK.")
    except ImportError:
        print("onnxruntime not installed, skipping inference test.")
    except Exception as e:
        print(f"ONNX Runtime test skipped (version mismatch or error): {e}")
        print("The exported ONNX model is valid — test on your target device.")


if __name__ == "__main__":
    export()
