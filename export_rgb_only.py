#!/usr/bin/env python3
"""
RGB-only deployment variant: fold the depth branch away.

With zero depth/mask input the depth encoders compute input-independent
constants, so an RGB-only deployment can cache them as buffers: the gated
fusion becomes f = g(f_rgb, c) * f_rgb + (1 - g) * c with constant c,
mathematically identical to feeding zeros, at a fraction of the compute.
ONNX constant folding performs this elimination automatically when the
zeros are graph constants.

Outputs: edge_onnx/physdnet_m_rgbonly.onnx (+ GMACs comparison printed).

    python export_rgb_only.py --ckpt runs/reside6k_physics/best_psnr.pth
"""

import argparse

import torch
import torch.nn as nn

from model import get_model


class RGBOnlyWrapper(nn.Module):
    """Bakes zero depth/mask as constants so export folds the depth branch."""

    def __init__(self, model, res=256):
        super().__init__()
        self.model = model
        self.register_buffer("zeros", torch.zeros(1, 1, res, res))

    def forward(self, rgb_hazy):
        return self.model(rgb_hazy, self.zeros, self.zeros)["restored"]


class StrippedPhysDNet(nn.Module):
    """True stripped module for FLOPs counting: depth-branch outputs cached
    as constant buffers (computed once from zeros)."""

    def __init__(self, model, res=256):
        super().__init__()
        m = model
        self.rgb_enc = nn.ModuleList([m.rgb_enc1, m.rgb_enc2,
                                      m.rgb_enc3, m.rgb_enc4])
        self.fuse = nn.ModuleList([m.fuse1, m.fuse2, m.fuse3, m.fuse4])
        self.bottleneck_attn = m.bottleneck_attn
        self.dec3, self.dec2, self.dec1 = m.dec3, m.dec2, m.dec1
        self.head_restore = m.head_restore
        self.pool = m.pool
        with torch.no_grad():  # constant depth features from zero input
            z = torch.zeros(1, 2, res, res)
            d = m.dep_enc1(z)
            self.register_buffer("d1", d)
            d = m.dep_enc2(m.pool(d))
            self.register_buffer("d2", d)
            d = m.dep_enc3(m.pool(d))
            self.register_buffer("d3", d)
            d = m.dep_enc4(m.pool(d))
            self.register_buffer("d4", d)

    def forward(self, rgb):
        r1 = self.rgb_enc[0](rgb)
        f1 = self.fuse[0](r1, self.d1)
        r2 = self.rgb_enc[1](self.pool(r1))
        f2 = self.fuse[1](r2, self.d2)
        r3 = self.rgb_enc[2](self.pool(r2))
        f3 = self.fuse[2](r3, self.d3)
        r4 = self.rgb_enc[3](self.pool(r3))
        f4 = self.bottleneck_attn(self.fuse[3](r4, self.d4))
        x = self.dec3(f4, f3)
        x = self.dec2(x, f2)
        x = self.dec1(x, f1)
        return self.head_restore(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="runs/reside6k_physics/best_psnr.pth")
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--out", default="edge_onnx/physdnet_m_rgbonly.onnx")
    args = ap.parse_args()

    model = get_model("robust")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()

    rgb = torch.randn(1, 3, args.res, args.res)

    # equivalence check: stripped vs zero-fed full model
    stripped = StrippedPhysDNet(model, args.res).eval()
    with torch.no_grad():
        z = torch.zeros(1, 1, args.res, args.res)
        full_out = model(rgb, z, z)["restored"]
        strip_out = stripped(rgb)
    diff = (full_out - strip_out).abs().max().item()
    print(f"[CHECK] stripped vs full max |diff| = {diff:.2e}")

    # FLOPs comparison (fvcore convention: total ~= MACs, GMACs = total/2)
    from fvcore.nn import FlopCountAnalysis

    def gmacs(m, inp):
        fa = FlopCountAnalysis(m, inp)
        fa.unsupported_ops_warnings(False)
        fa.uncalled_modules_warnings(False)
        return fa.total() / 2e9

    g_full = gmacs(model, (rgb, z, z))
    g_strip = gmacs(stripped, (rgb,))
    print(f"[FLOPS] full multimodal: {g_full:.2f} GMACs | "
          f"RGB-only stripped: {g_strip:.2f} GMACs | "
          f"reduction {100*(1-g_strip/g_full):.1f}%")

    # ONNX export with constants baked (TRT prunes the depth branch)
    wrapper = RGBOnlyWrapper(model, args.res).eval()
    torch.onnx.export(wrapper, (rgb,), args.out,
                      input_names=["rgb_hazy"], output_names=["restored"],
                      opset_version=16, do_constant_folding=True,
                      dynamo=False)
    import os
    print(f"[ONNX] {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
