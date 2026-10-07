"""
Baseline dehazing models for benchmarking.

Learned baselines plus a traditional Dark Channel Prior reference.

Usage:
    from baselines import get_baseline_model, list_baselines

    model = get_baseline_model("ffa_net", mode="lidar_physics", ch=64)
    out = model(rgb_hazy, sparse_depth, mask)
    # out["restored"]         — always present
    # out["physics_restored"] — only in lidar_physics mode
"""

from .registry import get_baseline_model, list_baselines, BASELINE_REGISTRY

__all__ = ["get_baseline_model", "list_baselines", "BASELINE_REGISTRY"]
