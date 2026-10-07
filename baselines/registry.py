"""
Baseline model registry.

Factory function to create any baseline model by name and mode.
"""

from .aod_net import AODNet
from .ffa_net import FFANet
from .plain_unet import PlainUNet
from .dehaze_former import DehazeFormer, DEHAZE_FORMER_PRESETS
from .dea_net import DEANet
from .dark_channel_prior import DarkChannelPrior


# Maps baseline name -> (class, default_kwargs)
BASELINE_REGISTRY = {
    "aod_net": (AODNet, {}),
    "ffa_net": (FFANet, {"ch": 64, "n_groups": 3, "n_blocks": 3}),
    "dehaze_former": (DehazeFormer, DEHAZE_FORMER_PRESETS["tiny"]),
    "dea_net": (DEANet, {"base_dim": 32, "n_blocks": 4, "n_bottleneck": 8}),
    "dark_channel_prior": (DarkChannelPrior, {}),
    "plain_unet": (PlainUNet, {"base_ch": 32}),
}


def get_baseline_model(name: str, mode: str = "rgb", **overrides):
    """
    Create a baseline model instance.

    Args:
        name: one of 'aod_net', 'ffa_net', 'dehaze_former', 'dea_net', 'dark_channel_prior', 'plain_unet'
        mode: 'rgb', 'lidar', or 'lidar_physics'
        **overrides: additional kwargs passed to the model constructor
                     (e.g., ch=128, n_blocks=36 for FFA-Net full size)

    Returns:
        nn.Module with same forward signature as LiDARDehazeNet
    """
    if name not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline '{name}', choose from {list(BASELINE_REGISTRY)}"
        )
    cls, defaults = BASELINE_REGISTRY[name]
    kwargs = {**defaults, "mode": mode, **overrides}
    return cls(**kwargs)


def list_baselines() -> list[str]:
    """Return available baseline names."""
    return list(BASELINE_REGISTRY.keys())
