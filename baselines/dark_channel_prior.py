"""
Dark Channel Prior (He et al., CVPR 2009) as a traditional dehazing baseline.

This is a parameter-free RGB dehazing method. It fits the baseline interface so it
can be evaluated through the same checkpoint-driven pipeline as the learned models.
Depth inputs, if provided by the caller, are ignored.
"""

import torch
import torch.nn.functional as F

from .common import BaseDehazeModel


class DarkChannelPrior(BaseDehazeModel):
    def __init__(
        self,
        mode: str = "rgb",
        patch_size: int = 15,
        omega: float = 0.95,
        t_min: float = 0.1,
        guided_radius: int = 20,
        guided_eps: float = 1e-3,
        airlight_top_ratio: float = 1e-3,
    ):
        super().__init__(mode=mode)
        if patch_size % 2 == 0:
            raise ValueError("patch_size must be odd")
        self.patch_size = patch_size
        self.omega = omega
        self.t_min = t_min
        self.guided_radius = guided_radius
        self.guided_eps = guided_eps
        self.airlight_top_ratio = airlight_top_ratio

    def _dark_channel(self, image: torch.Tensor) -> torch.Tensor:
        min_rgb = image.min(dim=1, keepdim=True).values
        pad = self.patch_size // 2
        return -F.max_pool2d(-min_rgb, kernel_size=self.patch_size, stride=1, padding=pad)

    def _estimate_airlight(self, image: torch.Tensor, dark: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = image.shape
        n_pixels = height * width
        topk = max(1, int(n_pixels * self.airlight_top_ratio))
        airlights = []
        for batch_idx in range(batch_size):
            dark_flat = dark[batch_idx, 0].reshape(-1)
            top_idx = torch.topk(dark_flat, k=topk, largest=True).indices
            image_flat = image[batch_idx].reshape(3, -1)
            brightness = image_flat.sum(dim=0)
            best_local = brightness[top_idx].argmax()
            airlights.append(image_flat[:, top_idx[best_local]])
        airlight = torch.stack(airlights, dim=0)
        return airlight.clamp(min=1e-3, max=1.0)

    def _guided_filter(self, guide: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        kernel = 2 * self.guided_radius + 1
        mean_guide = F.avg_pool2d(guide, kernel, stride=1, padding=self.guided_radius, count_include_pad=False)
        mean_src = F.avg_pool2d(src, kernel, stride=1, padding=self.guided_radius, count_include_pad=False)
        corr_guide = F.avg_pool2d(guide * guide, kernel, stride=1, padding=self.guided_radius, count_include_pad=False)
        corr_guide_src = F.avg_pool2d(guide * src, kernel, stride=1, padding=self.guided_radius, count_include_pad=False)

        var_guide = corr_guide - mean_guide * mean_guide
        cov_guide_src = corr_guide_src - mean_guide * mean_src

        a = cov_guide_src / (var_guide + self.guided_eps)
        b = mean_src - a * mean_guide

        mean_a = F.avg_pool2d(a, kernel, stride=1, padding=self.guided_radius, count_include_pad=False)
        mean_b = F.avg_pool2d(b, kernel, stride=1, padding=self.guided_radius, count_include_pad=False)
        return mean_a * guide + mean_b

    def _restore(self, image: torch.Tensor) -> torch.Tensor:
        dark = self._dark_channel(image)
        airlight = self._estimate_airlight(image, dark)
        normalized = image / airlight[:, :, None, None]
        transmission = 1.0 - self.omega * self._dark_channel(normalized)

        guide = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3])
        transmission = self._guided_filter(guide, transmission)
        transmission = transmission.clamp(min=self.t_min, max=1.0)

        restored = (image - airlight[:, :, None, None]) / transmission + airlight[:, :, None, None]
        return restored.clamp(0.0, 1.0)

    def _forward_features(
        self, x: torch.Tensor, depth_input: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        rgb = x[:, :3]
        return self._restore(rgb), None