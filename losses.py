"""
Loss functions for LiDAR-guided dehazing.

Combines:
  1. L1 / Charbonnier reconstruction loss
  2. Perceptual loss (VGG features)
  3. Edge / gradient loss
  4. Physics consistency loss (restored vs physics-reconstructed)
  5. Transmission smoothness regularization (depth-guided)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Individual loss components
# ---------------------------------------------------------------------------

class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss: sqrt(||x-y||^2 + eps^2)."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps2 = eps ** 2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))


class GradientLoss(nn.Module):
    """L1 loss on Sobel-like image gradients (edge preservation)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _gradient(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        px, py = self._gradient(pred)
        tx, ty = self._gradient(target)
        return F.l1_loss(px, tx) + F.l1_loss(py, ty)


class PerceptualLoss(nn.Module):
    """
    VGG-16 feature matching loss (layers relu1_2, relu2_2, relu3_3).

    NOTE: for Jetson deployment this is only used at training time.
    To keep deps light, we lazy-load VGG.
    """

    def __init__(self, layer_weights: dict[int, float] | None = None):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        # Freeze
        for p in vgg.parameters():
            p.requires_grad = False

        # We extract at indices: 3 (relu1_2), 8 (relu2_2), 15 (relu3_3)
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.children())[:4]),   # relu1_2
            nn.Sequential(*list(vgg.children())[4:9]),  # relu2_2
            nn.Sequential(*list(vgg.children())[9:16]), # relu3_3
        ])
        self.weights = layer_weights or {0: 1.0, 1: 1.0, 2: 1.0}

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.tensor(0.0, device=pred.device)
        x, y = pred, target
        for i, layer in enumerate(self.slices):
            x = layer(x)
            with torch.no_grad():
                y = layer(y)
            w = self.weights.get(i, 1.0)
            loss = loss + w * F.l1_loss(x, y)
        return loss


class TransmissionSmoothnessLoss(nn.Module):
    """
    Edge-aware smoothness on the predicted transmission map,
    guided by the input image edges (transmission should be smooth
    where the image is smooth, and can change at object boundaries).
    """

    def __init__(self):
        super().__init__()

    def forward(self, t_hat: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        # image gradients (as edge weights)
        img_dx = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean(dim=1, keepdim=True)
        img_dy = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean(dim=1, keepdim=True)

        # transmission gradients
        t_dx = torch.abs(t_hat[:, :, :, 1:] - t_hat[:, :, :, :-1])
        t_dy = torch.abs(t_hat[:, :, 1:, :] - t_hat[:, :, :-1, :])

        # weight: penalize transmission smoothness less where image has edges
        w_x = torch.exp(-10.0 * img_dx)
        w_y = torch.exp(-10.0 * img_dy)

        return (w_x * t_dx).mean() + (w_y * t_dy).mean()


class FFTLoss(nn.Module):
    """L1 loss on FFT magnitude — encourages recovery of high-frequency detail."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Force fp32: rfft2 on fp16 inputs (e.g. under torch.amp.autocast)
        # needs a complex-half multiply kernel that isn't precompiled into
        # libtorch and falls back to NVRTC JIT compilation, which fails with
        # "libnvrtc-builtins.so.<ver> not found" on machines without the
        # matching nvrtc-builtins package installed. FFT in fp32 sidesteps
        # this entirely and is cheap relative to the rest of the loss.
        f_pred = torch.fft.rfft2(pred.float(), norm="ortho")
        f_target = torch.fft.rfft2(target.float(), norm="ortho")
        return F.l1_loss(f_pred.abs(), f_target.abs())


class ContrastiveLoss(nn.Module):
    """Feature-space contrastive: pull restored→GT, push restored→hazy.

    Uses VGG relu2_2 features. Triplet-style margin loss.
    """

    def __init__(self, margin: float = 0.3):
        super().__init__()
        self.margin = margin
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features[:9]
        for p in vgg.parameters():
            p.requires_grad = False
        self.encoder = vgg

    def forward(self, restored: torch.Tensor, clear: torch.Tensor,
                hazy: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            f_clear = self.encoder(clear)
            f_hazy = self.encoder(hazy)
        f_restored = self.encoder(restored)
        d_pos = F.l1_loss(f_restored, f_clear)
        d_neg = F.l1_loss(f_restored, f_hazy)
        return F.relu(d_pos - d_neg + self.margin)


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class DehazeLoss(nn.Module):
    """
    Full training loss combining all components.

    Args:
        w_rec       : weight for reconstruction (Charbonnier) loss
        w_perc      : weight for perceptual loss (0 to disable VGG)
        w_grad      : weight for gradient/edge loss
        w_phys      : weight for physics consistency loss
        w_phys_rec  : weight for physics_restored vs GT loss
        w_smooth    : weight for transmission smoothness
        w_trans     : weight for transmission map supervision (vs GT)
    """

    def __init__(
        self,
        w_rec: float = 1.0,
        w_perc: float = 0.05,
        w_grad: float = 0.5,
        w_phys: float = 0.2,
        w_phys_rec: float = 1.0,
        w_smooth: float = 0.1,
        w_trans: float = 2.0,
        w_fft: float = 0.0,
        w_contrast: float = 0.0,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.w_rec = w_rec
        self.w_perc = w_perc
        self.w_grad = w_grad
        self.w_phys = w_phys
        self.w_phys_rec = w_phys_rec
        self.w_smooth = w_smooth
        self.w_trans = w_trans
        self.w_fft = w_fft
        self.w_contrast = w_contrast

        self.charbonnier = CharbonnierLoss()
        self.gradient = GradientLoss()
        self.trans_smooth = TransmissionSmoothnessLoss()
        self.fft_loss = FFTLoss() if w_fft > 0 else None
        self.contrastive = ContrastiveLoss() if w_contrast > 0 else None

        self.perceptual = None
        if use_perceptual and w_perc > 0:
            self.perceptual = PerceptualLoss()

    def forward(
        self,
        model_out: dict[str, torch.Tensor],
        hazy: torch.Tensor,
        clear_gt: torch.Tensor,
        trans_gt: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            model_out : dict from LiDARDehazeNet.forward()
            hazy      : B x 3 x H x W  input hazy image
            clear_gt  : B x 3 x H x W  ground-truth clear image
            trans_gt  : B x 1 x H x W  ground-truth transmission (optional)

        Returns:
            total_loss  : scalar
            loss_dict   : breakdown for logging
        """
        restored = model_out["restored"]
        physics_restored = model_out.get("physics_restored")
        t_hat = model_out.get("transmission")
        has_physics = physics_restored is not None and t_hat is not None

        # 1. Reconstruction (direct head vs GT)
        l_rec = self.charbonnier(restored, clear_gt)

        # 2. Perceptual
        l_perc = torch.tensor(0.0, device=hazy.device)
        if self.perceptual is not None:
            l_perc = self.perceptual(restored, clear_gt)

        # 3. Gradient / edge
        l_grad = self.gradient(restored, clear_gt)

        # 4. Physics consistency: restored and physics-reconstructed agree
        l_phys = torch.tensor(0.0, device=hazy.device)
        if has_physics:
            l_phys = self.charbonnier(restored, physics_restored)

        # 5. Physics reconstruction vs GT (prevents physics head shortcut)
        l_phys_rec = torch.tensor(0.0, device=hazy.device)
        if has_physics:
            l_phys_rec = self.charbonnier(physics_restored, clear_gt)

        # 6. Transmission smoothness (edge-aware)
        l_smooth = torch.tensor(0.0, device=hazy.device)
        if has_physics:
            l_smooth = self.trans_smooth(t_hat, hazy)

        # 7. Transmission supervision (if GT available from synthetic haze)
        l_trans = torch.tensor(0.0, device=hazy.device)
        if has_physics and trans_gt is not None:
            l_trans = F.l1_loss(t_hat, trans_gt)

        # 8. FFT frequency loss
        l_fft = torch.tensor(0.0, device=hazy.device)
        if self.fft_loss is not None:
            l_fft = self.fft_loss(restored, clear_gt)

        # 9. Contrastive loss (pull to GT, push from hazy)
        l_contrast = torch.tensor(0.0, device=hazy.device)
        if self.contrastive is not None:
            l_contrast = self.contrastive(restored, clear_gt, hazy)

        total = (self.w_rec * l_rec
                 + self.w_perc * l_perc
                 + self.w_grad * l_grad
                 + self.w_phys * l_phys
                 + self.w_phys_rec * l_phys_rec
                 + self.w_smooth * l_smooth
                 + self.w_trans * l_trans
                 + self.w_fft * l_fft
                 + self.w_contrast * l_contrast)

        loss_dict = {
            "rec": l_rec.item(),
            "perc": l_perc.item(),
            "grad": l_grad.item(),
            "phys": l_phys.item(),
            "phys_rec": l_phys_rec.item(),
            "smooth": l_smooth.item(),
            "trans": l_trans.item(),
            "fft": l_fft.item(),
            "contrast": l_contrast.item(),
            "total": total.item(),
        }
        return total, loss_dict


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    B, C, H, W = 2, 3, 128, 128
    model_out = {
        "restored": torch.rand(B, C, H, W),
        "physics_restored": torch.rand(B, C, H, W),
        "transmission": torch.rand(B, 1, H, W),
        "airlight": torch.rand(B, 3),
    }
    hazy = torch.rand(B, C, H, W)
    clear = torch.rand(B, C, H, W)

    criterion = DehazeLoss(use_perceptual=False)  # skip VGG for quick test
    trans_gt = torch.rand(B, 1, H, W)
    total, breakdown = criterion(model_out, hazy, clear, trans_gt=trans_gt)
    print(f"Total loss: {total.item():.4f}")
    for k, v in breakdown.items():
        print(f"  {k}: {v:.4f}")
