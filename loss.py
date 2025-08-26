import functools
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F
import torchvision


def _reduce(loss, reduction):
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"reduction={reduction}")


def _weighted(loss, weight, reduction="mean"):
    if weight is not None:
        assert weight.dim() == loss.dim()
        loss = loss * weight
    return _reduce(loss, reduction)


def weighted_loss(fn):
    @functools.wraps(fn)
    def wrap(pred, target, weight=None, reduction="mean", **kw):
        loss = fn(pred, target, **kw)
        return _weighted(loss, weight, reduction)

    return wrap


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-3):
    return torch.sqrt((pred - target) ** 2 + eps * eps)


def _rgb_to_ycbcr(x):
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    Y = 0.299 * r + 0.587 * g + 0.114 * b
    Cb = 0.564 * (b - Y)
    Cr = 0.713 * (r - Y)
    return Y, Cb, Cr


class ChromaLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction="mean", criterion="l1"):
        super().__init__()
        self.w = loss_weight
        self.reduction = reduction
        self.crit = F.l1_loss if criterion == "l1" else F.mse_loss

    def forward(self, pred, target):
        _, pCb, pCr = _rgb_to_ycbcr(pred.clamp(0, 1))
        _, tCb, tCr = _rgb_to_ycbcr(target.clamp(0, 1))
        l = self.crit(pCb, tCb, reduction=self.reduction) + self.crit(
            pCr, tCr, reduction=self.reduction
        )
        return self.w * l


def _sobel_kernels(device):
    kx = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)
    return kx, ky


class EdgeLoss(nn.Module):
    def __init__(self, loss_weight=0.5, reduction="mean", criterion="l1"):
        super().__init__()
        self.w = loss_weight
        self.reduction = reduction
        self.crit = F.l1_loss if criterion == "l1" else F.mse_loss

    def _edge(self, x):
        Y, _, _ = _rgb_to_ycbcr(x.clamp(0, 1))
        kx, ky = _sobel_kernels(Y.device)
        gx = F.conv2d(Y, kx, padding=1)
        gy = F.conv2d(Y, ky, padding=1)
        return torch.sqrt(gx * gx + gy * gy + 1e-6)

    def forward(self, pred, target):
        l = self.crit(self._edge(pred), self._edge(target), reduction=self.reduction)
        return self.w * l


class HighFreqLoss(nn.Module):
    def __init__(self, loss_weight=0.5, reduction="mean", criterion="l1"):
        super().__init__()
        self.w = loss_weight
        self.reduction = reduction
        self.crit = F.l1_loss if criterion == "l1" else F.mse_loss
        k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0])
        k = k[:, None] * k[None, :]
        k = k / k.sum()
        self.register_buffer("g", k.view(1, 1, 5, 5))

    def _hp(self, x):
        # x: (B, C, H, W)
        g = self.g.to(x.device)  # ensure kernel on same device
        g = g.expand(x.shape[1], 1, 5, 5)  # depthwise: one kernel per channel
        blur = F.conv2d(x, g, padding=2, groups=x.shape[1])
        return x - blur

    def forward(self, pred, target):
        hp_p, hp_t = self._hp(pred), self._hp(target)
        return self.w * self.crit(hp_p, hp_t, reduction=self.reduction)


class VGGLoss(nn.Module):
    def __init__(self, loss_weight=0.2, reduction="mean", criterion="l1"):
        super().__init__()
        vgg = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
        ).features
        self.layers = nn.ModuleList([vgg[:3], vgg[3:8], vgg[8:13]])
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self.w = loss_weight
        self.reduction = reduction
        self.crit = F.l1_loss if criterion == "l1" else F.mse_loss
        m = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        s = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", m)
        self.register_buffer("std", s)

    def _norm(self, x):
        return (x - self.mean) / self.std

    @torch.no_grad()
    def _feat(self, x):
        x = self._norm(x)
        feats = []
        h = x
        for block in self.layers:
            h = block(h)
            feats.append(h)
        return feats

    def forward(self, pred, target):
        f_p = self._feat(pred.clamp(0, 1))
        f_t = self._feat(target.clamp(0, 1))
        loss = 0
        for a, b in zip(f_p, f_t):
            loss = loss + self.crit(a, b, reduction=self.reduction)
        return self.w * loss


class LowLightLoss(nn.Module):
    def __init__(self, loss_weights: Dict, device):
        super().__init__()
        w = lambda k, d: loss_weights.get(k, d)
        self.charb = charbonnier_loss
        self.w_charb = w("charbonnier", 1.0)

        self.chroma = ChromaLoss(
            loss_weight=w("chroma", 1.0),
            reduction=loss_weights.get("chroma_reduction", "mean"),
            criterion=loss_weights.get("chroma_criterion", "l1"),
        ).to(device)

        self.edge = EdgeLoss(
            loss_weight=w("edge", 0.5),
            reduction=loss_weights.get("edge_reduction", "mean"),
            criterion=loss_weights.get("edge_criterion", "l1"),
        ).to(device)

        self.hfreq = HighFreqLoss(
            loss_weight=w("frequency", 0.5),
            reduction=loss_weights.get("frequency_reduction", "mean"),
            criterion=loss_weights.get("frequency_criterion", "l1"),
        ).to(device)

        self.vgg = VGGLoss(
            loss_weight=w("perceptual", 0.2),
            reduction=loss_weights.get("perceptual_reduction", "mean"),
            criterion=loss_weights.get("perceptual_criterion", "l1"),
        ).to(device)

        self.device = device

    def forward(self, pred, target):
        pred = pred.clamp(0, 1)
        target = target.clamp(0, 1)

        Lc = self.w_charb * self.charb(pred, target, reduction="mean")
        Lchroma = self.chroma(pred, target)
        Le = self.edge(pred, target)
        Lhf = self.hfreq(pred, target)
        Lp = self.vgg(pred, target)

        total = Lc + Lchroma + Le + Lhf + Lp

        return {
            "total": total,
            "charbonnier": Lc.detach(),
            "chroma": Lchroma.detach(),
            "edge": Le.detach(),
            "frequency": Lhf.detach(),
            "perceptual": Lp.detach(),
        }
