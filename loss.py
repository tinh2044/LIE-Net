import functools
from typing import Dict
import torch
from torch import nn as nn
from torchvision.models import vgg as vgg

from torch import autograd as autograd
from torch.nn import functional as F
import torchvision
import pytorch_msssim

_reduction_modes = ["none", "mean", "sum"]


def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction="mean"):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == "sum":
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == "mean":
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction="mean", **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction="none")


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction="none")


@weighted_loss
def log_mse_loss(pred, target):
    return torch.log(F.mse_loss(pred, target, reduction="none"))


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target) ** 2 + eps)


@weighted_loss
def psnr_loss(pred, target):
    mseloss = F.mse_loss(pred, target, reduction="none").mean((1, 2, 3))
    psnr_val = 10 * torch.log10(1 / mseloss).mean().item()
    return psnr_val


class PSNRLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(PSNRLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * psnr_loss(pred, target, weight) * -1.0


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(L1Loss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction
        )


class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction="mean"):
        super(MSELoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """

        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction
        )


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction="mean", eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(
            pred, target, weight, eps=self.eps, reduction=self.reduction
        )


class VGGLoss(nn.Module):
    """Combined VGG19 feature extractor and perceptual loss"""

    def __init__(self, loss_weight=1.0, criterion="l1", reduction="mean"):
        super(VGGLoss, self).__init__()

        # Initialize VGG19 feature extractor
        vgg_pretrained_features = torchvision.models.vgg19(
            weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1
        ).features

        # Create feature slices
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        del vgg_pretrained_features

        # Loss configuration
        if reduction not in ["none", "mean", "sum"]:
            raise ValueError(
                f"Unsupported reduction mode: {reduction}. "
                f"Supported ones are: {_reduction_modes}"
            )

        if criterion == "l1":
            self.criterion = nn.L1Loss(reduction=reduction)
        elif criterion == "l2":
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError("Unsupported criterion loss")

        # Feature weights for multi-scale loss
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.loss_weight = loss_weight

        # Move to GPU if available
        if torch.cuda.is_available():
            self.cuda()

    def extract_features(self, x):
        """Extract VGG19 features from input tensor"""
        h_relu1 = self.slice1(x)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

    def forward(self, x, y):
        """Compute perceptual loss between x and y"""
        # Clamp to [0,1] then normalize to ImageNet statistics expected by VGG
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=x.dtype).view(
            1, 3, 1, 1
        )
        x = (x.clamp(0.0, 1.0) - mean) / std
        y = (y.clamp(0.0, 1.0) - mean) / std

        x_features = self.extract_features(x)
        y_features = self.extract_features(y)

        loss = 0
        for i in range(len(x_features)):
            loss += self.weights[i] * self.criterion(
                x_features[i], y_features[i].detach()
            )

        return self.loss_weight * loss


def SSIM_loss(pred_img, real_img, data_range):
    loss = pytorch_msssim.ssim(pred_img, real_img, data_range=data_range)
    return loss


class SSIM(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.0):
        super(SSIM, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * SSIM_loss(pred, target, self.data_range)


class SSIMloss(nn.Module):
    def __init__(self, loss_weight=1.0, data_range=1.0):
        super(SSIMloss, self).__init__()
        self.loss_weight = loss_weight
        self.data_range = data_range

    def forward(self, pred, target, **kwargs):
        return self.loss_weight * (1 - SSIM_loss(pred, target, self.data_range))


class Gradient_Loss(nn.Module):
    def __init__(self, weight):
        super(Gradient_Loss, self).__init__()

        kernel_g = [
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)
        self.weight = weight

    def forward(self, x, xx):
        grad = 0
        y = x
        yy = xx
        gradient_x = F.conv2d(y, self.weight_g.to(y.device), groups=3)
        gradient_xx = F.conv2d(yy, self.weight_g.to(yy.device), groups=3)
        l = nn.L1Loss()
        a = l(gradient_x, gradient_xx)
        grad = grad + a
        return self.weight * grad


class LowLightLoss(nn.Module):
    """Combined loss for Low Light Image Enhancement"""

    def __init__(self, loss_weights: Dict, **kwargs):
        super(LowLightLoss, self).__init__()
        self.weights = loss_weights

        self.charbonnier_loss = CharbonnierLoss(
            loss_weight=loss_weights.get("charbonnier", 1),
            reduction=loss_weights.get("charbonnier_reduction", "mean"),
        )
        # self.perceptual_loss = VGGLoss(
        #     criterion=loss_weights.get("perceptual_criterion", "l2"),
        #     reduction=loss_weights.get("perceptual_reduction", "mean"),
        #     loss_weight=loss_weights.get("perceptual", 1),
        # )
        # Add SSIM loss (1-SSIM)
        self.ssim_loss = SSIMloss(
            loss_weight=loss_weights.get("ssim", 0.0),
            data_range=loss_weights.get("ssim_range", 1.0),
        )
        self.grad = Gradient_Loss(weight=loss_weights.get("grad", 0.0))

    def forward(self, pred, target):
        charbonnier_loss = self.charbonnier_loss(pred, target)
        # perceptual_loss = self.perceptual_loss(pred, target)
        ssim_loss = self.ssim_loss(pred, target)
        grad_loss = self.grad(pred, target)

        total_loss = charbonnier_loss + grad_loss + ssim_loss
        return {
            "total": total_loss,
            "charbonnier": charbonnier_loss,
            # "perceptual": perceptual_loss,
            "grad": grad_loss,
            "ssim": ssim_loss,
        }
