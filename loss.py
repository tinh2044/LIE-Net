import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


class GradientLoss(nn.Module):
    """Gradient Loss for edge preservation"""

    def __init__(self):
        super(GradientLoss, self).__init__()
        kernel_g = [
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        ]
        kernel_g = torch.FloatTensor(kernel_g).unsqueeze(0).permute(1, 0, 2, 3)
        self.weight_g = nn.Parameter(data=kernel_g, requires_grad=False)

    def forward(self, x, y):
        gradient_x = F.conv2d(x, self.weight_g, groups=3)
        gradient_y = F.conv2d(y, self.weight_g, groups=3)
        loss = F.l1_loss(gradient_x, gradient_y)
        return loss


class VGGPerceptualLoss(nn.Module):
    """VGG Perceptual Loss using torchvision weights API (no deprecation warnings)."""

    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        from torchvision.models import vgg16, VGG16_Weights

        # Initialize VGG16 with official weights (replaces deprecated 'pretrained=True')
        vgg_features = vgg16(weights=VGG16_Weights.DEFAULT).features.eval()

        # Slice feature blocks once from a single model to avoid repeated construction
        blocks = [
            vgg_features[:4].eval(),
            vgg_features[4:9].eval(),
            vgg_features[9:16].eval(),
            vgg_features[16:23].eval(),
        ]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, input, target, feature_layers=[0, 1, 2, 3]):
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(
                input, mode="bilinear", size=(224, 224), align_corners=False
            )
            target = self.transform(
                target, mode="bilinear", size=(224, 224), align_corners=False
            )
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
        return loss


class WeightedLoss(nn.Module):
    """Weighted Loss for combining multiple losses"""

    def __init__(self, weights):
        super(WeightedLoss, self).__init__()
        self.weights = weights

    def forward(self, losses_dict):
        total_loss = 0
        for loss_name, loss_value in losses_dict.items():
            if loss_name in self.weights:
                total_loss += self.weights[loss_name] * loss_value
        return total_loss


class IlluminationSmoothnessLoss(nn.Module):
    def __init__(self):
        super(IlluminationSmoothnessLoss, self).__init__()

    def forward(self, illumination_map):
        grad_x = torch.abs(
            illumination_map[:, :, :, :-1] - illumination_map[:, :, :, 1:]
        )
        grad_y = torch.abs(
            illumination_map[:, :, :-1, :] - illumination_map[:, :, 1:, :]
        )
        return torch.mean(grad_x) + torch.mean(grad_y)


class NoiseAwareReconstructionLoss(nn.Module):
    def __init__(self):
        super(NoiseAwareReconstructionLoss, self).__init__()

    def forward(self, pred, target, noise_map):
        # Reduce noise_map to 3 channels to match pred/target
        if noise_map.shape[1] != 3:
            # Use average pooling to reduce channels
            noise_map = torch.mean(noise_map, dim=1, keepdim=True)
            noise_map = noise_map.repeat(1, 3, 1, 1)

        weight = 1 - noise_map
        loss = torch.abs(pred - target) * weight
        return torch.mean(loss)


class ExposureControlLoss(nn.Module):
    def __init__(self, target_mean=0.6):
        super(ExposureControlLoss, self).__init__()
        self.target_mean = target_mean

    def forward(self, img):
        mean_val = torch.mean(img)
        loss = torch.abs(mean_val - self.target_mean)
        return loss


class LowLightLoss(nn.Module):
    """Combined loss for Low Light Image Enhancement"""

    def __init__(self, loss_weights):
        super(LowLightLoss, self).__init__()
        self.weights = loss_weights

        self.charbonnier_loss = CharbonnierLoss()
        self.perceptual_loss = VGGPerceptualLoss()
        self.smoothness_loss = IlluminationSmoothnessLoss()
        self.exposure_loss = ExposureControlLoss()
        self.noise_aware_loss = NoiseAwareReconstructionLoss()

    def forward(self, pred, target=None, illumination_map=None, noise_map=None):
        if target is None:
            target = pred
        losses = {}

        losses["charbonnier"] = self.charbonnier_loss(pred, target)
        losses["perceptual"] = self.perceptual_loss(pred, target)

        if illumination_map is not None:
            losses["smoothness"] = self.smoothness_loss(illumination_map)

        losses["exposure"] = self.exposure_loss(pred)

        if noise_map is not None:
            losses["noise_aware"] = self.noise_aware_loss(pred, target, noise_map)

        total_loss = 0
        for loss_name, loss_value in losses.items():
            if loss_name in self.weights:
                total_loss += self.weights[loss_name] * loss_value
        losses["total"] = total_loss
        return losses
