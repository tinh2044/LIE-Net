import torch
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips

lpips_model = lpips.LPIPS(net="vgg")


def calculate_psnr(img1, img2, max_val=2.0):
    # Denormalize images from [-1, 1] to [0, 1] for proper PSNR calculation
    img1_denorm = (img1 + 1) / 2
    img2_denorm = (img2 + 1) / 2
    mse = F.mse_loss(img1_denorm, img2_denorm)
    if mse == 0:
        return float("inf")
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2):
    # Denormalize images from [-1, 1] to [0, 1] for proper SSIM calculation
    img1_denorm = (img1 + 1) / 2
    img2_denorm = (img2 + 1) / 2
    return ssim(img1_denorm, img2_denorm, data_range=1.0, size_average=True).item()


def calculate_lpips(img1, img2, device="cuda"):
    global lpips_model
    lpips_model = lpips_model.to(device)
    with torch.no_grad():
        return lpips_model(img1, img2).mean().item()


def compute_metrics(img1, img2, device="cuda"):
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    lpips = calculate_lpips(img1, img2, device)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
    }
