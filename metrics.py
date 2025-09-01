import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim, ms_ssim
import lpips
import warnings
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Initialize LPIPS once; suppress pickle FutureWarning from inside the library load
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="You are using `torch.load` with `weights_only=False`",
        category=FutureWarning,
    )
    _lpips_model = lpips.LPIPS(net="vgg")


def calculate_psnr(img1, img2, max_val=1.0):
    img1 = img1.data.cpu().numpy().astype(np.float32)
    img2 = img2.data.cpu().numpy().astype(np.float32)
    value = 0
    for i in range(img1.shape[0]):
        psnr = compare_psnr(img1[i], img2[i], data_range=max_val)
        value += psnr

    return value / img1.shape[0]


def calculate_ssim(img1, img2):
    # SSIM expects (B, C, H, W)
    return ssim(img1, img2, data_range=1.0)


def calculate_ms_ssim(img1, img2):
    # MS-SSIM expects (B, C, H, W)
    return ms_ssim(img1, img2, data_range=1.0)


def calculate_lpips(img1, img2, device="cuda"):
    model = _lpips_model.to(device)
    with torch.no_grad():
        value = 0
        for i in range(img1.shape[0]):
            value += model(img1[i], img2[i])
        return value / img1.shape[0]


def compute_metrics(img1, img2, device="cuda"):
    psnr = calculate_psnr(img1, img2)
    ssim = calculate_ssim(img1, img2)
    ms_ssim = calculate_ms_ssim(img1, img2)
    lpips = calculate_lpips(img1, img2, device)
    return {
        "psnr": psnr,
        "ssim": ssim,
        "ms_ssim": ms_ssim,
        "lpips": lpips,
    }
