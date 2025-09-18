# noirnet_asp.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        # split channel dimension in half and multiply
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


def conv_dw(in_c, out_c, k=3, stride=1, dilation=1):
    pad = (k - 1) // 2 * dilation
    return nn.Sequential(
        nn.Conv2d(
            in_c,
            in_c,
            kernel_size=k,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=in_c,
            bias=True,
        ),
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=True),
    )


class ASPModule(nn.Module):
    def __init__(self, channels, patch_size=8, K=16, mlp_exp=4):
        super().__init__()
        self.C = channels
        self.p = patch_size
        self.B = patch_size * patch_size
        self.K = K

        # encoder (dictionary projection): B -> K (no bias)
        self.enc = nn.Linear(self.B, self.K, bias=False)
        # small coefficient MLP (shared across channels and spatial dims)
        self.mlp = nn.Sequential(
            nn.Linear(self.K, self.K * mlp_exp, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.K * mlp_exp, self.K, bias=True),
        )
        # decoder D: K -> B (no bias), reconstruct patch coefficients
        self.dec = nn.Linear(self.K, self.B, bias=False)

        # small learnable amplitude scale per-channel (start small)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # scalar per module (level)
        # follow with pointwise conv to mix channels after inverse fft
        self.post_pw = nn.Conv2d(channels, channels, 1, bias=True)

    def forward(self, x):
        """
        x: (B, C, H, W) float
        returns: same shape
        """
        B, C, H, W = x.shape
        assert C == self.C, f"ASPModule C mismatch ({C} vs {self.C})"
        p = self.p
        # compute FFT per sample/channel
        Xf = torch.fft.fft2(x)  # complex tensor (B, C, H, W)
        mag = torch.abs(Xf)  # (B, C, H, W)
        pha = torch.angle(Xf)  # (B, C, H, W)

        # extract low-frequency patch from amplitude (we use the top-left p x p block,
        # where DC is at index 0 in PyTorch's FFT)
        patch = mag[:, :, :p, :p].contiguous()  # (B, C, p, p)
        patch_flat = patch.view(B * C, -1)  # (B*C, B)

        # encode to coefficients
        alpha = self.enc(patch_flat)  # (B*C, K)
        # MLP correction
        alpha_corr = self.mlp(alpha)  # (B*C, K)
        # decode
        patch_recon = self.dec(alpha_corr)  # (B*C, B)
        patch_recon = patch_recon.view(B, C, p, p)  # (B, C, p, p)

        # delta amplitude patch (recon - orig)
        delta_patch = (patch_recon - patch).to(x.dtype)  # (B, C, p, p)

        # build delta amplitude full map (zeros except in low-freq region)
        delta_map = torch.zeros_like(
            mag, device=x.device, dtype=x.dtype
        )  # (B, C, H, W)
        delta_map[:, :, :p, :p] = delta_patch

        # apply correction to magnitude
        mag_new = mag + self.gamma * delta_map  # (B, C, H, W)

        # rebuild complex spectrum and inverse FFT
        real = mag_new * torch.cos(pha)
        imag = mag_new * torch.sin(pha)
        complex_new = torch.complex(real, imag)  # complex tensor (B, C, H, W)
        x_rec = torch.fft.ifft2(
            complex_new
        ).real  # back to real spatial domain (B, C, H, W)

        # light channel mixing
        x_rec = self.post_pw(x_rec)

        # we return multiplicative correction to preserve residual behavior:
        # apply sigmoid gating to keep scale stable and then multiply with original features
        gate = torch.sigmoid(x_rec)
        out = x * (1.0 + gate)  # encourage the ASP to modulate original features

        return out


class SpatialMixer(nn.Module):
    def __init__(self, channels, dw_kernel=3, r=8):
        super().__init__()
        self.channels = channels
        pad = (dw_kernel - 1) // 2
        # depthwise conv (groups=channels)
        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=dw_kernel,
            padding=pad,
            groups=channels,
            bias=True,
        )
        # pointwise conv
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # SCA (reduction r)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.dw(x)
        y = self.pw(y)
        att = self.sca(y)
        return y * att


class EBlock(nn.Module):
    """
    EBlock: SpatialMixer + ASP fused via 1x1 conv and residuals
    """

    def __init__(self, channels, patch_size=8, asp_K=16):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.spatial = SpatialMixer(channels)
        self.asp = ASPModule(channels, patch_size=patch_size, K=asp_K)
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # small gating parameters for residuals
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # normalization
        z = self.norm(x)
        # spatial path
        sp = self.spatial(z)
        # frequency path (works on original z)
        fr = self.asp(z)
        # fuse and residual
        fused = self.fuse(sp + fr)
        x1 = x + self.beta * fused
        # second tiny gated FFN (pointwise)
        y = F.relu(self.fuse(x1))
        y = self.fuse(y)
        out = x1 + self.gamma * y
        return out


class DBlock(nn.Module):
    def __init__(self, channels, dilations=[1, 4, 9], r=8):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.channels = channels
        # three dilated depthwise conv branches
        self.branches = nn.ModuleList()
        for d in dilations:
            pad = d
            self.branches.append(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=pad,
                    dilation=d,
                    groups=channels,
                    bias=True,
                )
            )
        # pointwise mixing
        self.pw = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # SCA
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // r, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // r, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # gated-FFN (lightweight)
        self.ffn_a = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.ffn_b = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.ffn_out = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        # gated params
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        y = self.norm(x)
        # branches
        bsum = 0
        for br in self.branches:
            bsum = bsum + br(y)
        bsum = F.gelu(bsum)
        mixed = self.pw(bsum)
        att = self.sca(mixed)
        mixed = mixed * att
        x1 = x + self.beta * mixed
        # gated FFN
        a = self.ffn_a(x1)
        b = torch.sigmoid(self.ffn_b(x1))
        f = a * b
        f = self.ffn_out(f)
        out = x1 + self.gamma * f
        return out


class NoirNetASP(nn.Module):
    def __init__(
        self,
        in_ch=3,
        c=[16, 32, 64, 128],
        enc_blocks=(2, 2, 3),
        dec_blocks=(2, 2, 3),
        asp_patch=8,
        asp_K=16,
    ):
        super().__init__()
        # Stem
        self.stem = nn.Conv2d(in_ch, c[0], kernel_size=3, padding=1, bias=True)

        # Encoder stacks
        self.enc0 = nn.Sequential(
            *[
                EBlock(c[0], patch_size=asp_patch, asp_K=asp_K)
                for _ in range(enc_blocks[0])
            ]
        )
        self.down0 = nn.Conv2d(
            c[0], c[1], kernel_size=3, stride=2, padding=1, bias=True
        )

        self.enc1 = nn.Sequential(
            *[
                EBlock(c[1], patch_size=asp_patch, asp_K=asp_K)
                for _ in range(enc_blocks[1])
            ]
        )
        self.down1 = nn.Conv2d(
            c[1], c[2], kernel_size=3, stride=2, padding=1, bias=True
        )

        self.enc2 = nn.Sequential(
            *[
                EBlock(c[2], patch_size=asp_patch, asp_K=asp_K)
                for _ in range(enc_blocks[2])
            ]
        )
        self.down2 = nn.Conv2d(
            c[2], c[3], kernel_size=3, stride=2, padding=1, bias=True
        )
        self.enc3 = nn.Sequential(
            *[
                EBlock(c[3], patch_size=asp_patch, asp_K=asp_K)
                for _ in range(enc_blocks[2])
            ]
        )

        # Bottleneck readout: low-res image (1x1 conv to RGB)
        self.readout = nn.Conv2d(c[3], in_ch, kernel_size=1, bias=True)

        # Decoder (upsample with PixelShuffle)
        self.up3_proj = nn.Conv2d(
            c[3], 4 * c[2], kernel_size=1, bias=True
        )  # for PixelShuffle r=2 -> c2
        self.dec2 = nn.Sequential(*[DBlock(c[2]) for _ in range(dec_blocks[0])])
        self.up2_proj = nn.Conv2d(
            c[2], 4 * c[1], kernel_size=1, bias=True
        )  # for PixelShuffle r=2 -> c1
        self.dec1 = nn.Sequential(*[DBlock(c[1]) for _ in range(dec_blocks[0])])

        self.up1_proj = nn.Conv2d(
            c[1], 4 * c[0], kernel_size=1, bias=True
        )  # for PixelShuffle r=2 -> c0
        self.dec0 = nn.Sequential(*[DBlock(c[0]) for _ in range(dec_blocks[1])])

        # final refinement blocks at full res
        self.refine = nn.Sequential(*[DBlock(c[0]) for _ in range(dec_blocks[2])])

        # final RGB conv
        self.final = nn.Conv2d(c[0], in_ch, kernel_size=3, padding=1, bias=True)

        # padder size ensures divisible by 4 (2^levels)
        self.padder_size = 8

    def forward(self, inp: torch.Tensor, side_loss: bool = False):
        """
        Input:
           inp: (B, 3, H, W)
        Return:
           out: (B, 3, H, W)
           if side_loss=True also returns (lowres_readout, out)
        """
        B, C, H, W = inp.shape
        x = self._pad_to_divisible(inp)

        # Stem
        x0 = self.stem(x)  # (B, c0, H, W)
        # Encoder level 0
        e0 = self.enc0(x0)
        d0 = self.down0(e0)  # (B, c1, H/2, W/2)
        # Encoder level 1
        e1 = self.enc1(d0)
        d1 = self.down1(e1)  # (B, c2, H/4, W/4)
        # Encoder level 2
        e2 = self.enc2(d1)  # (B, c2, H/4, W/4)
        # Encoder level 3
        d2 = self.down2(e2)  # (B, c3, H/8, W/8)
        e3 = self.enc3(d2)  # (B, c3, H/8, W/8)

        # Bottleneck readout (coarse low-res RGB) computed only if side_loss is requested

        # Decoder up 3->2
        u3 = self.up3_proj(e3)  # (B, 4*c2, H/8, W/8)
        u3 = nn.PixelShuffle(2)(u3)  # (B, c2, H/4, W/4)
        u3 = u3 + e2  # skip add
        d2 = self.dec2(u3)

        # Decoder up 2->1
        u2 = self.up2_proj(d2)  # (B, 4*c1, H/4, W/4)
        u2 = nn.PixelShuffle(2)(u2)  # (B, c1, H/2, W/2)
        u2 = u2 + e1  # skip add
        d1 = self.dec1(u2)

        # Decoder up 1->0
        u1 = self.up1_proj(d1)  # (B, 4*c0, H/2, W/2)
        u1 = nn.PixelShuffle(2)(u1)  # (B, c0, H, W)
        u1 = u1 + e0
        d0 = self.dec0(u1)

        # refinement
        r = self.refine(d0)

        out = self.final(r)
        # residual to input
        out = out + inp
        # unpad
        out = self._unpad_to_orig(out, H, W)

        if side_loss:
            # also return low-resolution readout (up-sample to original size if needed)
            lowres = self.readout(e3)  # (B, 3, H/8, W/8)
            lowres_ups = F.interpolate(
                lowres, size=(H, W), mode="bilinear", align_corners=False
            )
            return lowres_ups, out
        else:
            return out

    def _pad_to_divisible(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        mod_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_w = (self.padder_size - w % self.padder_size) % self.padder_size
        if mod_h == 0 and mod_w == 0:
            return x
        return F.pad(x, (0, mod_w, 0, mod_h), mode="reflect")

    def _unpad_to_orig(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return x[:, :, :H, :W]


if __name__ == "__main__":
    # small smoke test to verify forward shapes
    device = "cpu"
    model = NoirNetASP().to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,f}M")
    inp = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        out = model(inp)
    print("Input shape:", inp.shape, "Output shape:", out.shape)
