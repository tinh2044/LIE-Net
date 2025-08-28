import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss import LowLightLoss


class IlluminationExtractionModule(nn.Module):
    def __init__(self, channels):
        super(IlluminationExtractionModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        # Use ReLU6 for sharper activation instead of sigmoid
        self.activation = nn.ReLU6()
        # Add batch norm for better training stability
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Apply convolution and batch norm
        out = self.bn(self.conv(x))
        # Use ReLU6 for sharper, more defined illumination maps
        illumination_map = self.activation(out)
        return illumination_map


class NoiseEstimationModule(nn.Module):
    def __init__(self, channels):
        super(NoiseEstimationModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        # Use ReLU6 for sharper activation instead of sigmoid
        self.activation = nn.ReLU6()
        # Add batch norm for better training stability
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        # Apply convolution and batch norm
        out = self.bn(self.conv(x))
        # Use ReLU6 for sharper, more defined noise maps
        out = self.activation(out)
        return out


class IlluminationAwareGate(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(IlluminationAwareGate, self).__init__()
        self.iem = IlluminationExtractionModule(channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        kernel_s = self.kernel_size()
        self.conv_dw = nn.Conv1d(
            1, 1, kernel_size=kernel_s, padding=(kernel_s - 1) // 2, bias=False
        )
        self.act = nn.GELU()
        self.conv_pw = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        # Use LeakyReLU for sharper activation and better gradient flow
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        # Add edge-preserving mechanism
        self.edge_conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        return k if k % 2 else k + 1

    def forward(self, x):
        b, c, h, w = x.shape

        illumination_info = self.iem(x)

        y_avg = self.avg_pool(illumination_info)
        y_max = self.max_pool(illumination_info)
        y = 0.5 * (y_avg + y_max)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv_dw(y)
        y = self.act(y)
        y = self.conv_pw(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        # Use LeakyReLU for sharper activation
        y = self.activation(y)

        # Add edge-preserving mechanism
        edge_features = self.edge_conv(x)
        edge_weight = torch.sigmoid(edge_features.mean(dim=1, keepdim=True))

        # Combine illumination-aware gating with edge preservation
        gate_weight = 0.5 + 0.5 * y.expand_as(x)
        final_weight = gate_weight * (1.0 + 0.1 * edge_weight)

        out = x * final_weight
        return out


class IlluminationGuidedAttention(nn.Module):
    def __init__(self, channels, num_heads, window_size: int = 8):
        super(IlluminationGuidedAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1, 1, 1))
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        self.window_size = window_size

        self.iem = IlluminationExtractionModule(channels)
        self.nem = NoiseEstimationModule(channels)
        self.iem_dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )
        self.nem_dw = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, groups=channels, bias=False
        )

        # Linear projections for Q, K, V
        self.q_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape

        ws = self.window_size
        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        hp, wp = x.shape[-2:]

        # Illumination and noise priors with light locality smoothing
        iem_out = self.iem(x)
        illumination_map = self.iem_dw(iem_out)  # [B,C,Hp,Wp]

        nem_out = self.nem(x)
        noise_map = self.nem_dw(nem_out)  # [B,C,Hp,Wp]

        # Scalar priors per spatial position for modulation
        illum_scalar = illumination_map.mean(dim=1, keepdim=True)  # [B,1,Hp,Wp]
        noise_scalar = noise_map.mean(dim=1, keepdim=True)  # [B,1,Hp,Wp]

        # Projections
        q = self.q_conv(illumination_map)
        k = self.k_conv(noise_map)
        v = self.v_conv(x)

        # Reshape into non-overlapping windows
        heads = self.num_heads
        d = self.head_dim
        nh = hp // ws
        nw = wp // ws

        def to_windows(t):
            t = t.view(b, heads, d, hp, wp).permute(0, 1, 3, 4, 2)  # [B,H,Hp,Wp,D]
            t = t.view(b, heads, nh, ws, nw, ws, d).permute(0, 1, 2, 4, 3, 5, 6)
            t = t.reshape(b, heads, nh, nw, ws * ws, d)  # [B,H,Nh,Nw,S,D]
            return t

        def scalar_to_windows(t):
            t = t.view(b, 1, hp, wp).unsqueeze(1)  # [B,1,Hp,Wp] -> [B,1,1,Hp,Wp]
            t = t.view(b, 1, nh, ws, nw, ws).permute(0, 1, 2, 4, 3, 5)
            t = t.reshape(b, 1, nh, nw, ws * ws, 1)  # [B,1,Nh,Nw,S,1]
            return t

        qw = to_windows(q)
        kw = to_windows(k)
        vw = to_windows(v)

        illum_w = scalar_to_windows(illum_scalar)
        noise_w = scalar_to_windows(noise_scalar)

        # Normalize and modulate
        qw = F.normalize(qw, dim=-1)
        kw = F.normalize(kw, dim=-1)

        qw = qw * (1.0 + illum_w)
        kw = kw * (1.0 - noise_w).clamp(0.0, 1.0)

        # Windowed attention: [B,H,Nh,Nw,S,S]
        scale = d**-0.5
        attn = torch.matmul(qw, kw.transpose(-2, -1)) * scale
        attn = attn * self.temperature
        attn = torch.softmax(attn, dim=-1)
        outw = torch.matmul(attn, vw)  # [B,H,Nh,Nw,S,D]

        # Merge windows back
        outw = outw.view(b, heads, nh, nw, ws, ws, d).permute(0, 1, 6, 2, 4, 3, 5)
        out = outw.reshape(b, heads * d, nh * ws, nw * ws)  # [B,C,Hp,Wp]

        if pad_h or pad_w:
            out = out[:, :, :h, :w]

        out = self.project_out(out)
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(
            channels, hidden_channels * 2, kernel_size=1, bias=False
        )
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(
            hidden_channels, channels, kernel_size=1, bias=False
        )

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        out = self.project_out(F.gelu(x1) * x2)
        return out


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        out = self.body(x)
        return out


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        out = self.body(x)
        return out


class RestorationEnhancementAttention(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(RestorationEnhancementAttention, self).__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = IlluminationGuidedAttention(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm1 = (
            self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        x = x + self.attn(x_norm1)

        x_norm2 = (
            self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous())
            .transpose(-2, -1)
            .contiguous()
            .reshape(b, c, h, w)
        )
        x = x + self.ffn(x_norm2)
        return x


class EdgeEnhancementModule(nn.Module):
    """Module to enhance edges and reduce blurry effects"""

    def __init__(self, channels):
        super(EdgeEnhancementModule, self).__init__()
        self.edge_conv1 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.edge_conv2 = nn.Conv2d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        # Extract edge features
        edge1 = self.activation(self.bn1(self.edge_conv1(x)))
        edge2 = self.activation(self.bn2(self.edge_conv2(edge1)))

        # Enhance edges by adding edge information back
        enhanced = x + 0.1 * edge2
        return enhanced


class LIENet(nn.Module):
    def __init__(
        self,
        num_blocks=[2, 2, 2, 2],
        num_heads=[1, 2, 4, 8],
        channels=[16, 32, 64, 128],
        num_refinement=2,
        expansion_factor=2.66,
        loss={},
        in_channels=3,
        **kwargs,
    ):
        super(LIENet, self).__init__()
        self.embed_conv = nn.Conv2d(
            in_channels, channels[0], kernel_size=3, padding=1, bias=False
        )

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        RestorationEnhancementAttention(
                            num_ch, num_ah, expansion_factor
                        )
                        for _ in range(num_tb)
                    ]
                )
                for num_tb, num_ah, num_ch in zip(num_blocks, num_heads, channels)
            ]
        )

        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList(
            [UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]]
        )

        # IAG for skip connections
        self.iags = nn.ModuleList([IlluminationAwareGate(ch) for ch in channels[:-1]])

        self.reduces = nn.ModuleList(
            [
                nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                for i in reversed(
                    range(1, len(channels))
                )  # Adjust index for compatibility
            ]
        )

        self.decoders = nn.ModuleList(
            [
                nn.Sequential(
                    *[
                        RestorationEnhancementAttention(ch, nh, expansion_factor)
                        for _ in range(nb)
                    ]
                )
                for nb, nh, ch in zip(
                    reversed(num_blocks[:-1]),
                    reversed(num_heads[:-1]),
                    reversed(channels[:-1]),
                )
            ]
        )

        self.refinement = nn.Sequential(
            *[
                RestorationEnhancementAttention(
                    channels[0], num_heads[0], expansion_factor
                )
                for _ in range(num_refinement)
            ]
        )

        # Add edge enhancement module before output
        self.edge_enhancement = EdgeEnhancementModule(channels[0])

        self.out_conv = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)

        # Encoder
        encoder_features = []
        for i in range(len(self.encoders)):
            fo = self.encoders[i](fo)
            encoder_features.append(fo)
            if i < len(self.downs):
                fo = self.downs[i](fo)

        # Decoder
        # fo is now the output of the last encoder
        for i in range(len(self.decoders)):
            fo = self.ups[i](fo)
            # Get corresponding feature from encoder and apply OIB
            enc_feat = encoder_features[-(i + 2)]
            skip_feature = self.iags[-(i + 1)](enc_feat)
            fo = torch.cat([fo, skip_feature], dim=1)
            fo = self.reduces[i](fo)
            fo = self.decoders[i](fo)

        fr = self.refinement(fo)
        # Apply edge enhancement before final output
        fr = self.edge_enhancement(fr)
        output = self.out_conv(fr)

        return output


if __name__ == "__main__":
    model = LIENet(in_channels=3)
    input = torch.randn(1, 3, 256, 256)
    output = model(input)

    print(f"Output shape: {output.shape}")
