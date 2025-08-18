import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import sys
import os
import datetime
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from loss import LowLightLoss


class _DebugWriter:
    """Minimal debug writer that appends human-readable traces to a text file.

    Safe to call from multiple forwards; creates parent directory automatically.
    """

    def __init__(self, log_path: str):
        self.log_path = log_path
        self._header_written = False
        self._base_shape = None  # (B, C, H, W)
        self._tensor_sources = {}
        parent = os.path.dirname(os.path.abspath(log_path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

    def write(self, text: str) -> None:
        mode = "a"
        with open(self.log_path, mode, encoding="utf-8") as f:
            if not self._header_written:
                f.write("\n=== LIE-Net Debug Trace ===\n")
                f.write(f"Started: {datetime.datetime.now().isoformat()}\n")
                f.write("==========================\n")
                self._header_written = True
            # If base shape is known, augment any tensor shape tuples with relative notation
            line = text.rstrip("\n")
            if self._base_shape is not None:
                line = self._augment_shapes_with_relative(line)
            f.write(line + "\n")

    def set_base_shape(self, shape: tuple) -> None:
        # Expect (B, C, H, W)
        if len(shape) == 4:
            self._base_shape = tuple(int(x) for x in shape)

    def tag(self, tensor: torch.Tensor, name: str) -> None:
        try:
            self._tensor_sources[id(tensor)] = name
        except Exception:
            pass

    def source(self, tensor: torch.Tensor) -> str:
        return self._tensor_sources.get(id(tensor), None)

    def _augment_shapes_with_relative(self, line: str) -> str:
        # Replace every tuple of 4 integers in parentheses with appended relative form
        tuple_pattern = re.compile(r"\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)")

        def repl(m):
            b, c, h, w = [int(m.group(i)) for i in range(1, 5)]
            rel = self._format_relative((b, c, h, w))
            return f"({b}, {c}, {h}, {w}) => {rel}"

        return tuple_pattern.sub(repl, line)

    def _format_relative(self, shape: tuple) -> str:
        # Relative to input: show as (B, C, W_rel, H_rel) with W/H derived from input W0/H0
        if self._base_shape is None or len(shape) != 4:
            return str(shape)
        b0, c0, h0, w0 = self._base_shape
        b, c, h, w = shape

        def rel_wh(cur, base, symbol):
            if base <= 0:
                return f"{cur}"
            if cur == base:
                return symbol
            if base % cur == 0:
                return f"{symbol}/{base // cur}"
            if cur % base == 0:
                return f"{symbol}*{cur // base}"
            # Fallback decimal ratio
            ratio = cur / base
            return f"{symbol}*{ratio:.2f}"

        # Channels as absolute numeric value
        c_str = str(c)
        # Note: order as (W, H) per user request
        w_rel = rel_wh(w, w0, "W")
        h_rel = rel_wh(h, h0, "H")
        return f"(B, {c_str}, {w_rel}, {h_rel})"


def _attach_debug(module: nn.Module, writer: _DebugWriter, name: str = None) -> None:
    """Attach debug flags and writer to a module instance dynamically."""
    setattr(module, "debug_enabled", True)
    setattr(module, "_debug", writer)
    if name is not None:
        setattr(module, "debug_name", name)


def _log(module: nn.Module, message: str) -> None:
    """Write a debug message if module has debugging enabled."""
    if getattr(module, "debug_enabled", False) and getattr(module, "_debug", None):
        prefix = getattr(module, "debug_name", module.__class__.__name__)
        module._debug.write(f"[{prefix}] {message}")


class IlluminationExtractionModule(nn.Module):
    """
    Estimate the illumination map from the input features.
    """

    def __init__(self, channels):
        super(IlluminationExtractionModule, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        illumination_map = self.sigmoid(self.conv(x))
        _log(self, f"out L: {tuple(illumination_map.shape)}")
        # reflectance = x / (illumination_map + 1e-6) # Can return R if needed
        return illumination_map


class NoiseEstimationModule(nn.Module):
    """
    Estimate the noise map from the input features.
    """

    def __init__(self, channels):
        super(NoiseEstimationModule, self).__init__()
        # A more complex series of layers can give better results
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.sigmoid(self.conv(x))
        _log(self, f"out N: {tuple(out.shape)}")
        return out


class IlluminationAwareGate(nn.Module):
    """
    Illumination-Aware Gate (IAG)
    Use IEM to create "illumination gate" for skip-connections.
    """

    def __init__(self, channels, b=1, gamma=2):
        super(IlluminationAwareGate, self).__init__()
        self.iem = IlluminationExtractionModule(channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        kernel_s = self.kernel_size()
        # Lightweight gating stack: depthwise 1D conv (adaptive k) + GELU + pointwise 1D conv
        self.conv_dw = nn.Conv1d(
            1, 1, kernel_size=kernel_s, padding=(kernel_s - 1) // 2, bias=False
        )
        self.act = nn.GELU()
        self.conv_pw = nn.Conv1d(1, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels) / self.gamma) + self.b / self.gamma))
        return k if k % 2 else k + 1

    def forward(self, x):
        b, c, h, w = x.shape
        _log(self, f"in: {tuple(x.shape)}")
        _log(
            self,
            f"args: b={b}, c={c}, h={h}, w={w}, channels={self.channels}, b={self.b}, gamma={self.gamma}",
        )

        # Extract illumination map to serve as the basis for attention
        illumination_info = self.iem(x)
        _log(self, f"IEM output: {tuple(illumination_info.shape)}")

        # Fuse average and max pooled descriptors to capture stable and salient cues
        y_avg = self.avg_pool(illumination_info)
        _log(self, f"avg_pool: {tuple(y_avg.shape)}")
        y_max = self.max_pool(illumination_info)
        _log(self, f"max_pool: {tuple(y_max.shape)}")
        y = 0.5 * (y_avg + y_max)
        _log(self, f"fused: {tuple(y.shape)}")

        # Channel attention via 1D conv stack over channels
        y = y.squeeze(-1).transpose(-1, -2)
        _log(self, f"squeeze+transpose: {tuple(y.shape)}")
        y = self.conv_dw(y)
        _log(self, f"conv_dw: {tuple(y.shape)}")
        y = self.act(y)
        _log(self, f"GELU: {tuple(y.shape)}")
        y = self.conv_pw(y)
        _log(self, f"conv_pw: {tuple(y.shape)}")
        y = y.transpose(-1, -2).unsqueeze(-1)
        _log(self, f"transpose+unsqueeze: {tuple(y.shape)}")
        y = self.sigmoid(y)
        _log(self, f"sigmoid: {tuple(y.shape)}")
        out = x * y.expand_as(x)
        _log(self, f"expand_as: {tuple(y.expand_as(x).shape)}")
        _log(self, f"final output: {tuple(out.shape)}")
        return out


class IlluminationGuidedAttention(nn.Module):
    """
    Illumination-Guided Attention Module (IGAM)
    Use Illumination and Noise map as Query and Key.
    """

    def __init__(self, channels, num_heads, window_size: int = 8):
        super(IlluminationGuidedAttention, self).__init__()
        self.num_heads = num_heads
        # Per-head learnable temperature
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1, 1, 1))
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.debug_enabled = True
        self._debug = _DebugWriter("./igam_debug.txt")

        # Modules to extract illumination and noise
        self.iem = IlluminationExtractionModule(channels)
        self.nem = NoiseEstimationModule(channels)
        # Lightweight locality on priors
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
        _log(self, f"in: {tuple(x.shape)}")
        _log(
            self,
            f"args: b={b}, c={c}, h={h}, w={w}, num_heads={self.num_heads}, head_dim={self.head_dim}, window_size={self.window_size}",
        )

        ws = self.window_size
        pad_h = (ws - (h % ws)) % ws
        pad_w = (ws - (w % ws)) % ws
        _log(self, f"padding: pad_h={pad_h}, pad_w={pad_w}")

        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
            _log(self, f"after padding: {tuple(x.shape)}")
        hp, wp = x.shape[-2:]

        # Illumination and noise priors with light locality smoothing
        iem_out = self.iem(x)
        _log(self, f"IEM output: {tuple(iem_out.shape)}")
        illumination_map = self.iem_dw(iem_out)  # [B,C,Hp,Wp]
        _log(self, f"illumination_map: {tuple(illumination_map.shape)}")

        nem_out = self.nem(x)
        _log(self, f"NEM output: {tuple(nem_out.shape)}")
        noise_map = self.nem_dw(nem_out)  # [B,C,Hp,Wp]
        _log(self, f"noise_map: {tuple(noise_map.shape)}")

        # Scalar priors per spatial position for modulation
        illum_scalar = illumination_map.mean(dim=1, keepdim=True)  # [B,1,Hp,Wp]
        _log(self, f"illum_scalar: {tuple(illum_scalar.shape)}")
        noise_scalar = noise_map.mean(dim=1, keepdim=True)  # [B,1,Hp,Wp]
        _log(self, f"noise_scalar: {tuple(noise_scalar.shape)}")

        # Projections
        q = self.q_conv(illumination_map)
        _log(self, f"Q projection: {tuple(q.shape)}")
        k = self.k_conv(noise_map)
        _log(self, f"K projection: {tuple(k.shape)}")
        v = self.v_conv(x)
        _log(self, f"V projection: {tuple(v.shape)}")

        # Reshape into non-overlapping windows
        heads = self.num_heads
        d = self.head_dim
        nh = hp // ws
        nw = wp // ws
        _log(self, f"window params: heads={heads}, d={d}, nh={nh}, nw={nw}")

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
        _log(self, f"Q windows: {tuple(qw.shape)}")
        kw = to_windows(k)
        _log(self, f"K windows: {tuple(kw.shape)}")
        vw = to_windows(v)
        _log(self, f"V windows: {tuple(vw.shape)}")

        illum_w = scalar_to_windows(illum_scalar)
        _log(self, f"illum_windows: {tuple(illum_w.shape)}")
        noise_w = scalar_to_windows(noise_scalar)
        _log(self, f"noise_windows: {tuple(noise_w.shape)}")

        # Normalize and modulate
        qw = F.normalize(qw, dim=-1)
        _log(self, f"Q normalized: {tuple(qw.shape)}")
        kw = F.normalize(kw, dim=-1)
        _log(self, f"K normalized: {tuple(kw.shape)}")

        qw = qw * (1.0 + illum_w)
        _log(self, f"Q modulated: {tuple(qw.shape)}")
        kw = kw * (1.0 - noise_w).clamp(0.0, 1.0)
        _log(self, f"K modulated: {tuple(kw.shape)}")

        # Windowed attention: [B,H,Nh,Nw,S,S]
        scale = d**-0.5
        _log(self, f"attention scale: {scale}")
        attn = torch.matmul(qw, kw.transpose(-2, -1)) * scale
        _log(self, f"attention logits: {tuple(attn.shape)}")
        _log(self, f"temperature shape: {tuple(self.temperature.shape)}")
        attn = attn * self.temperature
        _log(self, f"temperature applied: {tuple(attn.shape)}")
        attn = torch.softmax(attn, dim=-1)
        _log(self, f"attention weights: {tuple(attn.shape)}")
        outw = torch.matmul(attn, vw)  # [B,H,Nh,Nw,S,D]
        _log(self, f"attention output: {tuple(outw.shape)}")

        # Merge windows back
        outw = outw.view(b, heads, nh, nw, ws, ws, d).permute(0, 1, 6, 2, 4, 3, 5)
        _log(self, f"reshaped windows: {tuple(outw.shape)}")
        out = outw.reshape(b, heads * d, nh * ws, nw * ws)  # [B,C,Hp,Wp]
        _log(self, f"merged windows: {tuple(out.shape)}")

        if pad_h or pad_w:
            out = out[:, :, :h, :w]
            _log(self, f"after unpadding: {tuple(out.shape)}")

        out = self.project_out(out)
        _log(self, f"final output: {tuple(out.shape)}")
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
        _log(self, f"in: {tuple(x.shape)}")
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        out = self.project_out(F.gelu(x1) * x2)
        _log(self, f"out: {tuple(out.shape)}")
        return out


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.body(x)
        _log(self, f"out: {tuple(out.shape)}")
        return out


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        _log(self, f"in: {tuple(x.shape)}")
        out = self.body(x)
        _log(self, f"out: {tuple(out.shape)}")
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
        _log(self, f"in: {tuple(x.shape)}")
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
        _log(self, f"out: {tuple(x.shape)}")
        return x


class LIENet(nn.Module):
    def __init__(
        self,
        num_blocks=[2, 3, 3, 4],
        num_heads=[1, 2, 4, 8],
        channels=[16, 32, 64, 128],
        num_refinement=4,
        expansion_factor=2.66,
        loss_weights={},
        in_channels=3,
        # Preprocessing controls
        use_preprocess=True,
        preprocess_gamma=0.8,
        preprocess_histogram_bounds=(0.01, 0.99),
        # Multi-scale controls
        use_multiscale=True,
        multiscale_loss_weight=0.5,
        # Debug controls
        debug_enabled=True,
        debug_log_path="./flow_debug.txt",
        debug_attach_all=True,
        **kwargs,
    ):
        super(LIENet, self).__init__()
        # -------- Preprocessing config --------
        self.use_preprocess = use_preprocess
        self.preprocess_gamma = preprocess_gamma
        self.preprocess_histogram_bounds = preprocess_histogram_bounds
        # -------- Multi-scale config --------
        self.use_multiscale = use_multiscale
        self.multiscale_loss_weight = multiscale_loss_weight
        # -------- Debug config --------
        self.debug_enabled = debug_enabled
        self._debug = _DebugWriter(debug_log_path) if debug_enabled else None
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

        self.out_conv = nn.Conv2d(channels[0], 3, kernel_size=3, padding=1, bias=False)

        # x2 output head for multi-scale supervision
        if self.use_multiscale:
            # Preserve channels while upsampling to x2
            self.up_final = nn.Sequential(
                nn.Conv2d(
                    channels[0], channels[0] * 4, kernel_size=3, padding=1, bias=False
                ),
                nn.PixelShuffle(2),
            )
            self.output_x2 = nn.Conv2d(
                channels[0], 3, kernel_size=3, padding=1, bias=False
            )

        # Add IEM and NEM for loss calculation
        self.iem = IlluminationExtractionModule(channels[0])
        self.nem = NoiseEstimationModule(channels[0])

        self.loss_func = LowLightLoss(loss_weights)

        # Attach deep debug to all modules
        if self.debug_enabled and self._debug is not None and debug_attach_all:
            self._attach_debug_to_all()

    def forward(self, x, target=None):
        if self.debug_enabled:
            self._debug.write("\n-- Forward pass --")
            self._debug.set_base_shape(tuple(x.shape))
            self._debug.write(
                f"input: {tuple(x.shape)} (min={x.min().item():.4f}, max={x.max().item():.4f})"
            )

        # Optional preprocessing: Gamma correction + histogram stretching
        if self.use_preprocess:
            x_in = self._preprocess_input(x)
        else:
            x_in = x

        fo = self.embed_conv(x_in)
        if self.debug_enabled:
            self._debug.tag(fo, "enc_0_stem")
        if self.debug_enabled:
            self._debug.write(f"embed_conv -> fo: {tuple(fo.shape)}")

        # Encoder
        encoder_features = []
        for i in range(len(self.encoders)):
            fo = self.encoders[i](fo)
            if self.debug_enabled:
                self._debug.tag(fo, f"enc_{i}")
                self._debug.write(f"encoder[{i}] -> {tuple(fo.shape)}")
            encoder_features.append(fo)
            if i < len(self.downs):
                fo = self.downs[i](fo)
                if self.debug_enabled:
                    self._debug.tag(fo, f"down_{i}")
                    self._debug.write(f"down[{i}] -> {tuple(fo.shape)}")

        # Decoder
        # fo is now the output of the last encoder
        for i in range(len(self.decoders)):
            fo = self.ups[i](fo)
            if self.debug_enabled:
                self._debug.tag(fo, f"up_{i}")
                self._debug.write(f"up[{i}] -> {tuple(fo.shape)}")
            # Get corresponding feature from encoder and apply OIB
            enc_feat = encoder_features[-(i + 2)]
            skip_feature = self.iags[-(i + 1)](enc_feat)
            if self.debug_enabled:
                self._debug.tag(skip_feature, f"iag_{len(self.iags) - (i + 1)}")
                self._debug.write(
                    f"iag[{len(self.iags) - (i + 1)}] enc_skip: {tuple(enc_feat.shape)} -> skip: {tuple(skip_feature.shape)}"
                )
            fo = torch.cat([fo, skip_feature], dim=1)
            if self.debug_enabled:
                src_left = (
                    self._debug.source(fo[:, : fo.shape[1] - skip_feature.shape[1]])
                    or f"up_{i}"
                )
                src_right = (
                    self._debug.source(skip_feature)
                    or f"iag_{len(self.iags) - (i + 1)}"
                )
                self._debug.write(
                    f"concat(up={src_left}, skip={src_right}) -> {tuple(fo.shape)}"
                )
            fo = self.reduces[i](fo)
            fo = self.decoders[i](fo)
            if self.debug_enabled:
                self._debug.write(f"reduce[{i}] + decoder[{i}] -> {tuple(fo.shape)}")

        fr = self.refinement(fo)
        output = self.out_conv(fr)
        if self.debug_enabled:
            self._debug.write(
                f"refinement -> fr: {tuple(fr.shape)}, output: {tuple(output.shape)}"
            )

        illumination_map = self.iem(fr)
        noise_map = self.nem(fr)

        # Base scale loss
        loss_dict_base = self.loss_func(
            pred=output,
            target=target,
            illumination_map=illumination_map,
            noise_map=noise_map,
        )

        # Multi-scale (x2) branch
        output_x2 = None
        illumination_map_x2 = None
        noise_map_x2 = None
        if self.use_multiscale:
            fr_up = self.up_final(fr)
            output_x2 = self.output_x2(fr_up)
            illumination_map_x2 = self.iem(fr_up)
            noise_map_x2 = self.nem(fr_up)
            if self.debug_enabled:
                self._debug.write(
                    f"x2: fr_up {tuple(fr_up.shape)} -> output_x2 {tuple(output_x2.shape)}"
                )

            # Create x2 target if provided
            target_x2 = None
            if target is not None:
                target_x2 = F.interpolate(
                    target, scale_factor=2, mode="bilinear", align_corners=False
                )
                if self.debug_enabled:
                    self._debug.write(f"target_x2 -> {tuple(target_x2.shape)}")

            loss_dict_x2 = self.loss_func(
                pred=output_x2,
                target=target_x2,
                illumination_map=illumination_map_x2,
                noise_map=noise_map_x2,
            )

            # Merge losses with weights
            merged_losses = {}
            # Keep detailed per-scale components for logging
            for k, v in loss_dict_base.items():
                if k == "total":
                    continue
                merged_losses[f"{k}_x1"] = v
            for k, v in loss_dict_x2.items():
                if k == "total":
                    continue
                merged_losses[f"{k}_x2"] = v

            # Totals
            merged_losses["total_x1"] = loss_dict_base.get(
                "total", torch.tensor(0.0, device=x.device)
            )
            merged_losses["total_x2"] = (
                loss_dict_x2.get("total", torch.tensor(0.0, device=x.device))
                * self.multiscale_loss_weight
            )
            merged_losses["total"] = (
                merged_losses["total_x1"] + merged_losses["total_x2"]
            )

            loss_out = merged_losses
        else:
            loss_out = loss_dict_base

        return {
            "input": x,
            "output": output,
            "output_x2": output_x2,
            "target": target,
            "illumination_map": illumination_map,
            "noise_map": noise_map,
            "illumination_map_x2": illumination_map_x2,
            "noise_map_x2": noise_map_x2,
            "loss": loss_out,
        }

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gamma correction and histogram stretching per image per channel.

        Assumes input in [0, 1]. If not, values are clamped to [0, 1] first.
        """
        x = x.clamp(0.0, 1.0)
        # Gamma correction
        if self.preprocess_gamma is not None and self.preprocess_gamma > 0:
            x = torch.pow(x, self.preprocess_gamma)
        # Histogram stretching using percentiles
        lower_q, upper_q = self.preprocess_histogram_bounds
        # Compute per-sample, per-channel quantiles across spatial dims
        b, c, h, w = x.shape
        x_view = x.view(b, c, -1)
        # torch.quantile supports dim; keepdim to broadcast back
        lo = torch.quantile(x_view, lower_q, dim=-1, keepdim=True)
        hi = torch.quantile(x_view, upper_q, dim=-1, keepdim=True)
        stretched = (x_view - lo) / (hi - lo + 1e-6)
        stretched = stretched.view(b, c, h, w)
        return stretched.clamp(0.0, 1.0)

    def _attach_debug_to_all(self) -> None:
        writer = self._debug
        # Stem
        _attach_debug(self.embed_conv, writer, name="Stem.Conv")
        # Encoders and internal REA components
        for i, enc_seq in enumerate(self.encoders):
            for j, blk in enumerate(enc_seq):
                _attach_debug(blk, writer, name=f"Enc[{i}].REA[{j}]")
                _attach_debug(blk.attn, writer, name=f"Enc[{i}].REA[{j}].IGAM")
                _attach_debug(blk.ffn, writer, name=f"Enc[{i}].REA[{j}].GDFN")
        # Down/Up samplers
        for i, d in enumerate(self.downs):
            _attach_debug(d, writer, name=f"Down[{i}]")
        for i, u in enumerate(self.ups):
            _attach_debug(u, writer, name=f"Up[{i}]")
        # Skip gates
        for i, gate in enumerate(self.iags):
            _attach_debug(gate, writer, name=f"IAG[{i}]")
            _attach_debug(gate.iem, writer, name=f"IAG[{i}].IEM")
        # Decoders
        for i, dec_seq in enumerate(self.decoders):
            for j, blk in enumerate(dec_seq):
                _attach_debug(blk, writer, name=f"Dec[{i}].REA[{j}]")
                _attach_debug(blk.attn, writer, name=f"Dec[{i}].REA[{j}].IGAM")
                _attach_debug(blk.ffn, writer, name=f"Dec[{i}].REA[{j}].GDFN")
        # Refinement blocks
        for j, blk in enumerate(self.refinement):
            _attach_debug(blk, writer, name=f"Ref[{j}].REA")
            _attach_debug(blk.attn, writer, name=f"Ref[{j}].REA.IGAM")
            _attach_debug(blk.ffn, writer, name=f"Ref[{j}].REA.GDFN")
        # Output heads and x2 head
        _attach_debug(self.out_conv, writer, name="Head.OutConv")
        if self.use_multiscale:
            if isinstance(self.up_final, nn.Sequential):
                for k, m in enumerate(self.up_final):
                    _attach_debug(m, writer, name=f"HeadX2.UpFinal[{k}]")
            else:
                _attach_debug(self.up_final, writer, name="HeadX2.UpFinal")
            _attach_debug(self.output_x2, writer, name="HeadX2.OutConv")
        # Loss priors
        _attach_debug(self.iem, writer, name="Loss.IEM")
        _attach_debug(self.nem, writer, name="Loss.NEM")


if __name__ == "__main__":
    # Example: test with RGB (default in_channels=3)
    model = LIENet(in_channels=3, debug_enabled=True)
    input = torch.randn(1, 3, 256, 256)
    output = model(input)
