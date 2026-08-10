from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - IMAGE_GEN currently pins a newer torch.
    SDPBackend = None
    sdpa_kernel = None

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.component_placement import place_component


def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


# Preserve the historical VAE attention implementation for ordinary decode sizes so
# existing seed/output behavior stays untouched. Very large hires decodes switch to
# global memory-efficient attention before the quadratic score matrix is materialized.
_VAE_ATTN_LEGACY_MAX_TOKENS = 12_288
_VAE_ATTN_FALLBACK_QUERY_CHUNK = 512


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, temb_channels: int = 0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.temb_proj = None
        if temb_channels > 0:
            self.temb_proj = nn.Linear(temb_channels, out_channels)

        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.nin_shortcut = None

    def forward(self, x: torch.Tensor, temb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)

        if temb is not None and self.temb_proj is not None:
            h = h + self.temb_proj(swish(temb))[:, :, None, None]

        h = self.norm2(h)
        h = swish(h)
        h = self.conv2(h)

        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels: int, *, optimize_large_tokens: bool = False):
        super().__init__()
        self.optimize_large_tokens = bool(optimize_large_tokens)
        self.norm = nn.GroupNorm(32, channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    @staticmethod
    def _legacy_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = q.shape
        tokens = h * w
        q_tokens = q.reshape(b, c, tokens).permute(0, 2, 1)
        k_channels = k.reshape(b, c, tokens)
        weights = torch.bmm(q_tokens, k_channels) * (c ** -0.5)
        weights = torch.softmax(weights, dim=2)
        v_channels = v.reshape(b, c, tokens)
        return torch.bmm(v_channels, weights.permute(0, 2, 1)).reshape(b, c, h, w)

    @staticmethod
    def _chunked_global_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Exact-global attention with bounded score memory.

        This is the compatibility fallback when CUDA cannot provide a fused SDPA
        kernel for the VAE's unusually wide single attention head. It preserves
        global key/value coverage; unlike VAE tiling, it does not split the image
        into independent spatial attention regions. Float32 score accumulation also
        avoids fp16 dot-product overflow seen in some checkpoint-embedded VAEs.
        """

        b, c, h, w = q.shape
        tokens = h * w
        result_dtype = q.dtype
        q_tokens = q.reshape(b, c, tokens).permute(0, 2, 1)
        k_tokens = k.reshape(b, c, tokens).permute(0, 2, 1)
        v_tokens = v.reshape(b, c, tokens).permute(0, 2, 1)
        k_transposed = k_tokens.float().transpose(1, 2)
        v_float = v_tokens.float()
        scale = c ** -0.5
        chunks: list[torch.Tensor] = []
        for start in range(0, tokens, _VAE_ATTN_FALLBACK_QUERY_CHUNK):
            end = min(tokens, start + _VAE_ATTN_FALLBACK_QUERY_CHUNK)
            scores = torch.bmm(q_tokens[:, start:end].float(), k_transposed) * scale
            probabilities = torch.softmax(scores, dim=-1)
            chunks.append(torch.bmm(probabilities, v_float).to(dtype=result_dtype))
        attended = torch.cat(chunks, dim=1)
        return attended.permute(0, 2, 1).reshape(b, c, h, w)

    @staticmethod
    def _memory_efficient_global_attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = q.shape
        tokens = h * w
        q_tokens = q.reshape(b, c, tokens).permute(0, 2, 1).unsqueeze(1)
        k_tokens = k.reshape(b, c, tokens).permute(0, 2, 1).unsqueeze(1)
        v_tokens = v.reshape(b, c, tokens).permute(0, 2, 1).unsqueeze(1)

        # Never permit SDPA's quadratic math backend on this large-token path.
        # Prefer fused CUDA kernels; if this GPU/head shape cannot use one, fall
        # back to bounded query chunks while keeping attention global.
        if q.is_cuda and sdpa_kernel is not None and SDPBackend is not None:
            backends = [
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
            try:
                with sdpa_kernel(backends=backends, set_priority=True):
                    attended = F.scaled_dot_product_attention(
                        q_tokens,
                        k_tokens,
                        v_tokens,
                        dropout_p=0.0,
                        is_causal=False,
                    )
                attended = attended.squeeze(1).permute(0, 2, 1).reshape(b, c, h, w)
                if bool(torch.isfinite(attended).all()):
                    return attended
            except RuntimeError:
                pass

        return AttnBlock._chunked_global_attention(q, k, v)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        token_count = int(q.shape[-2]) * int(q.shape[-1])
        if (
            self.optimize_large_tokens
            and token_count > _VAE_ATTN_LEGACY_MAX_TOKENS
        ):
            h_ = self._memory_efficient_global_attention(q, k, v)
        else:
            h_ = self._legacy_attention(q, k, v)

        h_ = self.proj_out(h_)
        return x + h_


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class EncoderDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_downsample: bool):
        super().__init__()
        self.block = nn.ModuleList([
            ResnetBlock(in_channels, out_channels),
            ResnetBlock(out_channels, out_channels),
        ])
        self.downsample = Downsample(out_channels) if add_downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class DecoderUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, add_upsample: bool):
        super().__init__()
        self.block = nn.ModuleList([
            ResnetBlock(in_channels, out_channels),
            ResnetBlock(out_channels, out_channels),
            ResnetBlock(out_channels, out_channels),
        ])
        self.upsample = Upsample(out_channels) if add_upsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.block:
            x = block(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MidBlock(nn.Module):
    def __init__(self, channels: int, *, optimize_large_tokens: bool = False):
        super().__init__()
        self.block_1 = ResnetBlock(channels, channels)
        self.attn_1 = AttnBlock(
            channels,
            optimize_large_tokens=optimize_large_tokens,
        )
        self.block_2 = ResnetBlock(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.attn_1(x)
        x = self.block_2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, channels: list[int], z_channels: int):
        super().__init__()
        self.conv_in = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1)

        self.down = nn.ModuleList([
            EncoderDownBlock(channels[0], channels[0], add_downsample=True),
            EncoderDownBlock(channels[0], channels[1], add_downsample=True),
            EncoderDownBlock(channels[1], channels[2], add_downsample=True),
            EncoderDownBlock(channels[2], channels[3], add_downsample=False),
        ])

        self.mid = MidBlock(channels[-1])

        self.norm_out = nn.GroupNorm(32, channels[-1], eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(channels[-1], 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        for block in self.down:
            x = block(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = swish(x)
        x = self.conv_out(x)
        return x


class Decoder(nn.Module):
    """
    Decoder built from explicit per-stage specs derived from checkpoint:
    specs[i] = (in_channels, out_channels)
    """
    def __init__(self, decoder_specs: list[tuple[int, int]], z_channels: int):
        super().__init__()

        if len(decoder_specs) != 4:
            raise ValueError(f"Expected 4 decoder specs, got {decoder_specs}")

        deepest_in = decoder_specs[-1][0]
        shallowest_out = decoder_specs[0][1]

        self.conv_in = nn.Conv2d(z_channels, deepest_in, kernel_size=3, stride=1, padding=1)
        # Only decoder-side large-token attention needs this protection. The
        # encoder already succeeds for the same hires request and must retain
        # its historical math so pixel-neural VAE re-encoding stays bit-for-bit
        # on the existing path as far as PyTorch permits.
        self.mid = MidBlock(deepest_in, optimize_large_tokens=True)

        self.up = nn.ModuleList([
            DecoderUpBlock(decoder_specs[0][0], decoder_specs[0][1], add_upsample=False),
            DecoderUpBlock(decoder_specs[1][0], decoder_specs[1][1], add_upsample=True),
            DecoderUpBlock(decoder_specs[2][0], decoder_specs[2][1], add_upsample=True),
            DecoderUpBlock(decoder_specs[3][0], decoder_specs[3][1], add_upsample=True),
        ])

        self.norm_out = nn.GroupNorm(32, shallowest_out, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(shallowest_out, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.mid(z)
        for block in reversed(self.up):
            z = block(z)
        z = self.norm_out(z)
        z = swish(z)
        z = self.conv_out(z)
        return z


class LDMVAE(nn.Module):
    """
    LDM-style VAE with names that match first_stage_model.* much more closely than diffusers.
    """
    def __init__(self, encoder_channels: list[int], decoder_specs: list[tuple[int, int]], z_channels: int = 4):
        super().__init__()
        self.encoder = Encoder(encoder_channels, z_channels)
        self.decoder = Decoder(decoder_specs, z_channels)
        self.quant_conv = nn.Conv2d(2 * z_channels, 2 * z_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(z_channels, z_channels, kernel_size=1)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mean + std * torch.randn_like(std)
        return self.decode(z)


@dataclass
class LDMVAEBuildResult:
    model: LDMVAE
    encoder_channels: list[int]
    decoder_specs: list[tuple[int, int]]
    loaded_keys: int = 0
    expected_keys: int = 0
    matched_keys: int = 0
    missing_keys: list[str] = field(default_factory=list)
    unexpected_keys: list[str] = field(default_factory=list)
    error: Optional[str] = None
    name: str = "vae"

    @property
    def success(self) -> bool:
        return self.error is None

    @property
    def coverage_ratio(self) -> float:
        if self.expected_keys <= 0:
            return 0.0
        return self.matched_keys / self.expected_keys

    def to_validation_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "success": self.success,
            "provided_keys": self.loaded_keys,
            "expected_keys": self.expected_keys,
            "matched_keys": self.matched_keys,
            "coverage_ratio": self.coverage_ratio,
            "missing_key_count": len(self.missing_keys),
            "unexpected_key_count": len(self.unexpected_keys),
            "missing_key_samples": self.missing_keys[:25],
            "unexpected_key_samples": self.unexpected_keys[:25],
            "encoder_channels": list(self.encoder_channels),
            "decoder_specs": [list(item) for item in self.decoder_specs],
            "error": self.error,
        }


class LDMVAEBuilder:
    def __init__(self, device: str | None = None, dtype: torch.dtype | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

    def debug_find_upsamplers(self, vae_state):
        print("\n=== VAE Upsampler Keys ===")
        for k in sorted(vae_state.keys()):
            if "upsample" in k:
                print(k)
    
    def derive_encoder_channels(self, vae_state: dict[str, Any]) -> list[int]:
        channels = []
        for i in range(4):
            key = f"encoder.down.{i}.block.0.conv1.weight"
            if key not in vae_state:
                raise KeyError(f"Missing VAE key for encoder channel derivation: {key}")
            channels.append(int(vae_state[key].shape[0]))
        return channels

    def derive_decoder_specs(self, vae_state: dict[str, Any]) -> list[tuple[int, int]]:
        """
        Derive explicit (in_channels, out_channels) per decoder stage from:
        decoder.up.{i}.block.0.conv1.weight shape == [out_channels, in_channels, 3, 3]
        """
        specs: list[tuple[int, int]] = []
        for i in range(4):
            key = f"decoder.up.{i}.block.0.conv1.weight"
            if key not in vae_state:
                raise KeyError(f"Missing VAE key for decoder spec derivation: {key}")
            weight = vae_state[key]
            out_c = int(weight.shape[0])
            in_c = int(weight.shape[1])
            specs.append((in_c, out_c))
        return specs

    def build_and_load(self, vae_state: dict[str, Any]) -> LDMVAEBuildResult:
        try:
            encoder_channels = self.derive_encoder_channels(vae_state)
            decoder_specs = self.derive_decoder_specs(vae_state)

            print(f"LDMVAEBuilder derived encoder_channels: {encoder_channels}")
            print(f"LDMVAEBuilder derived decoder_specs: {decoder_specs}")

            model = LDMVAE(
                encoder_channels=encoder_channels,
                decoder_specs=decoder_specs,
                z_channels=4,
            )
            
            #self.debug_find_upsamplers(vae_state)
            expected = set(model.state_dict().keys())
            provided = set(vae_state.keys())
            matched = expected.intersection(provided)
            incompatible = model.load_state_dict(vae_state, strict=False)

            place_component(
                model,
                device=self.device,
                dtype=self.dtype,
                owner="LDMVAEBuilder",
                component_name="vae",
            )

            return LDMVAEBuildResult(
                model=model,
                encoder_channels=encoder_channels,
                decoder_specs=decoder_specs,
                loaded_keys=len(vae_state),
                expected_keys=len(expected),
                matched_keys=len(matched),
                missing_keys=list(getattr(incompatible, "missing_keys", [])),
                unexpected_keys=list(getattr(incompatible, "unexpected_keys", [])),
                error=None,
            )
        except Exception as e:
            dummy = LDMVAE(
                encoder_channels=[128, 256, 512, 512],
                decoder_specs=[(256, 128), (512, 256), (512, 512), (512, 512)],
                z_channels=4,
            )
            return LDMVAEBuildResult(
                model=dummy,
                encoder_channels=[128, 256, 512, 512],
                decoder_specs=[(256, 128), (512, 256), (512, 512), (512, 512)],
                loaded_keys=len(vae_state),
                error=str(e),
            )