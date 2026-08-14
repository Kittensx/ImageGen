from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch


_VALID_VAE_DEVICES = {"auto", "cuda", "cpu"}


@dataclass(frozen=True)
class VAEExecutionSettings:
    """IMAGE_GEN-owned VAE memory controls.

    The active checkpoint VAE is a custom LDM implementation and does not expose
    Diffusers' ``enable_tiling`` or ``enable_slicing`` methods. These settings
    therefore drive IMAGE_GEN-owned batch slicing and overlap-add tiled
    encode/decode paths instead of calling APIs that may not exist.
    """

    tiling: bool = False
    slicing: bool = False
    device: str = "auto"
    decode_tile_size: int = 64
    decode_overlap: int = 8
    encode_tile_size: int = 512
    encode_overlap: int = 64
    latent_scale_factor: int = 8

    def normalized(self) -> "VAEExecutionSettings":
        device = str(self.device or "auto").strip().lower()
        if device not in _VALID_VAE_DEVICES:
            raise ValueError("VAE device must be one of: auto, cuda, cpu.")
        decode_tile_size = int(self.decode_tile_size)
        decode_overlap = int(self.decode_overlap)
        encode_tile_size = int(self.encode_tile_size)
        encode_overlap = int(self.encode_overlap)
        scale = int(self.latent_scale_factor)
        if scale <= 0:
            raise ValueError("VAE latent scale factor must be positive.")
        if decode_tile_size <= 0 or decode_overlap < 0:
            raise ValueError("VAE decode tile size must be positive and overlap non-negative.")
        if decode_overlap >= decode_tile_size:
            raise ValueError("VAE decode overlap must be smaller than the tile size.")
        if encode_tile_size <= 0 or encode_overlap < 0:
            raise ValueError("VAE encode tile size must be positive and overlap non-negative.")
        if encode_overlap >= encode_tile_size:
            raise ValueError("VAE encode overlap must be smaller than the tile size.")
        if encode_tile_size % scale or encode_overlap % scale:
            raise ValueError(
                "VAE encode tile size and overlap must be divisible by the latent scale factor."
            )
        return VAEExecutionSettings(
            tiling=bool(self.tiling),
            slicing=bool(self.slicing),
            device=device,
            decode_tile_size=decode_tile_size,
            decode_overlap=decode_overlap,
            encode_tile_size=encode_tile_size,
            encode_overlap=encode_overlap,
            latent_scale_factor=scale,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())


class VAEExecutionController:
    """Run VAE encode/decode with truthful device, slicing, and tiling behavior."""

    def __init__(
        self,
        vae: torch.nn.Module,
        *,
        settings: VAEExecutionSettings | None = None,
    ) -> None:
        self.vae = vae
        self.settings = (settings or VAEExecutionSettings()).normalized()
        self._last_report: dict[str, Any] = self._base_report()

    def configure(
        self,
        *,
        tiling: bool | None = None,
        slicing: bool | None = None,
        device: str | None = None,
    ) -> dict[str, Any]:
        current = self.settings
        self.settings = VAEExecutionSettings(
            tiling=current.tiling if tiling is None else bool(tiling),
            slicing=current.slicing if slicing is None else bool(slicing),
            device=current.device if device is None else str(device),
            decode_tile_size=current.decode_tile_size,
            decode_overlap=current.decode_overlap,
            encode_tile_size=current.encode_tile_size,
            encode_overlap=current.encode_overlap,
            latent_scale_factor=current.latent_scale_factor,
        ).normalized()
        self._last_report = self._base_report()
        return self.report()

    def report(self) -> dict[str, Any]:
        return dict(self._last_report)

    def _base_report(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "implementation": "image_gen_owned_ldm_vae_memory_controls",
            "diffusers_tiling_api_used": False,
            "diffusers_slicing_api_used": False,
            "requested": self.settings.to_dict(),
            "effective_device": None,
            "operation": None,
            "tiling_applied": False,
            "slicing_applied": False,
            "tile_count": 0,
            "batch_slices": 0,
        }

    @staticmethod
    def _module_device(module: torch.nn.Module) -> torch.device:
        for parameter in module.parameters():
            return parameter.device
        for buffer in module.buffers():
            return buffer.device
        return torch.device("cpu")

    @staticmethod
    def _module_dtype(module: torch.nn.Module) -> torch.dtype:
        for parameter in module.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
        for buffer in module.buffers():
            if buffer.is_floating_point():
                return buffer.dtype
        return torch.float32

    def _resolve_device(self) -> torch.device:
        requested = self.settings.device
        current = self._module_device(self.vae)
        if requested == "auto":
            return current
        if requested == "cpu":
            if current.type != "cpu":
                raise RuntimeError(
                    "VAE execution requested CPU, but the VAE component is still placed on "
                    f"{current}. Move/offload the component before encoding or decoding."
                )
            return current
        if not torch.cuda.is_available():
            raise RuntimeError("--vae-device cuda was requested, but CUDA is unavailable.")
        if current.type != "cuda":
            raise RuntimeError(
                "VAE execution requested CUDA, but the VAE component is still placed on "
                f"{current}. Acquire/place the component before encoding or decoding."
            )
        return current

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
        length = int(length)
        tile_size = min(int(tile_size), length)
        if tile_size >= length:
            return [0]
        step = tile_size - int(overlap)
        starts = list(range(0, max(1, length - tile_size + 1), step))
        last = length - tile_size
        if not starts or starts[-1] != last:
            starts.append(last)
        return starts

    @staticmethod
    def _blend_weight(
        height: int,
        width: int,
        *,
        top: int,
        bottom: int,
        left: int,
        right: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        weight_y = torch.ones((height,), device=device, dtype=dtype)
        weight_x = torch.ones((width,), device=device, dtype=dtype)
        if top > 0:
            weight_y[:top] = torch.linspace(1.0e-3, 1.0, top, device=device, dtype=dtype)
        if bottom > 0:
            weight_y[-bottom:] = torch.linspace(1.0, 1.0e-3, bottom, device=device, dtype=dtype)
        if left > 0:
            weight_x[:left] = torch.linspace(1.0e-3, 1.0, left, device=device, dtype=dtype)
        if right > 0:
            weight_x[-right:] = torch.linspace(1.0, 1.0e-3, right, device=device, dtype=dtype)
        return weight_y[:, None] * weight_x[None, :]

    @staticmethod
    def _extract_decode_tensor(value: Any) -> torch.Tensor:
        if torch.is_tensor(value):
            return value
        if hasattr(value, "sample") and torch.is_tensor(value.sample):
            return value.sample
        if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
            return value[0]
        raise TypeError("VAE decode must return a tensor, a .sample tensor, or a tensor tuple.")

    @staticmethod
    def _extract_encode_tensors(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(value, (tuple, list)) and len(value) >= 2:
            if torch.is_tensor(value[0]) and torch.is_tensor(value[1]):
                return value[0], value[1]
        latent_dist = getattr(value, "latent_dist", None)
        if latent_dist is not None:
            mean = getattr(latent_dist, "mean", None)
            logvar = getattr(latent_dist, "logvar", None)
            if torch.is_tensor(mean) and torch.is_tensor(logvar):
                return mean, logvar
        raise TypeError(
            "VAE encode must return (mean, logvar) tensors or an object with latent_dist.mean/logvar."
        )

    def _decode_direct(self, latents: torch.Tensor) -> torch.Tensor:
        return self._extract_decode_tensor(self.vae.decode(latents))

    def _decode_tiled(self, latents: torch.Tensor) -> tuple[torch.Tensor, int]:
        tile = self.settings.decode_tile_size
        overlap = self.settings.decode_overlap
        latent_height = int(latents.shape[-2])
        latent_width = int(latents.shape[-1])
        y_starts = self._tile_starts(latent_height, tile, overlap)
        x_starts = self._tile_starts(latent_width, tile, overlap)
        if len(y_starts) == 1 and len(x_starts) == 1:
            return self._decode_direct(latents), 1

        canvas: torch.Tensor | None = None
        weights: torch.Tensor | None = None
        scale_y: int | None = None
        scale_x: int | None = None
        tile_count = 0
        for y_index, y in enumerate(y_starts):
            y_end = min(latent_height, y + tile)
            for x_index, x in enumerate(x_starts):
                x_end = min(latent_width, x + tile)
                tile_latents = latents[..., y:y_end, x:x_end]
                decoded = self._decode_direct(tile_latents)
                current_scale_y = int(decoded.shape[-2]) // int(tile_latents.shape[-2])
                current_scale_x = int(decoded.shape[-1]) // int(tile_latents.shape[-1])
                if current_scale_y <= 0 or current_scale_x <= 0:
                    raise RuntimeError("VAE tiled decode produced an invalid spatial scale.")
                if (
                    int(tile_latents.shape[-2]) * current_scale_y != int(decoded.shape[-2])
                    or int(tile_latents.shape[-1]) * current_scale_x != int(decoded.shape[-1])
                ):
                    raise RuntimeError("VAE tiled decode requires an integer spatial scale.")
                if scale_y is None:
                    scale_y, scale_x = current_scale_y, current_scale_x
                    canvas = torch.zeros(
                        (
                            int(latents.shape[0]),
                            int(decoded.shape[1]),
                            latent_height * scale_y,
                            latent_width * scale_x,
                        ),
                        device=decoded.device,
                        dtype=decoded.dtype,
                    )
                    weights = torch.zeros_like(canvas[:, :1])
                if (current_scale_y, current_scale_x) != (scale_y, scale_x):
                    raise RuntimeError("VAE tiled decode returned inconsistent spatial scales.")

                assert canvas is not None and weights is not None
                out_y = y * scale_y
                out_x = x * scale_x
                out_h = int(decoded.shape[-2])
                out_w = int(decoded.shape[-1])
                top_overlap = 0 if y_index == 0 else max(0, (y_starts[y_index - 1] + tile - y) * scale_y)
                bottom_overlap = 0 if y_index == len(y_starts) - 1 else max(0, (y + tile - y_starts[y_index + 1]) * scale_y)
                left_overlap = 0 if x_index == 0 else max(0, (x_starts[x_index - 1] + tile - x) * scale_x)
                right_overlap = 0 if x_index == len(x_starts) - 1 else max(0, (x + tile - x_starts[x_index + 1]) * scale_x)
                weight = self._blend_weight(
                    out_h,
                    out_w,
                    top=min(top_overlap, out_h),
                    bottom=min(bottom_overlap, out_h),
                    left=min(left_overlap, out_w),
                    right=min(right_overlap, out_w),
                    device=decoded.device,
                    dtype=decoded.dtype,
                )[None, None]
                canvas[..., out_y : out_y + out_h, out_x : out_x + out_w] += decoded * weight
                weights[..., out_y : out_y + out_h, out_x : out_x + out_w] += weight
                tile_count += 1
        assert canvas is not None and weights is not None
        return canvas / weights.clamp_min(1.0e-6), tile_count

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(latents) or latents.ndim != 4:
            raise ValueError("VAE decode input must be a BCHW tensor.")
        device = self._resolve_device()
        dtype = self._module_dtype(self.vae)
        source = latents.to(device=device, dtype=dtype)
        slicing = bool(self.settings.slicing and int(source.shape[0]) > 1)
        batches: Iterable[torch.Tensor] = source.split(1, dim=0) if slicing else (source,)
        outputs: list[torch.Tensor] = []
        tile_count = 0
        tiling_applied = False
        for batch in batches:
            if self.settings.tiling:
                decoded, count = self._decode_tiled(batch)
                tiling_applied = tiling_applied or count > 1
                tile_count += count
            else:
                decoded = self._decode_direct(batch)
                tile_count += 1
            outputs.append(decoded)
        result = torch.cat(outputs, dim=0)
        self._last_report = {
            **self._base_report(),
            "effective_device": str(device),
            "execution_dtype": str(dtype),
            "input_dtype": str(latents.dtype),
            "operation": "decode",
            "tiling_applied": tiling_applied,
            "slicing_applied": slicing,
            "tile_count": tile_count,
            "batch_slices": len(outputs),
            "input_shape": [int(item) for item in latents.shape],
            "output_shape": [int(item) for item in result.shape],
        }
        return result

    def _encode_direct(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._extract_encode_tensors(self.vae.encode(images))

    def _encode_tiled(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        tile = self.settings.encode_tile_size
        overlap = self.settings.encode_overlap
        scale = self.settings.latent_scale_factor
        height = int(images.shape[-2])
        width = int(images.shape[-1])
        if height % scale or width % scale:
            raise ValueError("Tiled VAE encode requires image dimensions divisible by the latent scale factor.")
        y_starts = self._tile_starts(height, tile, overlap)
        x_starts = self._tile_starts(width, tile, overlap)
        if len(y_starts) == 1 and len(x_starts) == 1:
            mean, logvar = self._encode_direct(images)
            return mean, logvar, 1

        mean_canvas: torch.Tensor | None = None
        logvar_canvas: torch.Tensor | None = None
        weights: torch.Tensor | None = None
        tile_count = 0
        for y_index, y in enumerate(y_starts):
            y_end = min(height, y + tile)
            for x_index, x in enumerate(x_starts):
                x_end = min(width, x + tile)
                if y % scale or x % scale or (y_end - y) % scale or (x_end - x) % scale:
                    raise RuntimeError("VAE encode tiles must align to the latent scale factor.")
                mean, logvar = self._encode_direct(images[..., y:y_end, x:x_end])
                expected_h = (y_end - y) // scale
                expected_w = (x_end - x) // scale
                if tuple(mean.shape[-2:]) != (expected_h, expected_w):
                    raise RuntimeError("VAE tiled encode returned an unexpected latent shape.")
                if mean_canvas is None:
                    shape = (
                        int(images.shape[0]),
                        int(mean.shape[1]),
                        height // scale,
                        width // scale,
                    )
                    mean_canvas = torch.zeros(shape, device=mean.device, dtype=mean.dtype)
                    logvar_canvas = torch.zeros_like(mean_canvas)
                    weights = torch.zeros_like(mean_canvas[:, :1])
                assert mean_canvas is not None and logvar_canvas is not None and weights is not None
                out_y = y // scale
                out_x = x // scale
                out_h = int(mean.shape[-2])
                out_w = int(mean.shape[-1])
                top_overlap = 0 if y_index == 0 else max(0, (y_starts[y_index - 1] + tile - y) // scale)
                bottom_overlap = 0 if y_index == len(y_starts) - 1 else max(0, (y + tile - y_starts[y_index + 1]) // scale)
                left_overlap = 0 if x_index == 0 else max(0, (x_starts[x_index - 1] + tile - x) // scale)
                right_overlap = 0 if x_index == len(x_starts) - 1 else max(0, (x + tile - x_starts[x_index + 1]) // scale)
                weight = self._blend_weight(
                    out_h,
                    out_w,
                    top=min(top_overlap, out_h),
                    bottom=min(bottom_overlap, out_h),
                    left=min(left_overlap, out_w),
                    right=min(right_overlap, out_w),
                    device=mean.device,
                    dtype=mean.dtype,
                )[None, None]
                mean_canvas[..., out_y : out_y + out_h, out_x : out_x + out_w] += mean * weight
                logvar_canvas[..., out_y : out_y + out_h, out_x : out_x + out_w] += logvar * weight
                weights[..., out_y : out_y + out_h, out_x : out_x + out_w] += weight
                tile_count += 1
        assert mean_canvas is not None and logvar_canvas is not None and weights is not None
        divisor = weights.clamp_min(1.0e-6)
        return mean_canvas / divisor, logvar_canvas / divisor, tile_count

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("VAE encode input must be a BCHW tensor.")
        device = self._resolve_device()
        dtype = self._module_dtype(self.vae)
        source = images.to(device=device, dtype=dtype)
        slicing = bool(self.settings.slicing and int(source.shape[0]) > 1)
        batches: Iterable[torch.Tensor] = source.split(1, dim=0) if slicing else (source,)
        means: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        tile_count = 0
        tiling_applied = False
        for batch in batches:
            if self.settings.tiling:
                mean, logvar, count = self._encode_tiled(batch)
                tiling_applied = tiling_applied or count > 1
                tile_count += count
            else:
                mean, logvar = self._encode_direct(batch)
                tile_count += 1
            means.append(mean)
            logvars.append(logvar)
        mean_result = torch.cat(means, dim=0)
        logvar_result = torch.cat(logvars, dim=0)
        self._last_report = {
            **self._base_report(),
            "effective_device": str(device),
            "execution_dtype": str(dtype),
            "input_dtype": str(images.dtype),
            "operation": "encode",
            "tiling_applied": tiling_applied,
            "slicing_applied": slicing,
            "tile_count": tile_count,
            "batch_slices": len(means),
            "input_shape": [int(item) for item in images.shape],
            "input_device": str(images.device),
            "input_dtype": str(images.dtype),
            "effective_input_device": str(source.device),
            "effective_input_dtype": str(source.dtype),
            "mean_shape": [int(item) for item in mean_result.shape],
            "logvar_shape": [int(item) for item in logvar_result.shape],
        }
        return mean_result, logvar_result


__all__ = ["VAEExecutionController", "VAEExecutionSettings"]
