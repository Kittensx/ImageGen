from __future__ import annotations

import math

import torch

from image_gen.systems.denoising import DenoisingSystem
from image_gen.systems.diagnostics.output_quality import build_output_quality_report
from image_gen.systems.decoding.vae_memory import VAEExecutionController


class DecodingSystem:
    """Own latent-to-image tensor conversion; never writes files."""

    def __init__(
        self,
        vae: torch.nn.Module,
        *,
        vae_scaling_factor: float = 0.18215,
        vae_shift_factor: float = 0.0,
    ) -> None:
        self.vae = vae
        self.vae_scaling_factor = float(vae_scaling_factor)
        self.vae_shift_factor = float(vae_shift_factor)
        if self.vae_scaling_factor <= 0:
            raise ValueError("vae_scaling_factor must be positive.")
        if not math.isfinite(self.vae_shift_factor):
            raise ValueError("vae_shift_factor must be finite.")
        self._last_decode_report: dict[str, object] | None = None
        self._output_quality_diagnostics_enabled = False
        self._vae_memory = VAEExecutionController(vae)

    def configure_output_quality_diagnostics(self, enabled: bool) -> None:
        self._output_quality_diagnostics_enabled = bool(enabled)

    @torch.no_grad()
    def decode_with_diagnostics(
        self, latents: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, object]]:
        raw_images = self._vae_memory.decode(self.to_vae_latents(latents))
        raw_images = DenoisingSystem.extract_model_tensor(raw_images, owner="VAE decoder")
        if raw_images.ndim != 4:
            raise ValueError(f"VAE decoder must return BCHW data, got {tuple(raw_images.shape)}.")
        if raw_images.shape[0] != latents.shape[0]:
            raise ValueError("VAE decoder batch size does not match latent batch size.")
        if raw_images.shape[1] not in {1, 3, 4}:
            raise ValueError("VAE decoder must return 1, 3, or 4 image channels.")
        if not torch.isfinite(raw_images).all():
            raise ValueError("VAE decoder returned non-finite values.")
        images = (raw_images / 2.0 + 0.5).clamp(0.0, 1.0)
        report = build_output_quality_report(
            final_latents=latents,
            raw_vae_output=raw_images,
            normalized_images=images,
            vae_scaling_factor=self.vae_scaling_factor,
            vae_shift_factor=self.vae_shift_factor,
        )
        report["vae_memory_controls"] = self._vae_memory.report()
        self._last_decode_report = dict(report)
        return images, report

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        if self._output_quality_diagnostics_enabled:
            images, _report = self.decode_with_diagnostics(latents)
            return images

        raw_images = self._vae_memory.decode(self.to_vae_latents(latents))
        raw_images = DenoisingSystem.extract_model_tensor(raw_images, owner="VAE decoder")
        if raw_images.ndim != 4:
            raise ValueError(f"VAE decoder must return BCHW data, got {tuple(raw_images.shape)}.")
        if raw_images.shape[0] != latents.shape[0]:
            raise ValueError("VAE decoder batch size does not match latent batch size.")
        if raw_images.shape[1] not in {1, 3, 4}:
            raise ValueError("VAE decoder must return 1, 3, or 4 image channels.")
        if not torch.isfinite(raw_images).all():
            raise ValueError("VAE decoder returned non-finite values.")
        images = (raw_images / 2.0 + 0.5).clamp(0.0, 1.0)
        self._last_decode_report = {
            "contract_version": "image-gen-output-quality-v1",
            "suspect": False,
            "classification": "not_evaluated",
            "reasons": [],
            "capture_reason": "diagnostics_mode_did_not_request_output_quality",
            "vae_memory_controls": self._vae_memory.report(),
        }
        return images


    def to_vae_latents(self, sampling_latents: torch.Tensor) -> torch.Tensor:
        """Reverse the sampler-space scale/shift before VAE decode."""

        if not torch.is_tensor(sampling_latents):
            raise TypeError("sampling_latents must be a torch.Tensor.")
        return (
            sampling_latents / float(self.vae_scaling_factor)
            + float(self.vae_shift_factor)
        )

    def from_vae_latents(self, vae_latents: torch.Tensor) -> torch.Tensor:
        """Map VAE posterior latents into the sampler/model latent domain."""

        if not torch.is_tensor(vae_latents):
            raise TypeError("vae_latents must be a torch.Tensor.")
        return (
            vae_latents - float(self.vae_shift_factor)
        ) * float(self.vae_scaling_factor)

    def configure_memory_controls(
        self,
        *,
        tiling: bool | None = None,
        slicing: bool | None = None,
        device: str | None = None,
    ) -> dict[str, object]:
        return self._vae_memory.configure(
            tiling=tiling,
            slicing=slicing,
            device=device,
        )

    def memory_control_report(self) -> dict[str, object]:
        return self._vae_memory.report()

    @torch.no_grad()
    def encode_images(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IMAGE_GEN-owned VAE encode contract for future img2img/pixel hires."""

        return self._vae_memory.encode(images)

    def consume_last_decode_diagnostics(self) -> dict[str, object] | None:
        report = self._last_decode_report
        self._last_decode_report = None
        return dict(report) if report is not None else None

    @staticmethod
    def center_crop(
        images: torch.Tensor,
        *,
        width: int,
        height: int,
    ) -> torch.Tensor:
        """Center-crop BCHW image data to the exact user-requested dimensions."""

        if not torch.is_tensor(images) or images.ndim != 4:
            raise ValueError("Decoded images must be a BCHW tensor before cropping.")
        target_width = int(width)
        target_height = int(height)
        if target_width <= 0 or target_height <= 0:
            raise ValueError("Crop width and height must be positive.")

        source_height = int(images.shape[-2])
        source_width = int(images.shape[-1])
        if target_width > source_width or target_height > source_height:
            raise ValueError(
                "Requested crop dimensions exceed the decoded image dimensions: "
                f"requested={target_width}x{target_height}, "
                f"decoded={source_width}x{source_height}."
            )
        if target_width == source_width and target_height == source_height:
            return images

        left = (source_width - target_width) // 2
        top = (source_height - target_height) // 2
        return images[..., top : top + target_height, left : left + target_width]
