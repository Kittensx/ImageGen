from __future__ import annotations

from image_gen.program_metadata import PRODUCT_NAME

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from image_gen.systems.decoding import DecodingSystem, VAEExecutionController
from image_gen.contracts.vae_provenance import read_vae_provenance

VAE_ENCODE_CONTRACT_VERSION = "phase14n4-vae-encode-for-sampling-v1"
VAE_ROUND_TRIP_CONTRACT_VERSION = "phase14n4-vae-round-trip-v1"
VAE_EXECUTION_FINGERPRINT_VERSION = "phase14n5-vae-execution-fingerprint-v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class VAEEncodeResult:
    """Sampling latent plus the complete deterministic VAE encode record."""

    latents: torch.Tensor
    metadata: Mapping[str, Any]

    def to_serializable_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


@dataclass(frozen=True)
class VAERoundTripResult:
    """Diagnostic image produced by encode -> immediate decode with no noise."""

    image: torch.Tensor
    encoded: VAEEncodeResult
    metadata: Mapping[str, Any]

    def to_serializable_dict(self) -> dict[str, Any]:
        return dict(self.metadata)


@dataclass(frozen=True)
class _VAEBackend:
    owner: Any
    module: torch.nn.Module
    kind: str

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.kind == "decoding_system":
            return self.owner.encode_images(images)
        if self.kind in {"execution_controller", "module_controller"}:
            return self.owner.encode(images)
        raise RuntimeError(f"Unsupported VAE backend kind: {self.kind}")

    def decode_scaled_latents(
        self,
        latents: torch.Tensor,
        *,
        scaling_factor: float,
        shift_factor: float = 0.0,
    ) -> torch.Tensor:
        if self.kind == "decoding_system":
            return self.owner.decode(latents)
        controller = self.owner if self.kind in {"execution_controller", "module_controller"} else None
        if controller is None:
            raise RuntimeError(f"Unsupported VAE backend kind: {self.kind}")
        raw = controller.decode(
            latents / float(scaling_factor) + float(shift_factor)
        )
        raw = _extract_decode_tensor(raw)
        return (raw / 2.0 + 0.5).clamp(0.0, 1.0)

    def report(self) -> dict[str, Any]:
        if self.kind == "decoding_system":
            reporter = getattr(self.owner, "memory_control_report", None)
            return dict(reporter() or {}) if callable(reporter) else {}
        if self.kind in {"execution_controller", "module_controller"}:
            reporter = getattr(self.owner, "report", None)
            return dict(reporter() or {}) if callable(reporter) else {}
        return {}


def _extract_decode_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if hasattr(value, "sample") and torch.is_tensor(value.sample):
        return value.sample
    if isinstance(value, (tuple, list)) and value and torch.is_tensor(value[0]):
        return value[0]
    raise TypeError("VAE decode must return a tensor, a .sample tensor, or a tensor tuple.")


def _resolve_backend(vae: Any) -> _VAEBackend:
    if isinstance(vae, DecodingSystem):
        return _VAEBackend(owner=vae, module=vae.vae, kind="decoding_system")
    if isinstance(vae, VAEExecutionController):
        return _VAEBackend(owner=vae, module=vae.vae, kind="execution_controller")

    owned_module = getattr(vae, "vae", None)
    if isinstance(owned_module, torch.nn.Module):
        if (
            callable(getattr(vae, "encode_images", None))
            and callable(getattr(vae, "decode", None))
            and hasattr(vae, "vae_scaling_factor")
        ):
            return _VAEBackend(owner=vae, module=owned_module, kind="decoding_system")
        if callable(getattr(vae, "encode", None)) and callable(getattr(vae, "decode", None)):
            return _VAEBackend(owner=vae, module=owned_module, kind="execution_controller")

    if not isinstance(vae, torch.nn.Module):
        raise TypeError(
            f"vae must be a torch.nn.Module or a {PRODUCT_NAME}-compatible VAE execution owner."
        )
    return _VAEBackend(
        owner=VAEExecutionController(vae),
        module=vae,
        kind="module_controller",
    )


def _module_device_dtype(module: torch.nn.Module) -> tuple[torch.device, torch.dtype]:
    for parameter in module.parameters():
        dtype = parameter.dtype if parameter.is_floating_point() else torch.float32
        return parameter.device, dtype
    for buffer in module.buffers():
        dtype = buffer.dtype if buffer.is_floating_point() else torch.float32
        return buffer.device, dtype
    return torch.device("cpu"), torch.float32


def _normalize_hash(value: Any) -> tuple[str, str]:
    digest = str(value or "").strip()
    if not digest:
        return "", "unavailable"
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError("VAE hash must be a complete 64-character SHA-256 digest.")
    return digest.casefold(), "sha256"


def _resolve_vae_identity(
    backend: _VAEBackend,
    *,
    vae_identity: Mapping[str, Any] | str | None,
    vae_hash: str | None,
) -> dict[str, Any]:
    supplied: dict[str, Any]
    if isinstance(vae_identity, Mapping):
        supplied = dict(vae_identity)
    elif vae_identity is None:
        supplied = {}
    else:
        supplied = {"identity": str(vae_identity)}

    canonical = read_vae_provenance(backend.module)
    path_value = str(
        supplied.get("source_path")
        or supplied.get("path")
        or supplied.get("resolved_path")
        or canonical.get("source_path")
        or getattr(backend.module, "_image_gen_vae_path", "")
        or ""
    )
    identity_value = str(
        supplied.get("identity")
        or supplied.get("display_name")
        or supplied.get("name")
        or (Path(path_value).name if path_value else "")
        or canonical.get("identity")
        or getattr(backend.module, "_image_gen_vae_identity", "")
        or backend.module.__class__.__qualname__
    )
    hash_candidate = (
        vae_hash
        or supplied.get("sha256")
        or supplied.get("resolved_hash")
        or supplied.get("hash")
        or canonical.get("sha256")
        or getattr(backend.module, "_image_gen_vae_sha256", "")
    )
    digest, hash_type = _normalize_hash(hash_candidate)
    mode = str(
        supplied.get("source_kind")
        or supplied.get("mode")
        or supplied.get("source")
        or canonical.get("source_kind")
        or "runtime_component"
    )
    return {
        "identity": identity_value,
        "display_name": str(supplied.get("display_name") or identity_value),
        "path": path_value,
        "sha256": digest,
        "hash_type": hash_type,
        "hash_available": bool(digest),
        "mode": mode,
        "source_kind": mode,
        "embedded_in_checkpoint": bool(
            supplied.get("embedded_in_checkpoint", canonical.get("embedded_in_checkpoint", False))
        ),
        "provenance_contract_version": str(canonical.get("contract_version") or ""),
        "module_class": f"{backend.module.__class__.__module__}.{backend.module.__class__.__qualname__}",
        "backend_kind": backend.kind,
    }


def _validate_source(image: torch.Tensor, *, channel_order: str) -> None:
    if not torch.is_tensor(image) or image.ndim != 4:
        raise ValueError("VAE sampling encode requires a BCHW tensor.")
    if int(image.shape[0]) < 1:
        raise ValueError("VAE sampling encode requires at least one image.")
    if int(image.shape[1]) != 3:
        raise ValueError("VAE sampling encode requires exactly three RGB channels.")
    if str(channel_order or "").strip().casefold() != "rgb":
        raise ValueError("VAE sampling encode accepts RGB channel order only.")
    if not image.is_floating_point():
        raise ValueError("VAE sampling encode requires floating-point image data.")
    if not bool(torch.isfinite(image).all()):
        raise ValueError("VAE sampling encode input contains NaN or Inf.")
    minimum = float(image.detach().amin().cpu())
    maximum = float(image.detach().amax().cpu())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(
            f"VAE sampling encode expects {PRODUCT_NAME} RGB values in [0, 1]; "
            f"received range [{minimum:.8g}, {maximum:.8g}]."
        )


def _prepare_exact_image(
    image: torch.Tensor,
    *,
    target_width: int | None,
    target_height: int | None,
    allow_center_crop: bool,
) -> tuple[torch.Tensor, dict[str, Any]]:
    source_width = int(image.shape[-1])
    source_height = int(image.shape[-2])
    width = source_width if target_width is None else int(target_width)
    height = source_height if target_height is None else int(target_height)
    if width <= 0 or height <= 0:
        raise ValueError("VAE target width and height must be positive.")
    if (target_width is None) != (target_height is None):
        raise ValueError("VAE target_width and target_height must be supplied together.")

    cropped = False
    if (source_width, source_height) != (width, height):
        if not allow_center_crop:
            raise ValueError(
                "VAE encode input dimensions do not match the exact target dimensions: "
                f"input={source_width}x{source_height}, target={width}x{height}. "
                "Crop decoded padding before encoding or set allow_center_crop=True explicitly."
            )
        image = DecodingSystem.center_crop(image, width=width, height=height)
        cropped = True
    if tuple(image.shape[-2:]) != (height, width):
        raise RuntimeError("VAE exact-dimension preparation produced an unexpected shape.")
    return image, {
        "source_width": source_width,
        "source_height": source_height,
        "target_width": width,
        "target_height": height,
        "center_crop_applied": cropped,
        "crop_policy": "explicit_center_crop" if cropped else "exact_input_required",
    }


def _upscale_provenance(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if not metadata:
        return {
            "provided": False,
            "production_supported": None,
        }
    source = dict(metadata)
    if bool(source.get("builtin_resize", False)):
        upscaler_id = str(source.get("upscaler_id") or "").strip()
        upscaler_sha256 = str(source.get("upscaler_sha256") or "").strip().casefold()
        if upscaler_id != "builtin.pixel_resize.bicubic":
            raise ValueError("Built-in resize provenance has an unexpected upscaler_id.")
        if not _SHA256_RE.fullmatch(upscaler_sha256):
            raise ValueError("Built-in resize provenance must include a complete SHA-256 identity.")
        return {
            "provided": True,
            "production_supported": True,
            "schema_version": str(source.get("schema_version") or "image-gen-builtin-pixel-resize-v1"),
            "upscaler_id": upscaler_id,
            "upscaler_sha256": upscaler_sha256,
            "runtime_qualification_status": "builtin_qualified",
            "incoming_tensor_device": str(source.get("output_device") or "cpu"),
            "incoming_tensor_dtype": str(source.get("output_dtype") or "torch.float32"),
            "tile_coordinate_version": "not_applicable_builtin_resize",
            "blend_window_version": "not_applicable_builtin_resize",
            "overlap_unit": "not_applicable_builtin_resize",
            "exact_resize_filter": str(source.get("exact_resize_filter") or "bicubic"),
            "final_size_correction_filter": str(source.get("final_size_correction_filter") or "bicubic"),
            "aspect_policy": str(source.get("aspect_policy") or "stretch"),
            "padding_mode": str(source.get("padding_mode") or "reflect"),
        }
    runtime = dict(source.get("runtime_qualification") or {})
    status = str(runtime.get("status") or source.get("runtime_qualification_status") or "")
    if status not in {"qualified_cpu", "qualified_cuda"}:
        raise ValueError(
            "VAE re-encode accepts only a production-supported neural upscale result; "
            f"runtime qualification was {status or 'missing'!r}."
        )
    load_status = str(source.get("load_status") or "")
    if load_status and load_status != "supported":
        raise ValueError(
            "VAE re-encode accepts only a production-supported neural upscaler; "
            f"discovery load_status was {load_status!r}."
        )

    upscaler_id = str(source.get("upscaler_id") or "").strip()
    if not upscaler_id:
        raise ValueError("Neural upscale provenance must include a stable upscaler_id.")
    upscaler_sha256 = str(source.get("upscaler_sha256") or "").strip()
    if not _SHA256_RE.fullmatch(upscaler_sha256):
        raise ValueError(
            "Neural upscale provenance must include a complete 64-character upscaler SHA-256."
        )

    tile_plan = dict(source.get("tile_plan") or {})
    required_text = {
        "incoming tensor device": str(
            source.get("output_device") or source.get("runtime_device") or ""
        ),
        "incoming tensor dtype": str(
            source.get("output_dtype") or source.get("runtime_dtype") or ""
        ),
        "tile coordinate version": str(tile_plan.get("coordinate_version") or ""),
        "blend-window version": str(tile_plan.get("blend_window_version") or ""),
        "overlap unit": str(tile_plan.get("overlap_unit") or ""),
        "exact resize filter": str(source.get("exact_resize_filter") or ""),
    }
    missing = [name for name, value in required_text.items() if not value.strip()]
    if missing:
        raise ValueError(
            "Neural upscale provenance is incomplete; missing " + ", ".join(missing) + "."
        )

    return {
        "provided": True,
        "production_supported": True,
        "schema_version": str(source.get("schema_version") or ""),
        "upscaler_id": upscaler_id,
        "upscaler_sha256": upscaler_sha256.casefold(),
        "runtime_qualification_status": status,
        "incoming_tensor_device": required_text["incoming tensor device"],
        "incoming_tensor_dtype": required_text["incoming tensor dtype"],
        "tile_coordinate_version": required_text["tile coordinate version"],
        "blend_window_version": required_text["blend-window version"],
        "overlap_unit": required_text["overlap unit"],
        "exact_resize_filter": required_text["exact resize filter"],
        "final_size_correction_filter": str(source.get("final_size_correction_filter") or required_text["exact resize filter"]),
        "aspect_policy": str(source.get("aspect_policy") or "stretch"),
        "padding_mode": str(source.get("padding_mode") or "reflect"),
        "target_correction": dict(source.get("target_correction") or {}),
        "predicted_native_width": int(source.get("predicted_native_width") or 0),
        "predicted_native_height": int(source.get("predicted_native_height") or 0),
        "actual_native_width": int(source.get("actual_native_width") or 0),
        "actual_native_height": int(source.get("actual_native_height") or 0),
        "native_dimension_match": bool(source.get("native_dimension_match", False)),
        "target_width": int(source.get("target_width") or 0),
        "target_height": int(source.get("target_height") or 0),
        "output_shape": list(source.get("output_shape") or []),
    }



def _validate_upscale_dimensions(
    provenance: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> None:
    if not bool(provenance.get("provided")):
        return
    recorded_width = int(provenance.get("target_width") or 0)
    recorded_height = int(provenance.get("target_height") or 0)
    if recorded_width and recorded_height and (recorded_width, recorded_height) != (width, height):
        raise ValueError(
            "Neural upscale metadata target dimensions do not match the exact VAE encode tensor: "
            f"metadata={recorded_width}x{recorded_height}, tensor={width}x{height}."
        )
    output_shape = list(provenance.get("output_shape") or [])
    if len(output_shape) >= 4:
        output_height = int(output_shape[-2])
        output_width = int(output_shape[-1])
        if (output_width, output_height) != (width, height):
            raise ValueError(
                "Neural upscale metadata output shape does not match the exact VAE encode tensor: "
                f"metadata={output_width}x{output_height}, tensor={width}x{height}."
            )


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach()
    return {
        "shape": [int(item) for item in detached.shape],
        "device": str(detached.device),
        "dtype": str(detached.dtype),
        "finite": bool(torch.isfinite(detached).all()),
        "minimum": float(detached.amin().cpu()),
        "maximum": float(detached.amax().cpu()),
        "mean": float(detached.float().mean().cpu()),
    }


@torch.no_grad()
def vae_encode_for_sampling(
    *,
    image: torch.Tensor,
    vae: Any,
    scaling_factor: float,
    shift_factor: float = 0.0,
    deterministic: bool = True,
    target_width: int | None = None,
    target_height: int | None = None,
    allow_center_crop: bool = False,
    latent_downsample_factor: int = 8,
    channel_order: str = "rgb",
    vae_identity: Mapping[str, Any] | str | None = None,
    vae_hash: str | None = None,
    upscale_metadata: Mapping[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> VAEEncodeResult:
    """Convert IMAGE_GEN RGB output into a sampling latent exactly once.

    The owned boundary validates BCHW RGB [0, 1], performs the sole [-1, 1]
    conversion, uses the existing VAE execution controller, selects posterior
    mean for deterministic jobs, and applies the VAE scaling factor one time.
    """

    _validate_source(image, channel_order=channel_order)
    factor = float(scaling_factor)
    shift = float(shift_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("VAE scaling_factor must be a positive finite value.")
    if not math.isfinite(shift):
        raise ValueError("VAE shift_factor must be finite.")
    downsample = int(latent_downsample_factor)
    if downsample <= 0:
        raise ValueError("VAE latent_downsample_factor must be positive.")

    exact_image, crop_record = _prepare_exact_image(
        image,
        target_width=target_width,
        target_height=target_height,
        allow_center_crop=bool(allow_center_crop),
    )
    provenance = _upscale_provenance(upscale_metadata)
    _validate_upscale_dimensions(
        provenance,
        width=int(exact_image.shape[-1]),
        height=int(exact_image.shape[-2]),
    )
    backend = _resolve_backend(vae)
    if backend.kind == "decoding_system":
        configured_factor = float(backend.owner.vae_scaling_factor)
        configured_shift = float(getattr(backend.owner, "vae_shift_factor", 0.0))
        if not math.isclose(configured_factor, factor, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "The requested VAE scaling_factor does not match the active DecodingSystem: "
                f"requested={factor}, active={configured_factor}."
            )
        if not math.isclose(configured_shift, shift, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(
                "The requested VAE shift_factor does not match the active DecodingSystem: "
                f"requested={shift}, active={configured_shift}."
            )
    module_device, module_dtype = _module_device_dtype(backend.module)
    vae_input = exact_image.mul(2.0).sub(1.0)
    mean, logvar = backend.encode(vae_input)
    if not torch.is_tensor(mean) or not torch.is_tensor(logvar):
        raise TypeError("VAE encode must return posterior mean and log variance tensors.")
    if mean.ndim != 4 or logvar.ndim != 4 or mean.shape != logvar.shape:
        raise ValueError("VAE posterior mean and log variance must be matching BCHW tensors.")
    if int(mean.shape[0]) != int(exact_image.shape[0]):
        raise ValueError("VAE posterior batch order/size does not match the input batch.")
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(logvar).all()):
        raise ValueError("VAE posterior contains NaN or Inf.")

    expected_height = math.ceil(int(exact_image.shape[-2]) / downsample)
    expected_width = math.ceil(int(exact_image.shape[-1]) / downsample)
    if tuple(mean.shape[-2:]) != (expected_height, expected_width):
        raise ValueError(
            "VAE posterior dimensions are incompatible with the target image and downsample factor: "
            f"expected={expected_width}x{expected_height}, "
            f"received={int(mean.shape[-1])}x{int(mean.shape[-2])}."
        )

    if deterministic:
        posterior = mean
        posterior_selection = "mean"
    else:
        standard_deviation = torch.exp(0.5 * logvar.clamp(-30.0, 20.0))
        noise = torch.randn(
            mean.shape,
            device=mean.device,
            dtype=mean.dtype,
            generator=generator,
        )
        posterior = mean + standard_deviation * noise
        posterior_selection = "sample"
    latents = (posterior - shift) * factor
    if not bool(torch.isfinite(latents).all()):
        raise ValueError("Scaled VAE sampling latent contains NaN or Inf.")

    identity = _resolve_vae_identity(
        backend,
        vae_identity=vae_identity,
        vae_hash=vae_hash,
    )
    metadata = {
        "schema_version": VAE_ENCODE_CONTRACT_VERSION,
        "contract_version": VAE_ENCODE_CONTRACT_VERSION,
        "deterministic": bool(deterministic),
        "posterior_selection": posterior_selection,
        "posterior_sampling_randomness_used": not bool(deterministic),
        "source_contract": {
            "layout": "BCHW",
            "channel_order": "RGB",
            "range": [0.0, 1.0],
            "shape": [int(item) for item in image.shape],
            "device": str(image.device),
            "dtype": str(image.dtype),
        },
        "exact_image_contract": {
            **crop_record,
            "shape": [int(item) for item in exact_image.shape],
        },
        "vae_input_contract": {
            "layout": "BCHW",
            "channel_order": "RGB",
            "range": [-1.0, 1.0],
            "device": str(mean.device),
            "dtype": str(mean.dtype),
            "conversion": "image * 2 - 1",
        },
        "vae": {
            **identity,
            "module_device_before_encode": str(module_device),
            "module_dtype_before_encode": str(module_dtype),
        },
        "posterior": {
            "mean_shape": [int(item) for item in mean.shape],
            "logvar_shape": [int(item) for item in logvar.shape],
            "device": str(mean.device),
            "dtype": str(mean.dtype),
        },
        "sampling_latent": {
            "shape": [int(item) for item in latents.shape],
            "device": str(latents.device),
            "dtype": str(latents.dtype),
            "latent_downsample_factor": downsample,
            "expected_width": expected_width,
            "expected_height": expected_height,
            "scaling_factor": factor,
            "shift_factor": shift,
            "scaling_application_count": 1,
            "shift_application_count": 1 if shift != 0.0 else 0,
            "scaling_operation": "(posterior - shift_factor) * scaling_factor",
        },
        "vae_execution": backend.report(),
        "upscale_provenance": provenance,
    }
    return VAEEncodeResult(latents=latents, metadata=metadata)


def build_vae_execution_fingerprint(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Build an opt-in metadata-only diagnostic fingerprint.

    The fingerprint never hashes image or latent tensors, never performs an
    additional VAE call, and retains no CUDA tensor. It is therefore suitable
    only as an explicitly requested diagnostic record and has no expected VRAM
    overhead beyond the already executed encode metadata.
    """

    source = dict(metadata or {})
    payload = {
        "fingerprint_version": VAE_EXECUTION_FINGERPRINT_VERSION,
        "encode_contract_version": str(source.get("contract_version") or ""),
        "source_contract": dict(source.get("source_contract") or {}),
        "exact_image_contract": dict(source.get("exact_image_contract") or {}),
        "vae_input_contract": dict(source.get("vae_input_contract") or {}),
        "posterior_selection": str(source.get("posterior_selection") or ""),
        "sampling_latent": {
            key: value
            for key, value in dict(source.get("sampling_latent") or {}).items()
            if key in {
                "shape",
                "latent_downsample_factor",
                "expected_width",
                "expected_height",
                "scaling_factor",
                "shift_factor",
                "scaling_application_count",
                "shift_application_count",
                "scaling_operation",
            }
        },
        "vae": {
            key: value
            for key, value in dict(source.get("vae") or {}).items()
            if key in {
                "identity",
                "source_kind",
                "path",
                "sha256",
                "module_class",
                "backend_kind",
                "module_device_before_encode",
                "module_dtype_before_encode",
            }
        },
        "vae_execution": dict(source.get("vae_execution") or {}),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "schema_version": VAE_EXECUTION_FINGERPRINT_VERSION,
        "contract_version": VAE_EXECUTION_FINGERPRINT_VERSION,
        "diagnostic_only": True,
        "capture_mode": "metadata_only_no_tensor_hash",
        "additional_vae_execution": False,
        "retained_cuda_tensor": False,
        "expected_vram_overhead_bytes": 0,
        "sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


@torch.no_grad()
def vae_round_trip_from_encoded_for_diagnostics(
    *,
    image: torch.Tensor,
    encoded: VAEEncodeResult,
    vae: Any,
    scaling_factor: float,
    shift_factor: float = 0.0,
    allow_center_crop: bool = False,
) -> VAERoundTripResult:
    """Decode an existing encode result without repeating VAE encoding."""

    backend = _resolve_backend(vae)
    decoded = backend.decode_scaled_latents(
        encoded.latents,
        scaling_factor=float(scaling_factor),
        shift_factor=float(shift_factor),
    )
    exact = encoded.metadata["exact_image_contract"]
    width = int(exact["target_width"])
    height = int(exact["target_height"])
    if tuple(decoded.shape[-2:]) != (height, width):
        if int(decoded.shape[-1]) < width or int(decoded.shape[-2]) < height:
            raise ValueError(
                "VAE round-trip decode is smaller than the exact source dimensions: "
                f"decoded={int(decoded.shape[-1])}x{int(decoded.shape[-2])}, "
                f"source={width}x{height}."
            )
        decoded = DecodingSystem.center_crop(decoded, width=width, height=height)
    if int(decoded.shape[0]) != int(image.shape[0]) or int(decoded.shape[1]) != 3:
        raise ValueError("VAE round-trip output must preserve batch size and RGB channels.")
    if not bool(torch.isfinite(decoded).all()):
        raise ValueError("VAE round-trip image contains NaN or Inf.")

    source_exact, _ = _prepare_exact_image(
        image,
        target_width=width,
        target_height=height,
        allow_center_crop=bool(allow_center_crop),
    )
    comparable = decoded.to(device=source_exact.device, dtype=torch.float32)
    source_float = source_exact.float()
    absolute = (comparable - source_float).abs()
    channel_mae = absolute.mean(dim=(0, 2, 3)).detach().cpu().tolist()
    metadata = {
        "schema_version": VAE_ROUND_TRIP_CONTRACT_VERSION,
        "contract_version": VAE_ROUND_TRIP_CONTRACT_VERSION,
        "encode_contract": encoded.to_serializable_dict(),
        "decode_execution": backend.report(),
        "round_trip_image": _tensor_summary(decoded),
        "comparison": {
            "mean_absolute_error": float(absolute.mean().detach().cpu()),
            "maximum_absolute_error": float(absolute.max().detach().cpu()),
            "root_mean_square_error": float(
                torch.sqrt(torch.mean((comparable - source_float) ** 2)).detach().cpu()
            ),
            "per_channel_mean_absolute_error_rgb": [float(value) for value in channel_mae],
            "batch_order_preserved": int(decoded.shape[0]) == int(source_exact.shape[0]),
            "channel_order_compared_as": "RGB",
            "noise_added": False,
        },
    }
    return VAERoundTripResult(image=decoded, encoded=encoded, metadata=metadata)


@torch.no_grad()
def vae_round_trip_for_diagnostics(
    *,
    image: torch.Tensor,
    vae: Any,
    scaling_factor: float,
    shift_factor: float = 0.0,
    deterministic: bool = True,
    target_width: int | None = None,
    target_height: int | None = None,
    allow_center_crop: bool = False,
    latent_downsample_factor: int = 8,
    channel_order: str = "rgb",
    vae_identity: Mapping[str, Any] | str | None = None,
    vae_hash: str | None = None,
    upscale_metadata: Mapping[str, Any] | None = None,
    generator: torch.Generator | None = None,
) -> VAERoundTripResult:
    """Encode and immediately decode without adding second-pass noise."""

    encoded = vae_encode_for_sampling(
        image=image,
        vae=vae,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
        deterministic=deterministic,
        target_width=target_width,
        target_height=target_height,
        allow_center_crop=allow_center_crop,
        latent_downsample_factor=latent_downsample_factor,
        channel_order=channel_order,
        vae_identity=vae_identity,
        vae_hash=vae_hash,
        upscale_metadata=upscale_metadata,
        generator=generator,
    )
    return vae_round_trip_from_encoded_for_diagnostics(
        image=image,
        encoded=encoded,
        vae=vae,
        scaling_factor=scaling_factor,
        shift_factor=shift_factor,
        allow_center_crop=allow_center_crop,
    )


__all__ = [
    "VAE_ENCODE_CONTRACT_VERSION",
    "VAE_ROUND_TRIP_CONTRACT_VERSION",
    "VAE_EXECUTION_FINGERPRINT_VERSION",
    "VAEEncodeResult",
    "VAERoundTripResult",
    "build_vae_execution_fingerprint",
    "vae_encode_for_sampling",
    "vae_round_trip_for_diagnostics",
    "vae_round_trip_from_encoded_for_diagnostics",
]
