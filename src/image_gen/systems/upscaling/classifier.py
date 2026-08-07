from __future__ import annotations

import importlib.metadata
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from image_gen.systems.upscaling.contracts import SUPPORTED_NATIVE_SCALES
from image_gen.systems.upscaling.diagnostics import bounded_error_text

CANONICAL_LOADER_BACKEND = "spandrel"
INSPECTOR_VERSION = "imagegen-pth-shape-inspector-v2"


@dataclass(frozen=True)
class UpscalerClassification:
    architecture: str
    architecture_confidence: str
    native_scale: int
    input_channels: int
    output_channels: int
    supports_half: bool
    supports_bfloat16: bool
    tile_supported: bool
    load_status: str
    loader_backend: str
    compatibility_notes: tuple[str, ...]
    bounded_error: str = ""


def loader_backend_version() -> str:
    try:
        spandrel_version = importlib.metadata.version("spandrel")
    except importlib.metadata.PackageNotFoundError:
        spandrel_version = "not-installed"
    try:
        import torch

        torch_version = str(torch.__version__)
    except Exception:
        torch_version = "unavailable"
    return (
        f"{CANONICAL_LOADER_BACKEND}:{spandrel_version}|"
        f"{INSPECTOR_VERSION}|torch:{torch_version}"
    )


def _shape(value: Any) -> tuple[int, ...]:
    raw_shape = getattr(value, "shape", ())
    try:
        return tuple(int(item) for item in raw_shape)
    except (TypeError, ValueError):
        return ()


def _tensor_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    output: dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        if not isinstance(raw_key, str):
            continue
        if _shape(raw_value):
            output[raw_key] = raw_value
    return output or None


def _extract_state_dict(payload: Any) -> dict[str, Any] | None:
    direct = _tensor_mapping(payload)
    if direct is not None:
        return direct
    if not isinstance(payload, Mapping):
        return None
    for key in ("params_ema", "params", "state_dict", "model_state_dict", "model", "net_g"):
        nested = payload.get(key)
        direct = _tensor_mapping(nested)
        if direct is not None:
            return direct
    return None


def extract_tensor_state_dict(payload: Any) -> dict[str, Any] | None:
    """Return the supported tensor state dictionary without executing custom objects."""

    return _extract_state_dict(payload)


def _normalize_keys(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for raw_key, value in state_dict.items():
        key = str(raw_key)
        while key.startswith("module."):
            key = key[7:]
        output[key] = value
    return output


def _modern_rrdbnet(state_dict: Mapping[str, Any]) -> UpscalerClassification | None:
    keys = {key.casefold(): key for key in state_dict}
    first_key = keys.get("conv_first.weight")
    last_key = keys.get("conv_last.weight")
    has_rrdb = any(".rdb1.conv1.weight" in key.casefold() for key in state_dict)
    if first_key is None or last_key is None or not has_rrdb:
        return None

    first_shape = _shape(state_dict[first_key])
    last_shape = _shape(state_dict[last_key])
    if len(first_shape) != 4 or len(last_shape) != 4:
        return UpscalerClassification(
            "realesrgan_rrdbnet",
            "medium",
            0,
            0,
            0,
            False,
            False,
            False,
            "inspection_failed",
            CANONICAL_LOADER_BACKEND,
            ("Recognized RRDBNet key layout but convolution shapes were invalid.",),
            "Invalid conv_first or conv_last tensor shape.",
        )

    raw_input_channels = int(first_shape[1])
    output_channels = int(last_shape[0])
    upconv_indices = {
        int(match.group(1))
        for key in state_dict
        if (match := re.fullmatch(r"conv_up(\d+)\.weight", key.casefold()))
    }
    upsample_factor = 2 ** len(upconv_indices) if upconv_indices else 0

    input_channels = raw_input_channels
    pixel_unshuffle_factor = 1
    if output_channels > 0 and raw_input_channels % output_channels == 0:
        ratio = raw_input_channels // output_channels
        root = int(math.isqrt(ratio))
        if root * root == ratio and root in {1, 2, 4}:
            pixel_unshuffle_factor = root
            input_channels = output_channels

    native_scale = (
        upsample_factor // pixel_unshuffle_factor
        if upsample_factor and upsample_factor % pixel_unshuffle_factor == 0
        else 0
    )
    return _validated_supported_architecture(
        architecture="realesrgan_rrdbnet",
        native_scale=native_scale,
        input_channels=input_channels,
        output_channels=output_channels,
        confidence="high",
        notes=(
            "Modern BasicSR/RealESRGAN RRDBNet state-dictionary layout detected.",
            "Runtime model construction remains gated on Phase 14N-2 Spandrel qualification.",
        ),
    )


def _legacy_esrgan_rrdbnet(state_dict: Mapping[str, Any]) -> UpscalerClassification | None:
    keys = {key.casefold(): key for key in state_dict}
    first_key = keys.get("model.0.weight")
    has_rrdb = any(".rdb1.conv1." in key.casefold() for key in state_dict)
    if first_key is None or not has_rrdb:
        return None

    first_shape = _shape(state_dict[first_key])
    top_level: list[tuple[int, str, tuple[int, ...]]] = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"model\.(\d+)\.weight", key.casefold())
        shape = _shape(value)
        if match and len(shape) == 4:
            top_level.append((int(match.group(1)), key, shape))
    top_level.sort(key=lambda item: item[0])
    if len(first_shape) != 4 or len(top_level) < 2:
        return UpscalerClassification(
            "esrgan_rrdbnet",
            "medium",
            0,
            0,
            0,
            False,
            False,
            False,
            "inspection_failed",
            CANONICAL_LOADER_BACKEND,
            ("Recognized legacy ESRGAN RRDBNet keys but could not determine channel layout.",),
            "Incomplete legacy RRDBNet convolution metadata.",
        )

    final_shape = top_level[-1][2]
    input_channels = int(first_shape[1])
    output_channels = int(final_shape[0])
    feature_channels = int(first_shape[0])
    feature_convolutions = [
        shape
        for index, _key, shape in top_level[1:-1]
        if int(shape[0]) == feature_channels and int(shape[1]) == feature_channels
    ]
    upsample_layers = max(0, len(feature_convolutions) - 1)
    native_scale = 2 ** upsample_layers if upsample_layers else 0
    return _validated_supported_architecture(
        architecture="esrgan_rrdbnet",
        native_scale=native_scale,
        input_channels=input_channels,
        output_channels=output_channels,
        confidence="high",
        notes=(
            "Legacy ESRGAN RRDBNet sequential state-dictionary layout detected.",
            "Runtime model construction remains gated on Phase 14N-2 Spandrel qualification.",
        ),
    )


def _srvggnetcompact(state_dict: Mapping[str, Any]) -> UpscalerClassification | None:
    body_layers: list[tuple[int, tuple[int, ...]]] = []
    for key, value in state_dict.items():
        match = re.fullmatch(r"body\.(\d+)\.weight", key.casefold())
        shape = _shape(value)
        if match and len(shape) == 4:
            body_layers.append((int(match.group(1)), shape))
    body_layers.sort(key=lambda item: item[0])
    if len(body_layers) < 3:
        return None

    first_shape = body_layers[0][1]
    final_shape = body_layers[-1][1]
    input_channels = int(first_shape[1])
    packed_output_channels = int(final_shape[0])
    native_scale = 0
    output_channels = 0
    if input_channels > 0 and packed_output_channels % input_channels == 0:
        ratio = packed_output_channels // input_channels
        root = int(math.isqrt(ratio))
        if root * root == ratio:
            native_scale = root
            output_channels = input_channels
    return _validated_supported_architecture(
        architecture="realesrgan_srvggnetcompact",
        native_scale=native_scale,
        input_channels=input_channels,
        output_channels=output_channels,
        confidence="high",
        notes=(
            "RealESRGAN SRVGGNetCompact body layout detected.",
            "Runtime model construction remains gated on Phase 14N-2 Spandrel qualification.",
        ),
    )


def _validated_supported_architecture(
    *,
    architecture: str,
    native_scale: int,
    input_channels: int,
    output_channels: int,
    confidence: str,
    notes: tuple[str, ...],
) -> UpscalerClassification:
    if input_channels != 3 or output_channels != 3:
        return UpscalerClassification(
            architecture,
            confidence,
            native_scale,
            input_channels,
            output_channels,
            False,
            False,
            False,
            "unsupported_channels",
            CANONICAL_LOADER_BACKEND,
            notes + ("Initial Phase 14N support requires three-channel RGB input and output.",),
            bounded_error_text(
                f"Unsupported channel contract: input={input_channels}, output={output_channels}."
            ),
        )
    if native_scale not in SUPPORTED_NATIVE_SCALES:
        return UpscalerClassification(
            architecture,
            confidence,
            native_scale,
            input_channels,
            output_channels,
            False,
            False,
            False,
            "deferred_scale",
            CANONICAL_LOADER_BACKEND,
            notes
            + (
                "Native scales outside 2x, 4x, and 8x remain inventoried and deferred until target-aware memory qualification is complete.",
            ),
            bounded_error_text(f"Unsupported native scale: {native_scale}x."),
        )
    return UpscalerClassification(
        architecture,
        confidence,
        native_scale,
        input_channels,
        output_channels,
        True,
        True,
        True,
        "supported",
        CANONICAL_LOADER_BACKEND,
        notes
        + (
            "Inspection used torch.load(weights_only=True, map_location='meta'); no CUDA model was created.",
        ),
        "",
    )


def classify_state_dict(state_dict: Mapping[str, Any]) -> UpscalerClassification:
    normalized = _normalize_keys(state_dict)
    for classifier in (_modern_rrdbnet, _legacy_esrgan_rrdbnet, _srvggnetcompact):
        result = classifier(normalized)
        if result is not None:
            return result
    return UpscalerClassification(
        "unclassified",
        "none",
        0,
        0,
        0,
        False,
        False,
        False,
        "deferred_architecture",
        CANONICAL_LOADER_BACKEND,
        (
            "The file contains a tensor state dictionary. Its architecture is retained as deferred until Spandrel metadata and runtime qualification are complete.",
        ),
        "No current ESRGAN or RealESRGAN discovery signature was found; expanded Spandrel qualification is required.",
    )


def inspect_upscaler_file(path: str | Path) -> UpscalerClassification:
    selected = Path(path)
    try:
        import torch
    except Exception as exc:
        return UpscalerClassification(
            "unclassified",
            "none",
            0,
            0,
            0,
            False,
            False,
            False,
            "inspection_failed",
            CANONICAL_LOADER_BACKEND,
            ("PyTorch is required for safe tensor-only .pth inspection.",),
            bounded_error_text(exc),
        )

    try:
        payload = torch.load(
            selected,
            map_location="meta",
            weights_only=True,
        )
    except (EOFError, IndexError, OSError, RuntimeError, ValueError, pickle.UnpicklingError) as exc:
        raw_message = str(exc)
        message = bounded_error_text(raw_message)
        lowered = raw_message.casefold()
        unsafe_object_rejection = (
            "unsupported global" in lowered
            or "can only build tensor" in lowered
            or "was not an allowed global" in lowered
        )
        status = "inspection_failed" if unsafe_object_rejection else "corrupt"
        return UpscalerClassification(
            "unclassified",
            "none",
            0,
            0,
            0,
            False,
            False,
            False,
            status,
            CANONICAL_LOADER_BACKEND,
            (
                "The file was not executed. Tensor-only inspection rejected or could not decode it.",
            ),
            message,
        )
    except Exception as exc:
        return UpscalerClassification(
            "unclassified",
            "none",
            0,
            0,
            0,
            False,
            False,
            False,
            "inspection_failed",
            CANONICAL_LOADER_BACKEND,
            ("Unexpected error during bounded tensor-only model inspection.",),
            bounded_error_text(exc),
        )

    state_dict = _extract_state_dict(payload)
    if state_dict is None:
        return UpscalerClassification(
            "unclassified",
            "none",
            0,
            0,
            0,
            False,
            False,
            False,
            "unclassified",
            CANONICAL_LOADER_BACKEND,
            ("The file loaded safely but did not contain a recognized tensor state dictionary.",),
            "No tensor state dictionary was found in the supported wrapper keys.",
        )
    return classify_state_dict(state_dict)
