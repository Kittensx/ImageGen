from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


_KNOWN_MODEL_FAMILIES = {"sd1", "sd2", "sdxl", "sd3", "flux"}
LORA_SCAN_CACHE_SCHEMA_VERSION = 4
_HASH_CHUNK_SIZE = 1024 * 1024
_HEX_HASH_RE = re.compile(r"^[0-9a-fA-F]{12,128}$")


def _compact_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def _normalized_hash(value: Any) -> str:
    token = str(value or "").strip().lower()
    return token if _HEX_HASH_RE.fullmatch(token) else ""


def _sha256_stream(path: Path, *, offset: int = 0) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        if offset:
            stream.seek(offset)
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def addnet_hash_safetensors(path: str | Path) -> str:
    """Return the Kohya/AddNet SHA-256 used by A1111 for Safetensors LoRAs.

    The compatibility hash intentionally excludes the mutable Safetensors JSON
    header and hashes only the tensor-data payload. This is distinct from the
    full-file SHA-256 used by IMAGE_GEN for local integrity and replay identity.
    """

    resolved = Path(path).expanduser().resolve()
    file_size = resolved.stat().st_size
    with resolved.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError("Safetensors file is too short to contain a header length.")
    header_size = int.from_bytes(header, "little", signed=False)
    data_offset = header_size + 8
    if header_size <= 0 or data_offset > file_size:
        raise ValueError(
            "Safetensors header length is invalid for the current file size: "
            f"header_size={header_size}, file_size={file_size}."
        )
    return _sha256_stream(resolved, offset=data_offset)


def compute_lora_compatibility_hash(
    path: str | Path,
    *,
    safetensors_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Compute the hash identity expected by A1111/Civitai LoRA metadata."""

    resolved = Path(path).expanduser().resolve()
    metadata = dict(safetensors_metadata or {})
    embedded = _normalized_hash(metadata.get("sshs_model_hash"))
    if resolved.suffix.lower() == ".safetensors":
        if embedded:
            full_hash = embedded
            source = "safetensors:sshs_model_hash"
        else:
            full_hash = addnet_hash_safetensors(resolved)
            source = "safetensors_payload_sha256"
    else:
        full_hash = _sha256_stream(resolved)
        source = "full_file_sha256"
    return {
        "a1111_hash": full_hash,
        "a1111_short_hash": full_hash[:12],
        "a1111_hash_source": source,
    }


def lora_scan_cache_is_current(
    path: str | Path,
    cache: Mapping[str, Any] | None,
    *,
    require_compatibility_hash: bool = True,
) -> bool:
    """Return whether a persisted LoRA scan matches the current file."""

    resolved = Path(path).expanduser().resolve()
    payload = dict(cache or {})
    signature = payload.get("file_signature") if isinstance(payload.get("file_signature"), Mapping) else {}
    try:
        stat = resolved.stat()
        valid = bool(
            int(payload.get("schema_version") or 0) >= LORA_SCAN_CACHE_SCHEMA_VERSION
            and payload.get("scan_status")
            and int(signature.get("size_bytes") or 0) == int(stat.st_size)
            and int(signature.get("modified_ns") or 0) == int(stat.st_mtime_ns)
        )
    except (OSError, TypeError, ValueError):
        return False
    if require_compatibility_hash:
        valid = valid and bool(_normalized_hash(payload.get("a1111_hash")))
    return valid


def cached_or_compute_lora_compatibility_hash(
    path: str | Path,
    *,
    sidecar_metadata: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Use a current scan-cache hash, otherwise compute a compatibility hash."""

    sidecar = dict(sidecar_metadata or {})
    cache = dict(sidecar.get("_lora_scan_cache") or {})
    if lora_scan_cache_is_current(path, cache, require_compatibility_hash=True):
        full_hash = _normalized_hash(cache.get("a1111_hash"))
        short_hash = _normalized_hash(cache.get("a1111_short_hash")) or full_hash[:12]
        return {
            "a1111_hash": full_hash,
            "a1111_short_hash": short_hash[:12],
            "a1111_hash_source": str(cache.get("a1111_hash_source") or "scan_cache"),
        }

    metadata: dict[str, Any] = {}
    if Path(path).suffix.lower() == ".safetensors":
        try:
            from safetensors import safe_open

            with safe_open(str(Path(path).expanduser().resolve()), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
        except Exception:
            metadata = {}
    return compute_lora_compatibility_hash(path, safetensors_metadata=metadata)


def canonical_model_family(value: Any) -> str:
    """Return only a known architecture family.

    Unknown labels, checkpoint names, and repository identifiers must not be
    returned as families because doing so makes otherwise usable LoRAs appear
    definitively incompatible in the CLI.
    """

    text = str(value or "").strip().lower()
    if not text:
        return ""
    compact = _compact_token(text)

    if any(token in compact for token in ("sdxl", "stablediffusionxl")):
        return "sdxl"
    if "pony" in compact:
        # Pony checkpoints use the SDXL architecture. Keep ecosystem tags in
        # metadata, but use the architecture family for load compatibility.
        return "sdxl"
    if any(token in compact for token in ("sd3", "stablediffusion3")):
        return "sd3"
    if "flux" in compact:
        return "flux"

    if any(token in compact for token in ("sd2", "sdv2", "stablediffusion2", "stablediffusionv2")):
        return "sd2"
    if compact in {"2", "20", "21", "v2", "v20", "v21"} or compact.startswith(("v20", "v21")):
        return "sd2"

    if any(token in compact for token in ("sd1", "sdv1", "stablediffusion1", "stablediffusionv1")):
        return "sd1"
    if compact in {"1", "14", "15", "v1", "v14", "v15"} or compact.startswith(("v14", "v15")):
        return "sd1"

    if compact in _KNOWN_MODEL_FAMILIES:
        return compact
    return ""


def _first_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if text.startswith(("[", "{")):
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                return _first_text(parsed)
        return text
    if isinstance(value, Mapping):
        for key in (
            "activation_text",
            "trigger_words",
            "trained_words",
            "activation_words",
            "tags",
            "prompt",
        ):
            if key in value:
                found = _first_text(value.get(key))
                if found:
                    return found
        return ""
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(items)
    return str(value).strip()


def _activation_text_from_sources(
    sidecar_metadata: Mapping[str, Any],
    safetensors_metadata: Mapping[str, Any],
) -> tuple[str, str]:
    sidecar_candidates = (
        "activation_text",
        "trigger_words",
        "trained_words",
        "activation_words",
    )
    for key in sidecar_candidates:
        value = _first_text(sidecar_metadata.get(key))
        if value:
            return value, f"sidecar:{key}"

    nested = sidecar_metadata.get("metadata")
    if isinstance(nested, Mapping):
        for key in sidecar_candidates:
            value = _first_text(nested.get(key))
            if value:
                return value, f"sidecar_metadata:{key}"

    safetensors_candidates = (
        "activation_text",
        "trigger_words",
        "trained_words",
        "ss_trigger_words",
        "modelspec.trigger_words",
    )
    for key in safetensors_candidates:
        value = _first_text(safetensors_metadata.get(key))
        if value:
            return value, f"safetensors:{key}"
    return "", ""


def _family_from_metadata(metadata: Mapping[str, Any]) -> str:
    if str(metadata.get("ss_v2") or "").strip().lower() in {"true", "1", "yes"}:
        return "sd2"

    # These fields are intended to identify the architecture or training base.
    # Do not inspect free-form descriptions because unrelated version numbers
    # such as "2.0" can cause false SD2 classifications.
    for key in (
        "modelspec.architecture",
        "modelspec.base_model",
        "ss_base_model_version",
        "ss_sd_model_name",
    ):
        family = canonical_model_family(metadata.get(key))
        if family:
            return family
    return ""


def _tensor_format(keys: list[str]) -> str:
    lowered = [key.lower() for key in keys]
    if any("hada_w1_a" in key or "hada_w2_a" in key or "lokr_" in key for key in lowered):
        return "LyCORIS"
    if any(
        key.startswith("lora_unet_")
        or key.startswith("lora_te_")
        or key.startswith("lora_te1_")
        or key.startswith("lora_te2_")
        for key in lowered
    ):
        return "Kohya"
    if any(".lora_a.weight" in key or ".lora_b.weight" in key for key in lowered):
        return "Diffusers PEFT"
    if any("lora_down.weight" in key or "lora_up.weight" in key for key in lowered):
        return "LoRA up/down"
    return "Unknown"


def _is_family_probe_key(key: str) -> bool:
    lowered = key.lower()
    is_text_encoder = lowered.startswith("lora_te") or "text_encoder" in lowered
    is_cross_attention = any(
        token in lowered
        for token in (
            "attn2_to_k",
            "attn2_to_v",
            "attn2.to_k",
            "attn2.to_v",
        )
    )
    is_input_matrix = any(
        lowered.endswith(suffix)
        for suffix in (
            ".lora_down.weight",
            ".lora_a.weight",
            "lora_down.weight",
            "lora_a.weight",
        )
    )
    return bool((is_text_encoder or is_cross_attention) and is_input_matrix)


def _family_from_keys_and_shapes(keys: list[str], shape_map: Mapping[str, tuple[int, ...]]) -> str:
    lowered = [key.lower() for key in keys]
    if any(key.startswith("lora_te2_") or "text_encoder_2" in key for key in lowered):
        return "sdxl"

    input_dimensions: set[int] = set()
    for key, shape in shape_map.items():
        if len(shape) < 2 or not _is_family_probe_key(key):
            continue
        input_dimension = int(shape[-1])
        if input_dimension in {768, 1024, 1280, 2048}:
            input_dimensions.add(input_dimension)

    if 1280 in input_dimensions or 2048 in input_dimensions:
        return "sdxl"
    if input_dimensions == {1024}:
        return "sd2"
    if input_dimensions == {768}:
        return "sd1"
    return ""


def inspect_lora_file(
    path: str | Path,
    *,
    sidecar_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    sidecar = dict(sidecar_metadata or {})
    result: dict[str, Any] = {
        "path": str(resolved),
        "tensor_key_count": 0,
        "tensor_key_format": "Unknown",
        "detected_model_family": "",
        "activation_text": "",
        "activation_text_source": "",
        "safetensors_metadata": {},
        "a1111_hash": "",
        "a1111_short_hash": "",
        "a1111_hash_source": "",
        "a1111_hash_error": "",
        "inspection_error": "",
    }
    if resolved.suffix.lower() != ".safetensors":
        result["inspection_error"] = "LoRA metadata inspection currently supports .safetensors files only."
        return result

    try:
        from safetensors import safe_open

        keys: list[str] = []
        shapes: dict[str, tuple[int, ...]] = {}
        with safe_open(str(resolved), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = dict(handle.metadata() or {})
            for key in keys:
                if not _is_family_probe_key(key) and not (
                    key.lower().startswith("lora_te2_") or "text_encoder_2" in key.lower()
                ):
                    continue
                try:
                    shapes[key] = tuple(int(value) for value in handle.get_slice(key).get_shape())
                except Exception:
                    continue

        family = _family_from_metadata(metadata) or _family_from_keys_and_shapes(keys, shapes)
        activation_text, activation_source = _activation_text_from_sources(sidecar, metadata)
        try:
            compatibility_hash = compute_lora_compatibility_hash(
                resolved,
                safetensors_metadata=metadata,
            )
            compatibility_hash_error = ""
        except Exception as exc:
            compatibility_hash = {
                "a1111_hash": "",
                "a1111_short_hash": "",
                "a1111_hash_source": "",
            }
            compatibility_hash_error = f"{type(exc).__name__}: {exc}"
        result.update(
            {
                "tensor_key_count": len(keys),
                "tensor_key_format": _tensor_format(keys),
                "detected_model_family": family,
                "activation_text": activation_text,
                "activation_text_source": activation_source,
                "safetensors_metadata": metadata,
                **compatibility_hash,
                "a1111_hash_error": compatibility_hash_error,
                "inspection_error": "",
            }
        )
    except Exception as exc:
        result["inspection_error"] = f"{type(exc).__name__}: {exc}"
    return result
