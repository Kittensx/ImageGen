from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from image_gen.runtime.adapters.contracts import AdapterInspectionRecord


_KNOWN_MODEL_FAMILIES = {"sd1", "sd2", "sdxl", "sd3", "flux"}
LORA_SCAN_CACHE_SCHEMA_VERSION = 5
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
    if any(
        key.startswith("lora_te2_")
        or "text_encoder_2" in key
        or key.startswith("conditioner.embedders.1")
        for key in lowered
    ):
        return "sdxl"

    input_dimensions: set[int] = set()
    for key, shape in shape_map.items():
        lowered_key = key.lower()
        checkpoint_probe = (
            lowered_key.startswith("cond_stage_model.") and lowered_key.endswith("token_embedding.weight")
        ) or (
            lowered_key.startswith("model.diffusion_model.")
            and (".attn2.to_k.weight" in lowered_key or ".attn2.to_v.weight" in lowered_key)
        )
        if len(shape) < 2 or not (_is_family_probe_key(key) or checkpoint_probe):
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


def _adapter_parameter_key(key: str) -> bool:
    lowered = key.lower()
    return any(
        token in lowered
        for token in (
            ".lora_a.",
            ".lora_b.",
            ".lora_down.",
            ".lora_up.",
            "lora_down.weight",
            "lora_up.weight",
            "lora_a.weight",
            "lora_b.weight",
            "lora_magnitude_vector",
            "magnitude_vector",
            "hada_w1_",
            "hada_w2_",
            "lokr_",
        )
    ) or lowered.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"))


def _checkpoint_like_evidence(keys: list[str]) -> list[str]:
    lowered = [key.lower() for key in keys]
    evidence: list[str] = []
    signals = {
        "unet": any(key.startswith("model.diffusion_model.") for key in lowered),
        "vae": any(key.startswith("first_stage_model.") for key in lowered),
        "text_encoder": any(key.startswith("cond_stage_model.") for key in lowered),
        "sdxl_conditioner": any(key.startswith("conditioner.embedders.") for key in lowered),
    }
    for label, present in signals.items():
        if present:
            evidence.append(f"checkpoint_prefix:{label}")
    return evidence


def _network_metadata(metadata: Mapping[str, Any]) -> tuple[str, str]:
    for key in ("ss_network_module", "ss_network_type"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value, key
    return "", ""


def _adapter_format(keys: list[str], metadata: Mapping[str, Any]) -> tuple[str, list[str]]:
    lowered = [key.lower() for key in keys]
    network_type, network_source = _network_metadata(metadata)
    network_token = _compact_token(network_type)
    evidence: list[str] = []
    if network_type:
        evidence.append(f"{network_source}={network_type}")

    has_hada_w1 = any("hada_w1_" in key for key in lowered)
    has_hada_w2 = any("hada_w2_" in key for key in lowered)
    if has_hada_w1 and has_hada_w2:
        evidence.append("complete_loha_hada_groups")
        return "lycoris_loha", evidence
    if any("lokr_" in key for key in lowered):
        evidence.append("lokr_tensor_group")
        return "lycoris_lokr", evidence

    has_kohya = any(
        key.startswith(("lora_unet_", "lora_te_", "lora_te1_", "lora_te2_"))
        for key in lowered
    )
    has_peft = any(".lora_a.weight" in key or ".lora_b.weight" in key for key in lowered)
    has_up_down = any("lora_down.weight" in key or "lora_up.weight" in key for key in lowered)
    is_lycoris_metadata = "lycoris" in network_token
    locon_hint = "locon" in network_token or _compact_token(metadata.get("ss_network_args")).find("conv_dim") >= 0

    if is_lycoris_metadata and (locon_hint or has_up_down or has_kohya):
        evidence.append("lycoris_metadata_with_lora_up_down_or_kohya_keys")
        return "lycoris_locon" if locon_hint or has_up_down else "lycoris_other", evidence
    if is_lycoris_metadata:
        evidence.append("lycoris_metadata_without_recognized_algorithm_group")
        return "lycoris_other", evidence
    if has_kohya:
        evidence.append("kohya_component_prefixes")
        return "standard_kohya_lora", evidence
    if has_peft:
        evidence.append("peft_lora_a_b_pairs")
        return "standard_diffusers_peft_lora", evidence
    if has_up_down:
        evidence.append("lora_up_down_pairs")
        return "standard_lora_up_down", evidence

    checkpoint_evidence = _checkpoint_like_evidence(keys)
    if checkpoint_evidence and not any(_adapter_parameter_key(key) for key in keys):
        evidence.extend(checkpoint_evidence)
        evidence.append("no_recognized_adapter_parameter_groups")
        return "non_adapter_full_model", evidence
    evidence.append("no_recognized_adapter_format")
    return "unknown_adapter", evidence


def _adapter_extensions(keys: list[str]) -> tuple[str, ...]:
    lowered = [key.lower() for key in keys]
    extensions: list[str] = []
    if any("lora_magnitude_vector" in key or "magnitude_vector" in key for key in lowered):
        extensions.append("dora_magnitude")
    return tuple(extensions)


def _legacy_tensor_format(adapter_format: str) -> str:
    return {
        "standard_kohya_lora": "Kohya",
        "standard_diffusers_peft_lora": "Diffusers PEFT",
        "standard_lora_up_down": "LoRA up/down",
        "lycoris_loha": "LyCORIS",
        "lycoris_lokr": "LyCORIS",
        "lycoris_locon": "LyCORIS",
        "lycoris_other": "LyCORIS",
        "non_adapter_full_model": "Unknown",
        "unknown_adapter": "Unknown",
        "invalid": "Unknown",
    }.get(adapter_format, "Unknown")


def _adapter_group_base(key: str) -> str:
    lowered = key.lower()
    suffixes = (
        ".lora_down.weight",
        ".lora_up.weight",
        ".lora_a.weight",
        ".lora_b.weight",
        ".alpha",
        ".lora_magnitude_vector",
        ".magnitude_vector",
        ".hada_w1_a",
        ".hada_w1_b",
        ".hada_w2_a",
        ".hada_w2_b",
    )
    for suffix in suffixes:
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    marker = lowered.find(".lokr_")
    if marker >= 0:
        return lowered[:marker]
    return lowered


def _target_analysis(keys: list[str], shape_map: Mapping[str, tuple[int, ...]]) -> tuple[tuple[str, ...], dict[str, int]]:
    groups: dict[str, set[str]] = {
        "unet": set(),
        "text_encoder": set(),
        "text_encoder_2": set(),
        "linear": set(),
        "convolution": set(),
        "other": set(),
    }
    for key in keys:
        if not _adapter_parameter_key(key):
            continue
        lowered = key.lower()
        base = _adapter_group_base(key)
        if lowered.startswith("lora_te2_") or "text_encoder_2" in lowered:
            groups["text_encoder_2"].add(base)
        elif (
            lowered.startswith(("lora_te_", "lora_te1_", "text_encoder.", "text_model."))
            or ".text_encoder." in lowered
        ):
            groups["text_encoder"].add(base)
        elif (
            lowered.startswith((
                "lora_unet_",
                "unet.",
                "down_blocks.",
                "up_blocks.",
                "mid_block.",
                "conv_in.",
                "conv_out.",
                "time_embedding.",
                "add_embedding.",
            ))
            or ".unet." in lowered
        ):
            groups["unet"].add(base)
        else:
            groups["other"].add(base)

        shape = shape_map.get(key)
        if shape and any(token in lowered for token in ("lora_down", "lora_a", "hada_w1_a", "lokr_w1")):
            if len(shape) > 2:
                groups["convolution"].add(base)
            elif len(shape) == 2:
                groups["linear"].add(base)

    counts = {f"{name}_target_groups": len(values) for name, values in groups.items()}
    scopes = tuple(name for name in ("unet", "text_encoder", "text_encoder_2", "linear", "convolution", "other") if groups[name])
    return scopes, counts


def _sidecar_family(sidecar: Mapping[str, Any]) -> tuple[str, str]:
    for key in ("model_family", "base_model", "sd_version"):
        family = canonical_model_family(sidecar.get(key))
        if family:
            return family, f"sidecar:{key}"
    nested = sidecar.get("metadata")
    if isinstance(nested, Mapping):
        for key in ("model_family", "base_model", "sd_version"):
            family = canonical_model_family(nested.get(key))
            if family:
                return family, f"sidecar_metadata:{key}"
    return "", ""


def _cached_compatibility_hash(path: Path, sidecar: Mapping[str, Any]) -> dict[str, str]:
    cache = dict(sidecar.get("_lora_scan_cache") or {})
    if not lora_scan_cache_is_current(path, cache, require_compatibility_hash=True):
        return {"a1111_hash": "", "a1111_short_hash": "", "a1111_hash_source": ""}
    return {
        "a1111_hash": _normalized_hash(cache.get("a1111_hash")),
        "a1111_short_hash": _normalized_hash(cache.get("a1111_short_hash")),
        "a1111_hash_source": str(cache.get("a1111_hash_source") or ""),
    }


def inspect_lora_file(
    path: str | Path,
    *,
    sidecar_metadata: Mapping[str, Any] | None = None,
    include_compatibility_hash: bool = True,
) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    sidecar = dict(sidecar_metadata or {})
    try:
        stat = resolved.stat()
        signature = {
            "path": str(resolved),
            "size_bytes": int(stat.st_size),
            "modified_ns": int(stat.st_mtime_ns),
        }
    except OSError:
        signature = {"path": str(resolved), "size_bytes": 0, "modified_ns": 0}

    result: dict[str, Any] = {
        "path": str(resolved),
        "file_signature": signature,
        "tensor_key_count": 0,
        "tensor_key_format": "Unknown",
        "adapter_format": "invalid" if not resolved.exists() else "unknown_adapter",
        "adapter_format_evidence": [],
        "adapter_extensions": [],
        "target_scopes": [],
        "target_counts": {},
        "detected_model_family": "",
        "model_family_evidence": [],
        "network_type": "Unknown",
        "activation_text": "",
        "activation_text_source": "",
        "safetensors_metadata": {},
        "network_dimension": None,
        "network_alpha": None,
        "a1111_hash": "",
        "a1111_short_hash": "",
        "a1111_hash_source": "",
        "a1111_hash_error": "",
        "inspection_warnings": [],
        "inspection_errors": [],
        "inspection_error": "",
        "adapter_inspection": {},
    }
    if resolved.suffix.lower() != ".safetensors":
        message = "LoRA metadata inspection currently supports .safetensors files only."
        result["adapter_format"] = "invalid"
        result["inspection_error"] = message
        result["inspection_errors"] = [message]
        record = AdapterInspectionRecord(
            source_path=str(resolved),
            file_signature=signature,
            adapter_format="invalid",
            adapter_format_evidence=("unsupported_file_extension",),
            inspection_errors=(message,),
        )
        result["adapter_inspection"] = record.to_dict()
        return result

    try:
        from safetensors import safe_open

        keys: list[str] = []
        shapes: dict[str, tuple[int, ...]] = {}
        with safe_open(str(resolved), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            metadata = dict(handle.metadata() or {})
            for key in keys:
                lowered_key = key.lower()
                checkpoint_family_probe = (
                    lowered_key.startswith("cond_stage_model.") and lowered_key.endswith("token_embedding.weight")
                ) or (
                    lowered_key.startswith("model.diffusion_model.")
                    and (".attn2.to_k.weight" in lowered_key or ".attn2.to_v.weight" in lowered_key)
                ) or lowered_key.startswith("conditioner.embedders.1")
                if not (
                    _adapter_parameter_key(key)
                    or _is_family_probe_key(key)
                    or lowered_key.startswith("lora_te2_")
                    or "text_encoder_2" in lowered_key
                    or checkpoint_family_probe
                ):
                    continue
                try:
                    shapes[key] = tuple(int(value) for value in handle.get_slice(key).get_shape())
                except Exception:
                    continue

        metadata_family = _family_from_metadata(metadata)
        shape_family = _family_from_keys_and_shapes(keys, shapes)
        sidecar_family, sidecar_family_source = _sidecar_family(sidecar)
        family = metadata_family or shape_family or sidecar_family
        family_evidence: list[str] = []
        if metadata_family:
            family_evidence.append("safetensors_architecture_metadata")
        if shape_family:
            family_evidence.append(f"tensor_shape_family:{shape_family}")
        if sidecar_family and not metadata_family and not shape_family:
            family_evidence.append(sidecar_family_source)
        elif sidecar_family and family and sidecar_family != family:
            family_evidence.append(f"contradictory_{sidecar_family_source}:{sidecar_family}")

        adapter_format, format_evidence = _adapter_format(keys, metadata)
        adapter_extensions = _adapter_extensions(keys)
        target_scopes, target_counts = _target_analysis(keys, shapes)
        activation_text, activation_source = _activation_text_from_sources(sidecar, metadata)
        network_type, _ = _network_metadata(metadata)
        if not network_type:
            if adapter_format == "non_adapter_full_model":
                network_type = "full"
            elif adapter_format.startswith("lycoris_"):
                network_type = "LyCORIS"
            elif adapter_format.startswith("standard_"):
                network_type = "LoRA"
            else:
                network_type = "Unknown"

        compatibility_hash = _cached_compatibility_hash(resolved, sidecar)
        compatibility_hash_error = ""
        if include_compatibility_hash and not compatibility_hash.get("a1111_hash"):
            try:
                compatibility_hash = compute_lora_compatibility_hash(
                    resolved,
                    safetensors_metadata=metadata,
                )
            except Exception as exc:
                compatibility_hash = {
                    "a1111_hash": "",
                    "a1111_short_hash": "",
                    "a1111_hash_source": "",
                }
                compatibility_hash_error = f"{type(exc).__name__}: {exc}"

        warnings: list[str] = []
        if sidecar_family and family and sidecar_family != family:
            warnings.append(
                f"Sidecar/provider family '{sidecar_family}' contradicts file-derived family '{family}'; file evidence is authoritative."
            )
        if adapter_format == "unknown_adapter":
            warnings.append("No recognized adapter tensor representation was found.")
        if "dora_magnitude" in adapter_extensions:
            warnings.append("DoRA magnitude-vector extension detected; this extension requires explicit runtime qualification.")

        record = AdapterInspectionRecord(
            source_path=str(resolved),
            file_signature=signature,
            model_family=family,
            model_family_evidence=tuple(family_evidence),
            adapter_format=adapter_format,
            adapter_format_evidence=tuple(format_evidence),
            adapter_extensions=adapter_extensions,
            network_type=network_type,
            tensor_key_count=len(keys),
            target_scopes=target_scopes,
            target_counts=target_counts,
            source_rank=metadata.get("ss_network_dim"),
            source_alpha=metadata.get("ss_network_alpha"),
            inspection_warnings=tuple(warnings),
        )
        result.update(
            {
                "tensor_key_count": len(keys),
                "tensor_key_format": _legacy_tensor_format(adapter_format),
                "adapter_format": adapter_format,
                "adapter_format_evidence": format_evidence,
                "adapter_extensions": list(adapter_extensions),
                "target_scopes": list(target_scopes),
                "target_counts": target_counts,
                "detected_model_family": family,
                "model_family_evidence": family_evidence,
                "network_type": network_type,
                "activation_text": activation_text,
                "activation_text_source": activation_source,
                "safetensors_metadata": metadata,
                "network_dimension": metadata.get("ss_network_dim"),
                "network_alpha": metadata.get("ss_network_alpha"),
                **compatibility_hash,
                "a1111_hash_error": compatibility_hash_error,
                "inspection_warnings": warnings,
                "inspection_errors": [],
                "inspection_error": "",
                "adapter_inspection": record.to_dict(),
            }
        )
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        result["adapter_format"] = "invalid"
        result["inspection_error"] = message
        result["inspection_errors"] = [message]
        record = AdapterInspectionRecord(
            source_path=str(resolved),
            file_signature=signature,
            adapter_format="invalid",
            adapter_format_evidence=("safetensors_open_failed",),
            inspection_errors=(message,),
        )
        result["adapter_inspection"] = record.to_dict()
    return result
