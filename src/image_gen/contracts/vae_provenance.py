from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

VAE_PROVENANCE_CONTRACT_VERSION = "image-gen-vae-provenance-v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _full_sha256(value: Any) -> str:
    digest = str(value or "").strip().casefold()
    if digest and not _SHA256_RE.fullmatch(digest):
        raise ValueError("VAE provenance requires a complete 64-character SHA-256 digest.")
    return digest


def normalize_vae_provenance(
    value: Mapping[str, Any] | None = None,
    *,
    source_kind: str = "runtime_component",
    source_path: str = "",
    sha256: str = "",
    identity: str = "",
    display_name: str = "",
    embedded_in_checkpoint: bool | None = None,
) -> dict[str, Any]:
    """Return the canonical loader-owned identity for the effective VAE."""

    supplied = dict(value or {})
    resolved_path = str(
        supplied.get("source_path")
        or supplied.get("path")
        or supplied.get("resolved_path")
        or source_path
        or ""
    )
    digest = _full_sha256(
        supplied.get("sha256")
        or supplied.get("resolved_hash")
        or supplied.get("hash")
        or sha256
    )
    resolved_kind = str(
        supplied.get("source_kind")
        or supplied.get("mode")
        or supplied.get("source")
        or source_kind
        or "runtime_component"
    ).strip().casefold()
    resolved_identity = str(
        supplied.get("identity")
        or identity
        or (f"{resolved_kind}:{digest}" if digest else "")
        or (Path(resolved_path).name if resolved_path else "")
        or "runtime_vae"
    )
    resolved_embedded = (
        bool(supplied.get("embedded_in_checkpoint"))
        if "embedded_in_checkpoint" in supplied
        else (
            bool(embedded_in_checkpoint)
            if embedded_in_checkpoint is not None
            else resolved_kind in {"embedded", "embedded_checkpoint", "checkpoint_embedded"}
        )
    )
    return {
        "schema_version": VAE_PROVENANCE_CONTRACT_VERSION,
        "contract_version": VAE_PROVENANCE_CONTRACT_VERSION,
        "identity": resolved_identity,
        "display_name": str(
            supplied.get("display_name")
            or display_name
            or (Path(resolved_path).name if resolved_path else resolved_identity)
        ),
        "source_kind": resolved_kind,
        "source_path": resolved_path,
        "path": resolved_path,
        "sha256": digest,
        "hash_type": "sha256" if digest else "unavailable",
        "hash_available": bool(digest),
        "embedded_in_checkpoint": resolved_embedded,
        "external_vae": not resolved_embedded,
    }


def attach_vae_provenance(vae: Any, provenance: Mapping[str, Any]) -> dict[str, Any]:
    """Attach normalized provenance to the effective loaded VAE component."""

    normalized = normalize_vae_provenance(provenance)
    setattr(vae, "_image_gen_vae_provenance", dict(normalized))
    # Compatibility attributes remain available for Phase 14N-4 callers.
    setattr(vae, "_image_gen_vae_identity", normalized["identity"])
    setattr(vae, "_image_gen_vae_path", normalized["source_path"])
    setattr(vae, "_image_gen_vae_sha256", normalized["sha256"])
    return normalized


def read_vae_provenance(vae: Any) -> dict[str, Any]:
    value = getattr(vae, "_image_gen_vae_provenance", None)
    if isinstance(value, Mapping):
        return normalize_vae_provenance(value)
    return normalize_vae_provenance(
        source_path=str(getattr(vae, "_image_gen_vae_path", "") or ""),
        sha256=str(getattr(vae, "_image_gen_vae_sha256", "") or ""),
        identity=str(getattr(vae, "_image_gen_vae_identity", "") or ""),
    )


__all__ = [
    "VAE_PROVENANCE_CONTRACT_VERSION",
    "attach_vae_provenance",
    "normalize_vae_provenance",
    "read_vae_provenance",
]
