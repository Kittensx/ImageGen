from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from modules.model_qualification_registry import qualification_for_sha256
from modules.project_context import ProjectContext
from modules.sd2_runtime_assets import SD2RuntimeAssetResolver, SD2RuntimeAssets
from modules.sd2_runtime_profile import (
    SD2RuntimeProfile,
    normalize_prediction_type,
    profile_from_filename,
    profile_from_id,
)


@dataclass(frozen=True)
class SD2ResolvedModelContract:
    profile: SD2RuntimeProfile
    assets: SD2RuntimeAssets
    prediction_type: str
    prediction_type_source: str
    conditioning_dimension: int
    runtime_text_layers: int
    source_text_blocks: int
    qualification_source: str = ""
    checkpoint_sha256: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "prediction_type": self.prediction_type,
            "prediction_type_source": self.prediction_type_source,
            "conditioning_dimension": self.conditioning_dimension,
            "runtime_text_layers": self.runtime_text_layers,
            "source_text_blocks": self.source_text_blocks,
            "qualification_source": self.qualification_source,
            "checkpoint_sha256": self.checkpoint_sha256,
            "runtime_assets": self.assets.to_dict(),
        }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_sd2_model_contract(
    context: ProjectContext,
    *,
    checkpoint_filename: str,
    checkpoint_sha256: str | None = None,
    checkpoint_prediction_type: str | None = None,
    checkpoint_prediction_source: str | None = None,
    explicit_profile_id: str | None = None,
    runtime_assets_root: str | Path | None = None,
) -> SD2ResolvedModelContract:
    """Resolve a qualified SD2 variant contract without family-wide guessing.

    Evidence priority is deliberately conservative:
      1. explicit prediction metadata already read from the checkpoint header;
      2. an exact, previously-qualified checkpoint SHA-256 fingerprint;
      3. an explicitly configured runtime profile;
      4. a canonical filename profile hint.

    The selected profile is always checked against the retained scheduler, text
    encoder, and UNet runtime configs.  ``sd2.x`` by itself never selects
    epsilon or v-prediction.
    """

    digest = str(checkpoint_sha256 or "").strip().lower()
    qualification = qualification_for_sha256(digest)
    if qualification is not None and qualification.architecture != "sd2.x":
        raise ValueError(
            "Checkpoint fingerprint qualification conflicts with SD2 loading: "
            f"qualified architecture={qualification.architecture!r}"
        )

    explicit_profile = profile_from_id(explicit_profile_id) if explicit_profile_id else None
    if explicit_profile_id and explicit_profile is None:
        raise ValueError(f"Unknown SD2 runtime profile: {explicit_profile_id!r}")

    fingerprint_profile = (
        profile_from_id(qualification.profile_id) if qualification is not None else None
    )
    if qualification is not None and fingerprint_profile is None:
        raise ValueError(
            f"Checkpoint fingerprint references an unknown SD2 profile: {qualification.profile_id!r}"
        )
    if explicit_profile is not None and fingerprint_profile is not None:
        if explicit_profile.profile_id != fingerprint_profile.profile_id:
            raise ValueError(
                "Explicit SD2 runtime profile conflicts with the qualified checkpoint fingerprint: "
                f"explicit={explicit_profile.profile_id!r}, "
                f"fingerprint={fingerprint_profile.profile_id!r}"
            )

    filename_profile = profile_from_filename(checkpoint_filename)
    profile = fingerprint_profile or explicit_profile or filename_profile
    if profile is None:
        raise ValueError(
            "SD2 checkpoint variant is unresolved. The checkpoint has no recognized qualification fingerprint, "
            "explicit runtime profile, or canonical profile filename. IMAGE_GEN will not guess prediction type "
            "from the sd2.x family alone."
        )

    checkpoint_evidence_prediction = normalize_prediction_type(checkpoint_prediction_type)
    checkpoint_evidence_source = str(checkpoint_prediction_source or "").strip()
    metadata_prediction = (
        checkpoint_evidence_prediction
        if checkpoint_evidence_source != "checkpoint_fingerprint"
        else ""
    )
    if (
        fingerprint_profile is None
        and explicit_profile is None
        and not checkpoint_evidence_prediction
        and filename_profile is not None
    ):
        raise ValueError(
            "SD2 filename matched a runtime profile hint, but filenames are not prediction-type evidence. "
            "Qualify the checkpoint by fingerprint/metadata or configure an explicit SD2 runtime profile."
        )

    resolver = SD2RuntimeAssetResolver(context)
    assets = (
        resolver.resolve_from_root(runtime_assets_root)
        if runtime_assets_root is not None
        else resolver.resolve(profile)
    )

    scheduler = assets.scheduler_payload()
    scheduler_prediction = normalize_prediction_type(scheduler.get("prediction_type"))
    if not scheduler_prediction:
        raise ValueError(
            f"SD2 scheduler config does not declare a supported prediction_type: {assets.scheduler_config}"
        )
    if scheduler_prediction != profile.prediction_type:
        raise ValueError(
            "SD2 runtime profile conflicts with scheduler metadata: "
            f"profile={profile.prediction_type!r}, scheduler={scheduler_prediction!r}, "
            f"scheduler_config={assets.scheduler_config}"
        )

    fingerprint_prediction = (
        normalize_prediction_type(qualification.prediction_type) if qualification is not None else ""
    )
    if checkpoint_evidence_source == "checkpoint_fingerprint" and checkpoint_evidence_prediction:
        if fingerprint_prediction and checkpoint_evidence_prediction != fingerprint_prediction:
            raise ValueError(
                "Checkpoint fingerprint report conflicts with the qualification registry: "
                f"report={checkpoint_evidence_prediction!r}, registry={fingerprint_prediction!r}"
            )
        fingerprint_prediction = checkpoint_evidence_prediction
    if metadata_prediction and fingerprint_prediction and metadata_prediction != fingerprint_prediction:
        raise ValueError(
            "Checkpoint prediction metadata conflicts with the qualified checkpoint fingerprint: "
            f"metadata={metadata_prediction!r}, fingerprint={fingerprint_prediction!r}"
        )

    resolved_prediction = metadata_prediction or fingerprint_prediction or scheduler_prediction
    if resolved_prediction != scheduler_prediction:
        raise ValueError(
            "SD2 checkpoint prediction evidence conflicts with runtime scheduler metadata: "
            f"checkpoint={resolved_prediction!r}, scheduler={scheduler_prediction!r}"
        )
    if metadata_prediction:
        prediction_source = checkpoint_evidence_source or "checkpoint_metadata"
        qualification_source = "checkpoint_metadata"
    elif fingerprint_prediction:
        prediction_source = "checkpoint_fingerprint"
        qualification_source = "checkpoint_fingerprint"
    elif explicit_profile is not None:
        prediction_source = "runtime_scheduler_config"
        qualification_source = "explicit_runtime_profile"
    else:
        prediction_source = "runtime_scheduler_config"
        qualification_source = "filename_profile_verified_by_runtime_assets"

    text_config = _load_json(assets.text_encoder_config)
    hidden_size = int(text_config.get("hidden_size") or 0)
    num_hidden_layers = int(text_config.get("num_hidden_layers") or 0)
    max_positions = int(text_config.get("max_position_embeddings") or 0)
    if hidden_size != profile.conditioning_dimension:
        raise ValueError(
            f"SD2 text encoder hidden_size mismatch: expected {profile.conditioning_dimension}, got {hidden_size}"
        )
    if num_hidden_layers != profile.runtime_text_layers:
        raise ValueError(
            f"SD2 text encoder layer mismatch: expected {profile.runtime_text_layers}, got {num_hidden_layers}"
        )
    if max_positions != 77:
        raise ValueError(f"SD2 text encoder max_position_embeddings must be 77, got {max_positions}")

    unet_config = _load_json(assets.unet_config)
    cross_attention_dim = int(unet_config.get("cross_attention_dim") or 0)
    sample_size = int(unet_config.get("sample_size") or 0)
    if cross_attention_dim != profile.conditioning_dimension:
        raise ValueError(
            f"SD2 UNet cross_attention_dim mismatch: expected {profile.conditioning_dimension}, got {cross_attention_dim}"
        )
    if sample_size != profile.native_sample_size:
        raise ValueError(
            f"SD2 UNet sample_size mismatch: expected {profile.native_sample_size}, got {sample_size}"
        )
    if not bool(unet_config.get("use_linear_projection", False)):
        raise ValueError("SD2 UNet config must enable use_linear_projection for the qualified runtime profile.")

    return SD2ResolvedModelContract(
        profile=profile,
        assets=assets,
        prediction_type=resolved_prediction,
        prediction_type_source=prediction_source,
        conditioning_dimension=profile.conditioning_dimension,
        runtime_text_layers=profile.runtime_text_layers,
        source_text_blocks=profile.openclip_source_layers,
        qualification_source=qualification_source,
        checkpoint_sha256=digest,
    )
