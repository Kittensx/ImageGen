from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modules.sd2_runtime_profile import normalize_prediction_type


@dataclass(frozen=True)
class ModelQualification:
    """Known model fingerprint qualification used when checkpoint metadata is silent."""

    sha256: str
    architecture: str
    profile_id: str
    prediction_type: str
    conditioning_dimension: int
    source: str = "checkpoint_fingerprint"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sha256": self.sha256,
            "architecture": self.architecture,
            "profile_id": self.profile_id,
            "prediction_type": self.prediction_type,
            "conditioning_dimension": self.conditioning_dimension,
            "source": self.source,
        }


# Fingerprints are qualification evidence, not architecture heuristics.  Add an
# entry only after the exact checkpoint has been compared to trustworthy runtime
# metadata/reference components.  The SD2.1 Base 512 entry below was qualified
# against the retained scheduler/text/UNet/VAE reference set used by the SD2
# validation ladder.
_KNOWN_BY_SHA256: dict[str, ModelQualification] = {
    "df955bdf6b682338ea9b55dfc0d8f3475aadf4836e204893d28b82355e0956d2": ModelQualification(
        sha256="df955bdf6b682338ea9b55dfc0d8f3475aadf4836e204893d28b82355e0956d2",
        architecture="sd2.x",
        profile_id="sd2.1-base-512",
        prediction_type="epsilon",
        conditioning_dimension=1024,
    ),
}


def qualification_for_sha256(sha256: str | None) -> ModelQualification | None:
    digest = str(sha256 or "").strip().lower()
    if not digest:
        return None
    return _KNOWN_BY_SHA256.get(digest)


def validate_qualification(qualification: ModelQualification) -> None:
    if not qualification.sha256 or len(qualification.sha256) != 64:
        raise ValueError("Model qualification SHA-256 must contain 64 hexadecimal characters.")
    if not normalize_prediction_type(qualification.prediction_type):
        raise ValueError(
            f"Unsupported model qualification prediction type: {qualification.prediction_type!r}"
        )
    if qualification.conditioning_dimension <= 0:
        raise ValueError("Model qualification conditioning_dimension must be positive.")


for _qualification in _KNOWN_BY_SHA256.values():
    validate_qualification(_qualification)
