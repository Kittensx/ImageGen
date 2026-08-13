from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping


def normalize_prediction_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    mapping = {
        "eps": "epsilon",
        "epsilon": "epsilon",
        "epsilon_prediction": "epsilon",
        "v": "v-prediction",
        "v_prediction": "v-prediction",
        "vpred": "v-prediction",
        "v_pred": "v-prediction",
        "sample": "sample",
        "sample_prediction": "sample",
    }
    return mapping.get(text, "")


@dataclass(frozen=True)
class SD2RuntimeProfile:
    profile_id: str
    display_name: str
    prediction_type: str
    native_sample_size: int
    conditioning_dimension: int = 1024
    openclip_source_layers: int = 24
    runtime_text_layers: int = 23
    runtime_assets_subdir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "display_name": self.display_name,
            "prediction_type": self.prediction_type,
            "native_sample_size": self.native_sample_size,
            "conditioning_dimension": self.conditioning_dimension,
            "openclip_source_layers": self.openclip_source_layers,
            "runtime_text_layers": self.runtime_text_layers,
            "runtime_assets_subdir": self.runtime_assets_subdir,
        }


SD2_1_BASE_512 = SD2RuntimeProfile(
    profile_id="sd2.1-base-512",
    display_name="Stable Diffusion 2.1 Base 512",
    prediction_type="epsilon",
    native_sample_size=64,
    runtime_assets_subdir="stable_diffusion/sd2_1_base",
)

SD2_1_768_V = SD2RuntimeProfile(
    profile_id="sd2.1-768-v",
    display_name="Stable Diffusion 2.1 768-v",
    prediction_type="v-prediction",
    native_sample_size=96,
    runtime_assets_subdir="stable_diffusion/sd2_1_768",
)

SD2_0_BASE_512 = SD2RuntimeProfile(
    profile_id="sd2.0-base-512",
    display_name="Stable Diffusion 2.0 Base 512",
    prediction_type="epsilon",
    native_sample_size=64,
    runtime_assets_subdir="stable_diffusion/sd2_0_base",
)

SD2_0_768_V = SD2RuntimeProfile(
    profile_id="sd2.0-768-v",
    display_name="Stable Diffusion 2.0 768-v",
    prediction_type="v-prediction",
    native_sample_size=96,
    runtime_assets_subdir="stable_diffusion/sd2_0_768",
)


_PROFILES_BY_ID = {
    profile.profile_id: profile
    for profile in (SD2_1_BASE_512, SD2_1_768_V, SD2_0_BASE_512, SD2_0_768_V)
}


def profile_from_id(profile_id: str | None) -> SD2RuntimeProfile | None:
    return _PROFILES_BY_ID.get(str(profile_id or "").strip().lower())


_FILENAME_RULES: tuple[tuple[re.Pattern[str], SD2RuntimeProfile], ...] = (
    (re.compile(r"(?:^|[_-])v?2[-_.]?1[_-]?512(?:[_-]|$)", re.I), SD2_1_BASE_512),
    (re.compile(r"(?:^|[_-])v?2[-_.]?1[_-]?768(?:[_-]|$)", re.I), SD2_1_768_V),
    (re.compile(r"(?:^|[_-])512[-_]?base(?:[_-]|$)", re.I), SD2_0_BASE_512),
    (re.compile(r"(?:^|[_-])768[-_]?v(?:[_-]|$)", re.I), SD2_0_768_V),
)


def profile_from_filename(filename: str | None) -> SD2RuntimeProfile | None:
    stem = str(filename or "").strip()
    for pattern, profile in _FILENAME_RULES:
        if pattern.search(stem):
            return profile
    return None


def prediction_from_sd2_evidence(
    *,
    filename: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    scheduler_prediction_type: Any = None,
) -> tuple[str, str, SD2RuntimeProfile | None]:
    explicit_scheduler = normalize_prediction_type(scheduler_prediction_type)
    if explicit_scheduler:
        return explicit_scheduler, "scheduler_config", profile_from_filename(filename)

    source = dict(metadata or {})
    for key in (
        "modelspec.prediction_type",
        "prediction_type",
        "parameterization",
        "model.parameterization",
        "ss_prediction_type",
        "ss_parameterization",
    ):
        if key not in source:
            continue
        normalized = normalize_prediction_type(source.get(key))
        if normalized:
            return normalized, "metadata", profile_from_filename(filename)

    profile = profile_from_filename(filename)
    if profile is not None:
        # A filename may identify which runtime assets to try, but it is not
        # authoritative evidence of the checkpoint training target.
        return "", "filename_profile_hint", profile
    return "", "", None
