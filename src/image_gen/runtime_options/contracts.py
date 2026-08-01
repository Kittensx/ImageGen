from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disabled", ""}:
        return False
    raise ValueError(f"Expected a boolean value, received {value!r}.")


@dataclass(frozen=True)
class MSLKFMHAOptions:
    """Process-start MSLK FMHA environment values.

    Empty strings are intentional values for the MSLK variables that permit the
    installed backend to choose its own default.  They must not be collapsed to
    ``None`` during startup normalization.
    """

    policy: str = "blackwell_safe"
    debug: str = ""
    block_n: str = ""
    block_m: str = ""
    num_warps: str = ""
    num_stages: str = ""
    experimental_head_dims: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "policy": str(self.policy),
            "debug": str(self.debug),
            "block_n": str(self.block_n),
            "block_m": str(self.block_m),
            "num_warps": str(self.num_warps),
            "num_stages": str(self.num_stages),
            "experimental_head_dims": str(self.experimental_head_dims),
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "MSLKFMHAOptions":
        source = dict(values or {})
        defaults = cls()
        return cls(
            policy=str(source.get("policy", defaults.policy)),
            debug=str(source.get("debug", defaults.debug)),
            block_n=str(source.get("block_n", defaults.block_n)),
            block_m=str(source.get("block_m", defaults.block_m)),
            num_warps=str(source.get("num_warps", defaults.num_warps)),
            num_stages=str(source.get("num_stages", defaults.num_stages)),
            experimental_head_dims=str(
                source.get("experimental_head_dims", defaults.experimental_head_dims)
            ),
        )


@dataclass(frozen=True)
class RuntimeProfileSelection:
    """Identity and provenance for the profile template applied at startup."""

    profile_id: str = "auto"
    label: str = "Automatic"
    schema_version: int = 1
    source: str = "builtin"
    selector: str = "auto"
    selected_from: str = "default"
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile_id": str(self.profile_id),
            "label": str(self.label),
            "schema_version": int(self.schema_version),
            "source": str(self.source),
            "selector": str(self.selector),
            "selected_from": str(self.selected_from),
            "notes": list(self.notes),
        }

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "RuntimeProfileSelection":
        source = dict(values or {})
        defaults = cls()
        notes = source.get("notes", defaults.notes)
        if isinstance(notes, str):
            notes = (notes,)
        return cls(
            profile_id=str(source.get("profile_id", defaults.profile_id)),
            label=str(source.get("label", defaults.label)),
            schema_version=int(source.get("schema_version", defaults.schema_version)),
            source=str(source.get("source", defaults.source)),
            selector=str(source.get("selector", defaults.selector)),
            selected_from=str(source.get("selected_from", defaults.selected_from)),
            notes=tuple(str(item) for item in (notes or ())),
        )


@dataclass(frozen=True)
class RuntimeStartupOptions:
    """Normalized process and per-job runtime startup contract.

    Phase 14K subphases add the user-facing switches that populate this object.
    Phase 14K-1 establishes the shared schema and source-precedence behavior so
    CLI, WebUI, and resident workers do not grow separate settings models.
    """

    schema_version: int = 2
    runtime_profile: RuntimeProfileSelection = field(default_factory=RuntimeProfileSelection)
    attention_backend: str = "auto"
    memory_policy: str = "auto"
    vram_safety_margin_mb: int = 1024
    attention_slicing: str = "off"
    vae_tiling: bool = False
    vae_slicing: bool = False
    vae_device: str = "auto"
    retain_unet_between_jobs: bool = True
    retain_vae_between_jobs: bool = True
    retain_text_encoder_between_jobs: bool = True
    preview_policy: str = "normal"
    hires_memory_profile: str = "inherit"
    pre_hires_cleanup: bool = False
    oom_retry_profile: str = "cleanup"
    oom_retry_limit: int = 1
    mslk_fmha: MSLKFMHAOptions = field(default_factory=MSLKFMHAOptions)
    allocator_options: dict[str, str] = field(default_factory=dict)
    source_map: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "runtime_profile": self.runtime_profile.to_dict(),
            "attention_backend": str(self.attention_backend),
            "memory_policy": str(self.memory_policy),
            "vram_safety_margin_mb": int(self.vram_safety_margin_mb),
            "attention_slicing": str(self.attention_slicing),
            "vae_tiling": bool(self.vae_tiling),
            "vae_slicing": bool(self.vae_slicing),
            "vae_device": str(self.vae_device),
            "retain_unet_between_jobs": bool(self.retain_unet_between_jobs),
            "retain_vae_between_jobs": bool(self.retain_vae_between_jobs),
            "retain_text_encoder_between_jobs": bool(
                self.retain_text_encoder_between_jobs
            ),
            "preview_policy": str(self.preview_policy),
            "hires_memory_profile": str(self.hires_memory_profile),
            "pre_hires_cleanup": bool(self.pre_hires_cleanup),
            "oom_retry_profile": str(self.oom_retry_profile),
            "oom_retry_limit": int(self.oom_retry_limit),
            "mslk_fmha": self.mslk_fmha.to_dict(),
            "allocator_options": dict(self.allocator_options),
            "source_map": dict(self.source_map),
        }

    @classmethod
    def from_mapping(
        cls, values: Mapping[str, Any] | None
    ) -> "RuntimeStartupOptions":
        source = dict(values or {})
        defaults = cls()
        return cls(
            schema_version=int(source.get("schema_version", defaults.schema_version)),
            runtime_profile=RuntimeProfileSelection.from_mapping(
                source.get("runtime_profile")
            ),
            attention_backend=str(
                source.get("attention_backend", defaults.attention_backend)
            ),
            memory_policy=str(source.get("memory_policy", defaults.memory_policy)),
            vram_safety_margin_mb=int(
                source.get(
                    "vram_safety_margin_mb", defaults.vram_safety_margin_mb
                )
            ),
            attention_slicing=str(
                source.get("attention_slicing", defaults.attention_slicing)
            ),
            vae_tiling=_as_bool(source.get("vae_tiling"), defaults.vae_tiling),
            vae_slicing=_as_bool(source.get("vae_slicing"), defaults.vae_slicing),
            vae_device=str(source.get("vae_device", defaults.vae_device)),
            retain_unet_between_jobs=_as_bool(
                source.get("retain_unet_between_jobs"),
                defaults.retain_unet_between_jobs,
            ),
            retain_vae_between_jobs=_as_bool(
                source.get("retain_vae_between_jobs"),
                defaults.retain_vae_between_jobs,
            ),
            retain_text_encoder_between_jobs=_as_bool(
                source.get("retain_text_encoder_between_jobs"),
                defaults.retain_text_encoder_between_jobs,
            ),
            preview_policy=str(
                source.get("preview_policy", defaults.preview_policy)
            ),
            hires_memory_profile=str(
                source.get(
                    "hires_memory_profile", defaults.hires_memory_profile
                )
            ),
            pre_hires_cleanup=_as_bool(
                source.get("pre_hires_cleanup"), defaults.pre_hires_cleanup
            ),
            oom_retry_profile=str(
                source.get("oom_retry_profile", defaults.oom_retry_profile)
            ),
            oom_retry_limit=int(
                source.get("oom_retry_limit", defaults.oom_retry_limit)
            ),
            mslk_fmha=MSLKFMHAOptions.from_mapping(source.get("mslk_fmha")),
            allocator_options={
                str(key): str(value)
                for key, value in dict(source.get("allocator_options") or {}).items()
            },
            source_map={
                str(key): str(value)
                for key, value in dict(source.get("source_map") or {}).items()
            },
        )
