from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml


CANONICAL_CONFIG_RELATIVE_PATH = Path("user_config") / "user-config.yml"


class ProjectConfigurationError(RuntimeError):
    """Raised when the canonical project configuration cannot be loaded."""


class ProjectValidationError(RuntimeError):
    """Raised when required runtime paths are not ready for generation."""

    def __init__(self, report: "ProjectValidationReport") -> None:
        self.report = report
        super().__init__(report.format_text())


@dataclass(frozen=True)
class ProjectPathIssue:
    severity: str
    code: str
    label: str
    path: Path | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "label": self.label,
            "path": str(self.path) if self.path is not None else None,
            "message": self.message,
        }


@dataclass(frozen=True)
class ProjectValidationReport:
    project_root: Path
    config_path: Path
    issues: tuple[ProjectPathIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ProjectPathIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    @property
    def warnings(self) -> tuple[ProjectPathIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "warning")

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.is_valid,
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def format_text(self) -> str:
        status = "PASS" if self.is_valid else "FAIL"
        lines = [
            f"Project configuration validation: {status}",
            f"Project root: {self.project_root}",
            f"Config file:  {self.config_path}",
        ]
        if not self.issues:
            lines.append("No path or configuration issues detected.")
            return "\n".join(lines)

        for issue in self.issues:
            path_text = f" ({issue.path})" if issue.path is not None else ""
            lines.append(
                f"[{issue.severity.upper()}] {issue.code}: "
                f"{issue.label}{path_text} - {issue.message}"
            )
        return "\n".join(lines)


_DEFAULT_PATHS: dict[str, str] = {
    "models_root": "models/StableDiffusion",
    "checkpoints_dir": "models/StableDiffusion/CheckPoints",
    "vae_dir": "models/StableDiffusion/VAE",
    "vae_approx_dir": "models/StableDiffusion/VAE_approx",
    "lora_dir": "models/StableDiffusion/Lora",
    "blip_dir": "models/StableDiffusion/BLIP",
    "codeformer_dir": "models/StableDiffusion/Codeformer",
    "esrgan_dir": "models/StableDiffusion/ESRGAN",
    "gfpgan_dir": "models/StableDiffusion/GFPGAN",
    "realesrgan_dir": "models/StableDiffusion/RealESRGAN",
    "controlnet_dir": "models/StableDiffusion/ControlNet",
    "embeddings_dir": "models/Embeddings",
    "hypernetworks_dir": "models/StableDiffusion/Hypernetworks",
    "local_config_dir": "modules/local_configs",
    "tokenizer_dir": "modules/local_tokenizers/clip-vit-large-patch14",
    "data_dir": "data",
    "registry_db_path": "data/asset_registry.db",
    "output_dir": "output",
    "txt2img_output_dir": "output/txt2image",
    "cache_dir": "data/cache",
    "temporary_dir": "data/tmp",
    "diagnostics_dir": "artifacts/diagnostics",
}


def _deep_copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, item in value.items():
        if isinstance(item, Mapping):
            copied[str(key)] = _deep_copy_mapping(item)
        elif isinstance(item, list):
            copied[str(key)] = list(item)
        else:
            copied[str(key)] = item
    return copied


def _resolve_path(value: str | os.PathLike[str], project_root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


@dataclass(frozen=True)
class ProjectContext:
    """Authoritative project, configuration, and runtime path context.

    Construction is read-only: it never creates model, data, output, cache, or
    temporary directories. Runtime systems receive this context rather than
    inferring paths from the process working directory.
    """

    project_root: Path
    config_path: Path
    config: dict[str, Any]

    modules_root: Path
    registry_root: Path
    models_root: Path
    checkpoints_dir: Path
    vae_dir: Path
    vae_approx_dir: Path
    lora_dir: Path
    blip_dir: Path
    codeformer_dir: Path
    esrgan_dir: Path
    gfpgan_dir: Path
    realesrgan_dir: Path
    controlnet_dir: Path
    embeddings_dir: Path
    hypernetworks_dir: Path
    local_config_dir: Path
    tokenizer_root: Path
    data_root: Path
    registry_db_path: Path
    output_root: Path
    txt2img_output_root: Path
    cache_root: Path
    temporary_root: Path
    diagnostics_root: Path
    default_model_path: Path | None

    @classmethod
    def default_project_root(cls) -> Path:
        return Path(__file__).resolve().parents[1]

    @classmethod
    def load(
        cls,
        *,
        project_root: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
    ) -> "ProjectContext":
        root = (
            Path(project_root).expanduser().resolve()
            if project_root is not None
            else cls.default_project_root().resolve()
        )
        selected_config = Path(config_path).expanduser() if config_path is not None else CANONICAL_CONFIG_RELATIVE_PATH
        if not selected_config.is_absolute():
            selected_config = root / selected_config
        selected_config = selected_config.resolve()

        if not selected_config.is_file():
            raise ProjectConfigurationError(
                "Canonical project configuration file not found: "
                f"{selected_config}. Expected {CANONICAL_CONFIG_RELATIVE_PATH.as_posix()} "
                "under the project root unless --project-config is supplied."
            )

        try:
            loaded = yaml.safe_load(selected_config.read_text(encoding="utf-8-sig")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ProjectConfigurationError(
                f"Unable to load project configuration {selected_config}: {exc}"
            ) from exc
        if not isinstance(loaded, Mapping):
            raise ProjectConfigurationError(
                f"Project configuration must be a YAML mapping: {selected_config}"
            )

        config = _deep_copy_mapping(loaded)
        configured_paths = config.get("paths") or {}
        if not isinstance(configured_paths, Mapping):
            raise ProjectConfigurationError("The top-level 'paths' section must be a mapping.")
        path_values = dict(_DEFAULT_PATHS)
        for key, value in configured_paths.items():
            if value is not None and str(value).strip():
                path_values[str(key)] = str(value)

        resolved = {key: _resolve_path(value, root) for key, value in path_values.items()}

        defaults = config.get("defaults") or {}
        if not isinstance(defaults, Mapping):
            raise ProjectConfigurationError("The top-level 'defaults' section must be a mapping.")
        raw_default_model = defaults.get("model_path")
        default_model_path = (
            _resolve_path(str(raw_default_model), root)
            if raw_default_model is not None and str(raw_default_model).strip()
            else None
        )

        return cls(
            project_root=root,
            config_path=selected_config,
            config=config,
            modules_root=(root / "modules").resolve(),
            registry_root=(root / "modules" / "ss_registry").resolve(),
            models_root=resolved["models_root"],
            checkpoints_dir=resolved["checkpoints_dir"],
            vae_dir=resolved["vae_dir"],
            vae_approx_dir=resolved["vae_approx_dir"],
            lora_dir=resolved["lora_dir"],
            blip_dir=resolved["blip_dir"],
            codeformer_dir=resolved["codeformer_dir"],
            esrgan_dir=resolved["esrgan_dir"],
            gfpgan_dir=resolved["gfpgan_dir"],
            realesrgan_dir=resolved["realesrgan_dir"],
            controlnet_dir=resolved["controlnet_dir"],
            embeddings_dir=resolved["embeddings_dir"],
            hypernetworks_dir=resolved["hypernetworks_dir"],
            local_config_dir=resolved["local_config_dir"],
            tokenizer_root=resolved["tokenizer_dir"],
            data_root=resolved["data_dir"],
            registry_db_path=resolved["registry_db_path"],
            output_root=resolved["output_dir"],
            txt2img_output_root=resolved["txt2img_output_dir"],
            cache_root=resolved["cache_dir"],
            temporary_root=resolved["temporary_dir"],
            diagnostics_root=resolved["diagnostics_dir"],
            default_model_path=default_model_path,
        )

    def resolve_project_path(self, value: str | os.PathLike[str]) -> Path:
        return _resolve_path(value, self.project_root)

    def generation_defaults(self) -> dict[str, Any]:
        generation = self.config.get("generation") or {}
        defaults = self.config.get("defaults") or {}
        if not isinstance(generation, Mapping) or not isinstance(defaults, Mapping):
            raise ProjectConfigurationError("'generation' and 'defaults' must be mappings.")

        payload: dict[str, Any] = {
            "positive_prompt": generation.get("prompt", ""),
            "negative_prompt": generation.get("negative_prompt", ""),
            "seed": generation.get("seed"),
            "steps": generation.get("steps"),
            "cfg_scale": generation.get("cfg_scale"),
            "width": generation.get("width"),
            "height": generation.get("height"),
            "sampler_name": generation.get("sampler"),
            "scheduler_name": generation.get("scheduler"),
            "batch_size": generation.get("batch_size"),
            "batch_count": generation.get("batch_count", 1),
            "unlimited": bool(generation.get("unlimited", False)),
            "model_path": str(self.default_model_path) if self.default_model_path else None,
            "vae_path": defaults.get("vae_path"),
            "lora_paths": defaults.get("lora_paths", []),
            # User-facing generation saves by default. Library callers can still
            # construct GenerationRequest(save_images=False), and the CLI keeps
            # the explicit --no-save escape hatch.
            "save_images": bool(generation.get("save_images", True)),
            "output_dir": str(self.txt2img_output_root),
            "output_prefix": str(
                generation.get(
                    "filename_pattern",
                    generation.get("output_prefix", "{index:05d}-{seed}"),
                )
                or "{index:05d}-{seed}"
            ),
            "scheduler_kwargs": dict(generation.get("scheduler_kwargs") or {}),
            "sampler_kwargs": dict(generation.get("sampler_kwargs") or {}),
            "prompt_parser_name": str(generation.get("prompt_parser") or "legacy"),
            "prompt_parser_kwargs": dict(generation.get("prompt_parser_kwargs") or {}),
            "prompt_shortcut_profile_name": str(generation.get("prompt_shortcut_profile") or ("legacy_default" if str(generation.get("prompt_parser") or "legacy") == "legacy" else "parser21_native")),
            "prompt_parser_preset_name": str(generation.get("prompt_parser_preset") or ""),
            "base_prompt_parser_name": str(generation.get("prompt_parser") or "legacy"),
            "base_shortcut_profile_name": str(generation.get("prompt_shortcut_profile") or ("legacy_default" if str(generation.get("prompt_parser") or "legacy") == "legacy" else "parser21_native")),
            "hires_prompt_parser_mode": str(generation.get("hires_prompt_parser_mode") or "same_as_base"),
            "hires_prompt_parser_name": str(generation.get("hires_prompt_parser") or generation.get("prompt_parser") or "legacy"),
            "hires_prompt_parser_kwargs": dict(generation.get("hires_prompt_parser_kwargs") or {}),
            "hires_shortcut_profile_mode": str(generation.get("hires_shortcut_profile_mode") or "same_as_base"),
            "hires_shortcut_profile_name": str(generation.get("hires_shortcut_profile") or generation.get("prompt_shortcut_profile") or "legacy_default"),
            "hires_shortcut_profile_snapshot": dict(generation.get("hires_shortcut_profile_snapshot") or {}),
            "hires_enabled": bool(generation.get("hires_enabled", False)),
            "hires_positive_prompt": str(generation.get("hires_positive_prompt") or ""),
            "hires_negative_prompt": str(generation.get("hires_negative_prompt") or ""),
            "hires_size_mode": str(generation.get("hires_size_mode") or "scale_from_base"),
            "hires_scale": float(generation.get("hires_scale") or 1.5),
            "hires_width": int(generation.get("hires_width") or 0),
            "hires_height": int(generation.get("hires_height") or 0),
            "hires_steps": int(generation.get("hires_steps") or 20),
            "hires_denoising_strength": float(generation.get("hires_denoising_strength") or 0.4),
            "hires_upscaler": str(generation.get("hires_upscaler") or "latent_bicubic"),
            "hires_save_lowres": bool(generation.get("hires_save_lowres", True)),
            "parser_kwargs": dict(generation.get("parser_kwargs") or {}),
            "diagnostics": dict(self.config.get("diagnostics") or {}),
        }
        return {key: value for key, value in payload.items() if value is not None}

    def effective_config(self) -> dict[str, Any]:
        return {
            "project_root": str(self.project_root),
            "config_path": str(self.config_path),
            "paths": {
                "modules_root": str(self.modules_root),
                "registry_root": str(self.registry_root),
                "models_root": str(self.models_root),
                "checkpoints_dir": str(self.checkpoints_dir),
                "vae_dir": str(self.vae_dir),
                "vae_approx_dir": str(self.vae_approx_dir),
                "lora_dir": str(self.lora_dir),
                "blip_dir": str(self.blip_dir),
                "codeformer_dir": str(self.codeformer_dir),
                "esrgan_dir": str(self.esrgan_dir),
                "gfpgan_dir": str(self.gfpgan_dir),
                "realesrgan_dir": str(self.realesrgan_dir),
                "controlnet_dir": str(self.controlnet_dir),
                "embeddings_dir": str(self.embeddings_dir),
                "hypernetworks_dir": str(self.hypernetworks_dir),
                "local_config_dir": str(self.local_config_dir),
                "tokenizer_dir": str(self.tokenizer_root),
                "data_dir": str(self.data_root),
                "registry_db_path": str(self.registry_db_path),
                "output_dir": str(self.output_root),
                "txt2img_output_dir": str(self.txt2img_output_root),
                "cache_dir": str(self.cache_root),
                "temporary_dir": str(self.temporary_root),
                "diagnostics_dir": str(self.diagnostics_root),
            },
            "defaults": dict(self.config.get("defaults") or {}),
            "generation": dict(self.config.get("generation") or {}),
            "registry": dict(self.config.get("registry") or {}),
            "loader": dict(self.config.get("loader") or {}),
            "diagnostics": dict(self.config.get("diagnostics") or {}),
        }

    def effective_config_json(self) -> str:
        return json.dumps(self.effective_config(), indent=2)

    def validate(
        self,
        *,
        for_generation: bool = True,
        model_path: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        require_output: bool = True,
    ) -> ProjectValidationReport:
        issues: list[ProjectPathIssue] = []

        def add(severity: str, code: str, label: str, path: Path | None, message: str) -> None:
            issues.append(ProjectPathIssue(severity, code, label, path, message))

        def require_dir(code: str, label: str, path: Path) -> None:
            if not path.exists():
                add("error", code, label, path, "directory does not exist")
            elif not path.is_dir():
                add("error", code, label, path, "path exists but is not a directory")

        if not self.project_root.is_dir():
            add("error", "PROJECT_ROOT_MISSING", "project root", self.project_root, "directory does not exist")
        if not self.config_path.is_file():
            add("error", "CONFIG_MISSING", "project config", self.config_path, "file does not exist")

        require_dir("MODULES_ROOT_MISSING", "modules root", self.modules_root)
        require_dir("LOCAL_CONFIG_MISSING", "local architecture config root", self.local_config_dir)
        require_dir("TOKENIZER_ROOT_MISSING", "local tokenizer root", self.tokenizer_root)
        if self.tokenizer_root.is_dir():
            for filename in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
                expected = self.tokenizer_root / filename
                if not expected.is_file():
                    add(
                        "error",
                        "TOKENIZER_FILE_MISSING",
                        f"tokenizer file {filename}",
                        expected,
                        "required local tokenizer asset is missing",
                    )

        if for_generation:
            selected_model = (
                self.resolve_project_path(model_path)
                if model_path is not None and str(model_path).strip()
                else self.default_model_path
            )
            if selected_model is None:
                add("error", "MODEL_NOT_CONFIGURED", "model path", None, "no model path was supplied or configured")
            elif not selected_model.is_file():
                add("error", "MODEL_FILE_MISSING", "model file", selected_model, "configured model file does not exist")

            if selected_model is None or self.checkpoints_dir in selected_model.parents:
                require_dir("CHECKPOINT_ROOT_MISSING", "checkpoint root", self.checkpoints_dir)
            elif not self.checkpoints_dir.exists():
                add(
                    "warning",
                    "CHECKPOINT_ROOT_MISSING",
                    "checkpoint root",
                    self.checkpoints_dir,
                    "directory does not exist, but an external model path was supplied",
                )

            if require_output:
                selected_output = (
                    self.resolve_project_path(output_dir)
                    if output_dir is not None and str(output_dir).strip()
                    else self.txt2img_output_root
                )
                require_dir("OUTPUT_ROOT_MISSING", "txt2img output root", selected_output)

        for code, label, path in (
            ("DATA_ROOT_MISSING", "data root", self.data_root),
            ("CACHE_ROOT_MISSING", "cache root", self.cache_root),
            ("TEMP_ROOT_MISSING", "temporary root", self.temporary_root),
        ):
            if not path.exists():
                add(
                    "warning",
                    code,
                    label,
                    path,
                    "directory does not exist; create it before the subsystem that owns it is used",
                )
            elif not path.is_dir():
                add("error", code, label, path, "path exists but is not a directory")

        return ProjectValidationReport(
            project_root=self.project_root,
            config_path=self.config_path,
            issues=tuple(issues),
        )

    def require_generation_ready(
        self,
        *,
        model_path: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        require_output: bool = True,
    ) -> ProjectValidationReport:
        report = self.validate(
            for_generation=True,
            model_path=model_path,
            output_dir=output_dir,
            require_output=require_output,
        )
        if not report.is_valid:
            raise ProjectValidationError(report)
        return report
