from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.checkpoint_inspector import CheckpointInspector
from modules.model_qualification_registry import qualification_for_sha256
from modules.registry.component_selection import canonical_model_family
from modules.sd2_runtime_profile import profile_from_filename, profile_from_id
from modules.sd2_runtime_assets import SD2RuntimeAssetResolver
from modules.sdxl_runtime_assets import SDXLRuntimeAssetResolver
from modules.sdxl_model_contract import resolve_sdxl_model_contract
from modules.sdxl_runtime_profile import apply_sdxl_profile_to_request
from modules.sd3_model_contract import resolve_sd3_model_contract
from modules.sd3_runtime_profile import apply_sd3_profile_to_request

from image_gen.contracts import GenerationRequest


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return None

def _advanced_model_family(extras: dict[str, Any]) -> str:
    if not bool(extras.get("advanced_models_enabled")):
        return ""
    resolved = dict(extras.get("_advanced_model_resolved") or {})
    return canonical_model_family(
        resolved.get("family")
        or resolved.get("family_id")
        or extras.get("advanced_model_family")
    )


class ModelPreflightMixin:
    def _resolve_sd2_profile_hint(self, checkpoint_path: str | Path | None):
        """Resolve an SD2 runtime profile for pre-load assets without family guessing.

        Canonical filenames remain a cheap profile hint. If the user renamed a
        qualified checkpoint, fall back to the exact SHA-256 qualification
        registry. The result is cached for this runner so tokenizer identity
        checks do not hash the same large checkpoint repeatedly.
        """
        path = Path(str(checkpoint_path or "")).expanduser()
        profile = profile_from_filename(path.name)
        if profile is not None:
            return profile

        if path.is_file() and path.suffix.lower() == ".safetensors":
            try:
                stat = path.stat()
                cache_key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
                cache = getattr(self, "_sd2_profile_hint_cache", None)
                if cache is None:
                    cache = {}
                    setattr(self, "_sd2_profile_hint_cache", cache)
                if cache_key in cache:
                    return cache[cache_key]

                digest = ""
                registry = getattr(self.model_loader, "asset_registry", None)
                if registry is not None:
                    try:
                        asset_record = registry.get_asset_by_path(str(path.resolve()))
                    except Exception:
                        asset_record = None
                    digest = str(getattr(asset_record, "sha256", "") or "").strip().lower()
                if not digest:
                    digest = CheckpointInspector.sha256_file(path)

                qualification = qualification_for_sha256(digest)
                profile = (
                    profile_from_id(qualification.profile_id)
                    if qualification is not None
                    else None
                )
                cache[cache_key] = profile
                if profile is not None:
                    return profile
            except OSError:
                pass

        configured_profile = str(
            ((self.project_context.config.get("defaults") or {}).get("sd2_runtime_profile") or "")
        ).strip()
        return profile_from_id(configured_profile) if configured_profile else None

    def _build_local_tokenizer(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        checkpoint_family: str = "",
    ):
        family = str(checkpoint_family or "").strip().lower()
        local_dir = self.project_context.tokenizer_root
        identity = f"sd1:{local_dir.resolve()}"

        if "sd 2" in family or family.startswith("sd2") or "stable-diffusion-2" in family:
            profile = self._resolve_sd2_profile_hint(checkpoint_path)
            if profile is None:
                raise ValueError(
                    "SD2 tokenizer profile is unresolved for this checkpoint. IMAGE_GEN will not reuse the SD1 "
                    "tokenizer silently; select/declare a qualified SD2 runtime profile first."
                )
            assets = SD2RuntimeAssetResolver(self.project_context).resolve(profile)
            local_dir = assets.tokenizer_dir
            identity = f"{profile.profile_id}:{local_dir.resolve()}"
        elif family == "sdxl" or "stable-diffusion-xl" in family:
            assets = SDXLRuntimeAssetResolver(self.project_context).resolve()
            local_dir = assets.tokenizer_dir
            identity = f"sdxl:{local_dir.resolve()}"

        if not local_dir.exists():
            raise FileNotFoundError(f"Missing local tokenizer directory: {local_dir}")
        from transformers import CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(str(local_dir), local_files_only=True)
        return tokenizer, identity

    def _apply_sdxl_runtime_preflight(
        self,
        request: GenerationRequest,
        extras: dict[str, Any],
    ) -> None:
        if bool(extras.get("_sdxl_profile_preflight_complete")):
            return
        preflight_model_path = extras.get("model_path") or self.model_loading_system.default_model_path
        if not preflight_model_path:
            return
        candidate = Path(str(preflight_model_path)).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_context.resolve_project_path(candidate)
        if not candidate.is_file() or candidate.suffix.lower() != ".safetensors":
            return
        try:
            advanced_family = _advanced_model_family(extras)
            if advanced_family:
                if advanced_family != "sdxl":
                    extras["_sdxl_profile_preflight_complete"] = True
                    return
                # Advanced Models has already resolved the family and exact component
                # fingerprints through the component registry. Re-opening the donor
                # checkpoint here only to rediscover "sdxl" is both redundant and,
                # on Windows under memory pressure, can reserve enough mapped-file
                # commit to fail before selective component hydration begins.
                extras["checkpoint_preflight_architecture"] = {
                    "family": "sdxl",
                    "source": "advanced_model_registry_resolution",
                    "checkpoint_reinspection_skipped": True,
                }
            else:
                preflight_contract = CheckpointInspector().inspect_architecture_contract(str(candidate))
                if str(preflight_contract.family or "").strip().lower() != "sdxl":
                    extras["_sdxl_profile_preflight_complete"] = True
                    return
            explicit_profile = str(extras.get("sdxl_runtime_profile_override") or "").strip() or None
            sdxl_contract = resolve_sdxl_model_contract(
                self.project_context,
                checkpoint_filename=candidate.name,
                explicit_profile_id=explicit_profile,
            )
            application = apply_sdxl_profile_to_request(
                request,
                sdxl_contract.profile,
                enforce_steps=_optional_bool(
                    extras.get("model_enforce_recommended_steps", extras.get("sdxl_enforce_recommended_steps"))
                ),
                enforce_cfg=_optional_bool(
                    extras.get("model_enforce_recommended_cfg", extras.get("sdxl_enforce_recommended_cfg"))
                ),
            )
            extras["sdxl_runtime_profile_override"] = sdxl_contract.profile.profile_id
            extras["model_runtime_profile"] = sdxl_contract.profile.to_dict()
            extras["sdxl_profile_application"] = application
            scheduler_kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})
            if str(request.scheduler_name or "").strip().lower() == "sdxl_euler_trailing":
                scheduler_kwargs.setdefault(
                    "scheduler_config_path", str(sdxl_contract.assets.scheduler_config)
                )
            request.scheduler_kwargs = scheduler_kwargs
            # Profiles no longer force sampler/scheduler choices. Only discard
            # already-resolved descriptors if an explicit future request mutation
            # actually changes one of those selections.
            before_selection = dict(application.get("before") or {})
            after_selection = dict(application.get("after") or {})
            if (
                before_selection.get("sampler_name") != after_selection.get("sampler_name")
                or before_selection.get("scheduler_name") != after_selection.get("scheduler_name")
            ):
                for key in (
                    "resolved_scheduler_entry",
                    "resolved_scheduler_descriptor",
                    "resolved_sampler_entry",
                    "resolved_sampler_descriptor",
                ):
                    extras.pop(key, None)
            extras["_sdxl_profile_preflight_complete"] = True
        except Exception as exc:
            extras["sdxl_profile_preflight_error"] = f"{type(exc).__name__}: {exc}"
            raise

    def _apply_sd3_runtime_preflight(
        self,
        request: GenerationRequest,
        extras: dict[str, Any],
    ) -> None:
        if bool(extras.get("_sd3_profile_preflight_complete")):
            return
        preflight_model_path = extras.get("model_path") or self.model_loading_system.default_model_path
        if not preflight_model_path:
            return
        candidate = Path(str(preflight_model_path)).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_context.resolve_project_path(candidate)
        if not candidate.is_file() or candidate.suffix.lower() != ".safetensors":
            return
        try:
            report = CheckpointInspector().inspect(str(candidate), compute_sha256=False)
            if str(report.architecture or "").strip().lower() != "sd3.x":
                extras["_sd3_profile_preflight_complete"] = True
                return
            explicit_profile = str(extras.get("sd3_runtime_profile_override") or "").strip() or None
            sd3_contract = resolve_sd3_model_contract(
                self.project_context,
                checkpoint_variant=report.architecture_variant,
                explicit_profile_id=explicit_profile,
            )
            generic_steps = _optional_bool(
                extras.get("model_enforce_recommended_steps", extras.get("sdxl_enforce_recommended_steps"))
            )
            generic_cfg = _optional_bool(
                extras.get("model_enforce_recommended_cfg", extras.get("sdxl_enforce_recommended_cfg"))
            )
            application = apply_sd3_profile_to_request(
                request,
                sd3_contract.profile,
                enforce_steps=generic_steps,
                enforce_cfg=generic_cfg,
            )
            extras["sd3_runtime_profile_override"] = sd3_contract.profile.profile_id
            extras["model_runtime_profile"] = sd3_contract.profile.to_dict()
            extras["sd3_profile_application"] = application
            extras.setdefault("sd3_text_encoder_source", "auto")
            scheduler_kwargs = dict(getattr(request, "scheduler_kwargs", {}) or {})
            if str(request.scheduler_name or "").strip().lower() == "flow_match_euler":
                scheduler_kwargs.setdefault(
                    "scheduler_config_path", str(sd3_contract.assets.scheduler_config)
                )
            request.scheduler_kwargs = scheduler_kwargs
            extras["_sd3_profile_preflight_complete"] = True
        except Exception as exc:
            extras["sd3_profile_preflight_error"] = f"{type(exc).__name__}: {exc}"
            raise
