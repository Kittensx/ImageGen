from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from image_gen.systems.diagnostics.serialization import json_safe
from image_gen.systems.prompt_expansion import (
    SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION,
    select_prompt_expansion_slot,
)
from image_gen.systems.regional_prompting import (
    REGION_CONTRACT_VERSION,
    select_region_record_slot,
)
from modules.prompt_parsers.semantic_replay import (
    SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION,
    select_superhybrid_semantic_slot,
)
from modules.txt2img.generation_manifest import GenerationManifest
from modules.txt2img.manifest_io import save_manifest_json, save_manifest_txt
from modules.txt2img.png_metadata import build_pnginfo
from modules.txt2img.seed_utils import offset_seed


DEFAULT_FILENAME_PATTERN = "{index:05d}-{seed}"
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_LEADING_INDEX = re.compile(r"^(\d{5,})(?:[-_]|$)")
_TRAILING_INDEX = re.compile(r"(?:^|[-_])(\d{5,})$")


@dataclass
class SavedImageRecord:
    image_path: str
    txt_path: str | None = None
    json_path: str | None = None
    index: int | None = None
    seed: int | None = None


class GenerationOutputSaver:
    """Centralized save helper for txt2img outputs.

    ``prefix`` remains backward compatible with legacy callers. A plain value
    such as ``img`` produces ``img_0000001``. A value containing format fields
    is treated as a filename template. The default user-facing template is
    ``{index:05d}-{seed}``.

    Supported fields include: ``index``, ``seed``, ``datetime``, ``date``,
    ``time``, ``model``, ``model_name``, ``vae``, ``vae_name``, ``lora``,
    ``lora_names``, ``sampler``, ``scheduler``, ``width``, ``height``, and
    ``prefix``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        prefix: str = DEFAULT_FILENAME_PATTERN,
        image_ext: str = ".png",
    ):
        self.output_dir = Path(output_dir)
        self.prefix = self._clean_prefix(prefix)
        self.image_ext = self._normalize_ext(image_ext)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _clean_prefix(prefix: str) -> str:
        value = str(prefix or DEFAULT_FILENAME_PATTERN).strip()
        return value or DEFAULT_FILENAME_PATTERN

    @staticmethod
    def _normalize_ext(image_ext: str) -> str:
        value = str(image_ext or ".png").strip().lower()
        if not value.startswith("."):
            value = f".{value}"
        return value

    @staticmethod
    def tensor_to_pil_images(images: torch.Tensor) -> list[Image.Image]:
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW tensor, got shape {tuple(images.shape)}")

        batch = images.detach().float().cpu().clamp(0, 1)
        pil_images: list[Image.Image] = []
        for image in batch:
            image_hwc = image.permute(1, 2, 0)
            image_uint8 = (image_hwc * 255.0).round().to(torch.uint8).numpy()
            pil_images.append(Image.fromarray(image_uint8).convert("RGB"))
        return pil_images

    @staticmethod
    def _coerce_pil_images(images: torch.Tensor | Sequence[Image.Image]) -> list[Image.Image]:
        if torch.is_tensor(images):
            return GenerationOutputSaver.tensor_to_pil_images(images)

        output: list[Image.Image] = []
        for idx, image in enumerate(images):
            if not isinstance(image, Image.Image):
                raise TypeError(f"Item {idx} is not a PIL image: {type(image)}")
            output.append(image.convert("RGB"))
        return output

    @staticmethod
    def _safe_component(value: Any, fallback: str = "") -> str:
        text = str(value or fallback).strip()
        text = _INVALID_FILENAME_CHARS.sub("_", text)
        text = text.replace(".", "_")
        text = re.sub(r"\s+", "_", text)
        text = re.sub(r"_+", "_", text)
        return text.strip(" ._-")

    @staticmethod
    def _asset_label(asset: Any) -> str:
        if asset is None:
            return ""
        for name in (
            "resolved_display_name",
            "requested_display_name",
            "resolved_filename",
            "requested_filename",
            "resolved_path",
            "requested_path",
            "resolved_identifier",
            "requested_identifier",
        ):
            value = getattr(asset, name, "")
            if value:
                return Path(str(value)).stem
        return ""

    def _existing_indices(self) -> list[int]:
        found: set[int] = set()
        legacy_pattern = re.compile(
            rf"^{re.escape(self.prefix)}_(\d+)$"
        ) if "{" not in self.prefix else None

        for path in self.output_dir.glob(f"*{self.image_ext}"):
            stem = path.stem
            leading = _LEADING_INDEX.match(stem)
            if leading:
                found.add(int(leading.group(1)))
                continue
            trailing = _TRAILING_INDEX.search(stem)
            if trailing:
                found.add(int(trailing.group(1)))
                continue
            if legacy_pattern is not None:
                legacy = legacy_pattern.match(stem)
                if legacy:
                    found.add(int(legacy.group(1)))
        return sorted(found)

    def next_index(self) -> int:
        existing = self._existing_indices()
        if not existing:
            return 1
        return existing[-1] + 1

    @staticmethod
    def _manifest_copy(manifest: GenerationManifest) -> GenerationManifest:
        # Rehydrate from JSON-safe data rather than deepcopying runtime objects
        # such as mappingproxy, torch.device, or plugin descriptors.
        return GenerationManifest.from_dict(json_safe(manifest.to_dict()))

    @staticmethod
    def _project_prompt_expansion_for_image(
        manifest: GenerationManifest,
        image_offset: int,
    ) -> None:
        extra = manifest.optional_for_rerun.extra
        records_raw = extra.get("prompt_expansion_pass_records")
        if not isinstance(records_raw, dict):
            return

        records = dict(records_raw)
        projected: dict[str, Any] = {}
        used_per_image_record = False
        for pass_name, value in records.items():
            if not isinstance(value, dict):
                projected[str(pass_name)] = value
                continue
            if (
                value.get("contract_version")
                == SUPERHYBRID_EXPANSION_BATCH_CONTRACT_VERSION
                and int(value.get("slot_count", 0) or 0) > 1
            ):
                projected[str(pass_name)] = select_prompt_expansion_slot(value, image_offset)
                used_per_image_record = True
            else:
                projected[str(pass_name)] = value

        if not used_per_image_record:
            return

        extra["batch_prompt_expansion_pass_records"] = records
        extra["prompt_expansion_pass_records"] = projected

        semantic_records_raw = extra.get("prompt_semantic_pass_records")
        if isinstance(semantic_records_raw, dict):
            semantic_records = dict(semantic_records_raw)
            projected_semantics: dict[str, Any] = {}
            projected_any_semantics = False
            for pass_name, value in semantic_records.items():
                if (
                    isinstance(value, dict)
                    and value.get("contract_version")
                    == SUPERHYBRID_SEMANTIC_BATCH_CONTRACT_VERSION
                    and int(value.get("slot_count", 0) or 0) > 1
                ):
                    projected_semantics[str(pass_name)] = select_superhybrid_semantic_slot(
                        value, image_offset
                    )
                    projected_any_semantics = True
                else:
                    projected_semantics[str(pass_name)] = value
            if projected_any_semantics:
                extra["batch_prompt_semantic_pass_records"] = semantic_records
                extra["prompt_semantic_pass_records"] = projected_semantics

        for pass_name, record in projected.items():
            if not isinstance(record, dict):
                continue
            selection_seeds = list(record.get("selection_seeds") or [])
            if not selection_seeds:
                continue
            parser_kwargs_key = (
                "prompt_parser_kwargs" if str(pass_name) == "base" else "hires_prompt_parser_kwargs"
            )
            parser_kwargs = dict(extra.get(parser_kwargs_key) or {})
            parser_kwargs["seed"] = int(selection_seeds[0])
            extra[parser_kwargs_key] = parser_kwargs
        base_record = projected.get("base")
        if isinstance(base_record, dict):
            extra["prompt_expansion_record"] = base_record
            extra["prompt_expansion_contract_version"] = str(
                base_record.get("contract_version") or ""
            )
        extra["prompt_expansion_projected_image_slot"] = int(image_offset)
        manifest.required_for_rerun.batch_size = 1
        manifest.required_for_rerun.batch_count = 1

    @staticmethod
    def _project_region_runtime_record(record: dict[str, Any], image_offset: int) -> dict[str, Any]:
        selected = dict(record or {})
        regions = [
            dict(item or {})
            for item in list(selected.get("regions") or [])
            if int((item or {}).get("slot_index", 0) or 0) == int(image_offset)
        ]
        for item in regions:
            item["source_batch_slot_index"] = int(image_offset)
            item["slot_index"] = 0
        selected["regions"] = regions
        selected["source_batch_slot_index"] = int(image_offset)
        selected["regional_unet_calls"] = sum(int(item.get("unet_calls", 0) or 0) for item in regions)
        selected["regional_host_elapsed_ms"] = round(
            sum(float(item.get("host_elapsed_ms", item.get("duration_ms", 0.0)) or 0.0) for item in regions), 4
        )
        selected["regional_unet_duration_ms"] = selected["regional_host_elapsed_ms"]
        selected["active_region_instances"] = sum(
            int(item.get("active_step_count", 0) or 0) for item in regions
        )
        selected["steps_with_regions"] = sorted({
            int(step)
            for item in regions
            for step in list(item.get("active_steps") or [])
        })
        return selected

    @staticmethod
    def _project_regions_for_image(
        manifest: GenerationManifest,
        image_offset: int,
    ) -> None:
        extra = manifest.optional_for_rerun.extra
        records_raw = extra.get("region_pass_records")
        if not isinstance(records_raw, dict):
            return
        records = dict(records_raw)
        projected: dict[str, Any] = {}
        projected_any = False
        for pass_name, value in records.items():
            if (
                isinstance(value, dict)
                and value.get("contract_version") == REGION_CONTRACT_VERSION
                and int(value.get("slot_count", 0) or 0) > 1
            ):
                projected[str(pass_name)] = select_region_record_slot(value, image_offset)
                projected_any = True
            else:
                projected[str(pass_name)] = value
        if projected_any:
            extra["batch_region_pass_records"] = records
            extra["region_pass_records"] = projected
            extra["region_projected_image_slot"] = int(image_offset)

        runtime_raw = extra.get("regional_runtime_passes")
        if isinstance(runtime_raw, dict):
            runtime_records = dict(runtime_raw)
            projected_runtime = {
                str(pass_name): GenerationOutputSaver._project_region_runtime_record(
                    dict(value or {}), image_offset
                )
                if isinstance(value, dict) else value
                for pass_name, value in runtime_records.items()
            }
            extra["batch_regional_runtime_passes"] = runtime_records
            extra["regional_runtime_passes"] = projected_runtime
            preferred = projected_runtime.get("hires") or projected_runtime.get("base") or {}
            extra["regional_runtime"] = preferred

    @staticmethod
    def _resolve_image_seeds(
        manifest: GenerationManifest | None,
        image_count: int,
    ) -> list[int | None]:
        if manifest is None:
            return [None] * image_count

        raw = manifest.extra.get("resolved_seeds", [])
        if isinstance(raw, (list, tuple)) and len(raw) >= image_count:
            return [int(value) for value in raw[:image_count]]

        base_seed = int(manifest.required_for_rerun.seed)
        return [offset_seed(base_seed, index) for index in range(image_count)]

    def _template_values(
        self,
        *,
        index: int,
        seed: int | None,
        manifest: GenerationManifest | None,
        timestamp: datetime,
    ) -> dict[str, Any]:
        req = manifest.required_for_rerun if manifest is not None else None
        model_path = getattr(req, "model_path", "") if req is not None else ""
        model_name = Path(str(model_path)).stem if model_path else ""

        vae_name = self._asset_label(getattr(manifest, "vae", None)) if manifest else ""
        lora_names = []
        extra = dict(getattr(manifest, "extra", {}) or {}) if manifest is not None else {}
        if manifest is not None:
            lora_names = [
                self._asset_label(asset)
                for asset in getattr(manifest, "loras", [])
            ]
            lora_names = [name for name in lora_names if name]

        if not vae_name:
            vae_path = extra.get("vae_path") or extra.get("vae_name") or ""
            vae_name = Path(str(vae_path)).stem if vae_path else ""
        if not lora_names:
            raw_loras = extra.get("lora_paths") or extra.get("loras") or []
            if isinstance(raw_loras, str):
                raw_loras = [raw_loras]
            if isinstance(raw_loras, (list, tuple)):
                lora_names = [Path(str(value)).stem for value in raw_loras if value]

        legacy_prefix = self.prefix if "{" not in self.prefix else "img"
        values = {
            "index": int(index),
            "seed": "unknown" if seed is None else int(seed),
            "datetime": timestamp.strftime("%Y%m%d-%H%M%S-%f"),
            "date": timestamp.strftime("%Y%m%d"),
            "time": timestamp.strftime("%H%M%S-%f"),
            "model": self._safe_component(model_name, "model"),
            "model_name": self._safe_component(model_name, "model"),
            "vae": self._safe_component(vae_name, "vae"),
            "vae_name": self._safe_component(vae_name, "vae"),
            "lora": self._safe_component("+".join(lora_names), "no_lora"),
            "lora_names": self._safe_component("+".join(lora_names), "no_lora"),
            "sampler": self._safe_component(getattr(req, "sampler_name", ""), "sampler") if req else "sampler",
            "scheduler": self._safe_component(getattr(req, "scheduler_name", ""), "scheduler") if req else "scheduler",
            "width": int(getattr(req, "width", 0) or 0) if req else 0,
            "height": int(getattr(req, "height", 0) or 0) if req else 0,
            "prefix": self._safe_component(legacy_prefix, "img"),
            "prompt": self._safe_component(getattr(req, "prompt", ""), "prompt") if req else "prompt",
            "negative_prompt": self._safe_component(getattr(req, "negative_prompt", ""), "") if req else "",
            "steps": int(getattr(req, "steps", 0) or 0) if req else 0,
            "cfg_scale": getattr(req, "cfg_scale", 0) if req else 0,
        }
        for key, value in extra.items():
            if key in values or not str(key).isidentifier():
                continue
            if isinstance(value, (str, int, float, bool)):
                values[key] = self._safe_component(value) if isinstance(value, str) else value
            elif isinstance(value, (list, tuple)) and all(
                isinstance(item, (str, int, float, bool)) for item in value
            ):
                values[key] = self._safe_component("+".join(str(item) for item in value))
        return values

    def build_base_path(
        self,
        index: int,
        *,
        seed: int | None = None,
        manifest: GenerationManifest | None = None,
        timestamp: datetime | None = None,
    ) -> Path:
        timestamp = timestamp or datetime.now()
        if "{" in self.prefix:
            values = self._template_values(
                index=index,
                seed=seed,
                manifest=manifest,
                timestamp=timestamp,
            )
            try:
                filename = self.prefix.format_map(values)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"Invalid output filename pattern {self.prefix!r}: {exc}"
                ) from exc
        else:
            filename = f"{self.prefix}_{index:07d}"

        filename = self._safe_component(filename, f"{index:05d}-{seed}")
        if not filename:
            filename = f"{index:05d}-{seed}"
        return self.output_dir / filename

    def _avoid_collision(self, base_path: Path) -> Path:
        candidate = base_path
        suffix = 1
        while any(
            candidate.with_suffix(ext).exists()
            for ext in (self.image_ext, ".txt", ".json")
        ):
            candidate = base_path.with_name(f"{base_path.name}-{suffix:02d}")
            suffix += 1
        return candidate

    def _image_save_kwargs(
        self,
        *,
        image_path: Path,
        manifest: GenerationManifest | None,
        image_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        save_kwargs = dict(image_kwargs or {})
        if image_path.suffix.lower() == ".png" and manifest is not None:
            existing = save_kwargs.get("pnginfo")
            if existing is not None and not isinstance(existing, PngInfo):
                raise TypeError(f"pnginfo must be a PIL.PngImagePlugin.PngInfo instance, got {type(existing)}")
            save_kwargs["pnginfo"] = build_pnginfo(manifest, existing=existing)
        return save_kwargs

    def save_batch(
        self,
        images: torch.Tensor | Sequence[Image.Image],
        manifest: GenerationManifest | None = None,
        save_txt: bool = True,
        save_json: bool = True,
        image_kwargs: dict[str, Any] | None = None,
    ) -> list[SavedImageRecord]:
        pil_images = self._coerce_pil_images(images)
        image_kwargs = dict(image_kwargs or {})
        seeds = self._resolve_image_seeds(manifest, len(pil_images))

        records: list[SavedImageRecord] = []
        next_idx = self.next_index()

        for image_offset, image in enumerate(pil_images):
            image_seed = seeds[image_offset]
            image_manifest = self._manifest_copy(manifest) if manifest is not None else None
            if image_manifest is not None and image_seed is not None:
                image_manifest.required_for_rerun.seed = int(image_seed)
                batch_seeds = list(image_manifest.extra.get("resolved_seeds", []) or [])
                image_manifest.extra["batch_resolved_seeds"] = batch_seeds
                image_manifest.extra["resolved_seeds"] = [int(image_seed)]
                image_manifest.extra["image_seed"] = int(image_seed)
                self._project_prompt_expansion_for_image(image_manifest, image_offset)
                self._project_regions_for_image(image_manifest, image_offset)

            base_path = self._avoid_collision(
                self.build_base_path(
                    next_idx,
                    seed=image_seed,
                    manifest=image_manifest,
                )
            )
            image_path = base_path.with_suffix(self.image_ext)
            txt_path = base_path.with_suffix(".txt") if image_manifest is not None and save_txt else None
            json_path = base_path.with_suffix(".json") if image_manifest is not None and save_json else None

            save_kwargs = self._image_save_kwargs(
                image_path=image_path,
                manifest=image_manifest,
                image_kwargs=image_kwargs,
            )
            image.save(image_path, **save_kwargs)

            if image_manifest is not None:
                image_manifest.update_runtime_paths(
                    image_path=str(image_path),
                    txt_path=None if txt_path is None else str(txt_path),
                    json_path=None if json_path is None else str(json_path),
                )
                # Both sidecars receive complete, self-consistent paths.
                if txt_path is not None:
                    save_manifest_txt(image_manifest, txt_path)
                if json_path is not None:
                    save_manifest_json(image_manifest, json_path)

            records.append(
                SavedImageRecord(
                    image_path=str(image_path),
                    txt_path=None if txt_path is None else str(txt_path),
                    json_path=None if json_path is None else str(json_path),
                    index=next_idx,
                    seed=image_seed,
                )
            )
            next_idx += 1

        return records


def save_generation_batch(
    images: torch.Tensor | Sequence[Image.Image],
    output_dir: str | Path,
    prefix: str = DEFAULT_FILENAME_PATTERN,
    manifest: GenerationManifest | None = None,
    save_txt: bool = True,
    save_json: bool = True,
    image_ext: str = ".png",
    image_kwargs: dict[str, Any] | None = None,
) -> list[SavedImageRecord]:
    saver = GenerationOutputSaver(
        output_dir=output_dir,
        prefix=prefix,
        image_ext=image_ext,
    )
    return saver.save_batch(
        images=images,
        manifest=manifest,
        save_txt=save_txt,
        save_json=save_json,
        image_kwargs=image_kwargs,
    )
