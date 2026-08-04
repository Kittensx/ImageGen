from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from image_gen.runtime.lora_inspector import (
    canonical_model_family,
    inspect_lora_file,
    lora_scan_cache_is_current,
)
from image_gen.webui.asset_metadata import load_asset_metadata, save_asset_metadata
from modules.project_context import ProjectContext


LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}


@dataclass
class LoRAListEntry:
    index: int
    name: str
    path: str
    extension: str
    size_mb: float
    model_family: str
    compatibility: str
    tensor_key_format: str
    activation_text: str
    activation_text_source: str
    preferred_weight: float
    inspection_error: str = ""


class LoRASelector:
    """Interactive CLI LoRA discovery and conservative model-family filtering."""

    def __init__(self, project_context: ProjectContext | None = None) -> None:
        self.context = project_context or ProjectContext.load()

    def checkpoint_family(self, model_path: str) -> str:
        path = Path(model_path).expanduser().resolve()
        if path.suffix.lower() != ".safetensors":
            return ""
        try:
            from safetensors import safe_open

            dimensions: set[int] = set()
            has_sdxl_text_encoder = False
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    lowered = key.lower()
                    if "conditioner.embedders.1" in lowered or "text_encoder_2" in lowered:
                        has_sdxl_text_encoder = True
                    if not (
                        key.startswith("cond_stage_model.")
                        or (
                            key.startswith("model.diffusion_model.")
                            and (".attn2.to_k.weight" in key or ".attn2.to_v.weight" in key)
                        )
                    ):
                        continue
                    try:
                        shape = tuple(int(value) for value in handle.get_slice(key).get_shape())
                    except Exception:
                        continue
                    if len(shape) >= 2 and int(shape[-1]) in {768, 1024, 1280, 2048}:
                        dimensions.add(int(shape[-1]))
            if has_sdxl_text_encoder or 1280 in dimensions or 2048 in dimensions:
                return "sdxl"
            if 1024 in dimensions and 768 not in dimensions:
                return "sd2"
            if 768 in dimensions and 1024 not in dimensions:
                return "sd1"
        except Exception:
            return ""
        return ""

    @staticmethod
    def _scan_cache_is_current(path: Path, cache: dict[str, Any]) -> bool:
        return lora_scan_cache_is_current(
            path,
            cache,
            require_compatibility_hash=True,
        )

    def scan_loras(
        self,
        model_path: str,
    ) -> tuple[str, list[LoRAListEntry], list[LoRAListEntry], list[LoRAListEntry]]:
        model_family = self.checkpoint_family(model_path)
        root = self.context.lora_dir
        if not root.exists():
            return model_family, [], [], []

        paths = sorted(
            [
                path
                for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in LORA_EXTENSIONS
            ],
            key=lambda path: (path.name.casefold(), str(path).casefold()),
        )
        compatible: list[LoRAListEntry] = []
        unclassified: list[LoRAListEntry] = []
        incompatible: list[LoRAListEntry] = []

        for index, path in enumerate(paths, start=1):
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0.0
            sidecar = load_asset_metadata(path)
            cache = dict(sidecar.get("_lora_scan_cache") or {})
            cache_is_current = self._scan_cache_is_current(path, cache)
            analysis = (
                {
                    "detected_model_family": cache.get("detected_model_family") or "",
                    "activation_text": cache.get("activation_text") or "",
                    "activation_text_source": cache.get("activation_text_source") or "",
                    "tensor_key_format": cache.get("tensor_key_format") or "Unknown",
                    "inspection_error": cache.get("inspection_error") or "",
                }
                if cache_is_current
                else inspect_lora_file(path, sidecar_metadata=sidecar)
            )

            # Only known architecture-family labels participate in filtering.
            # Arbitrary sidecar values such as a checkpoint name must remain
            # unclassified rather than being treated as incompatible.
            family = canonical_model_family(
                sidecar.get("model_family")
                or sidecar.get("base_model")
                or analysis.get("detected_model_family")
            )
            activation_text = str(
                sidecar.get("activation_text")
                or analysis.get("activation_text")
                or ""
            ).strip()
            activation_source = str(
                analysis.get("activation_text_source")
                or ("sidecar:activation_text" if sidecar.get("activation_text") else "")
            ).strip()
            try:
                preferred_weight = float(sidecar.get("preferred_weight", 1.0))
            except (TypeError, ValueError):
                preferred_weight = 1.0

            if family and model_family:
                compatibility = "compatible" if family == model_family else "incompatible"
            else:
                compatibility = "unknown"

            entry = LoRAListEntry(
                index=index,
                name=path.stem,
                path=str(path.resolve()),
                extension=path.suffix.lower(),
                size_mb=round(size_mb, 2),
                model_family=family,
                compatibility=compatibility,
                tensor_key_format=str(analysis.get("tensor_key_format") or "Unknown"),
                activation_text=activation_text,
                activation_text_source=activation_source,
                preferred_weight=preferred_weight,
                inspection_error=str(analysis.get("inspection_error") or ""),
            )
            if compatibility == "compatible":
                compatible.append(entry)
            elif compatibility == "incompatible":
                incompatible.append(entry)
            else:
                unclassified.append(entry)
        return model_family, compatible, unclassified, incompatible

    @staticmethod
    def _print_entries(title: str, entries: list[LoRAListEntry]) -> None:
        print(f"\n=== {title} ===")
        if not entries:
            print("None found.")
            return
        for entry in entries:
            details = [entry.tensor_key_format]
            details.append(entry.model_family or "family unknown")
            details.append(f"{entry.size_mb:.2f} MB")
            print(f"{entry.index}. {entry.name} ({', '.join(details)})")
            if entry.activation_text:
                print(f"   Activation text: {entry.activation_text}")
            if entry.inspection_error:
                print(f"   Scan warning: {entry.inspection_error}")
            print(f"   {entry.path}")

    @staticmethod
    def _parse_selection(raw: str, available: list[LoRAListEntry]) -> list[LoRAListEntry] | None:
        text = str(raw or "").strip()
        lowered = text.lower()
        if not text or lowered in {"none", "n", "0"}:
            return []
        if lowered in {"a", "all"}:
            return list(available)

        index_map = {entry.index: entry for entry in available}
        name_map: dict[str, LoRAListEntry] = {}
        for entry in available:
            name_map[entry.name.casefold()] = entry
            name_map[Path(entry.path).name.casefold()] = entry
            name_map[entry.path.casefold()] = entry

        selected: list[LoRAListEntry] = []
        seen: set[int] = set()
        for token in [part.strip() for part in text.split(",") if part.strip()]:
            entry: LoRAListEntry | None = None
            try:
                entry = index_map.get(int(token))
            except ValueError:
                entry = name_map.get(token.casefold())
            if entry is None:
                return None
            if entry.index not in seen:
                selected.append(entry)
                seen.add(entry.index)
        return selected if selected else None

    def choose_loras(self, model_path: str) -> list[dict[str, Any]]:
        model_family, compatible, unclassified, incompatible = self.scan_loras(model_path)
        total = len(compatible) + len(unclassified) + len(incompatible)
        print("\n=== LoRA Selection ===")
        print(f"Selected checkpoint family: {model_family or 'unknown'}")
        print(
            "LoRA files discovered: "
            f"{total} total; {len(compatible)} verified compatible; "
            f"{len(unclassified)} unclassified; {len(incompatible)} known incompatible."
        )

        self._print_entries("Compatible LoRAs", compatible)
        # Unknown is not the same as incompatible. Show these by default so a
        # valid older Kohya LoRA without architecture metadata is not hidden.
        self._print_entries("Unclassified LoRAs - compatibility not verified", unclassified)
        available = [*compatible, *unclassified]

        if incompatible:
            print(
                f"\n{len(incompatible)} LoRA(s) are tagged for another architecture and are hidden by default."
            )
            show_incompatible = input("Show known-incompatible LoRAs too? (y/n) [n]: ").strip().lower()
            if show_incompatible in {"y", "yes", "1", "true", "on"}:
                self._print_entries("Known-incompatible LoRAs - advanced override", incompatible)
                available.extend(incompatible)

        if not available:
            print("No selectable LoRAs were found for this checkpoint.")
            return []

        while True:
            raw = input(
                "Choose LoRAs [blank=none, comma-separated numbers or exact names, a=all shown]: "
            )
            selected = self._parse_selection(raw, available)
            if selected is not None:
                break
            print("Invalid LoRA selection.")

        output: list[dict[str, Any]] = []
        for order, entry in enumerate(selected):
            weight_raw = input(
                f"Weight for {entry.name} [{entry.preferred_weight:g}]: "
            ).strip()
            try:
                weight = entry.preferred_weight if not weight_raw else float(weight_raw)
            except ValueError:
                print(f"Invalid weight; using {entry.preferred_weight:g}.")
                weight = entry.preferred_weight

            if entry.activation_text:
                activation_raw = input(
                    f"Activation text for {entry.name} [{entry.activation_text}]: "
                ).strip()
                activation_text = activation_raw or entry.activation_text
            else:
                activation_text = input(
                    f"Activation text for {entry.name} [blank=none]: "
                ).strip()
                if activation_text:
                    try:
                        save_asset_metadata(entry.path, {"activation_text": activation_text})
                        print(f"Saved activation text for {entry.name}.")
                    except Exception as exc:
                        print(
                            f"WARNING: Could not save activation text for {entry.name}: "
                            f"{type(exc).__name__}: {exc}"
                        )

            output.append(
                {
                    "asset_type": "lora",
                    "name": entry.name,
                    "path": entry.path,
                    "weight": weight,
                    "enabled": True,
                    "polarity": "positive",
                    "activation_text": activation_text,
                    "model_family": entry.model_family,
                    "source": "visual_selection",
                    "order": order,
                    "metadata": {
                        "tensor_key_format": entry.tensor_key_format,
                        "compatibility": entry.compatibility,
                        "activation_text_source": (
                            entry.activation_text_source if entry.activation_text else "cli_user_supplied"
                        ),
                    },
                }
            )
        return output


def choose_cli_loras_for_model(
    model_path: str,
    *,
    project_context: ProjectContext | None = None,
) -> list[dict[str, Any]]:
    return LoRASelector(project_context=project_context).choose_loras(model_path)
