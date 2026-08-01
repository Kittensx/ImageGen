from __future__ import annotations

import json
from pathlib import Path

from image_gen.systems.diagnostics.serialization import json_safe
from modules.txt2img.generation_manifest import GenerationManifest
from modules.txt2img.manifest_formatters import manifest_to_infotext


def save_manifest_json(
    manifest: GenerationManifest,
    json_path: str | Path,
    indent: int = 2,
) -> Path:
    """Write a manifest after normalizing all runtime values to JSON data.

    Sampler and scheduler metadata may legitimately contain objects such as
    ``torch.device``, ``torch.dtype``, registry descriptors, or diagnostic
    helpers. The manifest file must never retain those live objects.
    """

    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_safe(manifest.to_dict())

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False)

    manifest.update_runtime_paths(json_path=str(path))
    return path


def load_manifest_json(json_path: str | Path) -> GenerationManifest:
    path = Path(json_path)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    return GenerationManifest.from_dict(payload)


def save_manifest_txt(
    manifest: GenerationManifest,
    txt_path: str | Path,
    include_optional: bool = True,
    include_runtime: bool = True,
    include_assets: bool = True,
) -> Path:
    path = Path(txt_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    infotext = manifest_to_infotext(
        manifest=manifest,
        include_optional=include_optional,
        include_runtime=include_runtime,
        include_assets=include_assets,
    )

    with path.open("w", encoding="utf-8") as file:
        file.write(infotext)

    manifest.update_runtime_paths(txt_path=str(path))
    return path
