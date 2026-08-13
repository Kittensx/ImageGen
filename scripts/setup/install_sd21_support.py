from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping

# This file is deliberately executable as a standalone installer. When Python
# launches a script by path, sys.path starts at scripts/setup rather than at the
# IMAGE_GEN project root, so bootstrap the root before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.project_context import ProjectContext


DEFAULT_MANIFEST = Path("scripts/setup/manifests/sd21_support.json")
FORBIDDEN_WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".bin", ".pt", ".pth", ".onnx", ".gguf"}


class SD21InstallError(RuntimeError):
    pass


def _project_root() -> Path:
    return PROJECT_ROOT


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SD21InstallError(f"SD2.1 support manifest was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SD21InstallError(f"SD2.1 support manifest is invalid JSON: {exc}") from exc

    if int(payload.get("schema_version", 0)) != 1:
        raise SD21InstallError(f"Unsupported SD2.1 support manifest schema: {path}")
    if not str(payload.get("repo_id") or "").strip():
        raise SD21InstallError("SD2.1 support manifest is missing repo_id.")

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise SD21InstallError(f"SD2.1 support manifest contains no files: {path}")

    for raw in files:
        if not isinstance(raw, Mapping):
            raise SD21InstallError("SD2.1 support manifest contains an invalid file record.")
        remote = str(raw.get("remote_path") or "").replace("\\", "/")
        destination_kind = str(raw.get("destination_kind") or "").strip().casefold()
        if not remote or remote.startswith("/") or ".." in Path(remote).parts:
            raise SD21InstallError(f"Unsafe SD2.1 remote path in manifest: {remote!r}")
        if destination_kind != "runtime":
            raise SD21InstallError(
                "The SD2.1 support installer is runtime-assets-only; "
                f"destination kind {destination_kind!r} is not allowed."
            )
        if Path(remote).suffix.casefold() in FORBIDDEN_WEIGHT_SUFFIXES:
            raise SD21InstallError(
                "The SD2.1 support installer must never download model/checkpoint weights; "
                f"rejected {remote!r}."
            )
    return payload


def _destination_root(context: ProjectContext) -> Path:
    return (Path(context.runtime_assets_root) / "stable_diffusion" / "sd2_1_base").resolve()


def _destination_for(context: ProjectContext, record: Mapping[str, Any]) -> Path:
    root = _destination_root(context)
    relative = Path(str(record.get("destination_path") or "").replace("\\", "/"))
    if not str(relative) or relative.is_absolute() or ".." in relative.parts:
        raise SD21InstallError(f"Unsafe SD2.1 destination path: {relative}")
    destination = (root / relative).resolve()
    if destination != root and root not in destination.parents:
        raise SD21InstallError(f"SD2.1 destination escapes its managed root: {destination}")
    return destination


def _file_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _required_files(context: ProjectContext, manifest: Mapping[str, Any]) -> list[tuple[dict[str, Any], Path]]:
    result: list[tuple[dict[str, Any], Path]] = []
    for raw in manifest.get("files") or []:
        record = dict(raw or {})
        if record.get("required", True) is False:
            continue
        result.append((record, _destination_for(context, record)))
    return result


def _download_file(*, repo_id: str, revision: str, remote_path: str, destination: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SD21InstallError(
            "SD2.1 support installation requires huggingface_hub. "
            "Run the main IMAGE_GEN installer first so normal Python requirements are installed."
        ) from exc

    try:
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                revision=revision,
            )
        )
    except Exception as exc:  # huggingface_hub exposes several transport-specific error types.
        raise SD21InstallError(f"Could not download {remote_path} from Hugging Face: {exc}") from exc

    if not _file_ready(cached):
        raise SD21InstallError(f"Hugging Face returned an empty or missing file for {remote_path}.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copy2(cached, temporary)
        if not _file_ready(temporary):
            raise SD21InstallError(f"Downloaded support file failed local validation: {remote_path}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _write_receipt(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / "artifacts" / "install" / "sd21_support_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install IMAGE_GEN Stable Diffusion 2.1 runtime support files from Hugging Face."
    )
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--trigger", default="manual")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    context = ProjectContext.load(project_root=root)
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else (root / DEFAULT_MANIFEST).resolve()
    )
    manifest = _load_manifest(manifest_path)
    required = _required_files(context, manifest)
    missing = [(record, destination) for record, destination in required if not _file_ready(destination)]

    print("IMAGE_GEN Stable Diffusion 2.1 Runtime Support Installer")
    print("=========================================================")
    print(f"Source:     https://huggingface.co/{manifest['repo_id']}")
    print(f"Support ID: {manifest.get('support_id', 'sd2.1-runtime')}")
    print(f"Destination: {_destination_root(context)}")
    print("Model checkpoints are user-managed and are not downloaded by this installer.")
    print("Reference-component weights are not downloaded by this installer.")
    print(f"Required files: {len(required)}; missing: {len(missing)}")

    if args.status_only:
        for _record, destination in missing:
            print(f"MISSING: {destination}")
        return 0 if not missing else 3

    if not missing:
        print("SD2.1 runtime support is already installed.")
        _write_receipt(
            root,
            {
                "schema_version": 1,
                "support_id": manifest.get("support_id"),
                "repo_id": manifest.get("repo_id"),
                "revision": manifest.get("revision", "main"),
                "trigger": args.trigger,
                "status": "already_ready",
                "required_count": len(required),
            },
        )
        return 0

    if args.dry_run:
        print("Dry run: the following runtime support files would be downloaded:")
        for record, destination in missing:
            print(f"  {record['remote_path']} -> {destination}")
        return 0

    repo_id = str(manifest["repo_id"])
    revision = str(manifest.get("revision") or "main")
    installed: list[str] = []
    for index, (record, destination) in enumerate(missing, start=1):
        remote_path = str(record["remote_path"])
        print(f"[{index}/{len(missing)}] {remote_path}")
        _download_file(
            repo_id=repo_id,
            revision=revision,
            remote_path=remote_path,
            destination=destination,
        )
        installed.append(str(destination))

    remaining = [str(destination) for _record, destination in required if not _file_ready(destination)]
    if remaining:
        raise SD21InstallError(
            "SD2.1 runtime support installation finished with missing files:\n  " + "\n  ".join(remaining)
        )

    receipt = _write_receipt(
        root,
        {
            "schema_version": 1,
            "support_id": manifest.get("support_id"),
            "repo_id": repo_id,
            "revision": revision,
            "trigger": args.trigger,
            "status": "installed",
            "required_count": len(required),
            "downloaded_count": len(installed),
            "downloaded_files": installed,
        },
    )
    print(f"SD2.1 runtime support is ready. Receipt: {receipt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SD21InstallError as exc:
        print(f"SD2.1 support installer error: {exc}", file=sys.stderr)
        raise SystemExit(1)
