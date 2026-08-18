from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

# Executing this file by path places scripts/setup at sys.path[0]. Bootstrap the
# IMAGE_GEN project root before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.project_context import ProjectContext
from modules.sd3_runtime_profile import profile_from_id
from modules.sd3_shared_text_encoders import infer_text_encoder_role_from_path, register_shared_text_encoder_asset


DEFAULT_MANIFEST = Path("scripts/setup/manifests/sd3_support.json")
OFFICIAL_REPOSITORIES = {
    "stabilityai/stable-diffusion-3-medium",
    "stabilityai/stable-diffusion-3-medium-diffusers",
    "stabilityai/stable-diffusion-3.5-medium",
}
FORBIDDEN_RUNTIME_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".ckpt",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
}
ALLOWED_SHARED_TEXT_ENCODER_DESTINATIONS = {
    "clip_g.safetensors": Path("clip/clip_g.safetensors"),
    "clip_l.safetensors": Path("clip/clip_l.safetensors"),
    "t5xxl_fp8_e4m3fn.safetensors": Path("t5/t5xxl_fp8_e4m3fn.safetensors"),
}
ALLOWED_DESTINATION_KINDS = {"runtime", "text_encoder"}


class SD3InstallError(RuntimeError):
    pass


def _project_root() -> Path:
    return PROJECT_ROOT


def _safe_relative(value: Any, *, label: str) -> Path:
    text = str(value or "").replace("\\", "/").strip()
    relative = Path(text)
    if not text or text.startswith("/") or relative.is_absolute() or ".." in relative.parts:
        raise SD3InstallError(f"Unsafe {label}: {text!r}")
    return relative


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SD3InstallError(f"SD3 support manifest was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise SD3InstallError(f"SD3 support manifest is invalid JSON: {exc}") from exc

    if int(payload.get("schema_version", 0)) != 1:
        raise SD3InstallError(f"Unsupported SD3 support manifest schema: {path}")

    sources = payload.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise SD3InstallError("SD3 support manifest is missing sources.")

    for source_id, raw_source in sources.items():
        if not isinstance(raw_source, Mapping):
            raise SD3InstallError(f"Invalid SD3 source record: {source_id!r}")
        repo_id = str(raw_source.get("repo_id") or "").strip()
        if repo_id not in OFFICIAL_REPOSITORIES:
            raise SD3InstallError(
                f"SD3 support source {source_id!r} must use an approved Stability AI repository, got {repo_id!r}."
            )
        for fallback in raw_source.get("fallback_repo_ids") or []:
            fallback_repo = str(fallback or "").strip()
            if fallback_repo not in OFFICIAL_REPOSITORIES:
                raise SD3InstallError(
                    f"SD3 fallback source {fallback_repo!r} is not an approved Stability AI repository."
                )

    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise SD3InstallError(f"SD3 support manifest contains no files: {path}")

    for raw in files:
        if not isinstance(raw, Mapping):
            raise SD3InstallError("SD3 support manifest contains an invalid file record.")
        source_id = str(raw.get("source_id") or "").strip()
        if source_id not in sources:
            raise SD3InstallError(f"SD3 support file references unknown source_id {source_id!r}.")

        remote = _safe_relative(raw.get("remote_path"), label="SD3 remote path")
        destination = _safe_relative(raw.get("destination_path"), label="SD3 destination path")
        destination_kind = str(raw.get("destination_kind") or "").strip().casefold()
        if destination_kind not in ALLOWED_DESTINATION_KINDS:
            raise SD3InstallError(f"Unsupported SD3 destination kind: {destination_kind!r}")

        if destination_kind == "runtime":
            profile_id = str(raw.get("profile_id") or "").strip()
            if profile_from_id(profile_id) is None:
                raise SD3InstallError(f"Unknown SD3 runtime profile in manifest: {profile_id!r}")
            if remote.suffix.casefold() in FORBIDDEN_RUNTIME_WEIGHT_SUFFIXES:
                raise SD3InstallError(
                    "The SD3 runtime installer must never download transformer/VAE/checkpoint weights; "
                    f"rejected {remote.as_posix()!r}."
                )
        else:
            if remote.parent.as_posix() != "text_encoders":
                raise SD3InstallError(
                    "Shared SD3 text encoders must come from the official text_encoders directory; "
                    f"rejected {remote.as_posix()!r}."
                )
            expected_destination = ALLOWED_SHARED_TEXT_ENCODER_DESTINATIONS.get(remote.name)
            if expected_destination is None:
                raise SD3InstallError(f"Unapproved SD3 shared text encoder: {remote.name!r}")
            if destination != expected_destination:
                raise SD3InstallError(
                    "Shared SD3 text encoders must use the canonical encoder-family layout under "
                    f"TextEncoders: expected {expected_destination.as_posix()!r}, got {destination.as_posix()!r}."
                )

    return dict(payload)


def _runtime_root(context: ProjectContext, profile_id: str) -> Path:
    profile = profile_from_id(profile_id)
    if profile is None:
        raise SD3InstallError(f"Unknown SD3 runtime profile: {profile_id!r}")
    return (Path(context.runtime_assets_root) / profile.runtime_assets_subdir).resolve()


def _text_encoder_root(context: ProjectContext) -> Path:
    return (Path(context.models_root) / "TextEncoders").resolve()


def _destination_for(context: ProjectContext, record: Mapping[str, Any]) -> Path:
    kind = str(record.get("destination_kind") or "").strip().casefold()
    relative = _safe_relative(record.get("destination_path"), label="SD3 destination path")
    if kind == "runtime":
        root = _runtime_root(context, str(record.get("profile_id") or ""))
    elif kind == "text_encoder":
        root = _text_encoder_root(context)
    else:
        raise SD3InstallError(f"Unsupported SD3 destination kind: {kind!r}")

    destination = (root / relative).resolve()
    if destination != root and root not in destination.parents:
        raise SD3InstallError(f"SD3 destination escapes its managed root: {destination}")
    return destination


def _file_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _register_installed_shared_text_encoders(
    context: ProjectContext,
    required: Sequence[tuple[dict[str, Any], Path]],
) -> list[dict[str, str]]:
    registered: list[dict[str, str]] = []
    for record, destination in required:
        if str(record.get("destination_kind") or "").strip().casefold() != "text_encoder":
            continue
        if not _file_ready(destination):
            continue
        role = infer_text_encoder_role_from_path(destination)
        asset, snapshots = register_shared_text_encoder_asset(context, destination, role=role)
        component_sha256 = None
        for snapshot in snapshots:
            if snapshot.component_role in {"text_encoder", "text_encoder_2", "text_encoder_3"}:
                component_sha256 = snapshot.component_sha256
                break
        registered.append(
            {
                "role": role,
                "path": str(destination),
                "asset_sha256": str(asset.sha256 or ""),
                "component_sha256": str(component_sha256 or ""),
            }
        )
    return registered


def _required_files(
    context: ProjectContext,
    manifest: Mapping[str, Any],
    *,
    profile_filter: str,
) -> list[tuple[dict[str, Any], Path]]:
    result: list[tuple[dict[str, Any], Path]] = []
    for raw in manifest.get("files") or []:
        record = dict(raw or {})
        if record.get("required", True) is False:
            continue
        if record.get("destination_kind") == "runtime" and profile_filter != "all":
            if str(record.get("profile_id")) != profile_filter:
                continue
        result.append((record, _destination_for(context, record)))
    return result


def _source_candidates(manifest: Mapping[str, Any], record: Mapping[str, Any]) -> list[tuple[str, str]]:
    sources = manifest.get("sources") or {}
    source = sources[str(record["source_id"])]
    revision = str(source.get("revision") or "main")
    repos = [str(source["repo_id"])]
    repos.extend(str(item) for item in (source.get("fallback_repo_ids") or []))
    return [(repo_id, revision) for repo_id in repos]


def _download_file(
    *,
    source_candidates: Sequence[tuple[str, str]],
    remote_path: str,
    destination: Path,
) -> str:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SD3InstallError(
            "SD3 support installation requires huggingface_hub. "
            "Run the main IMAGE_GEN installer first so normal Python requirements are installed."
        ) from exc

    failures: list[str] = []
    for repo_id, revision in source_candidates:
        try:
            cached = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=remote_path,
                    revision=revision,
                )
            )
        except Exception as exc:  # huggingface_hub exposes several transport-specific error types.
            failures.append(f"{repo_id}: {type(exc).__name__}: {exc}")
            continue

        if not _file_ready(cached):
            failures.append(f"{repo_id}: downloaded file is empty or missing")
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            shutil.copy2(cached, temporary)
            if not _file_ready(temporary):
                raise SD3InstallError(f"Downloaded support file failed local validation: {remote_path}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return repo_id

    joined = "\n  ".join(failures)
    raise SD3InstallError(
        f"Could not download {remote_path} from the approved Stability AI source(s).\n"
        "These repositories are gated. Accept their Hugging Face access terms and authenticate with "
        "`hf auth login` or an HF_TOKEN environment variable, then rerun this installer.\n"
        f"Attempts:\n  {joined}"
    )


def _write_receipt(root: Path, payload: Mapping[str, Any]) -> Path:
    path = root / "artifacts" / "install" / "sd3_support_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install IMAGE_GEN Stable Diffusion 3/3.5 runtime assets and shared CLIP-L, CLIP-G, "
            "and FP8 T5-XXL text encoders from official Stability AI Hugging Face repositories."
        )
    )
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--profile",
        choices=("all", "sd3-medium", "sd3.5-medium"),
        default="all",
        help="Runtime asset profile to install. Shared text encoders are installed for every selection.",
    )
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
    required = _required_files(context, manifest, profile_filter=args.profile)
    missing = [(record, destination) for record, destination in required if not _file_ready(destination)]

    print("IMAGE_GEN Stable Diffusion 3 / 3.5 Support Installer")
    print("=====================================================")
    print(f"Support ID: {manifest.get('support_id', 'sd3-family-support')}")
    print(f"Runtime selection: {args.profile}")
    print(f"SD3 Medium runtime: {_runtime_root(context, 'sd3-medium')}")
    print(f"SD3.5 Medium runtime: {_runtime_root(context, 'sd3.5-medium')}")
    print(f"Shared text-encoder library: {_text_encoder_root(context)}")
    print(f"  CLIP: {_text_encoder_root(context) / 'clip'}")
    print(f"  T5:   {_text_encoder_root(context) / 't5'}")
    print("Main SD3/SD3.5 model checkpoints are user-managed and are never downloaded by this installer.")
    print("Shared encoder policy: CLIP-L + CLIP-G + T5-XXL FP8 E4M3FN.")
    print(f"Required files for this selection: {len(required)}; missing: {len(missing)}")

    if args.status_only:
        for _record, destination in missing:
            print(f"MISSING: {destination}")
        return 0 if not missing else 3

    if not missing:
        print("SD3 runtime assets and shared text encoders are already installed for this selection.")
        registered_text_encoders = _register_installed_shared_text_encoders(context, required)
        receipt = _write_receipt(
            root,
            {
                "schema_version": 1,
                "support_id": manifest.get("support_id"),
                "trigger": args.trigger,
                "profile": args.profile,
                "status": "already_ready",
                "required_count": len(required),
                "text_encoder_root": str(_text_encoder_root(context)),
                "registered_text_encoders": registered_text_encoders,
            },
        )
        print(f"Receipt: {receipt}")
        return 0

    if args.dry_run:
        print("Dry run: the following support files would be downloaded:")
        for record, destination in missing:
            candidates = ", ".join(repo for repo, _revision in _source_candidates(manifest, record))
            print(f"  [{record['destination_kind']}] {candidates}:{record['remote_path']} -> {destination}")
        return 0

    installed: list[dict[str, str]] = []
    for index, (record, destination) in enumerate(missing, start=1):
        remote_path = str(record["remote_path"])
        candidates = _source_candidates(manifest, record)
        print(f"[{index}/{len(missing)}] {remote_path}")
        repo_id = _download_file(
            source_candidates=candidates,
            remote_path=remote_path,
            destination=destination,
        )
        installed.append(
            {
                "repo_id": repo_id,
                "remote_path": remote_path,
                "destination": str(destination),
                "destination_kind": str(record["destination_kind"]),
            }
        )

    remaining = [str(destination) for _record, destination in required if not _file_ready(destination)]
    if remaining:
        raise SD3InstallError(
            "SD3 support installation finished with missing files:\n  " + "\n  ".join(remaining)
        )

    registered_text_encoders = _register_installed_shared_text_encoders(context, required)
    receipt = _write_receipt(
        root,
        {
            "schema_version": 1,
            "support_id": manifest.get("support_id"),
            "trigger": args.trigger,
            "profile": args.profile,
            "status": "installed",
            "required_count": len(required),
            "downloaded_count": len(installed),
            "downloaded_files": installed,
            "runtime_roots": {
                "sd3-medium": str(_runtime_root(context, "sd3-medium")),
                "sd3.5-medium": str(_runtime_root(context, "sd3.5-medium")),
            },
            "text_encoder_root": str(_text_encoder_root(context)),
            "registered_text_encoders": registered_text_encoders,
        },
    )
    print(f"SD3 support is ready. Receipt: {receipt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SD3InstallError as exc:
        print(f"SD3 support installer error: {exc}", file=sys.stderr)
        raise SystemExit(1)
