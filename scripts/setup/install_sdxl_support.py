from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MANIFEST = Path("scripts/setup/manifests/sdxl_support.json")
CORE_PROFILE_IDS = ("base", "refiner", "turbo")
JAPANESE_PROFILE_IDS = ("japanese-sdxl", "japanese-clip")
APPROVED_REPOSITORIES = {
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    "stabilityai/sdxl-turbo",
    "stabilityai/japanese-stable-diffusion-xl",
    "stabilityai/japanese-stable-clip-vit-l-16",
}
FORBIDDEN_WEIGHT_SUFFIXES = {
    ".safetensors",
    ".ckpt",
    ".bin",
    ".pt",
    ".pth",
    ".onnx",
    ".gguf",
}


class SDXLSupportInstallError(RuntimeError):
    pass


def _project_root() -> Path:
    return PROJECT_ROOT


def _safe_relative(value: Any, *, label: str) -> Path:
    text = str(value or "").replace("\\", "/").strip()
    relative = Path(text)
    if not text or text.startswith("/") or relative.is_absolute() or ".." in relative.parts:
        raise SDXLSupportInstallError(f"Unsafe {label}: {text!r}")
    return relative


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SDXLSupportInstallError(f"Missing {label}: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise SDXLSupportInstallError(f"Unable to read {label} JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SDXLSupportInstallError(f"{label} must contain a JSON object: {path}")
    return payload


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json(path, label="SDXL support manifest")
    if int(payload.get("schema_version", 0)) != 1:
        raise SDXLSupportInstallError(f"Unsupported SDXL support manifest schema: {path}")

    profiles = payload.get("profiles")
    if not isinstance(profiles, Mapping) or not profiles:
        raise SDXLSupportInstallError("SDXL support manifest is missing profiles.")

    for profile_id, raw_profile in profiles.items():
        if not isinstance(raw_profile, Mapping):
            raise SDXLSupportInstallError(f"Invalid SDXL support profile: {profile_id!r}")
        repo_id = str(raw_profile.get("repo_id") or "").strip()
        if repo_id not in APPROVED_REPOSITORIES:
            raise SDXLSupportInstallError(
                f"SDXL profile {profile_id!r} must use an approved Stability AI repository; got {repo_id!r}."
            )

        runtime_subdir = _safe_relative(raw_profile.get("runtime_subdir"), label="runtime subdir")
        if not runtime_subdir.parts or runtime_subdir.parts[0].casefold() != "stable_diffusion":
            raise SDXLSupportInstallError(
                f"SDXL profile {profile_id!r} must install below runtime_assets/stable_diffusion."
            )

        files = raw_profile.get("files")
        if not isinstance(files, list) or not files:
            raise SDXLSupportInstallError(f"SDXL profile {profile_id!r} contains no support files.")
        for raw_file in files:
            remote = _safe_relative(raw_file, label=f"{profile_id} remote path")
            if remote.suffix.casefold() in FORBIDDEN_WEIGHT_SUFFIXES:
                raise SDXLSupportInstallError(
                    "The SDXL support installer must never download checkpoint, UNet, VAE, or text-encoder weights; "
                    f"rejected {repo_id}:{remote.as_posix()}."
                )
    return payload


def _runtime_assets_root(project_root: Path) -> Path:
    try:
        from modules.project_context import ProjectContext
    except Exception:
        return (project_root / "runtime_assets").resolve()

    try:
        context = ProjectContext.load(project_root=project_root)
        return Path(context.runtime_assets_root).expanduser().resolve()
    except Exception:
        # The installer should still be able to repair runtime support assets even
        # if the wider application configuration cannot currently initialize.
        return (project_root / "runtime_assets").resolve()


def _profile_root(runtime_assets_root: Path, profile: Mapping[str, Any]) -> Path:
    relative = _safe_relative(profile.get("runtime_subdir"), label="runtime subdir")
    root = (runtime_assets_root / relative).resolve()
    if root != runtime_assets_root and runtime_assets_root not in root.parents:
        raise SDXLSupportInstallError(f"SDXL runtime destination escapes runtime_assets: {root}")
    return root


def _file_ready(path: Path) -> bool:
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _selected_profiles(
    manifest: Mapping[str, Any],
    *,
    profile: str,
    include_japanese: bool,
) -> list[tuple[str, dict[str, Any]]]:
    profiles = manifest.get("profiles") or {}
    if profile == "all":
        ids = list(CORE_PROFILE_IDS)
        if include_japanese:
            ids.extend(JAPANESE_PROFILE_IDS)
    elif profile == "japanese":
        ids = list(JAPANESE_PROFILE_IDS)
    else:
        ids = [profile]
        if include_japanese and profile not in JAPANESE_PROFILE_IDS:
            ids.extend(JAPANESE_PROFILE_IDS)

    selected: list[tuple[str, dict[str, Any]]] = []
    for profile_id in ids:
        raw = profiles.get(profile_id)
        if not isinstance(raw, Mapping):
            raise SDXLSupportInstallError(f"SDXL support manifest does not define profile {profile_id!r}.")
        selected.append((profile_id, dict(raw)))
    return selected


def _required_files(
    runtime_assets_root: Path,
    selected: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[tuple[str, dict[str, Any], str, Path]]:
    required: list[tuple[str, dict[str, Any], str, Path]] = []
    for profile_id, raw_profile in selected:
        profile = dict(raw_profile)
        root = _profile_root(runtime_assets_root, profile)
        for raw_remote in profile.get("files") or []:
            remote = _safe_relative(raw_remote, label=f"{profile_id} remote path").as_posix()
            destination = (root / Path(remote)).resolve()
            if destination != root and root not in destination.parents:
                raise SDXLSupportInstallError(f"SDXL support destination escapes profile root: {destination}")
            required.append((profile_id, profile, remote, destination))
    return required


def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


def _download_file(
    *,
    repo_id: str,
    revision: str,
    remote_path: str,
    destination: Path,
    refresh: bool,
    gated: bool,
) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SDXLSupportInstallError(
            "SDXL support installation requires huggingface_hub. Run the main IMAGE_GEN installer first."
        ) from exc

    try:
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                revision=revision,
                token=_hf_token(),
                force_download=bool(refresh),
            )
        )
    except Exception as exc:
        gated_hint = ""
        if gated:
            gated_hint = (
                " This is a gated Stability AI repository. Accept its Hugging Face access terms, then "
                "authenticate with `hf auth login` or set HF_TOKEN before rerunning the installer."
            )
        raise SDXLSupportInstallError(
            f"Could not download {repo_id}:{remote_path}: {type(exc).__name__}: {exc}.{gated_hint}"
        ) from exc

    if not _file_ready(cached):
        raise SDXLSupportInstallError(f"Downloaded support file is empty or missing: {repo_id}:{remote_path}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        shutil.copy2(cached, temporary)
        if not _file_ready(temporary):
            raise SDXLSupportInstallError(f"Downloaded support file failed local validation: {remote_path}")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_common_json(root: Path, files: Sequence[str], *, profile_id: str) -> None:
    for relative in files:
        path = root / relative
        if path.suffix.casefold() == ".json":
            _load_json(path, label=f"{profile_id} support file")


def _validate_profile(profile_id: str, profile: Mapping[str, Any], root: Path) -> dict[str, Any]:
    files = [str(item) for item in (profile.get("files") or [])]
    missing = [str(root / item) for item in files if not _file_ready(root / item)]
    if missing:
        raise SDXLSupportInstallError(
            f"{profile_id} support installation is incomplete:\n  " + "\n  ".join(missing)
        )

    _validate_common_json(root, files, profile_id=profile_id)
    result: dict[str, Any] = {
        "profile": profile_id,
        "root": str(root),
        "file_count": len(files),
    }

    if profile_id == "base":
        model_index = _load_json(root / "model_index.json", label="SDXL Base model index")
        scheduler = _load_json(root / "scheduler/scheduler_config.json", label="SDXL Base scheduler")
        text_encoder = _load_json(root / "text_encoder/config.json", label="SDXL Base text encoder")
        text_encoder_2 = _load_json(root / "text_encoder_2/config.json", label="SDXL Base text encoder 2")
        unet = _load_json(root / "unet/config.json", label="SDXL Base UNet")
        vae = _load_json(root / "vae/config.json", label="SDXL Base VAE")
        checks = {
            "pipeline_class": model_index.get("_class_name") == "StableDiffusionXLPipeline",
            "scheduler_class": scheduler.get("_class_name") == "EulerDiscreteScheduler",
            "prediction_type": scheduler.get("prediction_type") == "epsilon",
            "text_encoder_hidden_size": text_encoder.get("hidden_size") == 768,
            "text_encoder_2_hidden_size": text_encoder_2.get("hidden_size") == 1280,
            "unet_cross_attention_dim": unet.get("cross_attention_dim") == 2048,
            "vae_scaling_factor": abs(float(vae.get("scaling_factor", 0.0)) - 0.13025) <= 1e-8,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise SDXLSupportInstallError(
                "SDXL_Base assets do not match the canonical SDXL Base 1.0 architecture contract: "
                + ", ".join(failed)
            )
        result["architecture_checks"] = checks

    elif profile_id == "refiner":
        model_index = _load_json(root / "model_index.json", label="SDXL Refiner model index")
        scheduler = _load_json(root / "scheduler/scheduler_config.json", label="SDXL Refiner scheduler")
        unet = _load_json(root / "unet/config.json", label="SDXL Refiner UNet")
        checks = {
            "pipeline_class": model_index.get("_class_name") == "StableDiffusionXLImg2ImgPipeline",
            "scheduler_class": scheduler.get("_class_name") == "EulerDiscreteScheduler",
            "prediction_type": scheduler.get("prediction_type") == "epsilon",
            "unet_cross_attention_dim": unet.get("cross_attention_dim") == 1280,
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise SDXLSupportInstallError(
                "SDXL_Base_Refiner assets do not match the qualified SDXL Refiner contract: "
                + ", ".join(failed)
            )
        result["architecture_checks"] = checks

    elif profile_id == "turbo":
        model_index = _load_json(root / "model_index.json", label="SDXL Turbo model index")
        scheduler = _load_json(root / "scheduler/scheduler_config.json", label="SDXL Turbo scheduler")
        checks = {
            "pipeline_class": model_index.get("_class_name") == "StableDiffusionXLPipeline",
            "scheduler_class": scheduler.get("_class_name") == "EulerAncestralDiscreteScheduler",
            "prediction_type": scheduler.get("prediction_type") == "epsilon",
            "timestep_spacing": scheduler.get("timestep_spacing") == "trailing",
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise SDXLSupportInstallError(
                "SDXL_Turbo assets do not match the official SDXL-Turbo support contract: "
                + ", ".join(failed)
            )
        result["architecture_checks"] = checks

    return result


def _write_receipt(project_root: Path, payload: Mapping[str, Any]) -> Path:
    path = project_root / "artifacts" / "install" / "sdxl_support_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Install IMAGE_GEN Stable Diffusion XL lightweight runtime support assets from official "
            "Stability AI Hugging Face repositories. Heavy model weights are never downloaded."
        )
    )
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--profile",
        choices=("all", "base", "refiner", "turbo", "japanese"),
        default="all",
        help=(
            "Support set to install. 'all' installs Base, Refiner, and Turbo. 'japanese' installs the "
            "two gated Japanese support trees."
        ),
    )
    parser.add_argument(
        "--include-japanese",
        action="store_true",
        help="Also install gated Japanese SDXL + Japanese Stable CLIP lightweight support assets.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Redownload and replace support files even when a non-empty local copy already exists.",
    )
    parser.add_argument("--trigger", default="manual")
    parser.add_argument("--status-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    project_root = args.project_root.expanduser().resolve()
    runtime_assets_root = _runtime_assets_root(project_root)
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else (project_root / DEFAULT_MANIFEST).resolve()
    )
    manifest = _load_manifest(manifest_path)
    selected = _selected_profiles(
        manifest,
        profile=args.profile,
        include_japanese=bool(args.include_japanese),
    )
    required = _required_files(runtime_assets_root, selected)
    pending = [
        item
        for item in required
        if args.refresh or not _file_ready(item[3])
    ]

    print("IMAGE_GEN Stable Diffusion XL Support Installer")
    print("===============================================")
    print(f"Support ID: {manifest.get('support_id', 'sdxl-family-support')}")
    print(f"Runtime assets root: {runtime_assets_root}")
    print("Selected support trees:")
    for profile_id, profile in selected:
        gated = " [gated]" if bool(profile.get("gated")) else ""
        print(f"  {profile_id}: {_profile_root(runtime_assets_root, profile)}{gated}")
    print("Heavy checkpoint, UNet, VAE, and text-encoder weight files are never downloaded by this installer.")
    print(f"Required lightweight files: {len(required)}; downloads/replacements pending: {len(pending)}")

    if any(bool(profile.get("gated")) for _profile_id, profile in selected):
        print(
            "Japanese support uses gated Stability AI repositories. Accept the Hugging Face terms and "
            "authenticate with `hf auth login` or HF_TOKEN before installation."
        )

    if args.status_only:
        missing = [item for item in required if not _file_ready(item[3])]
        for profile_id, _profile, remote, destination in missing:
            print(f"MISSING [{profile_id}] {remote} -> {destination}")
        if not missing:
            print("All selected SDXL support files are present.")
        return 0 if not missing else 3

    if args.dry_run:
        print("Dry run: files that would be downloaded/replaced:")
        for profile_id, profile, remote, destination in pending:
            print(f"  [{profile_id}] {profile['repo_id']}:{remote} -> {destination}")
        return 0

    downloaded: list[dict[str, str]] = []
    for index, (profile_id, profile, remote, destination) in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] [{profile_id}] {remote}")
        _download_file(
            repo_id=str(profile["repo_id"]),
            revision=str(profile.get("revision") or "main"),
            remote_path=remote,
            destination=destination,
            refresh=bool(args.refresh),
            gated=bool(profile.get("gated")),
        )
        downloaded.append(
            {
                "profile": profile_id,
                "repo_id": str(profile["repo_id"]),
                "remote_path": remote,
                "destination": str(destination),
            }
        )

    validation: list[dict[str, Any]] = []
    for profile_id, profile in selected:
        root = _profile_root(runtime_assets_root, profile)
        validation.append(_validate_profile(profile_id, profile, root))

    receipt = _write_receipt(
        project_root,
        {
            "schema_version": 1,
            "support_id": manifest.get("support_id"),
            "trigger": args.trigger,
            "selection": args.profile,
            "include_japanese": bool(args.include_japanese or args.profile == "japanese"),
            "status": "installed" if downloaded else "already_ready",
            "runtime_assets_root": str(runtime_assets_root),
            "downloaded_count": len(downloaded),
            "downloaded_files": downloaded,
            "validation": validation,
        },
    )
    print(f"SDXL support is ready. Receipt: {receipt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SDXLSupportInstallError as exc:
        print(f"SDXL support installer error: {exc}", file=sys.stderr)
        raise SystemExit(1)
