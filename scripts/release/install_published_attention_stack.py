from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + subprocess.list2cmdline(command))
    subprocess.run(command, cwd=cwd, check=True)


def _matching_install_paths(site_packages: Path, package: str) -> list[Path]:
    normalized = package.replace("-", "_")
    paths: list[Path] = []
    direct = site_packages / normalized
    if direct.exists():
        paths.append(direct)
    for pattern in (
        f"{normalized}-*.dist-info",
        f"{package}-*.dist-info",
        f"{normalized}-*.egg-info",
        f"{package}-*.egg-info",
    ):
        paths.extend(site_packages.glob(pattern))
    return sorted({path.resolve() for path in paths})


def _remove(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            path.unlink()


def _backup_package(site_packages: Path, package: str, target: Path) -> list[str]:
    target.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in _matching_install_paths(site_packages, package):
        destination = target / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        copied.append(source.name)
    return copied


def _restore_package(site_packages: Path, package: str, backup: Path) -> None:
    _remove(_matching_install_paths(site_packages, package))
    if not backup.is_dir():
        return
    for source in backup.iterdir():
        destination = site_packages / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    with urllib.request.urlopen(url) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _resolve_wheel(
    record: dict[str, Any], wheel_dir: Path, *, allow_download: bool
) -> Path:
    target = wheel_dir / str(record["filename"])
    if not target.is_file():
        if not allow_download:
            raise FileNotFoundError(f"Required wheel was not found: {target}")
        _download(str(record["url"]), target)
    actual = _sha256(target)
    expected = str(record["sha256"]).lower()
    if actual.lower() != expected:
        raise RuntimeError(
            f"Wheel SHA256 mismatch for {target.name}: expected {expected}, got {actual}"
        )
    return target


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transactionally install the published SM120 MSLK and xFormers wheels."
    )
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--skip-gpu-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    python = args.python.expanduser().resolve()
    manifest_path = root / "modules" / "attention_runtime" / "release_stack_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    packages = dict(manifest.get("packages") or {})
    if set(packages) != {"mslk", "xformers"}:
        raise RuntimeError("Release manifest must contain exactly mslk and xformers packages")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    artifact_dir = root / "artifacts" / "attention_validation" / "published_wheel_install" / run_id
    wheel_dir = (
        args.wheel_dir.expanduser().resolve()
        if args.wheel_dir is not None
        else artifact_dir / "wheelhouse"
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    wheel_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": 2,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": str(python),
        "project_root": str(root),
        "site_packages": None,
        "wheels": {},
        "backups": {},
        "success": False,
        "rolled_back": False,
        "dry_run": bool(args.dry_run),
    }
    state_path = artifact_dir / "install_state.json"

    try:
        for name in ("mslk", "xformers"):
            wheel = _resolve_wheel(
                dict(packages[name]), wheel_dir, allow_download=not args.no_download
            )
            state["wheels"][name] = {
                "path": str(wheel),
                "sha256": _sha256(wheel),
            }
        if args.dry_run:
            state["success"] = True
            state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
            _write_json(state_path, state)
            print(f"PASS: wheel files and hashes verified. Report: {state_path}")
            return 0

        freeze_before = subprocess.check_output(
            [str(python), "-m", "pip", "freeze"], text=True
        )
        (artifact_dir / "pip_freeze_before.txt").write_text(
            freeze_before, encoding="utf-8"
        )
        site_packages_raw = subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_paths()['purelib'])",
            ],
            text=True,
        ).strip()
        site_packages = Path(site_packages_raw).resolve()
        state["site_packages"] = str(site_packages)
        backup_root = artifact_dir / "backup"
        for name in ("mslk", "xformers"):
            backup_dir = backup_root / name
            state["backups"][name] = {
                "path": str(backup_dir),
                "entries": _backup_package(site_packages, name, backup_dir),
            }
        _write_json(state_path, state)

        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                state["wheels"]["mslk"]["path"],
            ],
            cwd=root,
        )
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--force-reinstall",
                state["wheels"]["xformers"]["path"],
            ],
            cwd=root,
        )
        verification_dir = artifact_dir / "verification"
        command = [
            str(python),
            "-m",
            "image_gen.tools.verify_attention_stack",
            "--environment-only",
            "--strict",
            "--output-dir",
            str(verification_dir / "environment"),
        ]
        _run(command, cwd=root)
        if not args.skip_gpu_smoke:
            _run(
                [
                    str(python),
                    "-m",
                    "image_gen.tools.verify_attention_stack",
                    "--known-good-release-test",
                    "--strict",
                    "--output-dir",
                    str(verification_dir / "known_good_k512"),
                ],
                cwd=root,
            )
        freeze_after = subprocess.check_output(
            [str(python), "-m", "pip", "freeze"], text=True
        )
        (artifact_dir / "pip_freeze_after.txt").write_text(
            freeze_after, encoding="utf-8"
        )
        state["success"] = True
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(state_path, state)
        print(f"PASS: published attention stack installed. Backup: {artifact_dir}")
        return 0
    except Exception as exc:
        state["error"] = f"{type(exc).__name__}: {exc}"
        if state.get("site_packages"):
            site_packages = Path(str(state["site_packages"]))
            print("Installation or validation failed. Restoring previous packages...")
            for name in ("mslk", "xformers"):
                _restore_package(site_packages, name, artifact_dir / "backup" / name)
            state["rolled_back"] = True
        state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_json(state_path, state)
        print(f"FAIL: {state['error']}")
        print(f"Rollback/report directory: {artifact_dir}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
