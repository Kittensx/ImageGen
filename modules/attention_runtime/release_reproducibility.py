from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import platform
import re
import sys
from pathlib import Path
from threading import RLock
from typing import Any, Mapping
from urllib.parse import unquote, urlparse

_MANIFEST_NAME = "release_stack_manifest.json"
_CACHE_LOCK = RLock()
_CACHED_RUNTIME_REPORT: dict[str, Any] | None = None


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def manifest_path() -> Path:
    return Path(__file__).resolve().with_name(_MANIFEST_NAME)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_release_manifest(path: str | Path | None = None) -> dict[str, Any]:
    target = Path(path).resolve() if path is not None else manifest_path()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Release manifest must contain a JSON object: {target}")
    result = dict(payload)
    result["_manifest_path"] = str(target)
    result["_manifest_sha256"] = sha256_file(target)
    return result


def _distribution(name: str) -> importlib.metadata.Distribution | None:
    names = (name, "triton" if name == "triton-windows" else name)
    for candidate in names:
        try:
            return importlib.metadata.distribution(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _distribution_version(name: str) -> str | None:
    distribution = _distribution(name)
    if distribution is not None:
        try:
            return distribution.version
        except Exception:
            pass
    module_name = "triton" if name == "triton-windows" else name.replace("-", "_")
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _module_path(name: str) -> str | None:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if spec is None or spec.origin is None:
        return None
    try:
        return str(Path(spec.origin).resolve())
    except Exception:
        return str(spec.origin)


def _distribution_root(distribution: importlib.metadata.Distribution | None) -> str | None:
    if distribution is None:
        return None
    try:
        return str(Path(distribution.locate_file("")).resolve())
    except Exception:
        return None


def _direct_url(distribution: importlib.metadata.Distribution | None) -> dict[str, Any] | None:
    if distribution is None:
        return None
    try:
        raw = distribution.read_text("direct_url.json")
    except Exception:
        return None
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except Exception:
        return {"parse_error": True, "raw": raw}
    return value if isinstance(value, dict) else None


def _archive_hash_from_direct_url(value: Mapping[str, Any] | None) -> str | None:
    if not value:
        return None
    archive = value.get("archive_info")
    if not isinstance(archive, Mapping):
        return None
    hashes = archive.get("hashes")
    if isinstance(hashes, Mapping) and hashes.get("sha256"):
        return str(hashes["sha256"]).lower()
    raw = archive.get("hash")
    if isinstance(raw, str) and raw.lower().startswith("sha256="):
        return raw.split("=", 1)[1].lower()
    return None


def _filename_from_url(value: Mapping[str, Any] | None) -> str | None:
    if not value:
        return None
    raw = value.get("url")
    if not isinstance(raw, str):
        return None
    try:
        return unquote(Path(urlparse(raw).path).name)
    except Exception:
        return None


def _read_requirement_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _requirement_include_target(line: str) -> str | None:
    value = line.strip()
    for prefix in ("--requirement=", "--requirement ", "-r "):
        if value.startswith(prefix):
            target = value[len(prefix):].strip()
            return target.strip('"\'') or None
    if value.startswith("-r") and len(value) > 2 and not value[2].isspace():
        target = value[2:].strip()
        return target.strip('"\'') or None
    return None


def resolve_requirement_tree(path: Path) -> dict[str, Any]:
    """Resolve a pip requirement entry point including nested ``-r`` files.

    The setup layout intentionally splits the validated environment across several
    requirement files. Runtime compatibility checks must therefore evaluate the
    effective requirement graph rather than only the top-level wrapper file.
    """

    root = path.resolve()
    lines: list[str] = []
    files: list[Path] = []
    missing: list[Path] = []
    cycles: list[list[Path]] = []
    visited: set[Path] = set()

    def visit(current: Path, stack: tuple[Path, ...]) -> None:
        resolved = current.resolve()
        if resolved in stack:
            start = stack.index(resolved)
            cycles.append([*stack[start:], resolved])
            return
        if resolved in visited:
            return
        visited.add(resolved)
        if not resolved.is_file():
            missing.append(resolved)
            return
        files.append(resolved)
        next_stack = (*stack, resolved)
        for line in _read_requirement_lines(resolved):
            include_target = _requirement_include_target(line)
            if include_target is None:
                lines.append(line)
                continue
            include_path = Path(include_target)
            if not include_path.is_absolute():
                include_path = resolved.parent / include_path
            visit(include_path, next_stack)

    visit(root, ())
    return {
        "lines": lines,
        "files": files,
        "missing": missing,
        "cycles": cycles,
    }


def _find_requirement(lines: list[str], package: str) -> str | None:
    pattern = re.compile(rf"^{re.escape(package)}(?:\s*@|==|>=|<=|~=|!=|>|<|\s*$)", re.I)
    return next((line for line in lines if pattern.match(line)), None)


def _package_provenance(name: str, expected: Mapping[str, Any]) -> dict[str, Any]:
    distribution = _distribution(name)
    direct = _direct_url(distribution)
    archive_hash = _archive_hash_from_direct_url(direct)
    observed_version = _distribution_version(name)
    expected_hash = str(expected.get("sha256") or "").lower() or None
    expected_filename = str(expected.get("filename") or "") or None
    observed_filename = _filename_from_url(direct)
    return {
        "package": name,
        "installed": distribution is not None or _module_path(name) is not None,
        "version": observed_version,
        "expected_observed_version": expected.get("observed_version"),
        "version_is_compatibility_gate": bool(
            expected.get("version_is_compatibility_gate", False)
        ),
        "module_path": _module_path(name),
        "distribution_root": _distribution_root(distribution),
        "direct_url": direct,
        "observed_archive_sha256": archive_hash,
        "expected_archive_sha256": expected_hash,
        "archive_sha256_matches": (
            archive_hash == expected_hash
            if archive_hash is not None and expected_hash is not None
            else None
        ),
        "observed_filename": observed_filename,
        "expected_filename": expected_filename,
        "filename_matches": (
            observed_filename == expected_filename
            if observed_filename is not None and expected_filename is not None
            else None
        ),
        "release_tag": expected.get("release_tag"),
        "release_url": expected.get("url"),
    }


def _runtime_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_major_minor": [sys.version_info.major, sys.version_info.minor],
        "platform": platform.platform(),
        "torch_version": _distribution_version("torch"),
        "torch_cuda_version": None,
        "cuda_available": False,
        "gpu_name": None,
        "compute_capability": None,
        "triton_distribution_version": _distribution_version("triton-windows"),
        "triton_module_version": _distribution_version("triton"),
    }
    try:
        import torch

        result["torch_version"] = str(torch.__version__)
        result["torch_cuda_version"] = (
            None if torch.version.cuda is None else str(torch.version.cuda)
        )
        result["cuda_available"] = bool(torch.cuda.is_available())
        if result["cuda_available"]:
            index = int(torch.cuda.current_device())
            result["gpu_name"] = str(torch.cuda.get_device_name(index))
            result["compute_capability"] = list(
                torch.cuda.get_device_capability(index)
            )
    except Exception as exc:
        result["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _append(
    collection: list[dict[str, Any]],
    field: str,
    expected: Any,
    actual: Any,
    detail: str,
) -> None:
    collection.append(
        {
            "field": field,
            "expected": expected,
            "actual": actual,
            "detail": detail,
        }
    )


def _check_requirements(
    manifest: Mapping[str, Any], root: Path, errors: list[dict[str, Any]]
) -> dict[str, Any]:
    requirements = dict(manifest.get("requirements") or {})
    reports: dict[str, Any] = {}
    packages = dict(manifest.get("packages") or {})
    for key, relative in requirements.items():
        path = root / str(relative)
        tree = resolve_requirement_tree(path)
        lines = list(tree["lines"])
        resolved_files = list(tree["files"])
        missing_files = list(tree["missing"])
        cycles = list(tree["cycles"])
        reports[key] = {
            "path": str(relative).replace("\\", "/"),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "resolved_files": [
                str(item.relative_to(root)).replace("\\", "/")
                if item.is_relative_to(root)
                else str(item)
                for item in resolved_files
            ],
            "missing_includes": [str(item) for item in missing_files],
            "include_cycles": [
                [str(item) for item in cycle]
                for cycle in cycles
            ],
        }
        if not path.is_file():
            _append(errors, f"requirements.{key}", "file exists", "missing", str(path))
            continue
        for missing_path in missing_files:
            _append(
                errors,
                f"requirements.{key}.include",
                "included requirement file exists",
                "missing",
                str(missing_path),
            )
        for cycle in cycles:
            _append(
                errors,
                f"requirements.{key}.include_cycle",
                "acyclic requirement includes",
                [str(item) for item in cycle],
                "Recursive requirement include cycle detected.",
            )
        text = "\n".join(lines)
        if re.search(r"(?im)^logging==0\.4\.9\.6\s*$", text):
            _append(
                errors,
                f"requirements.{key}.logging",
                "obsolete package absent",
                "logging==0.4.9.6",
                "Use Python's standard-library logging module.",
            )
        for package_name, package_record in packages.items():
            requirement = _find_requirement(lines, package_name)
            reports[key].setdefault("packages", {})[package_name] = requirement
            if requirement is None:
                _append(
                    errors,
                    f"requirements.{key}.{package_name}",
                    package_record.get("url"),
                    None,
                    "Published wheel requirement is missing.",
                )
                continue
            expected_hash = str(package_record.get("sha256") or "").lower()
            if expected_hash and f"sha256={expected_hash}" not in requirement.lower():
                _append(
                    errors,
                    f"requirements.{key}.{package_name}.sha256",
                    expected_hash,
                    requirement,
                    "Requirement does not pin the published wheel hash.",
                )
    return reports


def _collect_xformers_contract(required_api: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "imported": False,
        "required_api": required_api,
        "available_api": {},
        "api_complete": False,
        "runtime_provenance": None,
        "provider_registry": None,
        "error": None,
    }
    try:
        fmha = importlib.import_module("xformers.ops.fmha")
        report["imported"] = True
        report["available_api"] = {
            name: callable(getattr(fmha, name, None)) for name in required_api
        }
        report["api_complete"] = all(report["available_api"].values())
        provenance_getter = getattr(fmha, "get_runtime_package_provenance", None)
        registry_getter = getattr(fmha, "get_fmha_provider_registry", None)
        if callable(provenance_getter):
            report["runtime_provenance"] = provenance_getter()
        if callable(registry_getter):
            report["provider_registry"] = registry_getter()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def _collect_mslk_profile() -> dict[str, Any]:
    try:
        splitk = importlib.import_module("mslk.attention.fmha.triton_splitk")
        getter = getattr(splitk, "get_production_validation_diagnostics")
        result = dict(getter())
        result["module_path"] = str(Path(splitk.__file__).resolve())
        return result
    except Exception as exc:
        return {
            "valid": False,
            "validated_head_dimensions": [],
            "error": f"{type(exc).__name__}: {exc}",
        }


def verify_release_stack(
    *,
    root: str | Path | None = None,
    manifest: Mapping[str, Any] | None = None,
    include_runtime: bool = True,
    force_reload: bool = False,
) -> dict[str, Any]:
    global _CACHED_RUNTIME_REPORT
    use_cache = root is None and manifest is None and include_runtime
    with _CACHE_LOCK:
        if use_cache and _CACHED_RUNTIME_REPORT is not None and not force_reload:
            return json.loads(json.dumps(_CACHED_RUNTIME_REPORT))

    resolved_root = Path(root).resolve() if root is not None else project_root()
    contract = dict(manifest) if manifest is not None else load_release_manifest()
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    requirements_report = _check_requirements(contract, resolved_root, errors)
    package_records = {
        name: _package_provenance(name, record)
        for name, record in dict(contract.get("packages") or {}).items()
    }

    for name, record in package_records.items():
        if include_runtime and not record["installed"]:
            _append(
                errors,
                f"packages.{name}.installed",
                True,
                False,
                "Required package is not installed in the active Python environment.",
            )
        if record.get("archive_sha256_matches") is False:
            _append(
                warnings,
                f"packages.{name}.published_wheel_sha256",
                record.get("expected_archive_sha256"),
                record.get("observed_archive_sha256"),
                "The active install did not originate from the published wheel hash. Runtime compatibility is evaluated separately from release provenance.",
            )
        if include_runtime and record.get("archive_sha256_matches") is None:
            _append(
                warnings,
                f"packages.{name}.published_wheel_sha256",
                record.get("expected_archive_sha256"),
                None,
                "The installer provenance hash is unavailable, usually because the package was installed from an unpacked directory or copied into site-packages.",
            )
        expected_version = record.get("expected_observed_version")
        if (
            include_runtime
            and expected_version
            and record.get("version") != expected_version
        ):
            _append(
                warnings,
                f"packages.{name}.version",
                expected_version,
                record.get("version"),
                "Package versions are diagnostic only and are not used as the runtime compatibility gate.",
            )

    runtime = _runtime_snapshot() if include_runtime else None
    runtime_contract = dict(contract.get("runtime_contract") or {})
    xformers_contract = None
    mslk_profile = None
    if include_runtime:
        required_api = [str(v) for v in runtime_contract.get("required_xformers_api") or ()]
        xformers_contract = _collect_xformers_contract(required_api)
        mslk_profile = _collect_mslk_profile()

        if not xformers_contract.get("api_complete"):
            _append(
                errors,
                "xformers.required_api",
                required_api,
                xformers_contract.get("available_api"),
                xformers_contract.get("error") or "Required FMHA diagnostics or explicit-execution API is missing.",
            )
        if not mslk_profile.get("valid"):
            _append(
                errors,
                "mslk.production_profile.valid",
                True,
                mslk_profile.get("valid"),
                mslk_profile.get("error")
                or repr(mslk_profile.get("mismatches") or []),
            )
        for field in ("provider", "operator", "dtype"):
            expected = runtime_contract.get(field)
            actual = mslk_profile.get(field)
            if expected is not None and actual != expected:
                _append(
                    errors,
                    f"mslk.production_profile.{field}",
                    expected,
                    actual,
                    "Active MSLK capability profile does not satisfy the provider contract.",
                )
        expected_mappings = {
            str(key): int(value)
            for key, value in dict(
                runtime_contract.get("logical_tile_mappings") or {}
            ).items()
        }
        actual_mappings = {
            str(key): int(value)
            for key, value in dict(
                mslk_profile.get("logical_tile_mappings") or {}
            ).items()
        }
        if expected_mappings != actual_mappings:
            _append(
                errors,
                "mslk.production_profile.logical_tile_mappings",
                expected_mappings,
                actual_mappings,
                "Logical-to-tile mappings differ from the validated runtime contract.",
            )
        expected_dims = sorted(int(v) for v in expected_mappings)
        actual_dims = sorted(
            int(v) for v in mslk_profile.get("validated_head_dimensions") or ()
        )
        if not set(expected_dims).issubset(actual_dims):
            _append(
                errors,
                "mslk.production_profile.validated_head_dimensions",
                expected_dims,
                actual_dims,
                "One or more required logical dimensions are not enabled for normal dispatch.",
            )
        expected_capability = runtime_contract.get("compute_capability")
        if runtime and runtime.get("cuda_available"):
            if runtime.get("compute_capability") != expected_capability:
                _append(
                    errors,
                    "runtime.compute_capability",
                    expected_capability,
                    runtime.get("compute_capability"),
                    "This published validation contract targets SM120.",
                )
        elif runtime and runtime.get("cuda_available") is False:
            _append(
                errors,
                "runtime.cuda_available",
                True,
                False,
                "The custom provider requires CUDA; automatic mode may fall back to a PyTorch backend.",
            )

    release_provenance_valid = all(
        record.get("archive_sha256_matches") is True for record in package_records.values()
    ) if include_runtime else True
    runtime_compatible = not errors
    report = {
        "schema_version": 2,
        "release_id": contract.get("release_id"),
        "installation_mode": contract.get("installation_mode"),
        "manifest_path": contract.get("_manifest_path"),
        "manifest_sha256": contract.get("_manifest_sha256"),
        "valid": runtime_compatible,
        "runtime_compatible": runtime_compatible,
        "release_provenance_valid": release_provenance_valid,
        "package_versions_are_diagnostic_only": True,
        "errors": errors,
        "warnings": warnings,
        "requirements": requirements_report,
        "packages": package_records,
        "runtime": runtime,
        "runtime_contract": runtime_contract,
        "mslk_profile": mslk_profile,
        "xformers_contract": xformers_contract,
    }
    with _CACHE_LOCK:
        if use_cache:
            _CACHED_RUNTIME_REPORT = json.loads(json.dumps(report))
    return report


def require_release_compatible_stack() -> dict[str, Any]:
    report = verify_release_stack()
    if not report.get("runtime_compatible"):
        details = "; ".join(
            f"{item.get('field')}: {item.get('detail') or item.get('actual')}"
            for item in report.get("errors") or ()
        )
        raise RuntimeError(
            "Published SM120 attention stack is not runtime-compatible"
            + (f": {details}" if details else ".")
        )
    return report


def summarize_release_report(report: Mapping[str, Any]) -> list[str]:
    return [
        f"Release ID: {report.get('release_id')}",
        f"Installation mode: {report.get('installation_mode')}",
        f"Runtime compatible: {bool(report.get('runtime_compatible'))}",
        f"Published-wheel provenance verified: {bool(report.get('release_provenance_valid'))}",
        f"Errors: {len(report.get('errors') or ())}",
        f"Warnings: {len(report.get('warnings') or ())}",
    ]


def reset_release_verification_cache_for_testing() -> None:
    global _CACHED_RUNTIME_REPORT
    with _CACHE_LOCK:
        _CACHED_RUNTIME_REPORT = None


__all__ = [
    "load_release_manifest",
    "manifest_path",
    "project_root",
    "resolve_requirement_tree",
    "require_release_compatible_stack",
    "reset_release_verification_cache_for_testing",
    "sha256_file",
    "summarize_release_report",
    "verify_release_stack",
]
