from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


class ResourceMaterializationError(RuntimeError):
    """Raised when a requested filesystem resource is invalid or unsafe."""


@dataclass(frozen=True)
class MaterializationResult:
    kind: str
    path: str
    status: str
    detail: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "path": self.path,
            "status": self.status,
            "detail": self.detail,
        }


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _deep_get(mapping: Mapping[str, Any], dotted_key: str) -> Any:
    current: Any = mapping
    for part in dotted_key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted_key)
        current = current[part]
    return current


def _strip_inline_comment(value: str) -> str:
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote == '"':
            escaped = True
            continue
        if char in {'"', "'"}:
            if quote is None:
                quote = char
            elif quote == char:
                quote = None
            continue
        if char == "#" and quote is None:
            return value[:index].rstrip()
    return value.rstrip()


def _parse_simple_scalar(value: str) -> Any:
    value = _strip_inline_comment(value).strip()
    if not value:
        return None
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if value == "[]":
        return []
    if value == "{}":
        return {}
    return value


def _load_paths_fallback(config_path: Path) -> dict[str, Any]:
    """Read the top-level YAML ``paths`` mapping without third-party packages.

    The hardware-aware installer is intentionally able to start from the host
    Python before the IMAGE_GEN virtual environment exists.  For that reason
    this fallback only reads the simple scalar path values needed by setup.
    Once PyYAML is available, ``load_project_config`` uses the full YAML parser.
    """

    if not config_path.is_file():
        return {}
    paths: dict[str, Any] = {}
    in_paths = False
    paths_indent = 0
    for raw_line in config_path.read_text(encoding="utf-8-sig").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if not in_paths:
            if indent == 0 and stripped == "paths:":
                in_paths = True
                paths_indent = indent
            continue
        if indent <= paths_indent:
            break
        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            continue
        parsed = _parse_simple_scalar(value)
        if parsed is not None:
            paths[key] = parsed
    return {"paths": paths}


def load_project_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.is_file():
        return {}
    if config_path.suffix.lower() == ".json":
        payload = json.loads(config_path.read_text(encoding="utf-8-sig"))
        if not isinstance(payload, Mapping):
            raise ResourceMaterializationError(
                f"Project config must be an object: {config_path}"
            )
        return dict(payload)
    try:
        import yaml  # type: ignore
    except ImportError:
        return _load_paths_fallback(config_path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) or {}
    except Exception as exc:  # PyYAML exposes several parser exception classes.
        raise ResourceMaterializationError(
            f"Unable to parse project config {config_path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise ResourceMaterializationError(
            f"Project config must be a mapping: {config_path}"
        )
    return dict(payload)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceMaterializationError(f"Unable to load manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ResourceMaterializationError(f"Manifest must be a JSON object: {path}")
    if int(payload.get("schema_version", 0)) != 1:
        raise ResourceMaterializationError(
            f"Unsupported resource manifest schema_version in {path}: "
            f"{payload.get('schema_version')!r}"
        )
    resources = payload.get("resources")
    if not isinstance(resources, list):
        raise ResourceMaterializationError(
            f"Manifest 'resources' must be a list: {path}"
        )
    return dict(payload)


def _expand_path(value: str, *, project_root: Path) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(value))
    path = Path(expanded)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def resolve_path_spec(
    spec: Any,
    *,
    project_root: Path,
    config: Mapping[str, Any],
) -> Path:
    append: str | None = None
    if isinstance(spec, str):
        raw_path = spec
    elif isinstance(spec, Mapping):
        config_key = spec.get("config_key")
        fallback = spec.get("fallback")
        if config_key:
            try:
                raw_path = _deep_get(config, str(config_key))
            except KeyError:
                raw_path = fallback
        else:
            raw_path = spec.get("value", fallback)
        append_value = spec.get("append")
        append = str(append_value) if append_value is not None else None
    else:
        raise ResourceMaterializationError(
            f"Resource path must be a string or object, got {type(spec).__name__}."
        )
    if raw_path is None or not str(raw_path).strip():
        raise ResourceMaterializationError(
            f"Resource path could not be resolved from specification: {spec!r}"
        )
    target = _expand_path(str(raw_path), project_root=project_root)
    if append:
        target = (target / append).resolve()
    return target


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_target(path: Path, *, project_root: Path, allow_external: bool) -> None:
    if allow_external:
        return
    if not _is_within(path, project_root):
        raise ResourceMaterializationError(
            f"Refusing to create a resource outside the project root: {path}. "
            "Set allow_external=true only for an explicitly trusted request."
        )


def _directory_has_user_content(directory: Path, ignored_names: set[str]) -> bool:
    if not directory.exists():
        return False
    try:
        return any(child.name not in ignored_names for child in directory.iterdir())
    except OSError as exc:
        raise ResourceMaterializationError(
            f"Unable to inspect directory before writing placeholder content: {directory}: {exc}"
        ) from exc


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _render_content(request: Mapping[str, Any]) -> str:
    kind = str(request.get("kind") or "").lower()
    content = request.get("content")
    if kind == "json":
        return json.dumps(content, indent=2, sort_keys=True) + "\n"
    if not isinstance(content, str):
        raise ResourceMaterializationError(
            f"{kind!r} resource content must be a string."
        )
    return content if content.endswith("\n") else content + "\n"


def materialize_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    project_root: Path,
    config: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> list[MaterializationResult]:
    root = project_root.expanduser().resolve()
    selected_config: Mapping[str, Any] = config or {}
    results: list[MaterializationResult] = []

    for index, request in enumerate(requests):
        if not isinstance(request, Mapping):
            raise ResourceMaterializationError(
                f"Resource request #{index + 1} must be an object."
            )
        kind = str(request.get("kind") or "").strip().lower()
        if kind not in {"directory", "text", "json"}:
            raise ResourceMaterializationError(
                f"Unsupported resource kind {kind!r} in request #{index + 1}."
            )
        target = resolve_path_spec(
            request.get("path"), project_root=root, config=selected_config
        )
        allow_external = bool(request.get("allow_external", False))
        _validate_target(target, project_root=root, allow_external=allow_external)

        if kind == "directory":
            if target.exists() and not target.is_dir():
                raise ResourceMaterializationError(
                    f"Directory target already exists as a file: {target}"
                )
            if target.is_dir():
                results.append(
                    MaterializationResult(kind, str(target), "exists", "directory already exists")
                )
                continue
            if not dry_run:
                target.mkdir(parents=True, exist_ok=True)
            results.append(MaterializationResult(kind, str(target), "planned" if dry_run else "created"))
            continue

        parent = target.parent
        if parent.exists() and not parent.is_dir():
            raise ResourceMaterializationError(
                f"Parent path is not a directory: {parent}"
            )
        if not dry_run:
            parent.mkdir(parents=True, exist_ok=True)

        write_policy = str(request.get("write_policy") or "if_missing").lower()
        if write_policy not in {"if_missing", "overwrite"}:
            raise ResourceMaterializationError(
                f"Unsupported write_policy {write_policy!r} for {target}."
            )
        if target.exists() and target.is_dir():
            raise ResourceMaterializationError(
                f"File target already exists as a directory: {target}"
            )
        if target.exists() and write_policy == "if_missing":
            results.append(
                MaterializationResult(kind, str(target), "exists", "file preserved")
            )
            continue

        if bool(request.get("only_if_parent_empty", False)):
            ignored_names = {target.name}
            if _directory_has_user_content(parent, ignored_names):
                results.append(
                    MaterializationResult(
                        kind,
                        str(target),
                        "skipped",
                        "parent directory already contains user assets",
                    )
                )
                continue

        payload = _render_content(request)
        if not dry_run:
            _atomic_write_text(target, payload)
        results.append(
            MaterializationResult(kind, str(target), "planned" if dry_run else "created")
        )

    return results


def apply_manifest(
    manifest_path: Path,
    *,
    project_root: Path,
    project_config_path: Path | None = None,
    dry_run: bool = False,
) -> list[MaterializationResult]:
    manifest = load_manifest(manifest_path)
    config = load_project_config(project_config_path)
    resources = manifest["resources"]
    return materialize_requests(
        resources,
        project_root=project_root,
        config=config,
        dry_run=dry_run,
    )


def _parse_request_json(values: Iterable[str]) -> list[Mapping[str, Any]]:
    requests: list[Mapping[str, Any]] = []
    for raw in values:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ResourceMaterializationError(
                f"Invalid --request-json payload: {exc}"
            ) from exc
        if isinstance(payload, Mapping):
            requests.append(payload)
        elif isinstance(payload, list) and all(isinstance(item, Mapping) for item in payload):
            requests.extend(payload)
        else:
            raise ResourceMaterializationError(
                "--request-json must contain one resource object or a list of resource objects."
            )
    return requests


def _write_report(
    path: Path,
    *,
    project_root: Path,
    manifests: Sequence[Path],
    results: Sequence[MaterializationResult],
    dry_run: bool,
) -> None:
    payload = {
        "schema_version": 1,
        "project_root": str(project_root),
        "dry_run": dry_run,
        "manifests": [str(item) for item in manifests],
        "created": sum(result.status == "created" for result in results),
        "planned": sum(result.status == "planned" for result in results),
        "existing": sum(result.status == "exists" for result in results),
        "skipped": sum(result.status == "skipped" for result in results),
        "results": [result.to_dict() for result in results],
    }
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create IMAGE_GEN folders and safe text/JSON resources from declarative manifests "
            "or request objects. Existing files are preserved unless overwrite is explicit."
        )
    )
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help="JSON resource manifest to apply. May be supplied more than once.",
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        help=(
            "Optional IMAGE_GEN config used by manifest path specs such as "
            "paths.checkpoints_dir."
        ),
    )
    parser.add_argument(
        "--request-json",
        action="append",
        default=[],
        help="One JSON resource request or a JSON list of requests.",
    )
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    config_path = args.project_config
    if config_path is None:
        config_path = root / "user_config" / "user-config.yml"
    elif not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.expanduser().resolve()

    manifest_paths: list[Path] = []
    for path in args.manifest:
        selected = path if path.is_absolute() else root / path
        manifest_paths.append(selected.expanduser().resolve())

    try:
        if not manifest_paths and not args.request_json:
            raise ResourceMaterializationError(
                "At least one --manifest or --request-json input is required."
            )
        results: list[MaterializationResult] = []
        for manifest_path in manifest_paths:
            results.extend(
                apply_manifest(
                    manifest_path,
                    project_root=root,
                    project_config_path=config_path,
                    dry_run=args.dry_run,
                )
            )
        direct_requests = _parse_request_json(args.request_json)
        if direct_requests:
            config = load_project_config(config_path)
            results.extend(
                materialize_requests(
                    direct_requests,
                    project_root=root,
                    config=config,
                    dry_run=args.dry_run,
                )
            )

        for result in results:
            detail = f" - {result.detail}" if result.detail else ""
            print(f"[{result.status.upper()}] {result.kind}: {result.path}{detail}")

        if args.report_json:
            report_path = args.report_json
            if not report_path.is_absolute():
                report_path = root / report_path
            if not args.dry_run:
                _write_report(
                    report_path.resolve(),
                    project_root=root,
                    manifests=manifest_paths,
                    results=results,
                    dry_run=args.dry_run,
                )
        return 0
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
