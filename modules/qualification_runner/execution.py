from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import _pixel_sha256, _safe_folder_name, _slug, _utc_now, _write_yaml
from .contracts import QUALIFICATION_RUNNER_SCHEMA_VERSION, QualificationCase


class QualificationExecutionMixin:
    """Cohesive qualification-runner responsibility mixin used by the public facade."""

    def default_output_root(self) -> Path:
        return Path(self.context.output_root).expanduser().resolve() / "component_qualification_runner"

    @staticmethod
    def model_output_folder_name(model_filename: str | Path) -> str:
        return _safe_folder_name(model_filename)

    def model_output_root(self, model_filename: str | Path) -> Path:
        return self.default_output_root() / self.model_output_folder_name(model_filename)

    def _write_output_indexes(self, *, root_dir: Path, model_dir: Path, run_record: Mapping[str, Any], run_dir: Path) -> None:
        model_index_path = model_dir / "index.json"
        root_index_path = root_dir / "index.json"
        summary = {
            "run_id": run_record.get("run_id"),
            "created_at_utc": run_record.get("created_at_utc"),
            "model_filename": (run_record.get("blueprint") or {}).get("model_filename"),
            "model_family": (run_record.get("blueprint") or {}).get("family"),
            "pattern_id": (run_record.get("pattern") or {}).get("pattern_id"),
            "pattern_label": (run_record.get("pattern") or {}).get("label"),
            "run_dir": str(run_dir),
            "relative_run_dir": str(run_dir.relative_to(model_dir.parent)),
            "case_count": len(list(run_record.get("cases") or [])),
        }

        def _merge_index(path: Path, key: str | None = None) -> dict[str, Any]:
            if path.is_file():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, dict):
                    payload = {}
            else:
                payload = {}
            return payload

        model_index = _merge_index(model_index_path)
        model_index.setdefault("schema_version", QUALIFICATION_RUNNER_SCHEMA_VERSION)
        model_index["model_folder"] = model_dir.name
        model_index["model_filename"] = summary["model_filename"]
        model_index["model_family"] = summary["model_family"]
        runs = list(model_index.get("runs") or [])
        runs = [item for item in runs if isinstance(item, dict) and item.get("run_id") != summary["run_id"]]
        runs.append(summary)
        runs.sort(key=lambda item: str(item.get("created_at_utc") or ""), reverse=True)
        model_index["runs"] = runs
        model_index_path.write_text(json.dumps(model_index, indent=2), encoding="utf-8")

        root_index = _merge_index(root_index_path)
        root_index.setdefault("schema_version", QUALIFICATION_RUNNER_SCHEMA_VERSION)
        models = dict(root_index.get("models") or {})
        models[model_dir.name] = {
            "model_folder": model_dir.name,
            "model_filename": summary["model_filename"],
            "model_family": summary["model_family"],
            "model_dir": str(model_dir),
            "relative_model_dir": str(model_dir.relative_to(root_dir)),
            "latest_run_id": summary["run_id"],
            "latest_run_dir": str(run_dir),
            "latest_relative_run_dir": str(run_dir.relative_to(root_dir)),
            "run_count": len(runs),
            "index_file": str(model_index_path),
        }
        root_index["models"] = dict(sorted(models.items(), key=lambda item: item[0].lower()))
        root_index_path.write_text(json.dumps(root_index, indent=2), encoding="utf-8")

    def _discover_case_outputs(self, case_dir: Path) -> tuple[list[Path], list[Path]]:
        manifests: list[Path] = []
        images: list[Path] = []
        for path in sorted(case_dir.glob("*.json")):
            if path.name in {"case.json", "effective_request.json"} or path.name.startswith("request-"):
                continue
            if path.name.endswith("-diagnostics.json"):
                continue
            manifests.append(path)
            for suffix in (".png", ".webp", ".jpg", ".jpeg"):
                image = path.with_suffix(suffix)
                if image.is_file():
                    images.append(image)
                    break
        return manifests, images

    @staticmethod
    def _relative_or_absolute(path: Path | None, base: Path) -> str | None:
        if path is None:
            return None
        try:
            return str(path.resolve().relative_to(base.resolve()))
        except ValueError:
            return str(path.resolve())

    def run_plan(
        self,
        plan: Mapping[str, Any],
        *,
        output_root: str | Path,
        stop_after_control_failure: bool = True,
    ) -> Path:
        selected_root = Path(output_root).expanduser().resolve()
        blueprint = dict(plan.get("blueprint") or {})
        pattern = dict(plan.get("pattern") or {})
        root_dir = selected_root
        root_dir.mkdir(parents=True, exist_ok=True)
        model_dir = root_dir / self.model_output_folder_name(str(blueprint.get("model_filename") or "model"))
        model_dir.mkdir(parents=True, exist_ok=True)
        run_id = (
            datetime.now().strftime("%Y%m%d-%H%M%S")
            + "-"
            + _slug(pattern.get("pattern_id") or "pattern")[:24]
        )
        run_dir = model_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_yaml(run_dir / "plan.yaml", dict(plan))

        run_record: dict[str, Any] = {
            "schema_version": QUALIFICATION_RUNNER_SCHEMA_VERSION,
            "run_id": run_id,
            "created_at_utc": _utc_now(),
            "project_root": str(self.context.project_root),
            "registry_db_path": str(self.context.registry_db_path),
            "registry_db_mtime_ns": (
                self.context.registry_db_path.stat().st_mtime_ns
                if self.context.registry_db_path.is_file()
                else None
            ),
            "python": str(self.python_executable),
            "blueprint": blueprint,
            "pattern": pattern,
            "cases": [],
        }

        control_pixel_sha256 = ""
        for case_payload in plan.get("cases") or []:
            case = QualificationCase(
                case_id=str(case_payload["case_id"]),
                label=str(case_payload.get("label") or case_payload["case_id"]),
                mutation_kind=str(case_payload.get("mutation_kind") or ""),
                mutation=dict(case_payload.get("mutation") or {}),
                request_payload=dict(case_payload.get("request_payload") or {}),
                resolved_composition=dict(case_payload.get("resolved_composition") or {}),
                parent_case_id=str(case_payload.get("parent_case_id") or ""),
            )
            case_dir = run_dir / "cases" / case.case_id
            case_dir.mkdir(parents=True, exist_ok=True)
            request_payload = dict(case.request_payload)
            request_payload["output_dir"] = str(case_dir)
            request_payload["output_prefix"] = f"{case.case_id}-{{index:05d}}-{{seed}}"
            request_payload["save_images"] = True
            request_path = case_dir / "request.yaml"
            _write_yaml(request_path, request_payload)
            effective_request = case_dir / "effective_request.json"
            console_log = case_dir / "console.log"

            command = [
                str(self.python_executable),
                "-m",
                "modules.txt2img.cli",
                "run",
                "--project-root",
                str(self.context.project_root),
                "--config",
                str(request_path),
                "--effective-request-out",
                str(effective_request),
                "--save",
                "--output-dir",
                str(case_dir),
                "--prefix",
                request_payload["output_prefix"],
            ]
            completed = subprocess.run(
                command,
                cwd=self.context.project_root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            console_log.write_text(
                (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or ""),
                encoding="utf-8",
            )
            manifests, images = self._discover_case_outputs(case_dir)
            technical_status = "success" if completed.returncode == 0 and manifests and images else "failure"
            pixel_sha256 = _pixel_sha256(images[0] if images else None)
            if case.case_id == "control":
                control_pixel_sha256 = pixel_sha256
            automatic_parity = "not_applicable"
            if case.mutation_kind == "component_source_parity":
                if pixel_sha256 and control_pixel_sha256:
                    automatic_parity = "exact_pixel_match" if pixel_sha256 == control_pixel_sha256 else "pixel_difference"
                else:
                    automatic_parity = "unavailable"
            case_record = {
                "schema_version": QUALIFICATION_RUNNER_SCHEMA_VERSION,
                "case_id": case.case_id,
                "label": case.label,
                "parent_case_id": case.parent_case_id or None,
                "mutation_kind": case.mutation_kind,
                "mutation": dict(case.mutation),
                "technical_status": technical_status,
                "return_code": int(completed.returncode),
                "command": command,
                "request": self._relative_or_absolute(request_path, run_dir),
                "effective_request": self._relative_or_absolute(effective_request if effective_request.is_file() else None, run_dir),
                "console_log": self._relative_or_absolute(console_log, run_dir),
                "manifests": [self._relative_or_absolute(path, run_dir) for path in manifests],
                "images": [self._relative_or_absolute(path, run_dir) for path in images],
                "image_pixel_sha256": pixel_sha256 or None,
                "automatic_parity": automatic_parity,
                "resolved_composition": dict(case.resolved_composition),
            }
            _write_yaml(case_dir / "case.yaml", case_record)
            review = self._review_payload(
                case=case,
                case_dir=case_dir,
                run_dir=run_dir,
                manifest=manifests[0] if manifests else None,
                image=images[0] if images else None,
                technical_status=technical_status,
                runtime_choices=dict(plan.get("runtime_choices") or {}),
                component_choices=dict(plan.get("component_choices") or {}),
            )
            if case.mutation_kind == "component_source_parity":
                review["automatic_parity"] = {
                    "status": automatic_parity,
                    "control_pixel_sha256": control_pixel_sha256 or None,
                    "case_pixel_sha256": pixel_sha256 or None,
                    "note": "Exact pixel equality is strong automatic evidence, but manual review remains required during this experimental phase.",
                }
            _write_yaml(case_dir / "review.yaml", review)
            run_record["cases"].append(case_record)
            _write_yaml(run_dir / "run.yaml", run_record)

            if case.case_id == "control" and technical_status != "success" and stop_after_control_failure:
                run_record["stopped_after_control_failure"] = True
                run_record["finished_at_utc"] = _utc_now()
                _write_yaml(run_dir / "run.yaml", run_record)
                return run_dir

        run_record["finished_at_utc"] = _utc_now()
        _write_yaml(run_dir / "run.yaml", run_record)
        self._write_output_indexes(root_dir=root_dir, model_dir=model_dir, run_record=run_record, run_dir=run_dir)
        return run_dir
