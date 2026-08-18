from __future__ import annotations

import copy
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .artifact_io import _load_yaml_mapping, _slug, _utc_now, _write_yaml
from .contracts import (
    BlueprintSnapshot,
    QUALIFICATION_RUNNER_SCHEMA_VERSION,
    QualificationCase,
    RETEST_DEFAULT_TOKEN,
    RETEST_SCALAR_FIELDS,
    REVIEW_CHOICES,
    REVIEW_SCHEMA_VERSION,
)


class QualificationReviewsMixin:
    """Cohesive qualification-runner responsibility mixin used by the public facade."""

    def _review_payload(
        self,
        *,
        case: QualificationCase,
        case_dir: Path,
        run_dir: Path,
        manifest: Path | None,
        image: Path | None,
        technical_status: str,
        runtime_choices: Mapping[str, Any],
        component_choices: Mapping[str, Any],
    ) -> dict[str, Any]:
        current_components = {
            role: str(item.get("component_sha256") or "")
            for role, item in dict(case.resolved_composition.get("components") or {}).items()
        }
        payload = {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "case_id": case.case_id,
            "parent_case_id": case.parent_case_id or None,
            "image": self._relative_or_absolute(image, case_dir),
            "image_manifest": self._relative_or_absolute(manifest, case_dir),
            "technical_status": technical_status,
            "review": {
                # Human quality judgment is intentionally independent of technical success.
                "choice": "pending",
                "notes": "",
            },
            "review_choice_help": {
                "accepted": "Image quality is acceptable; positive quality evidence for this exact configuration.",
                "output_poor": "Generation completed but this output is not acceptable; do not infer technical incompatibility.",
                "component_compatibility_suspect": "The changed component combination appears responsible; still user evidence, not an automatic hard exclusion.",
                "execution_profile_suspect": "Sampler/scheduler/steps/guidance/backend behavior may be responsible.",
                "technical_failure": "The case failed technically or the output is unusable because of a runtime failure.",
                "inconclusive": "No quality conclusion yet.",
                "parity_match": "This output matches the control closely enough to accept digital runtime parity.",
                "parity_mismatch": "This output does not match the control; digital extraction parity is not proven.",
                "pending": "Not reviewed yet.",
            },
            "retest_with": {
                "enabled": False,
                "notes": "",
                "overrides": {
                    # Keep 'default' for no change. Replace only fields you want to test.
                    "steps": RETEST_DEFAULT_TOKEN,
                    "cfg_scale": RETEST_DEFAULT_TOKEN,
                    "seed": RETEST_DEFAULT_TOKEN,
                    "width": RETEST_DEFAULT_TOKEN,
                    "height": RETEST_DEFAULT_TOKEN,
                    "sampler_name": RETEST_DEFAULT_TOKEN,
                    "scheduler_name": RETEST_DEFAULT_TOKEN,
                },
                "components": {
                    role: RETEST_DEFAULT_TOKEN for role in sorted(current_components)
                },
            },
            "current_configuration": {
                "steps": case.request_payload.get("steps"),
                "cfg_scale": case.request_payload.get("cfg_scale"),
                "seed": case.request_payload.get("seed"),
                "width": case.request_payload.get("width"),
                "height": case.request_payload.get("height"),
                "sampler_name": case.request_payload.get("sampler_name"),
                "scheduler_name": case.request_payload.get("scheduler_name"),
                "components": current_components,
                "composition_sha256": case.resolved_composition.get("composition_sha256"),
            },
            "available_choices": {
                "samplers": list(runtime_choices.get("samplers") or []),
                "schedulers": list(runtime_choices.get("schedulers") or []),
                "components": copy.deepcopy(dict(component_choices)),
                "instructions": (
                    "For retest_with.components, type the exact component 'value' fingerprint shown above, "
                    "or leave 'default' for no change."
                ),
            },
        }
        if case.mutation_kind == "component_source_parity":
            payload["parity"] = {
                "reference_case_id": str(case.mutation.get("reference_case_id") or "control"),
                "component_role": str(case.mutation.get("component_role") or ""),
                "component_sha256": str(case.mutation.get("component_sha256") or ""),
                "source_kind": str(case.mutation.get("source_kind") or ""),
                "source_asset_id": case.mutation.get("source_asset_id"),
                "source_path": str(case.mutation.get("source_path") or ""),
                "instruction": "Compare this image against the control image. Use parity_match only when the visual result is equivalent enough to approve the digital extraction path.",
            }
        return payload

    @staticmethod
    def _load_review(review_path: Path) -> dict[str, Any]:
        payload = _load_yaml_mapping(review_path)
        if int(payload.get("schema_version") or 0) != REVIEW_SCHEMA_VERSION:
            raise ValueError(f"Unsupported review schema in {review_path}.")
        choice = str((payload.get("review") or {}).get("choice") or "").strip().lower()
        if choice not in REVIEW_CHOICES:
            raise ValueError(
                f"Invalid review.choice {choice!r} in {review_path}. Allowed: {', '.join(REVIEW_CHOICES)}"
            )
        return payload

    def collect_reviews(self, run_dir: str | Path) -> dict[str, Any]:
        root = Path(run_dir).expanduser().resolve()
        if not (root / "run.yaml").is_file():
            raise ValueError(f"Not a qualification run directory: {root}")
        items: list[dict[str, Any]] = []
        counts = {choice: 0 for choice in REVIEW_CHOICES}
        retest_requested = 0
        errors: list[str] = []
        for review_path in sorted((root / "cases").glob("*/review.yaml")):
            try:
                payload = self._load_review(review_path)
                choice = str(payload["review"]["choice"]).strip().lower()
                counts[choice] += 1
                if bool((payload.get("retest_with") or {}).get("enabled", False)):
                    retest_requested += 1
                image_value = payload.get("image")
                manifest_value = payload.get("image_manifest")
                case_dir = review_path.parent
                image = (case_dir / str(image_value)).resolve() if image_value else None
                manifest = (case_dir / str(manifest_value)).resolve() if manifest_value else None
                if image is not None and not image.is_file():
                    raise ValueError(f"Referenced image is missing: {image}")
                if manifest is not None and not manifest.is_file():
                    raise ValueError(f"Referenced image manifest is missing: {manifest}")
                items.append(
                    {
                        "case_id": str(payload.get("case_id") or case_dir.name),
                        "choice": choice,
                        "notes": str((payload.get("review") or {}).get("notes") or ""),
                        "technical_status": str(payload.get("technical_status") or ""),
                        "retest_requested": bool((payload.get("retest_with") or {}).get("enabled", False)),
                        "review_path": str(review_path.relative_to(root)),
                    }
                )
            except Exception as exc:
                errors.append(f"{review_path}: {exc}")
        summary = {
            "schema_version": QUALIFICATION_RUNNER_SCHEMA_VERSION,
            "collected_at_utc": _utc_now(),
            "run_dir": str(root),
            "counts": counts,
            "retest_requested": retest_requested,
            "errors": errors,
            "reviews": items,
        }
        _write_yaml(root / "review_summary.yaml", summary)
        return summary

    @staticmethod
    def _normalize_retest_scalar(field: str, value: Any) -> Any:
        if value is None or (isinstance(value, str) and value.strip().lower() == RETEST_DEFAULT_TOKEN):
            return None
        if field in {"steps", "seed", "width", "height"}:
            parsed = int(value)
            if field != "seed" and parsed <= 0:
                raise ValueError(f"Retest {field} must be positive.")
            return parsed
        if field == "cfg_scale":
            return float(value)
        return str(value).strip()

    def _retest_case_from_review(
        self,
        *,
        run_dir: Path,
        review_path: Path,
        review: Mapping[str, Any],
        blueprint: BlueprintSnapshot,
        retest_index: int,
    ) -> QualificationCase | None:
        retest = dict(review.get("retest_with") or {})
        if not bool(retest.get("enabled", False)):
            return None
        original_request = _load_yaml_mapping(review_path.parent / "request.yaml")
        original_case = _load_yaml_mapping(review_path.parent / "case.yaml")
        request_overrides: dict[str, Any] = {}
        for field in RETEST_SCALAR_FIELDS:
            value = self._normalize_retest_scalar(field, (retest.get("overrides") or {}).get(field))
            if value is not None:
                request_overrides[field] = value
        if "steps" in request_overrides:
            request_overrides["model_enforce_recommended_steps"] = False
        if "cfg_scale" in request_overrides:
            request_overrides["model_enforce_recommended_cfg"] = False

        original_components = dict(original_request.get("advanced_model_components") or blueprint.components)
        component_overrides: dict[str, str] = {}
        component_requested = dict(retest.get("components") or {})
        valid_choices = self.component_choices(blueprint)
        by_role = {
            role: {str(item.get("value") or "") for item in items}
            for role, items in valid_choices.items()
        }
        for role, raw_value in component_requested.items():
            value = str(raw_value or "").strip().lower()
            if not value or value == RETEST_DEFAULT_TOKEN:
                continue
            if value not in by_role.get(str(role), set()):
                raise ValueError(
                    f"Retest component {role}={value!r} is not an eligible {blueprint.family} choice."
                )
            component_overrides[str(role)] = value
        components = dict(original_components)
        components.update(component_overrides)
        label_parts = []
        for key, value in request_overrides.items():
            if not key.startswith("model_enforce_"):
                label_parts.append(f"{key}={value}")
        for role, value in component_overrides.items():
            label_parts.append(f"{role}={value[:8]}")
        label = "Retest " + (", ".join(label_parts) if label_parts else "unchanged configuration")
        parent = str(review.get("case_id") or original_case.get("case_id") or review_path.parent.name)
        case_id = f"retest-{_slug(parent)}-{retest_index:03d}"
        return self._build_case(
            blueprint=blueprint,
            base_request=original_request,
            case_id=case_id,
            label=label,
            mutation_kind="retest",
            mutation={
                "review_source": str(review_path.relative_to(run_dir)),
                "request_overrides": request_overrides,
                "component_overrides": component_overrides,
                "user_notes": str(retest.get("notes") or ""),
            },
            component_overrides=components,
            request_overrides=request_overrides,
            parent_case_id=parent,
        )

    def run_requested_retests(self, run_dir: str | Path) -> list[Path]:
        root = Path(run_dir).expanduser().resolve()
        run_record = _load_yaml_mapping(root / "run.yaml")
        blueprint_payload = dict(run_record.get("blueprint") or {})
        blueprint = BlueprintSnapshot(
            asset_id=int(blueprint_payload["asset_id"]),
            model_path=str(blueprint_payload["model_path"]),
            model_filename=str(blueprint_payload["model_filename"]),
            model_sha256=str(blueprint_payload.get("model_sha256") or ""),
            family=str(blueprint_payload["family"]),
            family_label=str(blueprint_payload.get("family_label") or blueprint_payload["family"]),
            base_weight_role=str(blueprint_payload["base_weight_role"]),
            components=dict(blueprint_payload.get("components") or {}),
            component_details=dict(blueprint_payload.get("component_details") or {}),
        )
        choices = self.runtime_choices()
        component_choices = self.component_choices(blueprint)
        created: list[Path] = []
        existing = sorted((root / "cases").glob("retest-*"))
        next_index = len(existing) + 1
        for review_path in sorted((root / "cases").glob("*/review.yaml")):
            review = self._load_review(review_path)
            case = self._retest_case_from_review(
                run_dir=root,
                review_path=review_path,
                review=review,
                blueprint=blueprint,
                retest_index=next_index,
            )
            if case is None:
                continue
            case_dir = root / "cases" / case.case_id
            if case_dir.exists():
                next_index += 1
                continue
            case_dir.mkdir(parents=True, exist_ok=False)
            request_payload = dict(case.request_payload)
            request_payload["output_dir"] = str(case_dir)
            request_payload["output_prefix"] = f"{case.case_id}-{{index:05d}}-{{seed}}"
            request_payload["save_images"] = True
            request_path = case_dir / "request.yaml"
            _write_yaml(request_path, request_payload)
            effective_request = case_dir / "effective_request.json"
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
            console_log = case_dir / "console.log"
            console_log.write_text(
                (completed.stdout or "") + ("\n" if completed.stdout and completed.stderr else "") + (completed.stderr or ""),
                encoding="utf-8",
            )
            manifests, images = self._discover_case_outputs(case_dir)
            technical_status = "success" if completed.returncode == 0 and manifests and images else "failure"
            case_record = {
                "schema_version": QUALIFICATION_RUNNER_SCHEMA_VERSION,
                "case_id": case.case_id,
                "label": case.label,
                "parent_case_id": case.parent_case_id,
                "mutation_kind": case.mutation_kind,
                "mutation": dict(case.mutation),
                "technical_status": technical_status,
                "return_code": int(completed.returncode),
                "command": command,
                "request": str(request_path.relative_to(root)),
                "effective_request": str(effective_request.relative_to(root)) if effective_request.is_file() else None,
                "console_log": str(console_log.relative_to(root)),
                "manifests": [str(path.relative_to(root)) for path in manifests],
                "images": [str(path.relative_to(root)) for path in images],
                "resolved_composition": dict(case.resolved_composition),
            }
            _write_yaml(case_dir / "case.yaml", case_record)
            _write_yaml(
                case_dir / "review.yaml",
                self._review_payload(
                    case=case,
                    case_dir=case_dir,
                    run_dir=root,
                    manifest=manifests[0] if manifests else None,
                    image=images[0] if images else None,
                    technical_status=technical_status,
                    runtime_choices=choices,
                    component_choices=component_choices,
                ),
            )
            run_record.setdefault("cases", []).append(case_record)
            run_record["last_retest_at_utc"] = _utc_now()
            _write_yaml(root / "run.yaml", run_record)
            # Disable the request only after a child case has been materialized.
            review_edit = copy.deepcopy(review)
            review_edit.setdefault("retest_with", {})["enabled"] = False
            review_edit["retest_with"]["materialized_case_id"] = case.case_id
            _write_yaml(review_path, review_edit)
            created.append(case_dir)
            next_index += 1
        return created
