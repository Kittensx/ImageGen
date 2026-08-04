from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AcceptanceCase:
    name: str
    phase: int
    sampler: str
    scheduler: str
    prompt: str
    expected: str
    parser_kwargs: dict[str, Any]
    batch_size: int = 1
    hires: bool = False


def _cases() -> list[AcceptanceCase]:
    return [
        AcceptanceCase(
            name="phase5_kes_horizontal",
            phase=5,
            sampler="kes",
            scheduler="simple_kes",
            prompt=(
                "wide coastal scene REGION{a small red brick house@0,0.5,0,1*1~none|"
                "a large blue whale breaching from the ocean@0.5,1,0,1*1~none}"
            ),
            expected="House is concentrated on the left; whale/ocean subject is concentrated on the right.",
            parser_kwargs={"region_overlap_policy": "normalize"},
        ),
        AcceptanceCase(
            name="phase5_euler_vertical",
            phase=5,
            sampler="simple_euler",
            scheduler="standard_karras",
            prompt="cinematic landscape REGION{dramatic storm clouds|green mountain valley}:V:0.4,0.6",
            expected="Cloud subject is concentrated in the upper region; valley subject is concentrated below.",
            parser_kwargs={"region_overlap_policy": "normalize"},
        ),
        AcceptanceCase(
            name="phase5_dpm_overlap",
            phase=5,
            sampler="dpmpp_2m",
            scheduler="standard_karras",
            prompt=(
                "surreal scene REGION{a glowing red house@0,0.7,0,1*0.9~none|"
                "a luminous blue whale@0.3,1,0,1*0.9~none}"
            ),
            expected="Both subjects appear in their zones; the center overlap shows additive regional influence without a crash.",
            parser_kwargs={"region_overlap_policy": "additive"},
        ),
        AcceptanceCase(
            name="phase5_hires_reconstruction",
            phase=5,
            sampler="simple_euler",
            scheduler="standard_karras",
            prompt=(
                "detailed illustration REGION{an old stone cottage@0,0.5,0,1|"
                "a white whale in deep water@0.5,1,0,1}"
            ),
            expected="The left/right boundary remains in the same relative position after the hires pass.",
            parser_kwargs={"region_overlap_policy": "normalize"},
            hires=True,
        ),
        AcceptanceCase(
            name="phase6_curves_blur_timeline",
            phase=6,
            sampler="kes",
            scheduler="simple_kes",
            prompt=(
                "fantasy panorama REGION{golden castle@0,0.5,0,1*1~ease-in|"
                "silver whale@0.5,1,0,1*1~sine-in-out|start=0.1|stop=0.9|"
                "blur=0.12|base_ratio=0.2}"
            ),
            expected="Both regions activate only during the configured window, with a visibly softer center boundary.",
            parser_kwargs={"region_overlap_policy": "normalize"},
        ),
        AcceptanceCase(
            name="phase6_batch_slot_telemetry",
            phase=6,
            sampler="simple_euler",
            scheduler="standard_karras",
            prompt=(
                "storybook scene REGION{<random:house,castle>@0,0.5,0,1|"
                "<random:whale,sailboat>@0.5,1,0,1}"
            ),
            expected="Two batch images may resolve different subjects; each saved sidecar must contain only its own slot telemetry.",
            parser_kwargs={
                "region_overlap_policy": "normalize",
                "prompt_expansion_scope": "per_image",
            },
            batch_size=2,
        ),
    ]


def _request_payload(case: AcceptanceCase, *, model: Path, output_dir: Path, steps: int) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "positive_prompt": case.prompt,
        "negative_prompt": "blurry, malformed, low quality",
        "seed": 230500 + case.phase * 100 + len(case.name),
        "steps": int(steps),
        "cfg_scale": 7.0,
        "width": 512,
        "height": 512,
        "batch_size": int(case.batch_size),
        "batch_count": 1,
        "unlimited": False,
        "model_path": str(model),
        "sampler_name": case.sampler,
        "scheduler_name": case.scheduler,
        "scheduler_kwargs": {},
        "sampler_kwargs": {},
        "parser_kwargs": {},
        "prompt_parser_name": "superhybrid",
        "prompt_parser_kwargs": dict(case.parser_kwargs),
        "prompt_shortcut_profile_name": "superhybrid_native",
        "save_images": True,
        "output_dir": str(output_dir),
        "output_prefix": f"{case.name}-{{index:05d}}-{{seed}}",
    }
    if case.hires:
        payload.update(
            {
                "hires_enabled": True,
                "hires_size_mode": "scale_from_base",
                "hires_scale": 1.5,
                "hires_steps": max(6, int(steps)),
                "hires_denoising_strength": 0.45,
                "hires_step_policy": "a1111_fixed_steps_v1",
                "hires_upscaler": "latent_bilinear",
                "hires_prompt_parser_mode": "same_as_base",
                "hires_shortcut_profile_mode": "same_as_base",
            }
        )
    return payload


def _sidecars(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.glob("*.json")
        if not path.name.endswith("request.json") and not path.name.endswith("report.json")
    )


def _validate_sidecar(path: Path, case: AcceptanceCase) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    extra = dict((payload.get("optional_for_rerun") or {}).get("extra") or {})
    region_passes = dict(extra.get("region_pass_records") or {})
    runtime_passes = dict(extra.get("regional_runtime_passes") or {})
    required_passes = ["base", "hires"] if case.hires else ["base"]
    checks: list[dict[str, Any]] = []
    for pass_name in required_passes:
        region = dict(region_passes.get(pass_name) or {})
        runtime = dict(runtime_passes.get(pass_name) or {})
        checks.extend(
            [
                {"name": f"{pass_name}_region_record", "passed": bool(region)},
                {"name": f"{pass_name}_runtime_estimate", "passed": bool(region.get("runtime_estimate"))},
                {"name": f"{pass_name}_runtime_telemetry", "passed": bool(runtime)},
                {
                    "name": f"{pass_name}_regional_unet_calls",
                    "passed": int(runtime.get("regional_unet_calls", 0) or 0) > 0,
                    "value": int(runtime.get("regional_unet_calls", 0) or 0),
                },
                {
                    "name": f"{pass_name}_regional_host_elapsed",
                    "passed": float(runtime.get("regional_host_elapsed_ms", runtime.get("regional_unet_duration_ms", -1.0)) or 0.0) >= 0.0,
                    "value": float(runtime.get("regional_host_elapsed_ms", runtime.get("regional_unet_duration_ms", 0.0)) or 0.0),
                },
                {
                    "name": f"{pass_name}_timing_semantics",
                    "passed": str(runtime.get("timing_semantics") or "") == "host_elapsed_unsynchronized",
                    "value": str(runtime.get("timing_semantics") or ""),
                },
            ]
        )
    if case.batch_size > 1:
        base = dict(region_passes.get("base") or {})
        runtime = dict(runtime_passes.get("base") or {})
        checks.extend(
            [
                {"name": "projected_region_slot", "passed": int(base.get("slot_count", 0) or 0) == 1},
                {"name": "projected_runtime_slot", "passed": len(list(runtime.get("regions") or [])) > 0},
                {
                    "name": "original_batch_runtime_retained",
                    "passed": bool(extra.get("batch_regional_runtime_passes")),
                },
            ]
        )
    return {
        "sidecar": str(path),
        "passed": all(bool(item["passed"]) for item in checks),
        "checks": checks,
        "region_passes": list(region_passes),
        "runtime_passes": list(runtime_passes),
    }


def _run_command(command: list[str], *, cwd: Path, log_path: Path) -> tuple[int, float]:
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        process = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT, check=False)
    return process.returncode, time.perf_counter() - started


def _write_checklist(root: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# SuperHybrid REGION GPU Acceptance Checklist",
        "",
        "Open each saved image and confirm the visual expectation. Automated metadata checks do not prove composition quality.",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"## {result['name']}",
                "",
                f"- Phase: {result['phase']}",
                f"- Sampler: `{result['sampler']}` / `{result['scheduler']}`",
                f"- Automated status: **{'PASS' if result['passed'] else 'FAIL'}**",
                f"- Visual expectation: {result['expected']}",
                "- Manual result: [ ] PASS  [ ] FAIL",
                "- Notes:",
                "",
            ]
        )
        for image in result.get("images", []):
            lines.append(f"  - `{image}`")
        lines.append("")
    lines.extend(
        [
            "## Phase 6 WebUI checks",
            "",
            "- [ ] Open **Open REGION Builder** and confirm drag, resize, grid, and paint-mask modes work.",
            "- [ ] Apply the builder result to the base prompt and validate the prompt preview.",
            "- [ ] Repeat using the hires positive-prompt target.",
            "- [ ] Confirm the base and hires activation timelines match `start` and `stop`.",
            "- [ ] Confirm the runtime estimate appears before generation.",
            "- [ ] In Output Details, confirm per-region UNet calls, duration, and mask-cache counters appear.",
            "- [ ] Replay at least one saved image exactly and confirm REGION replay remains locked.",
            "",
        ]
    )
    (root / "ACCEPTANCE_CHECKLIST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-GPU SuperHybrid Phase 5 and Phase 6 REGION acceptance jobs.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite", choices=("quick", "full"), default="full")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--runtime-args", default="", help="Optional IMAGE_GEN runtime flags such as --xformers --lowvram.")
    args = parser.parse_args()

    project_root = args.project_root.expanduser().resolve()
    model = args.model.expanduser()
    if not model.is_absolute():
        model = (project_root / model).resolve()
    if not model.is_file():
        parser.error(f"Model does not exist: {model}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    root = (args.output_dir or (project_root / "test_results" / "superhybrid_region_acceptance" / stamp)).expanduser()
    if not root.is_absolute():
        root = (project_root / root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    cases = _cases()
    if args.suite == "quick":
        cases = [cases[0], cases[3], cases[4]]
    runtime_args = [item.strip("\"") for item in shlex.split(str(args.runtime_args or ""), posix=False)]
    results: list[dict[str, Any]] = []
    first_sidecar: Path | None = None
    for case in cases:
        case_root = root / case.name
        output_dir = case_root / "images"
        output_dir.mkdir(parents=True, exist_ok=True)
        request_path = case_root / "request.json"
        request_path.write_text(
            json.dumps(
                _request_payload(case, model=model, output_dir=output_dir, steps=max(4, int(args.steps))),
                indent=2,
                ensure_ascii=False,
            ) + "\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            "-m",
            "modules.txt2img.cli",
            "run",
            *runtime_args,
            "--project-root",
            str(project_root),
            "--config",
            str(request_path),
            "--save",
        ]
        code, duration = _run_command(command, cwd=project_root, log_path=case_root / "generation.log")
        sidecars = _sidecars(output_dir) if code == 0 else []
        if first_sidecar is None and sidecars:
            first_sidecar = sidecars[0]
        sidecar_reports = [_validate_sidecar(path, case) for path in sidecars]
        expected_count = max(1, int(case.batch_size))
        passed = code == 0 and len(sidecars) >= expected_count and all(item["passed"] for item in sidecar_reports)
        images = sorted(
            str(path)
            for path in output_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        result = {
            "name": case.name,
            "phase": case.phase,
            "sampler": case.sampler,
            "scheduler": case.scheduler,
            "expected": case.expected,
            "return_code": code,
            "duration_seconds": round(duration, 3),
            "sidecar_count": len(sidecars),
            "images": images,
            "passed": passed,
            "sidecars": sidecar_reports,
            "request": str(request_path),
            "log": str(case_root / "generation.log"),
        }
        results.append(result)
        print(f"[{('PASS' if passed else 'FAIL')}] {case.name} ({duration:.1f}s)")
        if not passed and not args.keep_going:
            break

    replay_result: dict[str, Any] | None = None
    if first_sidecar is not None and (not results or all(item["passed"] for item in results)):
        replay_root = root / "phase6_exact_replay"
        replay_root.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "modules.txt2img.cli",
            "run",
            *runtime_args,
            "--project-root",
            str(project_root),
            "--manifest",
            str(first_sidecar),
            "--output-dir",
            str(replay_root / "images"),
            "--prefix",
            "region-replay-{index:05d}-{seed}",
            "--save",
        ]
        code, duration = _run_command(command, cwd=project_root, log_path=replay_root / "generation.log")
        replay_sidecars = _sidecars(replay_root / "images") if code == 0 else []
        replay_locked = False
        if replay_sidecars:
            payload = json.loads(replay_sidecars[0].read_text(encoding="utf-8"))
            passes = dict(((payload.get("optional_for_rerun") or {}).get("extra") or {}).get("region_pass_records") or {})
            replay_locked = bool((passes.get("base") or {}).get("replay_locked"))
        replay_result = {
            "return_code": code,
            "duration_seconds": round(duration, 3),
            "source_sidecar": str(first_sidecar),
            "output_sidecars": [str(item) for item in replay_sidecars],
            "region_replay_locked": replay_locked,
            "passed": code == 0 and bool(replay_sidecars) and replay_locked,
        }
        print(f"[{('PASS' if replay_result['passed'] else 'FAIL')}] phase6_exact_replay ({duration:.1f}s)")

    report = {
        "contract_version": "image-gen-superhybrid-region-acceptance-v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(project_root),
        "python": sys.executable,
        "model": str(model),
        "suite": args.suite,
        "steps": max(4, int(args.steps)),
        "runtime_args": runtime_args,
        "results": results,
        "exact_replay": replay_result,
    }
    report["passed"] = bool(results) and all(item["passed"] for item in results) and bool(replay_result and replay_result["passed"])
    (root / "acceptance_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_checklist(root, results)
    print(f"Acceptance report: {root / 'acceptance_report.json'}")
    print(f"Manual checklist:  {root / 'ACCEPTANCE_CHECKLIST.md'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
