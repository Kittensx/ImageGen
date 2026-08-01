from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def _project_root() -> Path:
    """Return the IMAGE_GEN source root, not the ``src`` directory.

    This module lives at ``src/image_gen/tools/verify_attention_stack.py``.
    The previous implementation used ``parents[2]``, which resolves to
    ``<project>/src`` and caused release-contract checks to look for the
    requirements files under ``src/requirements``.  Discovering the root by
    its release manifest and requirements directory is resilient to editable
    installs and source-tree relocation.
    """

    resolved = Path(__file__).resolve()
    for candidate in resolved.parents:
        if (
            (candidate / "modules" / "attention_runtime" / "release_stack_manifest.json").is_file()
            and (candidate / "requirements" / "requirements-blackwell.txt").is_file()
        ):
            return candidate

    # Expected source-tree layout: <project>/src/image_gen/tools/<file>.
    return resolved.parents[3]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _failure(failures: list[dict[str, Any]], stage: str, exc: BaseException) -> None:
    failures.append(
        {
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    )


def _environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "IMAGE_GEN_ATTENTION_BACKEND",
                "MSLK_FMHA_POLICY",
                "MSLK_FMHA_DEBUG",
                "MSLK_FMHA_BLOCK_M",
                "MSLK_FMHA_BLOCK_N",
                "MSLK_FMHA_NUM_WARPS",
                "MSLK_FMHA_NUM_STAGES",
                "MSLK_FMHA_EXPERIMENTAL_HEAD_DIMS",
                "MSLK_FMHA_VALIDATION_PROFILE",
                "MSLK_FMHA_DISABLE_VALIDATED_PROFILE",
            )
        },
    }
    try:
        import torch

        result.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            index = int(torch.cuda.current_device())
            result.update(
                {
                    "cuda_device_index": index,
                    "gpu_name": torch.cuda.get_device_name(index),
                    "compute_capability": list(
                        torch.cuda.get_device_capability(index)
                    ),
                }
            )
    except Exception as exc:
        result["torch_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _execute_case(
    *,
    head_dim: int,
    query_length: int,
    key_value_length: int,
    heads: int,
    operator: str,
    compare_sdpa: bool,
    benchmark: bool,
    repeat_count: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from xformers.ops.fmha import execute_forward_with_evidence

    device = torch.device("cuda")
    dtype = torch.float16
    torch.manual_seed(140000 + head_dim + query_length + key_value_length + heads)
    query = torch.randn(
        (1, query_length, heads, head_dim), device=device, dtype=dtype
    )
    key = torch.randn(
        (1, key_value_length, heads, head_dim), device=device, dtype=dtype
    )
    value = torch.randn_like(key)
    scale = 1.0 / math.sqrt(head_dim)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    output, evidence = execute_forward_with_evidence(
        query,
        key,
        value,
        operator=operator,
        scale=scale,
    )
    torch.cuda.synchronize(device)
    first_ms = (time.perf_counter() - started) * 1000.0
    item: dict[str, Any] = evidence.to_dict()
    item.update(
        {
            "head_dimension": head_dim,
            "heads": heads,
            "query_length": query_length,
            "key_value_length": key_value_length,
            "attention_kind": (
                "cross" if query_length != key_value_length else "self"
            ),
            "scale": scale,
            "first_call_duration_ms": round(first_ms, 3),
            "first_call_peak_allocated_vram_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "finite": bool(
                output is not None and torch.isfinite(output).all().item()
            ),
        }
    )
    if output is None:
        return item

    if compare_sdpa:
        reference = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3),
            key.permute(0, 2, 1, 3),
            value.permute(0, 2, 1, 3),
            dropout_p=0.0,
            scale=scale,
        ).permute(0, 2, 1, 3)
        diff = (output.float() - reference.float()).abs()
        item["numerical_comparison"] = {
            "allclose": bool(
                torch.allclose(
                    output.float(), reference.float(), atol=0.05, rtol=0.05
                )
            ),
            "max_absolute_error": float(diff.max().item()),
            "mean_absolute_error": float(diff.mean().item()),
            "reference_finite": bool(torch.isfinite(reference).all().item()),
        }

    if benchmark:
        timings: list[float] = []
        repeat_outputs: list[Any] = []
        for _ in range(max(1, repeat_count)):
            torch.cuda.synchronize(device)
            warm_started = time.perf_counter()
            warm_output, warm_evidence = execute_forward_with_evidence(
                query,
                key,
                value,
                operator=operator,
                scale=scale,
            )
            torch.cuda.synchronize(device)
            timings.append((time.perf_counter() - warm_started) * 1000.0)
            if warm_output is not None and warm_evidence.executed:
                repeat_outputs.append(warm_output)
        repeat_exact = all(
            torch.equal(output, repeated) for repeated in repeat_outputs
        ) and len(repeat_outputs) == max(1, repeat_count)
        item["benchmark"] = {
            "repeat_count": max(1, repeat_count),
            "warm_call_duration_ms": [round(value, 3) for value in timings],
            "warm_call_mean_ms": (
                round(sum(timings) / len(timings), 3) if timings else None
            ),
            "repeat_outputs_exact": repeat_exact,
        }
    return item


def _default_cases(include_k512: bool) -> list[dict[str, int]]:
    # Eight self/cross sequence layouts x two head counts = 16 cases per
    # logical dimension, preserving the published 48-case validation shape.
    sequence_pairs = (
        (2, 2),
        (77, 77),
        (256, 256),
        (1024, 1024),
        (2, 77),
        (77, 2),
        (256, 77),
        (1024, 77),
    )
    cases = [
        {"head_dim": dim, "query_length": q, "key_value_length": kv, "heads": heads}
        for dim in (40, 80, 160)
        for heads in (1, 8)
        for q, kv in sequence_pairs
    ]
    if include_k512:
        cases.append(
            {
                "head_dim": 512,
                "query_length": 9600,
                "key_value_length": 9600,
                "heads": 1,
            }
        )
    return cases


def _model_signature_cases(path: Path) -> list[dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    layouts = payload.get("unique_layouts") if isinstance(payload, dict) else None
    result: list[dict[str, int]] = []
    for layout in layouts or ():
        if not isinstance(layout, dict):
            continue
        values = {
            "head_dim": layout.get("q_head_dim"),
            "heads": layout.get("heads"),
            "query_length": 2,
            "key_value_length": (
                77 if layout.get("attention_kind") == "cross" else 2
            ),
        }
        if all(isinstance(value, int) and value > 0 for value in values.values()):
            result.append({key: int(value) for key, value in values.items()})
    return result


def _summary_lines(report: dict[str, Any]) -> list[str]:
    matrix = list(report.get("compatibility_matrix") or ())
    passed_cases = sum(
        1
        for item in matrix
        if item.get("executed")
        and item.get("finite")
        and item.get("numerical_comparison", {}).get("allclose", True)
    )
    release = report.get("release", {})
    contract_errors = list(release.get("errors") or ())
    return [
        "# Published SM120 attention-stack validation",
        "",
        f"- Release ID: `{release.get('release_id')}`",
        f"- Runtime contract: `{'PASS' if release.get('runtime_compatible') else 'FAIL'}`",
        f"- Runtime contract errors: `{len(contract_errors)}`",
        f"- Published-wheel provenance: `{'PASS' if release.get('release_provenance_valid') else 'UNVERIFIED'}`",
        f"- Explicit operator: `{report.get('operator')}`",
        f"- GPU cases passed: `{passed_cases}/{len(matrix)}`",
        f"- Execution failures: `{len(report.get('failures') or ())}`",
        f"- Overall: `{'PASS' if report.get('passed') else 'FAIL'}`",
        "",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify the published Windows SM120 xFormers/MSLK attention stack."
    )
    modes = parser.add_argument_group("validation modes")
    modes.add_argument("--environment-only", action="store_true")
    modes.add_argument("--known-good-release-test", action="store_true")
    modes.add_argument("--model", type=Path)
    modes.add_argument("--static-model-signature", action="store_true")
    modes.add_argument("--model-signature", type=Path)
    modes.add_argument("--capture-first-unet-step", action="store_true")
    modes.add_argument("--model-layouts", action="store_true")
    modes.add_argument("--compare-sdpa", action="store_true")
    modes.add_argument("--benchmark", action="store_true")
    modes.add_argument("--full", action="store_true")
    parser.add_argument(
        "--operator",
        choices=("triton_splitk", "cutlass_blackwell"),
        default="triton_splitk",
    )
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    root = _project_root()
    for candidate in (root, root / "src"):
        value = str(candidate)
        if value not in sys.path:
            sys.path.insert(0, value)

    os.environ.setdefault("MSLK_FMHA_POLICY", "blackwell_safe")
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    output = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else root / "artifacts" / "attention_validation" / run_id
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "failures").mkdir(exist_ok=True)
    failures: list[dict[str, Any]] = []

    environment = _environment()
    _write_json(output / "environment.json", environment)

    from modules.attention_runtime.release_reproducibility import verify_release_stack

    release = verify_release_stack(root=root, force_reload=True)
    _write_json(output / "package_provenance.json", release)

    xformers_build: dict[str, Any] = {}
    provider_registry: dict[str, Any] = {}
    mslk_settings: dict[str, Any] = {}
    try:
        from xformers.ops.fmha import (
            get_fmha_provider_registry,
            get_runtime_package_provenance,
        )

        xformers_build = get_runtime_package_provenance()
        provider_registry = get_fmha_provider_registry(include_backward=True)
    except Exception as exc:
        _failure(failures, "xformers_diagnostics", exc)
    _write_json(output / "xformers_build.json", xformers_build)
    _write_json(output / "provider_registry.json", provider_registry)

    try:
        from mslk.attention.fmha import triton_splitk
        from mslk.attention.fmha.fmha_tuning_policy import (
            get_fmha_environment_snapshot,
            get_fmha_first_use_evidence,
        )

        mslk_settings = {
            "environment": get_fmha_environment_snapshot(),
            "first_use": get_fmha_first_use_evidence(),
            "production_profile": triton_splitk.get_production_validation_diagnostics(),
            "head_dimension_policy": triton_splitk.get_experimental_head_dim_diagnostics(),
        }
    except Exception as exc:
        _failure(failures, "mslk_diagnostics", exc)
    _write_json(output / "mslk_active_settings.json", mslk_settings)

    run_gpu = not args.environment_only and (
        args.full
        or args.known_good_release_test
        or args.model_layouts
        or args.compare_sdpa
        or args.benchmark
    )
    matrix: list[dict[str, Any]] = []
    operator_name = (
        "triton_splitKF" if args.operator == "triton_splitk" else "cutlassF-blackwell"
    )
    if run_gpu:
        if not environment.get("cuda_available"):
            failures.append(
                {
                    "stage": "gpu_validation",
                    "error_type": "CUDAUnavailable",
                    "error": "CUDA is unavailable in the active Python environment.",
                }
            )
        else:
            cases: list[dict[str, int]] = []
            if args.full or args.compare_sdpa or args.benchmark:
                cases.extend(_default_cases(include_k512=True))
            elif args.known_good_release_test:
                cases.extend(_default_cases(include_k512=True)[-1:])
            if args.model_layouts:
                signature_path = args.model_signature
                if signature_path is None and args.model and args.static_model_signature:
                    candidate = args.model.with_suffix(args.model.suffix + ".attention-signature.json")
                    if candidate.is_file():
                        signature_path = candidate
                if signature_path is None:
                    failures.append(
                        {
                            "stage": "model_layouts",
                            "error_type": "ModelSignatureRequired",
                            "error": "Use --model-signature PATH for model-layout execution. A static signature cannot be derived from checkpoint weights without loading the UNet.",
                        }
                    )
                else:
                    try:
                        cases.extend(_model_signature_cases(signature_path.resolve()))
                        _write_json(
                            output / "model_attention_signature.json",
                            json.loads(signature_path.read_text(encoding="utf-8")),
                        )
                    except Exception as exc:
                        _failure(failures, "model_signature", exc)
            unique: list[dict[str, int]] = []
            seen: set[tuple[int, int, int, int]] = set()
            for case in cases:
                key = (
                    case["head_dim"],
                    case["query_length"],
                    case["key_value_length"],
                    case["heads"],
                )
                if key not in seen:
                    seen.add(key)
                    unique.append(case)
            for case in unique:
                try:
                    matrix.append(
                        _execute_case(
                            **case,
                            operator=operator_name,
                            compare_sdpa=bool(args.full or args.compare_sdpa),
                            benchmark=bool(args.full or args.benchmark),
                            repeat_count=args.repeat_count,
                        )
                    )
                except Exception as exc:
                    _failure(
                        failures,
                        f"operator_{case['head_dim']}_{case['query_length']}_{case['key_value_length']}_{case['heads']}",
                        exc,
                    )
                    matrix.append({**case, "executed": False, "error": str(exc)})

    _write_json(output / "compatibility_matrix.json", matrix)
    _write_json(
        output / "numerical_comparison.json",
        [
            {
                "head_dimension": item.get("head_dimension"),
                "heads": item.get("heads"),
                "query_length": item.get("query_length"),
                "key_value_length": item.get("key_value_length"),
                "comparison": item.get("numerical_comparison"),
            }
            for item in matrix
            if item.get("numerical_comparison") is not None
        ],
    )
    _write_json(
        output / "performance.json",
        [
            {
                "head_dimension": item.get("head_dimension"),
                "heads": item.get("heads"),
                "query_length": item.get("query_length"),
                "key_value_length": item.get("key_value_length"),
                "first_call_duration_ms": item.get("first_call_duration_ms"),
                "first_call_peak_allocated_vram_bytes": item.get(
                    "first_call_peak_allocated_vram_bytes"
                ),
                "benchmark": item.get("benchmark"),
            }
            for item in matrix
        ],
    )

    if args.capture_first_unet_step:
        command = [
            str(root / ".venv" / "Scripts" / "python.exe"),
            "-m",
            "modules.txt2img.cli",
            "run",
            "--xformers",
            "--steps",
            "1",
            "--seed",
            "140522",
            "--width",
            "512",
            "--height",
            "512",
            "--prompt",
            "attention validation",
            "--negative-prompt",
            "",
            "--no-save",
        ]
        if args.model:
            command.extend(["--model", str(args.model.resolve())])
        _write_json(
            output / "first_unet_step_command.json",
            {
                "executed": False,
                "reason": "The utility records the deterministic application command; use the combined GPU acceptance BAT to execute model and hires tests.",
                "command": command,
            },
        )

    for index, item in enumerate(failures, start=1):
        _write_json(output / "failures" / f"failure-{index:03d}.json", item)

    case_pass = all(
        item.get("executed")
        and item.get("finite")
        and item.get("numerical_comparison", {}).get("allclose", True)
        for item in matrix
    )
    passed = bool(release.get("runtime_compatible")) and not failures and case_pass
    if args.environment_only:
        passed = bool(release.get("runtime_compatible")) and not failures
    report = {
        "schema_version": 2,
        "output_directory": str(output),
        "operator": operator_name,
        "release": release,
        "environment": environment,
        "compatibility_matrix": matrix,
        "failures": failures,
        "passed": passed,
    }
    _write_json(output / "validation_summary.json", report)
    lines = _summary_lines(report)
    (output / "benchmark_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"Artifacts: {output}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
