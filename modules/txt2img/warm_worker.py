from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import traceback
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from image_gen.runtime_options import (
    RuntimeStartupOptions,
    build_runtime_startup_status,
    add_runtime_startup_arguments,
    argv_for_primary_parser,
    prebootstrap_runtime_startup,
)

# Apply import-time runtime environment before Torch, xformers, MSLK, Triton,
# registry, or model modules are imported.
_PREBOOTSTRAP_RUNTIME_STARTUP_OPTIONS = prebootstrap_runtime_startup(sys.argv[1:])

import torch

from image_gen.runtime.scheduler_settings import normalize_scheduler_payload
from image_gen.systems.registry import RuntimeRegistrySystem
from modules.project_context import ProjectContext
from modules.txt2img.cli import _build_prompt_adapter
from modules.txt2img.request_loader import load_request_payload, payload_to_generation_request
from modules.txt2img.seed_utils import iter_batch_base_seeds, offset_seed
from modules.txt2img.txt2img_runner import Txt2ImgRunner

_STATUS_PREFIX = "WARM_WORKER_STATUS_JSON: "
_READY_PREFIX = "WARM_WORKER_READY_JSON: "
_COMPLETE_PREFIX = "WARM_WORKER_COMMAND_COMPLETE_JSON: "


def _emit(prefix: str, payload: dict[str, Any]) -> None:
    print(prefix + json.dumps(payload, ensure_ascii=False, sort_keys=True), flush=True)


def _utc_timestamp() -> float:
    return time.time()


class PersistentTxt2ImgWarmWorker:
    def __init__(
        self,
        context: ProjectContext,
        runtime_startup_options: RuntimeStartupOptions | None = None,
    ) -> None:
        self.context = context
        self.runtime_startup_options = (
            runtime_startup_options or _PREBOOTSTRAP_RUNTIME_STARTUP_OPTIONS
        )
        self.registry_system = RuntimeRegistrySystem(project_context=context)
        self.live_sampler_map = self.registry_system.legacy_map("sampler")
        self.live_scheduler_map = self.registry_system.legacy_map("scheduler")
        self.runner: Txt2ImgRunner | None = None
        self.stage = "idle"
        self.last_error: str | None = None
        self.last_transition_unix = _utc_timestamp()
        self.current_job_id: str | None = None
        self.selected_model_path: str | None = None
        self.timings: dict[str, Any] = {}
        self.runtime_settings: dict[str, Any] = {
            "warm_worker_execution_device": "cuda_preferred",
            "warm_worker_retention_device": "auto",
            "runtime_startup_options": self.runtime_startup_options.to_dict(),
        }

    def _ensure_runner(self) -> Txt2ImgRunner:
        if self.runner is not None:
            return self.runner
        from modules.load_safetensors_model import LoadModel

        self.runner = Txt2ImgRunner(
            prompt_adapter_factory=_build_prompt_adapter,
            model_loader=LoadModel(project_context=self.context),
            project_context=self.context,
            registry_system=self.registry_system,
        )
        return self.runner

    def _status_payload(self, *, stage: str | None = None, **extra: Any) -> dict[str, Any]:
        if stage is not None:
            self.stage = str(stage)
            self.last_transition_unix = _utc_timestamp()
        warm = self.runner.warm_model_status() if self.runner is not None else {
            "warm": False,
            "model_path": None,
            "cache_entries": 0,
            "cpu_loaded": False,
            "gpu_loaded": False,
            "component_devices": {},
            "cuda_memory": {"allocated_bytes": 0, "reserved_bytes": 0},
        }
        return {
            "schema_version": 1,
            "worker_pid": os.getpid(),
            "stage": self.stage,
            "warm_state": "warm" if warm.get("warm") else "cold",
            "selected_model_path": self.selected_model_path,
            "current_model_path": warm.get("model_path"),
            "cpu_loaded": bool(warm.get("cpu_loaded")),
            "gpu_loaded": bool(warm.get("gpu_loaded")),
            "component_devices": dict(warm.get("component_devices") or {}),
            "memory": dict(warm.get("cuda_memory") or {}),
            "cache_entries": int(warm.get("cache_entries") or 0),
            "current_job_id": self.current_job_id,
            "last_transition_unix": self.last_transition_unix,
            "last_error": self.last_error,
            "cuda_available": bool(torch.cuda.is_available()),
            "execution_device_policy": str(self.runtime_settings.get("warm_worker_execution_device") or "cuda_preferred"),
            "retention_device_policy": str(self.runtime_settings.get("warm_worker_retention_device") or "auto"),
            "execution_device": str(self.runtime_settings.get("last_execution_device") or (warm.get("component_devices") or {}).get("unet") or ("cuda" if torch.cuda.is_available() else "cpu")),
            "cpu_fallback_reason": self.runtime_settings.get("cpu_fallback_reason"),
            "timings": dict(self.timings),
            **extra,
        }

    def emit_status(self, stage: str | None = None, **extra: Any) -> dict[str, Any]:
        payload = self._status_payload(stage=stage, **extra)
        _emit(_STATUS_PREFIX, payload)
        return payload

    def _runner_event(self, payload: dict[str, Any]) -> None:
        stage = str(payload.get("stage") or "preparing_model")
        if stage == "model_ready":
            self.timings["checkpoint_hydration_time_ms"] = payload.get(
                "checkpoint_hydration_time_ms"
            )
            self.timings["gpu_transfer_included"] = payload.get("gpu_transfer_included")
            self.runtime_settings["cpu_fallback_reason"] = payload.get("cpu_fallback_reason")
            self.runtime_settings["last_execution_device"] = payload.get("execution_device")
        self.emit_status(stage, **{key: value for key, value in payload.items() if key != "stage"})

    def preload(self, command: dict[str, Any]) -> dict[str, Any]:
        model_path = str(command.get("model_path") or "").strip()
        if not model_path:
            raise ValueError("A model_path is required for warm preload.")
        self.selected_model_path = model_path
        self.current_job_id = None
        self.last_error = None
        self.emit_status("preparing_model", action="preload")
        started = time.perf_counter()
        extras = dict(command.get("runtime_settings") or {})
        self.runtime_settings.update(extras)
        runner = self._ensure_runner()
        current = runner.warm_model_status()
        current_path = str(current.get("model_path") or "")
        if current_path and os.path.normcase(str(Path(current_path).resolve())) != os.path.normcase(str(Path(model_path).resolve())):
            self.emit_status("unloading", action="automatic_model_swap", previous_model_path=current_path, next_model_path=model_path)
            runner.clear_model_cache()
        extras["warm_worker_event_callback"] = self._runner_event
        result = runner.preload_model(model_path, extras)
        self.emit_status("applying_retention_policy")
        retention = runner.apply_warm_retention(extras)
        self.timings.update(
            {
                "preload_time_ms": result.get("preload_time_ms"),
                "checkpoint_hydration_time_ms": (
                    result.get("model_provenance") or {}
                ).get("checkpoint_hydration_time_ms"),
                "first_step_warmup_time_ms": None,
                "first_step_warmup_performed": False,
            }
        )
        elapsed = round((time.perf_counter() - started) * 1000.0, 3)
        return self.emit_status(
            "ready",
            action="preload_complete",
            preload_time_ms=elapsed,
            model_provenance=dict(result.get("model_provenance") or {}),
            retention=retention,
        )

    def unload(self, command: dict[str, Any]) -> dict[str, Any]:
        del command
        self.current_job_id = None
        self.emit_status("unloading")
        result = self.runner.clear_model_cache() if self.runner is not None else {
            "cached_entries_released": 0,
            "previous_model_path": None,
            "unload_time_ms": 0.0,
        }
        self.timings["unload_time_ms"] = result.get("unload_time_ms")
        self.selected_model_path = None
        return self.emit_status("idle", action="unload_complete", unload=result)

    def _load_payload(self, config_path: str) -> tuple[Any, dict[str, Any]]:
        payload = load_request_payload(
            config_path=config_path,
            base_payload=self.context.generation_defaults(),
        )
        payload, _resolution = normalize_scheduler_payload(payload)
        request, payload_extras = payload_to_generation_request(payload)
        return request, payload_extras

    def run_job(self, command: dict[str, Any]) -> dict[str, Any]:
        job_id = str(command.get("job_id") or uuid.uuid4().hex[:12])
        config_path = str(command.get("config_path") or "").strip()
        if not config_path:
            raise ValueError("A config_path is required for warm generation.")
        self.current_job_id = job_id
        self.last_error = None
        runner = self._ensure_runner()
        runner.reset_runtime_state()
        request, payload_extras = self._load_payload(config_path)
        self.selected_model_path = str(payload_extras.get("model_path") or "") or None
        warm_before = runner.warm_model_status()
        warm_reuse_candidate = bool(
            warm_before.get("warm")
            and warm_before.get("model_path")
            and os.path.normcase(str(Path(str(warm_before.get("model_path"))).resolve()))
            == os.path.normcase(str(Path(str(self.selected_model_path)).resolve()))
        )
        self.emit_status(
            "preparing_model",
            action="run_job",
            warm_reuse_candidate=warm_reuse_candidate,
        )

        extras = {
            "live_sampler_map": self.live_sampler_map,
            "live_scheduler_map": self.live_scheduler_map,
            "warm_worker_event_callback": self._runner_event,
        }
        extras.update(payload_extras)
        self.runtime_settings.update({key: value for key, value in payload_extras.items() if key.startswith("warm_worker_") or key == "memory_policy"})
        batch_count = int(extras.pop("batch_count", 1) or 1)
        unlimited = bool(extras.pop("unlimited", False))
        if batch_count < 1:
            raise ValueError("batch_count must be at least 1")
        if request.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        requested_seed = request.seed
        base_seed_iterator = iter_batch_base_seeds(
            requested_seed,
            batch_size=request.batch_size,
        )
        completed_batches = 0
        total_saved = 0
        job_started = time.perf_counter()
        model_cache_reused = False
        first_generation_started: float | None = None

        while unlimited or completed_batches < batch_count:
            # Keep the warmed checkpoint/components resident, but clear any
            # request-scoped mutable runtime state before every batch-count
            # iteration so later images do not inherit stale state from the
            # first completed image.
            runner.reset_runtime_state()
            batch_number = completed_batches + 1
            batch_request = replace(
                request,
                seed=next(base_seed_iterator),
                resolved_seeds=[],
                scheduler_kwargs=dict(request.scheduler_kwargs),
                sampler_kwargs=dict(request.sampler_kwargs),
                prompt_parser_name=str(request.prompt_parser_name or "legacy"),
                prompt_parser_kwargs=dict(request.prompt_parser_kwargs),
                prompt_shortcut_profile_name=str(request.prompt_shortcut_profile_name or "legacy_default"),
                prompt_shortcut_profile_snapshot=dict(request.prompt_shortcut_profile_snapshot),
                prompt_parser_preset_name=str(request.prompt_parser_preset_name or ""),
                base_prompt_parser_name=str(request.base_prompt_parser_name or request.prompt_parser_name or "legacy"),
                base_shortcut_profile_name=str(request.base_shortcut_profile_name or request.prompt_shortcut_profile_name or "legacy_default"),
                hires_prompt_parser_mode=str(request.hires_prompt_parser_mode or "same_as_base"),
                hires_prompt_parser_name=str(request.hires_prompt_parser_name or request.prompt_parser_name or "legacy"),
                hires_prompt_parser_kwargs=dict(request.hires_prompt_parser_kwargs),
                hires_shortcut_profile_mode=str(request.hires_shortcut_profile_mode or "same_as_base"),
                hires_shortcut_profile_name=str(request.hires_shortcut_profile_name or request.prompt_shortcut_profile_name or "legacy_default"),
                hires_shortcut_profile_snapshot=dict(request.hires_shortcut_profile_snapshot),
                hires_positive_prompt=str(request.hires_positive_prompt or request.positive_prompt),
                hires_negative_prompt=str(request.hires_negative_prompt if request.hires_negative_prompt is not None else request.negative_prompt),
                hires_size_mode=str(request.hires_size_mode or "same_as_base"),
                hires_scale=float(request.hires_scale or 2.0),
                hires_width=int(request.hires_width or 0),
                hires_height=int(request.hires_height or 0),
                hires_dimension_plan=dict(request.hires_dimension_plan),
                prompt_preflight=dict(request.prompt_preflight),
                prompt_shadow_compare=bool(request.prompt_shadow_compare),
                prompt_route_plan=dict(request.prompt_route_plan),
                hires_prompt_route_plan=dict(request.hires_prompt_route_plan),
                parser_kwargs=dict(request.parser_kwargs),
                diagnostics=dict(request.diagnostics),
            )
            resolved_image_seeds = [
                offset_seed(int(batch_request.seed), index)
                for index in range(int(batch_request.batch_size))
            ]
            print(
                "GENERATION_SEED_JSON: "
                + json.dumps(
                    {
                        "batch_number": batch_number,
                        "base_seed": int(batch_request.seed),
                        "image_seeds": resolved_image_seeds,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
            batch_extras = dict(extras)
            batch_extras.update(
                {
                    "batch_number": batch_number,
                    "batch_count": batch_count,
                    "unlimited": unlimited,
                    "generation_mode": "unlimited" if unlimited else "batch_count",
                }
            )
            if first_generation_started is None:
                first_generation_started = time.perf_counter()
            self.emit_status("running", batch_number=batch_number, batch_count=batch_count)
            result = runner.run_request(
                batch_request,
                batch_extras,
                save_txt=bool(command.get("save_txt", True)),
                save_json=bool(command.get("save_json", True)),
            )

            live_preview_summary = dict(result.pipeline_result.metadata.get("live_preview") or {})
            print(
                "LIVE_PREVIEW_SUMMARY_JSON: "
                + json.dumps(live_preview_summary, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            output_quality_diagnostic = dict(
                result.pipeline_result.metadata.get("output_quality") or {}
            )
            print(
                "OUTPUT_QUALITY_DIAGNOSTIC_JSON: "
                + json.dumps(output_quality_diagnostic, ensure_ascii=False, sort_keys=True),
                flush=True,
            )

            prompt_parser_diagnostic = dict(
                result.pipeline_result.metadata.get("prompt_parser") or {}
            )
            print(
                "PROMPT_PARSER_DIAGNOSTIC_JSON: "
                + json.dumps(prompt_parser_diagnostic, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            model_diagnostic = dict(result.request_extras.get("model_provenance") or {})
            model_cache_reused = model_cache_reused or bool(model_diagnostic.get("cache_reused"))
            print(
                "MODEL_DIAGNOSTIC_JSON: "
                + json.dumps(model_diagnostic, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
            saved_paths = [record.image_path for record in result.saved_records]
            if result.request.save_images and not saved_paths:
                raise RuntimeError(
                    "Generation completed, but image saving was requested and no image was persisted."
                )
            if requested_seed is not None and int(requested_seed) >= 0:
                expected_base_seed = offset_seed(
                    int(requested_seed), completed_batches * int(request.batch_size)
                )
                if int(result.request.seed) != expected_base_seed:
                    raise RuntimeError(
                        "Fixed seed changed unexpectedly: "
                        f"requested batch base {expected_base_seed}, runtime used {result.request.seed}."
                    )
                expected_image_seeds = [
                    offset_seed(expected_base_seed, index)
                    for index in range(int(request.batch_size))
                ]
                if list(result.request.resolved_seeds or []) != expected_image_seeds:
                    raise RuntimeError(
                        "Fixed per-image seed sequence changed unexpectedly: "
                        f"expected {expected_image_seeds}, runtime used {result.request.resolved_seeds}."
                    )
            completed_batches += 1
            total_saved += len(saved_paths)
            for record in result.saved_records:
                seed_label = "unknown" if record.seed is None else str(record.seed)
                print(f"  Image [seed {seed_label}]: {record.image_path}", flush=True)
                if record.txt_path:
                    print(f"  TXT:   {record.txt_path}", flush=True)
                if record.json_path:
                    print(f"  JSON:  {record.json_path}", flush=True)

        self.emit_status("applying_retention_policy")
        retention = runner.apply_warm_retention(payload_extras)
        total_ms = round((time.perf_counter() - job_started) * 1000.0, 3)
        self.timings.update(
            {
                "last_job_total_ms": total_ms,
                "warm_reuse_benefited_last_job": model_cache_reused,
                "cold_or_switch_load_last_job": not model_cache_reused,
            }
        )
        self.current_job_id = None
        status = self.emit_status(
            "ready",
            action="job_complete",
            completed_batches=completed_batches,
            total_saved=total_saved,
            warm_reuse_benefited=model_cache_reused,
            job_total_ms=total_ms,
            retention=retention,
        )
        return {
            "job_id": job_id,
            "completed_batches": completed_batches,
            "total_saved": total_saved,
            "warm_reuse_benefited": model_cache_reused,
            "status": status,
        }

    def handle(self, command: dict[str, Any]) -> dict[str, Any]:
        name = str(command.get("command") or "status").strip().lower()
        if name == "status":
            return self.emit_status()
        if name == "preload":
            return self.preload(command)
        if name == "unload":
            return self.unload(command)
        if name == "run":
            return self.run_job(command)
        if name == "shutdown":
            result = self.unload(command)
            result["shutdown"] = True
            return result
        raise ValueError(f"Unsupported warm worker command: {name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IMAGE_GEN persistent warm txt2img worker")
    parser.add_argument("--project-root")
    parser.add_argument("--project-config")
    add_runtime_startup_arguments(parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    parser_argv = argv_for_primary_parser(raw_argv)
    args = build_parser().parse_args(parser_argv)
    args._runtime_argv = raw_argv
    context = ProjectContext.load(
        project_root=args.project_root,
        config_path=args.project_config,
    )
    runtime_startup_options = _PREBOOTSTRAP_RUNTIME_STARTUP_OPTIONS
    args.runtime_startup_options = runtime_startup_options
    worker = PersistentTxt2ImgWarmWorker(context, runtime_startup_options)
    _emit(
        _READY_PREFIX,
        {
            "schema_version": 1,
            "pid": os.getpid(),
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "project_root": str(context.project_root),
            "runtime_startup_options": runtime_startup_options.to_dict(),
            "runtime_startup_status": build_runtime_startup_status(
                runtime_startup_options,
                {"mslk_fmha": runtime_startup_options.mslk_fmha.to_dict()},
            ),
        },
    )
    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        command_id = ""
        command: dict[str, Any] = {}
        try:
            command = json.loads(line)
            if not isinstance(command, dict):
                raise ValueError("Warm worker commands must be JSON objects.")
            command_id = str(command.get("command_id") or uuid.uuid4().hex)
            result = worker.handle(command)
            _emit(
                _COMPLETE_PREFIX,
                {
                    "command_id": command_id,
                    "command": command.get("command"),
                    "ok": True,
                    "result": result,
                },
            )
            if str(command.get("command") or "").lower() == "shutdown":
                return 0
        except BaseException as exc:
            worker.last_error = f"{type(exc).__name__}: {exc}"
            worker.current_job_id = None
            worker.emit_status("failed", error=worker.last_error)
            _emit(
                _COMPLETE_PREFIX,
                {
                    "command_id": command_id,
                    "command": command.get("command"),
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
