from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from modules.project_context import ProjectContext
from image_gen.runtime.residency_policy import (
    RESIDENCY_STATE_HOT_GPU,
    RESIDENCY_STATE_HOT_STAGED,
    build_residency_diagnostics,
    normalize_model_residency_mode,
    resolve_effective_residency_state,
)

_READY_PREFIX = "MODEL_RUNTIME_READY_JSON: "
_STATUS_PREFIX = "MODEL_RUNTIME_STATUS_JSON: "
_COMPLETE_PREFIX = "MODEL_RUNTIME_COMMAND_COMPLETE_JSON: "
_MODEL_RUNTIME_STREAM_LIMIT = 16 * 1024 * 1024


class ModelRuntimeUnavailable(RuntimeError):
    pass


class ModelRuntimeCommandSuperseded(ModelRuntimeUnavailable):
    """Raised when a newer user selection intentionally replaces an in-flight activation."""

    pass


def _process_exit_record(code: int) -> dict[str, Any]:
    value = int(code)
    unsigned = value & 0xFFFFFFFF
    record: dict[str, Any] = {
        "code": value,
        "hex": f"0x{unsigned:08X}",
    }
    if unsigned == 0xC000013A:
        record["meaning"] = "STATUS_CONTROL_C_EXIT"
        record["summary"] = (
            "The worker was terminated by a Windows control event "
            "(for example Ctrl+C or console shutdown)."
        )
    return record


def _process_exit_message(code: int) -> str:
    record = _process_exit_record(code)
    suffix = f" ({record['hex']}"
    if record.get("meaning"):
        suffix += f", {record['meaning']}"
    return suffix + ")"


class ResidentModelRuntimeClient:
    """JSONL controller for the long-lived txt2img model runtime process."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.process: asyncio.subprocess.Process | None = None
        self._command_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self._current_command_id: str | None = None
        self._current_command_name: str | None = None
        self._current_command_model_path: str | None = None
        self._current_job_id: str | None = None
        self._superseded_command_id: str | None = None
        self._supersede_reason: str | None = None
        self._residency_mode_requested = "managed"
        self._status: dict[str, Any] = {
            "schema_version": 1,
            "online": False,
            "stage": "offline",
            "residency_state": "empty",
            "selected_model_path": None,
            "current_model_path": None,
            "cpu_loaded": False,
            "gpu_loaded": False,
            "hot_gpu_ready": False,
            "current_job_id": None,
            "last_error": None,
            "residency_mode_requested": "managed",
            "residency_state_effective": "empty",
            "hot_model_path": None,
            "hot_model_identity": None,
            "hot_composition_sha256": None,
            "hot_load_variant_fingerprint": None,
            "hot_since": None,
            "last_residency_transition": None,
            "last_residency_reason": "client_initialized",
            "retention_suppressed_for_hot": False,
            "hot_reuse_count": 0,
            "cold_or_switch_load_count": 0,
            "last_generation_residency_classification": None,
        }
        self._started_at_unix: float | None = None
        self._restart_count = 0
        self._cancel_requested = False
        self._cancel_reason: str | None = None
        self._stdout_buffer = bytearray()

    async def _read_stdout_line(self, process: asyncio.subprocess.Process) -> bytes:
        """Read one newline-delimited worker record without asyncio's line-size limit."""
        assert process.stdout is not None
        while True:
            newline_at = self._stdout_buffer.find(b"\n")
            if newline_at >= 0:
                end = newline_at + 1
                raw = bytes(self._stdout_buffer[:end])
                del self._stdout_buffer[:end]
                return raw
            chunk = await process.stdout.read(64 * 1024)
            if not chunk:
                if not self._stdout_buffer:
                    return b""
                raw = bytes(self._stdout_buffer)
                self._stdout_buffer.clear()
                return raw
            self._stdout_buffer.extend(chunk)
            if len(self._stdout_buffer) > _MODEL_RUNTIME_STREAM_LIMIT:
                raise ModelRuntimeUnavailable(
                    "Model runtime emitted a single unterminated output record larger "
                    f"than {_MODEL_RUNTIME_STREAM_LIMIT} bytes."
                )

    def set_residency_mode_requested(self, value: Any) -> str:
        mode = normalize_model_residency_mode(value)
        self._residency_mode_requested = mode
        self._status["residency_mode_requested"] = mode
        return mode

    def status(self) -> dict[str, Any]:
        process_online = self.process is not None and self.process.returncode is None
        raw = dict(self._status)
        raw_stage = str(raw.get("stage") or "offline")
        resident = bool(
            raw.get("current_model_path")
            or str(raw.get("residency_state") or "").strip().lower() == "resident"
        )
        worker_effective = str(raw.get("residency_state_effective") or "").strip().lower()
        hot_active = worker_effective in {RESIDENCY_STATE_HOT_GPU, RESIDENCY_STATE_HOT_STAGED}
        effective = resolve_effective_residency_state(
            requested_mode=self._residency_mode_requested,
            stage=raw_stage,
            resident=resident,
            gpu_loaded=bool(raw.get("gpu_loaded")),
            staged_runtime=bool(raw.get("staged_runtime")),
            hot_residency_active=hot_active,
            hot_gpu_ready=bool(raw.get("hot_gpu_ready", raw.get("gpu_loaded"))),
        )
        resident_status = {
            "model_path": raw.get("current_model_path"),
            "model_identity": raw.get("model_identity"),
            "composition_sha256": raw.get("composition_sha256"),
            "runtime_load_variant_fingerprint": raw.get("runtime_load_variant_fingerprint"),
        }
        diagnostics = build_residency_diagnostics(
            requested_mode=self._residency_mode_requested,
            effective_state=effective,
            resident_status=resident_status,
            last_residency_transition=raw.get("last_residency_transition"),
            last_residency_reason=raw.get("last_residency_reason"),
            retention_suppressed_for_hot=bool(raw.get("retention_suppressed_for_hot", False)),
            hot_reuse_count=int(raw.get("hot_reuse_count") or 0),
            cold_or_switch_load_count=int(raw.get("cold_or_switch_load_count") or 0),
            last_generation_residency_classification=raw.get("last_generation_residency_classification"),
            hot_since=raw.get("hot_since"),
        )
        return {
            **raw,
            **diagnostics,
            "online": process_online,
            "pid": self.process.pid if process_online else None,
            "started_at_unix": self._started_at_unix,
            "restart_count": self._restart_count,
            "current_command_id": self._current_command_id,
            "current_command_name": self._current_command_name,
            "current_command_model_path": self._current_command_model_path,
            "current_job_id": self._current_job_id or raw.get("current_job_id"),
        }

    def _environment(self) -> dict[str, str]:
        env = os.environ.copy()
        source_root = str(self.context.project_root / "src")
        env["PYTHONPATH"] = os.pathsep.join(
            [source_root, str(self.context.project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        return env

    async def ensure_started(self) -> None:
        if self.process is not None and self.process.returncode is None:
            return
        async with self._start_lock:
            if self.process is not None and self.process.returncode is None:
                return
            command = [
                sys.executable,
                "-m",
                "modules.txt2img.model_runtime",
                "--project-root",
                str(self.context.project_root),
            ]
            try:
                process = await asyncio.create_subprocess_exec(
                    *command,
                    cwd=str(self.context.project_root),
                    env=self._environment(),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    # Runtime diagnostics are emitted as JSONL. Memory telemetry
                    # can legitimately exceed asyncio's 64 KiB default line limit.
                    limit=_MODEL_RUNTIME_STREAM_LIMIT,
                )
            except Exception as exc:
                self._status.update(
                    {
                        "online": False,
                        "stage": "failed",
                        "last_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                raise ModelRuntimeUnavailable(str(exc)) from exc
            self.process = process
            self._stdout_buffer.clear()
            self._started_at_unix = time.time()
            self._restart_count += 1
            assert process.stdout is not None
            try:
                raw = await asyncio.wait_for(self._read_stdout_line(process), timeout=45.0)
            except asyncio.TimeoutError as exc:
                process.terminate()
                self.process = None
                raise ModelRuntimeUnavailable("Model runtime did not become ready within 45 seconds.") from exc
            if not raw:
                code = await process.wait()
                self.process = None
                if self._cancel_requested or self._status.get("stage") == "recovering":
                    raise ModelRuntimeUnavailable(
                        self._cancel_reason or str(self._status.get("last_cancellation") or "Model-runtime startup was cancelled.")
                    )
                self._status["last_exit"] = _process_exit_record(code)
                raise ModelRuntimeUnavailable(
                    f"Model runtime exited before readiness with code {code}{_process_exit_message(code)}."
                )
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line.startswith(_READY_PREFIX):
                process.terminate()
                self.process = None
                raise ModelRuntimeUnavailable(
                    f"Model runtime returned an unexpected startup response: {line[:300]}"
                )
            try:
                ready = json.loads(line[len(_READY_PREFIX):])
            except json.JSONDecodeError:
                ready = {}
            self._status.update(
                {
                    "online": True,
                    "stage": "idle",
                    "residency_state": "empty",
                    "last_error": None,
                    "last_exit": None,
                    "cuda_available": ready.get("cuda_available"),
                    "worker_python_executable": ready.get("python_executable"),
                    "worker_torch_version": ready.get("torch_version"),
                    "ready": ready,
                }
            )

    async def _call_line_handler(
        self,
        handler: Callable[[str], Any] | None,
        line: str,
    ) -> None:
        if handler is None:
            return
        result = handler(line)
        if inspect.isawaitable(result):
            await result

    async def execute(
        self,
        command: dict[str, Any],
        *,
        on_line: Callable[[str], Any] | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        async with self._command_lock:
            await self.ensure_started()
            process = self.process
            if process is None or process.returncode is not None:
                raise ModelRuntimeUnavailable("Model runtime is not running.")
            assert process.stdin is not None
            assert process.stdout is not None
            command_id = str(command.get("command_id") or uuid.uuid4().hex)
            payload = {**command, "command_id": command_id}
            self._current_command_id = command_id
            self._current_command_name = str(payload.get("command") or "").strip().lower() or None
            self._current_command_model_path = str(payload.get("model_path") or "").strip() or None
            self._current_job_id = str(payload.get("job_id") or "") or None
            try:
                process.stdin.write(
                    (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
                )
                await process.stdin.drain()
            except Exception as exc:
                await self._mark_dead(f"Unable to send model runtime command: {exc}")
                raise ModelRuntimeUnavailable(str(exc)) from exc

            async def read_until_complete() -> dict[str, Any]:
                while True:
                    raw = await self._read_stdout_line(process)
                    if not raw:
                        code = await process.wait()
                        if command_id == self._superseded_command_id:
                            reason = self._supersede_reason or "Model activation was superseded by a newer checkpoint selection."
                            await self._mark_dead(None, stage="superseded")
                            self._status["last_cancellation"] = reason
                            raise ModelRuntimeCommandSuperseded(reason)
                        if self._cancel_requested or self._status.get("stage") == "recovering":
                            reason = self._cancel_reason or str(
                                self._status.get("last_cancellation") or "Model-runtime operation was cancelled."
                            )
                            await self._mark_dead(None, stage="recovering")
                            self._status["last_cancellation"] = reason
                            raise ModelRuntimeUnavailable(reason)
                        self._status["last_exit"] = _process_exit_record(code)
                        await self._mark_dead(
                            f"Model runtime exited during command {command_id} with code {code}{_process_exit_message(code)}."
                        )
                        raise ModelRuntimeUnavailable(
                            f"Model runtime exited during command with code {code}{_process_exit_message(code)}."
                        )
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line.startswith(_STATUS_PREFIX):
                        try:
                            status = json.loads(line[len(_STATUS_PREFIX):])
                        except json.JSONDecodeError:
                            status = {}
                        if isinstance(status, dict):
                            self._status.update(status)
                            self._status["online"] = True
                    if line.startswith(_COMPLETE_PREFIX):
                        try:
                            completion = json.loads(line[len(_COMPLETE_PREFIX):])
                        except json.JSONDecodeError as exc:
                            raise ModelRuntimeUnavailable(
                                "Model runtime returned malformed command completion JSON."
                            ) from exc
                        if str(completion.get("command_id") or "") != command_id:
                            await self._call_line_handler(on_line, line)
                            continue
                        return completion
                    await self._call_line_handler(on_line, line)

            try:
                completion = (
                    await asyncio.wait_for(read_until_complete(), timeout=timeout)
                    if timeout is not None
                    else await read_until_complete()
                )
            finally:
                self._current_command_id = None
                self._current_command_name = None
                self._current_command_model_path = None
                self._current_job_id = None
                # A completed/cancelled command cannot continue owning the resident
                # checkpoint. Clear the cached worker field too so status() never
                # revives a stale job id after the private field is released.
                self._status["current_job_id"] = None
                if self._superseded_command_id == command_id:
                    self._superseded_command_id = None
                    self._supersede_reason = None
                if not self._cancel_requested:
                    self._cancel_reason = None
            return completion

    async def supersede_activation(self, requested_model_path: str) -> bool:
        """Cancel an obsolete in-flight model activation when the user selects a different checkpoint.

        The resident runtime is intentionally single-command. Without this escape hatch,
        rapid checkpoint changes queue behind a large checkpoint load and make the newer
        selection appear hung. Running generation commands are never superseded here.
        """
        if self._current_command_name != "activate" or not self._current_command_id:
            return False
        current_target = str(self._current_command_model_path or "").strip()
        requested_target = str(requested_model_path or "").strip()
        if not requested_target:
            return False

        def _identity(value: str) -> str:
            try:
                return os.path.normcase(str(Path(value).expanduser().resolve(strict=False)))
            except OSError:
                return os.path.normcase(value)

        if current_target and _identity(current_target) == _identity(requested_target):
            return False

        command_id = self._current_command_id
        reason = "Model activation superseded by a newer checkpoint selection."
        self._superseded_command_id = command_id
        self._supersede_reason = reason
        process = self.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        await self._mark_dead(None, stage="superseded")
        self._status["last_cancellation"] = reason
        return True

    async def activate(
        self,
        model_path: str,
        *,
        runtime_settings: dict[str, Any] | None = None,
        on_line: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        settings = dict(runtime_settings or {})
        self.set_residency_mode_requested(settings.get("model_residency_mode"))
        return await self.execute(
            {
                "command": "activate",
                "model_path": str(model_path),
                "runtime_settings": settings,
            },
            on_line=on_line,
        )

    async def apply_residency_mode(
        self,
        value: Any,
        *,
        runtime_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        mode = self.set_residency_mode_requested(value)
        settings = dict(runtime_settings or {})
        settings["model_residency_mode"] = mode
        return await self.execute(
            {
                "command": "set_residency_mode",
                "runtime_settings": settings,
            }
        )

    async def unload(self) -> dict[str, Any]:
        return await self.execute({"command": "unload"})

    async def run_job(
        self,
        *,
        job_id: str,
        config_path: str | Path,
        save_txt: bool = True,
        save_json: bool = True,
        save_diagnostics_json: bool = True,
        on_line: Callable[[str], Any] | None = None,
    ) -> dict[str, Any]:
        return await self.execute(
            {
                "command": "run",
                "job_id": str(job_id),
                "config_path": str(config_path),
                "save_txt": bool(save_txt),
                "save_json": bool(save_json),
                "save_diagnostics_json": bool(save_diagnostics_json),
            },
            on_line=on_line,
        )

    async def cancel_active(self, job_id: str | None = None) -> None:
        if job_id and self._current_job_id and str(job_id) != self._current_job_id:
            return
        process = self.process
        self._cancel_requested = True
        self._cancel_reason = "Generation cancelled while the model runtime was preparing or running."
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        await self._mark_dead(None, stage="recovering")
        self._status["last_cancellation"] = self._cancel_reason
        self._status["last_residency_reason"] = "cancelled_worker_deferred_reactivation"
        self._status["automatic_hot_reactivation_deferred"] = True
        self._cancel_requested = False
        self._cancel_reason = None

    async def stop(self) -> None:
        process = self.process
        if self._current_command_id is not None:
            await self.cancel_active(self._current_job_id)
            return
        if process is not None and process.returncode is None:
            try:
                await asyncio.wait_for(
                    self.execute({"command": "shutdown"}),
                    timeout=15.0,
                )
            except Exception:
                if process.returncode is None:
                    process.terminate()
                    try:
                        await asyncio.wait_for(process.wait(), timeout=5.0)
                    except asyncio.TimeoutError:
                        process.kill()
                        await process.wait()
        await self._mark_dead(None)

    async def _mark_dead(self, error: str | None, *, stage: str | None = None) -> None:
        self.process = None
        self._stdout_buffer.clear()
        self._current_command_id = None
        self._current_command_name = None
        self._current_command_model_path = None
        self._current_job_id = None
        self._status.update(
            {
                "online": False,
                "stage": stage or ("failed" if error else "offline"),
                "residency_state": "empty",
                "current_model_path": None,
                "cpu_loaded": False,
                "gpu_loaded": False,
                "hot_gpu_ready": False,
                "current_job_id": None,
                "last_error": error,
                "residency_state_effective": "recovering" if (stage == "recovering") else "empty",
                "hot_model_path": None,
                "hot_model_identity": None,
                "hot_composition_sha256": None,
                "hot_load_variant_fingerprint": None,
                "hot_since": None,
                "retention_suppressed_for_hot": False,
                "automatic_hot_reactivation_deferred": bool(stage == "recovering"),
            }
        )


__all__ = ["ResidentModelRuntimeClient", "ModelRuntimeUnavailable", "ModelRuntimeCommandSuperseded"]
