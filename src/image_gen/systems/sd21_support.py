from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from modules.project_context import ProjectContext


MANIFEST_RELATIVE_PATH = Path("scripts/setup/manifests/sd21_support.json")
RUNTIME_SUBDIR = Path("stable_diffusion/sd2_1_base")
SD2_RUNTIME_MINIMUM_VRAM_GIB_EXCLUSIVE = 13.0


@dataclass(frozen=True)
class SD21SupportStatus:
    ready: bool
    support_id: str
    manifest_path: str
    missing_files: tuple[str, ...]
    required_count: int
    present_count: int
    installer_running: bool = False
    installer_pid: int | None = None
    # These fields qualify SD2.x model execution, not support-file installation.
    hardware_eligible: bool = False
    hardware_reason: str = ""
    selected_gpu_name: str = ""
    selected_gpu_memory_mib: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "support_id": self.support_id,
            "manifest_path": self.manifest_path,
            "missing_files": list(self.missing_files),
            "required_count": self.required_count,
            "present_count": self.present_count,
            "installer_running": self.installer_running,
            "installer_pid": self.installer_pid,
            "hardware_eligible": self.hardware_eligible,
            "hardware_reason": self.hardware_reason,
            "selected_gpu_name": self.selected_gpu_name,
            "selected_gpu_memory_mib": self.selected_gpu_memory_mib,
        }


def runtime_hardware_qualification(
    *, minimum_vram_gib_exclusive: float = SD2_RUNTIME_MINIMUM_VRAM_GIB_EXCLUSIVE,
) -> tuple[bool, str, str, int | None, int | None]:
    """Qualify the selected NVIDIA GPU for SD2.x model execution.

    This check deliberately does not gate installation of the lightweight SD2.1
    runtime support files. It is consulted only when an SD2.x model is activated.
    """

    executable = shutil.which("nvidia-smi")
    if not executable:
        return False, "nvidia-smi is unavailable; SD2.x GPU VRAM could not be qualified.", "", None, None
    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=index,name,memory.total,uuid",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return False, f"Unable to query SD2.x GPU VRAM: {exc}", "", None, None

    records: list[tuple[int, str, int | None, str]] = []
    for line in completed.stdout.splitlines():
        parts = [item.strip() for item in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        digits = "".join(ch for ch in parts[2] if ch.isdigit())
        memory_mib = int(digits) if digits else None
        uuid = parts[3] if len(parts) >= 4 else ""
        records.append((index, parts[1], memory_mib, uuid))
    if not records:
        return False, "No NVIDIA GPU was reported by nvidia-smi.", "", None, None

    selected = None
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip().split(",", 1)[0].strip()
    if visible:
        if visible.isdigit():
            selected = next((item for item in records if item[0] == int(visible)), None)
        if selected is None:
            selected = next((item for item in records if item[3] and item[3].casefold() == visible.casefold()), None)
    if selected is None:
        selected = max(records, key=lambda item: item[2] if item[2] is not None else -1)

    index, name, memory_mib, _uuid = selected
    if memory_mib is None:
        return False, f"VRAM could not be determined for {name}.", name, memory_mib, index
    threshold_mib = int(float(minimum_vram_gib_exclusive) * 1024.0)
    if memory_mib <= threshold_mib:
        return (
            False,
            f"{name} reports {memory_mib / 1024.0:.2f} GiB VRAM; SD2.x model execution requires more than "
            f"{minimum_vram_gib_exclusive:g} GiB.",
            name,
            memory_mib,
            index,
        )
    return (
        True,
        f"{name} reports {memory_mib / 1024.0:.2f} GiB VRAM and passes the >"
        f"{minimum_vram_gib_exclusive:g} GiB SD2.x execution gate.",
        name,
        memory_mib,
        index,
    )


class SD21SupportManager:
    """Shared readiness and installer-launch contract for SD2.1 runtime assets.

    The support bundle contains only lightweight tokenizer/configuration files.
    IMAGE_GEN never downloads a checkpoint or reference-component weight through
    this manager. Missing support files are installed by the same standalone
    setup script used during the main IMAGE_GEN installation.
    """

    _launch_lock = threading.RLock()
    _installer_process: subprocess.Popen[Any] | None = None

    def __init__(self, context: ProjectContext, *, manifest_path: str | Path | None = None) -> None:
        self.context = context
        self.manifest_path = (
            Path(manifest_path).expanduser().resolve()
            if manifest_path is not None
            else (context.project_root / MANIFEST_RELATIVE_PATH).resolve()
        )
        self._manifest = self._load_manifest(self.manifest_path)

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", 0)) != 1:
            raise ValueError(f"Unsupported SD2.1 support manifest schema: {path}")
        files = payload.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"SD2.1 support manifest has no files: {path}")
        forbidden_suffixes = {".safetensors", ".ckpt", ".bin", ".pt", ".pth", ".onnx", ".gguf"}
        for raw in files:
            record = dict(raw or {})
            if str(record.get("destination_kind") or "").strip().casefold() != "runtime":
                raise ValueError("SD2.1 support manifest may contain runtime assets only.")
            if Path(str(record.get("remote_path") or "")).suffix.casefold() in forbidden_suffixes:
                raise ValueError("SD2.1 support manifest may not contain model-weight files.")
        return payload

    def _destination_root(self) -> Path:
        return (Path(self.context.runtime_assets_root) / RUNTIME_SUBDIR).resolve()

    def destination_for(self, record: Mapping[str, Any]) -> Path:
        root = self._destination_root()
        relative = Path(str(record.get("destination_path") or "").replace("\\", "/"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe SD2.1 support destination: {relative}")
        target = (root / relative).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"SD2.1 support destination escapes its managed root: {target}")
        return target

    def required_files(self) -> list[tuple[dict[str, Any], Path]]:
        result: list[tuple[dict[str, Any], Path]] = []
        for raw in self._manifest.get("files") or []:
            record = dict(raw or {})
            if record.get("required", True) is False:
                continue
            result.append((record, self.destination_for(record)))
        return result

    @classmethod
    def _running_process(cls) -> subprocess.Popen[Any] | None:
        with cls._launch_lock:
            process = cls._installer_process
            if process is not None and process.poll() is not None:
                cls._installer_process = None
                process = None
            return process

    def status(self) -> SD21SupportStatus:
        required = self.required_files()
        missing: list[str] = []
        for _record, destination in required:
            try:
                ready = destination.is_file() and destination.stat().st_size > 0
            except OSError:
                ready = False
            if not ready:
                missing.append(str(destination))
        process = self._running_process()
        eligible, hardware_reason, gpu_name, gpu_memory_mib, _gpu_index = runtime_hardware_qualification()
        return SD21SupportStatus(
            ready=not missing,
            support_id=str(self._manifest.get("support_id") or "sd2.1-runtime"),
            manifest_path=str(self.manifest_path),
            missing_files=tuple(missing),
            required_count=len(required),
            present_count=len(required) - len(missing),
            installer_running=process is not None,
            installer_pid=int(process.pid) if process is not None else None,
            hardware_eligible=eligible,
            hardware_reason=hardware_reason,
            selected_gpu_name=gpu_name,
            selected_gpu_memory_mib=gpu_memory_mib,
        )

    def launch_installer(self, *, reason: str = "sd2_checkpoint_detected") -> SD21SupportStatus:
        current = self.status()
        if current.ready:
            return current

        with self._launch_lock:
            process = self._installer_process
            if process is not None and process.poll() is None:
                return self.status()

            script = (self.context.project_root / "scripts/setup/install_sd21_support.py").resolve()
            if not script.is_file():
                return self.status()

            command = [
                sys.executable,
                str(script),
                "--project-root",
                str(self.context.project_root),
                "--trigger",
                str(reason or "sd2_checkpoint_detected"),
            ]
            creationflags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0) if sys.platform == "win32" else 0
            try:
                self.__class__._installer_process = subprocess.Popen(
                    command,
                    cwd=str(self.context.project_root),
                    creationflags=creationflags,
                )
            except OSError:
                self.__class__._installer_process = None
            return self.status()

    def ensure_for_architecture(
        self,
        architecture: str,
        *,
        launch_if_missing: bool = True,
        reason: str = "sd2_checkpoint_detected",
    ) -> SD21SupportStatus:
        normalized = str(architecture or "").strip().casefold()
        if normalized not in {"sd2", "sd2.x", "sd2x", "stable-diffusion-2", "stable_diffusion_2"}:
            return self.status()
        status = self.status()
        if not status.ready and launch_if_missing:
            status = self.launch_installer(reason=reason)
        return status


__all__ = [
    "SD21SupportManager",
    "SD21SupportStatus",
    "SD2_RUNTIME_MINIMUM_VRAM_GIB_EXCLUSIVE",
    "runtime_hardware_qualification",
]
