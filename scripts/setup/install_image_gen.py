from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import venv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


REQUIRED_PYTHON = (3, 10, 20)


class InstallError(RuntimeError):
    pass


@dataclass(frozen=True)
class GPUInfo:
    index: int
    name: str
    compute_capability: str
    driver_version: str
    memory_mib: int | None = None
    uuid: str | None = None
    pci_bus_id: str | None = None


@dataclass(frozen=True)
class CudaToolkit:
    version: str
    path: str
    nvcc_path: str | None
    sources: tuple[str, ...]


@dataclass(frozen=True)
class InstallChoice:
    profile_id: str
    profile_label: str
    mode: str
    cuda_runtime: str
    toolkit_path: str | None
    toolkit_version: str | None
    description: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _version_tuple(value: str | None) -> tuple[int, ...]:
    if not value:
        return ()
    return tuple(int(part) for part in re.findall(r"\d+", str(value)))


def _version_at_least(actual: str | None, required: str | None) -> bool:
    actual_parts = _version_tuple(actual)
    required_parts = _version_tuple(required)
    if not actual_parts or not required_parts:
        return False
    width = max(len(actual_parts), len(required_parts))
    return actual_parts + (0,) * (width - len(actual_parts)) >= required_parts + (0,) * (
        width - len(required_parts)
    )


def _run_capture(command: Sequence[str], *, env: dict[str, str] | None = None) -> str:
    completed = subprocess.run(
        list(command),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return completed.stdout


def _run(
    command: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> None:
    printable = subprocess.list2cmdline([str(item) for item in command])
    print(f"+ {printable}")
    if dry_run:
        return
    subprocess.run([str(item) for item in command], cwd=cwd, env=env, check=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalize_compute_capability(value: str) -> str:
    match = re.search(r"(\d+)\D+(\d+)", str(value).strip())
    if match:
        return f"{int(match.group(1))}.{int(match.group(2))}"
    digits = re.findall(r"\d+", str(value))
    if len(digits) >= 2:
        return f"{int(digits[0])}.{int(digits[1])}"
    return str(value).strip()


def parse_nvidia_smi_csv(output: str) -> list[GPUInfo]:
    gpus: list[GPUInfo] = []
    for row in csv.reader(line for line in output.splitlines() if line.strip()):
        if len(row) < 4:
            continue
        index_raw, name, compute_capability, driver_version = [item.strip() for item in row[:4]]
        memory_mib: int | None = None
        if len(row) >= 5:
            memory_match = re.search(r"\d+", row[4])
            if memory_match:
                memory_mib = int(memory_match.group(0))
        try:
            index = int(index_raw)
        except ValueError:
            continue
        uuid = row[5].strip() if len(row) >= 6 and row[5].strip() else None
        pci_bus_id = row[6].strip() if len(row) >= 7 and row[6].strip() else None
        gpus.append(
            GPUInfo(
                index=index,
                name=name,
                compute_capability=_normalize_compute_capability(compute_capability),
                driver_version=driver_version,
                memory_mib=memory_mib,
                uuid=uuid,
                pci_bus_id=pci_bus_id,
            )
        )
    return gpus


def parse_driver_cuda_version(output: str) -> str | None:
    match = re.search(r"CUDA Version\s*:\s*(\d+(?:\.\d+)?)", output, re.IGNORECASE)
    return match.group(1) if match else None


def scan_nvidia() -> tuple[list[GPUInfo], str | None, str]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        raise InstallError(
            "nvidia-smi was not found. Install or repair the NVIDIA display driver, then rerun install.bat."
        )
    try:
        csv_output = _run_capture(
            [
                executable,
                "--query-gpu=index,name,compute_cap,driver_version,memory.total,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ]
        )
    except (subprocess.CalledProcessError, OSError) as exc:
        raise InstallError(f"Unable to query NVIDIA GPUs with nvidia-smi: {exc}") from exc
    gpus = parse_nvidia_smi_csv(csv_output)
    if not gpus:
        raise InstallError("No NVIDIA CUDA GPU was reported by nvidia-smi.")
    try:
        full_output = _run_capture([executable])
    except (subprocess.CalledProcessError, OSError):
        full_output = ""
    return gpus, parse_driver_cuda_version(full_output), executable


def _cuda_version_from_path(path: Path) -> str | None:
    version_json = path / "version.json"
    if version_json.is_file():
        try:
            payload = json.loads(version_json.read_text(encoding="utf-8"))
            for key_path in (
                ("cuda", "version"),
                ("cuda_cudart", "version"),
            ):
                current: Any = payload
                for key in key_path:
                    current = current[key]
                match = re.search(r"\d+\.\d+", str(current))
                if match:
                    return match.group(0)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    nvcc = path / "bin" / "nvcc.exe"
    if nvcc.is_file():
        try:
            output = _run_capture([str(nvcc), "--version"])
            match = re.search(r"release\s+(\d+\.\d+)", output, re.IGNORECASE)
            if match:
                return match.group(1)
        except (OSError, subprocess.CalledProcessError):
            pass
    name_match = re.fullmatch(r"v?(\d+)\.(\d+)", path.name, re.IGNORECASE)
    if name_match:
        return f"{int(name_match.group(1))}.{int(name_match.group(2))}"
    return None


def _registry_cuda_paths() -> list[tuple[Path, str]]:
    if os.name != "nt":
        return []
    try:
        import winreg
    except ImportError:
        return []
    results: list[tuple[Path, str]] = []
    roots = (
        r"SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA",
        r"SOFTWARE\WOW6432Node\NVIDIA Corporation\GPU Computing Toolkit\CUDA",
    )
    for root in roots:
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, root) as parent:
                count = winreg.QueryInfoKey(parent)[0]
                for index in range(count):
                    child_name = winreg.EnumKey(parent, index)
                    try:
                        with winreg.OpenKey(parent, child_name) as child:
                            install_dir, _ = winreg.QueryValueEx(child, "InstallDir")
                        results.append((Path(str(install_dir)), f"registry:{child_name}"))
                    except OSError:
                        continue
        except OSError:
            continue
    return results


def discover_cuda_toolkits(
    *,
    environ: dict[str, str] | None = None,
    extra_paths: Iterable[Path] = (),
) -> list[CudaToolkit]:
    env = dict(os.environ if environ is None else environ)
    candidates: dict[str, dict[str, Any]] = {}

    def add_candidate(path_value: str | Path | None, source: str) -> None:
        if not path_value:
            return
        path = Path(str(path_value).strip().strip('"')).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        if not resolved.is_dir():
            return
        key = os.path.normcase(str(resolved))
        record = candidates.setdefault(key, {"path": resolved, "sources": set()})
        record["sources"].add(source)

    for key, value in env.items():
        upper = key.upper()
        if upper in {"CUDA_PATH", "CUDA_HOME"} or upper.startswith("CUDA_PATH_V"):
            add_candidate(value, f"environment:{key}")

    program_files_values = [env.get("ProgramFiles"), env.get("ProgramW6432")]
    for program_files in program_files_values:
        if not program_files:
            continue
        root = Path(program_files) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        if root.is_dir():
            for child in root.glob("v*"):
                add_candidate(child, "filesystem")

    for path, source in _registry_cuda_paths():
        add_candidate(path, source)

    nvcc = shutil.which("nvcc", path=env.get("PATH"))
    if nvcc:
        add_candidate(Path(nvcc).resolve().parent.parent, "PATH:nvcc")

    for path in extra_paths:
        add_candidate(path, "explicit")

    toolkits: list[CudaToolkit] = []
    for record in candidates.values():
        path = Path(record["path"])
        version = _cuda_version_from_path(path)
        if not version:
            continue
        nvcc_path = path / "bin" / "nvcc.exe"
        toolkits.append(
            CudaToolkit(
                version=version,
                path=str(path),
                nvcc_path=str(nvcc_path) if nvcc_path.is_file() else None,
                sources=tuple(sorted(record["sources"])),
            )
        )
    return sorted(toolkits, key=lambda item: (_version_tuple(item.version), item.path), reverse=True)


def load_hardware_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InstallError(f"Hardware profile manifest was not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InstallError(f"Hardware profile manifest is invalid JSON: {exc}") from exc
    if int(payload.get("schema_version", 0)) != 1:
        raise InstallError("Unsupported hardware profile schema version.")
    profiles = payload.get("profiles")
    if not isinstance(profiles, list) or not profiles:
        raise InstallError("Hardware profile manifest does not contain any profiles.")
    return payload




def validate_profile_contract(root: Path, profile: dict[str, Any]) -> dict[str, Any]:
    required_paths = {
        "base_requirements": root / str(profile.get("base_requirements") or ""),
        "attention_manifest": root / str(profile.get("attention_manifest") or ""),
    }
    for field, path in required_paths.items():
        if not str(profile.get(field) or ""):
            raise InstallError(f"Profile {profile.get('id')} does not define {field}.")
        if not path.is_file():
            raise InstallError(f"Profile {profile.get('id')} references a missing file: {path}")

    try:
        attention = json.loads(required_paths["attention_manifest"].read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise InstallError(f"Attention manifest is invalid JSON: {exc}") from exc
    packages = dict(attention.get("packages") or {})
    if set(packages) != {"mslk", "xformers"}:
        raise InstallError("The custom attention manifest must contain exactly mslk and xformers.")
    for name, record in packages.items():
        if not record.get("url") or not re.fullmatch(r"[0-9a-fA-F]{64}", str(record.get("sha256") or "")):
            raise InstallError(f"Attention package {name} is missing a URL or valid SHA256.")

    compatibility = dict(attention.get("compatibility") or {})
    if not compatibility:
        raise InstallError("The custom attention manifest does not define installer compatibility.")
    expected_pairs = {
        "torch_version": str(dict(profile.get("torch") or {}).get("version") or ""),
        "torch_cuda_runtime": str(dict(profile.get("torch") or {}).get("cuda_runtime") or ""),
        "triton_requirement": str(profile.get("triton_requirement") or ""),
    }
    for field, expected in expected_pairs.items():
        actual = str(compatibility.get(field) or "")
        if actual != expected:
            raise InstallError(
                f"Profile/attention manifest mismatch for {field}: expected {expected}, got {actual}."
            )
    profile_caps = {
        str(value)
        for value in profile.get(
            "attention_contract_compute_capabilities",
            profile.get("compute_capabilities", []),
        )
    }
    attention_caps = {str(value) for value in compatibility.get("compute_capabilities", [])}
    if profile_caps != attention_caps:
        raise InstallError(
            "Profile/attention manifest compute-capability lists do not match: "
            f"profile={sorted(profile_caps)}, attention={sorted(attention_caps)}."
        )
    return attention


def _profile_matches_gpu(profile: dict[str, Any], gpu: GPUInfo) -> bool:
    if str(profile.get("gpu_vendor", "")).upper() != "NVIDIA":
        return False
    accepted = {str(value) for value in profile.get("compute_capabilities", [])}
    return gpu.compute_capability in accepted or "*" in accepted


def _profile_matches_host(profile: dict[str, Any]) -> bool:
    target_platform = str(profile.get("platform", ""))
    if target_platform and target_platform != sys.platform:
        return False
    machine = str(profile.get("machine", "")).lower()
    host_machine = platform.machine().lower()
    host_aliases = {host_machine}
    if host_machine in {"amd64", "x86_64"}:
        host_aliases.update({"amd64", "x86_64"})
    if machine and machine not in host_aliases:
        return False
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    return python_version in {str(value) for value in profile.get("python_versions", [])}


def matching_profiles(
    manifest: dict[str, Any],
    gpu: GPUInfo,
    driver_cuda_version: str | None,
    *,
    enforce_host: bool = True,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    allow_unvalidated = bool(
        dict(manifest.get("selection_policy") or {}).get("allow_unvalidated_combinations", False)
    )
    for raw in manifest.get("profiles", []):
        profile = dict(raw)
        if not allow_unvalidated and str(profile.get("status", "")).lower() != "validated":
            continue
        if enforce_host and not _profile_matches_host(profile):
            continue
        if not _profile_matches_gpu(profile, gpu):
            continue
        required_runtime = str(dict(profile.get("torch") or {}).get("cuda_runtime") or "")
        if driver_cuda_version and required_runtime and not _version_at_least(
            driver_cuda_version, required_runtime
        ):
            continue
        matches.append(profile)
    return sorted(matches, key=lambda item: int(item.get("priority", 0)), reverse=True)


def community_matching_profiles(
    manifest: dict[str, Any],
    gpu: GPUInfo,
    driver_cuda_version: str | None,
    *,
    enforce_host: bool = True,
) -> list[dict[str, Any]]:
    """Build community/unverified candidates from published reference stacks.

    A community candidate means the package combination is available for a real
    capability probe on this NVIDIA GPU. It does not promote the GPU to a
    validated hardware target.
    """

    candidates: list[dict[str, Any]] = []
    for raw in manifest.get("profiles", []):
        reference = dict(raw)
        community = dict(reference.get("community_validation") or {})
        if not bool(community.get("enabled", False)):
            continue
        if enforce_host and not _profile_matches_host(reference):
            continue
        accepted = {
            str(value)
            for value in community.get("compatible_compute_capabilities", [])
        }
        if gpu.compute_capability not in accepted and "*" not in accepted:
            continue
        required_runtime = str(dict(reference.get("torch") or {}).get("cuda_runtime") or "")
        if driver_cuda_version and required_runtime and not _version_at_least(
            driver_cuda_version, required_runtime
        ):
            continue

        profile = json.loads(json.dumps(reference))
        source_id = str(reference.get("id") or "reference")
        capability_tag = gpu.compute_capability.replace(".", "")
        torch_cfg = dict(reference.get("torch") or {})
        torch_version = str(torch_cfg.get("version") or "unknown")
        cuda_tag = str(torch_cfg.get("cuda_tag") or torch_cfg.get("cuda_runtime") or "cuda")
        profile["source_profile_id"] = source_id
        profile["id"] = f"windows-nvidia-community-sm{capability_tag}-torch{torch_version.replace('.', '')}-{cuda_tag}"
        profile["label"] = (
            f"NVIDIA {gpu.name} / compute {gpu.compute_capability} / "
            f"PyTorch {torch_version} / {cuda_tag} / Community qualification"
        )
        profile["status"] = str(community.get("status") or "community_unverified")
        profile["qualification"] = "community_unverified"
        profile["compute_capabilities"] = [gpu.compute_capability]
        profile["strict_attention_validation"] = False
        runtime_environment = dict(reference.get("runtime_environment") or {})
        runtime_environment.update(dict(community.get("runtime_environment") or {}))
        profile["runtime_environment"] = runtime_environment
        candidates.append(profile)

    return sorted(candidates, key=lambda item: int(item.get("priority", 0)), reverse=True)


def _profile_qualification(profile: dict[str, Any]) -> str:
    if str(profile.get("qualification") or "").strip():
        return str(profile["qualification"]).strip()
    if str(profile.get("status") or "").lower() == "validated":
        return "validated"
    return "community_unverified"


def build_install_choices(
    profiles: Sequence[dict[str, Any]], toolkits: Sequence[CudaToolkit]
) -> list[InstallChoice]:
    choices: list[InstallChoice] = []
    for profile in profiles:
        profile_id = str(profile["id"])
        label = str(profile.get("label") or profile_id)
        torch = dict(profile.get("torch") or {})
        runtime = str(torch.get("cuda_runtime") or "unknown")
        if bool(profile.get("allow_bundled_cuda_runtime", False)):
            choices.append(
                InstallChoice(
                    profile_id=profile_id,
                    profile_label=label,
                    mode="bundled",
                    cuda_runtime=runtime,
                    toolkit_path=None,
                    toolkit_version=None,
                    description=(
                        f"{label}: use the PyTorch CUDA {runtime} bundled runtime "
                        "(recommended; no local CUDA toolkit required)"
                    ),
                )
            )
        validated = {str(value) for value in profile.get("validated_local_cuda_toolkits", [])}
        for toolkit in toolkits:
            if toolkit.version not in validated:
                continue
            choices.append(
                InstallChoice(
                    profile_id=profile_id,
                    profile_label=label,
                    mode="local_toolkit",
                    cuda_runtime=runtime,
                    toolkit_path=toolkit.path,
                    toolkit_version=toolkit.version,
                    description=(
                        f"{label}: use installed CUDA {toolkit.version} toolkit at {toolkit.path}"
                    ),
                )
            )
    return choices


def _select_menu(items: Sequence[Any], label: str, formatter) -> Any:
    if not items:
        raise InstallError(f"No selectable {label} options are available.")
    if len(items) == 1:
        print(f"Selected {label}: {formatter(items[0])}")
        return items[0]
    print(f"\nAvailable {label} options:")
    for index, item in enumerate(items, start=1):
        print(f"  {index}. {formatter(item)}")
    while True:
        response = input(f"Select {label} [1-{len(items)}]: ").strip()
        try:
            selected = int(response)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= selected <= len(items):
            return items[selected - 1]
        print("Selection is outside the available range.")


def _find_profile(profiles: Sequence[dict[str, Any]], profile_id: str) -> dict[str, Any]:
    for profile in profiles:
        if str(profile.get("id")) == profile_id:
            return dict(profile)
    raise InstallError(f"Unknown hardware profile: {profile_id}")


def _choose_install_choice(
    choices: Sequence[InstallChoice],
    *,
    requested_profile: str | None,
    requested_cuda: str | None,
    non_interactive: bool,
) -> InstallChoice:
    filtered = list(choices)
    if requested_profile:
        filtered = [item for item in filtered if item.profile_id == requested_profile]
    if requested_cuda:
        normalized = requested_cuda.strip().lower()
        if normalized == "bundled":
            filtered = [item for item in filtered if item.mode == "bundled"]
        else:
            requested_path = os.path.normcase(os.path.abspath(requested_cuda))
            filtered = [
                item
                for item in filtered
                if item.toolkit_version == requested_cuda
                or (
                    item.toolkit_path
                    and os.path.normcase(os.path.abspath(item.toolkit_path)) == requested_path
                )
            ]
    if not filtered:
        raise InstallError(
            "The requested CUDA/profile combination is not present in the validated hardware manifest."
        )
    if non_interactive:
        bundled = [item for item in filtered if item.mode == "bundled"]
        return (bundled or filtered)[0]
    return _select_menu(filtered, "CUDA environment", lambda item: item.description)


def _check_host_requirements() -> None:
    if sys.platform != "win32":
        raise InstallError("This installer currently supports Windows only.")
    running_python = (
        sys.version_info.major,
        sys.version_info.minor,
        sys.version_info.micro,
    )
    if running_python != REQUIRED_PYTHON:
        required = ".".join(str(part) for part in REQUIRED_PYTHON)
        detected = ".".join(str(part) for part in running_python)
        raise InstallError(
            f"IMAGE_GEN requires Python {required} x64 exactly. "
            f"This installer is running with Python {detected}. "
            "Other Python 3.10 patch releases are not accepted for this build."
        )
    if struct.calcsize("P") * 8 != 64:
        raise InstallError("Python 3.10.20 must be a 64-bit installation.")


def _environment_for_choice(
    root: Path, gpu: GPUInfo, profile: dict[str, Any], choice: InstallChoice
) -> dict[str, str]:
    env = dict(os.environ)
    python_path_entries = [str(root / "src"), str(root)]
    existing_python_path = env.get("PYTHONPATH")
    if existing_python_path:
        python_path_entries.append(existing_python_path)
    env["PYTHONPATH"] = os.pathsep.join(python_path_entries)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = gpu.uuid or str(gpu.index)
    env["IMAGE_GEN_HARDWARE_PROFILE"] = choice.profile_id
    env["IMAGE_GEN_PYTORCH_CUDA_RUNTIME"] = choice.cuda_runtime
    for key, value in dict(profile.get("runtime_environment") or {}).items():
        env[str(key)] = str(value)
    if choice.toolkit_path:
        env["CUDA_PATH"] = choice.toolkit_path
        env["CUDA_HOME"] = choice.toolkit_path
        env["PATH"] = str(Path(choice.toolkit_path) / "bin") + os.pathsep + env.get("PATH", "")
    return env


def write_runtime_environment(
    path: Path, gpu: GPUInfo, profile: dict[str, Any], choice: InstallChoice
) -> None:
    lines = [
        "@echo off",
        "rem Generated by install.bat. Rerun the installer to change this environment.",
        f'set "IMAGE_GEN_HARDWARE_PROFILE={choice.profile_id}"',
        f'set "IMAGE_GEN_PYTORCH_CUDA_RUNTIME={choice.cuda_runtime}"',
        'set "CUDA_DEVICE_ORDER=PCI_BUS_ID"',
        f'set "CUDA_VISIBLE_DEVICES={gpu.uuid or gpu.index}"',
    ]
    if choice.toolkit_path:
        lines.extend(
            [
                f'set "CUDA_PATH={choice.toolkit_path}"',
                f'set "CUDA_HOME={choice.toolkit_path}"',
                'set "PATH=%CUDA_PATH%\\bin;%PATH%"',
            ]
        )
    for key, value in dict(profile.get("runtime_environment") or {}).items():
        lines.append(f'set "{key}={value}"')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\r\n".join(lines) + "\r\n", encoding="utf-8")


def _create_venv(root: Path, *, dry_run: bool) -> tuple[Path, Path | None]:
    target = root / ".venv"
    backup: Path | None = None
    if target.exists():
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = root / f".venv.backup-{stamp}"
        print(f"Existing .venv will be preserved at: {backup}")
        if not dry_run:
            target.rename(backup)
    print(f"Creating clean virtual environment: {target}")
    if not dry_run:
        venv.EnvBuilder(with_pip=True, clear=False).create(target)
    python = target / "Scripts" / "python.exe"
    return python, backup


def _restore_venv(root: Path, backup: Path | None) -> None:
    target = root / ".venv"
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    if backup and backup.exists():
        backup.rename(target)


def _install_profile(
    root: Path,
    python: Path,
    profile: dict[str, Any],
    choice: InstallChoice,
    gpu: GPUInfo,
    *,
    skip_gpu_smoke: bool,
    no_download: bool,
    dry_run: bool,
) -> dict[str, Any]:
    env = _environment_for_choice(root, gpu, profile, choice)
    torch = dict(profile.get("torch") or {})
    torch_packages = [str(item) for item in torch.get("packages", [])]
    index_url = str(torch.get("index_url") or "")
    if not torch_packages or not index_url:
        raise InstallError(f"Profile {profile['id']} does not define the PyTorch packages/index.")

    _run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        cwd=root,
        env=env,
        dry_run=dry_run,
    )
    _run(
        [str(python), "-m", "pip", "install", "--index-url", index_url, *torch_packages],
        cwd=root,
        env=env,
        dry_run=dry_run,
    )
    triton_requirement = str(profile.get("triton_requirement") or "")
    if triton_requirement:
        _run(
            [str(python), "-m", "pip", "install", triton_requirement],
            cwd=root,
            env=env,
            dry_run=dry_run,
        )

    requirements_path = root / str(profile.get("base_requirements"))
    _run(
        [str(python), "-m", "pip", "install", "-r", str(requirements_path)],
        cwd=root,
        env=env,
        dry_run=dry_run,
    )

    attention_manifest = root / str(profile.get("attention_manifest"))
    qualification = _profile_qualification(profile)
    community_mode = qualification != "validated"
    attention_command = [
        str(python),
        str(root / "scripts" / "release" / "install_published_attention_stack.py"),
        "--project-root",
        str(root),
        "--python",
        str(python),
        "--manifest",
        str(attention_manifest),
    ]
    if community_mode:
        attention_command.append("--community-compatibility")
    if skip_gpu_smoke:
        attention_command.append("--skip-gpu-smoke")
    if no_download:
        attention_command.append("--no-download")
    if dry_run:
        attention_command.append("--dry-run")
    _run(attention_command, cwd=root, env=env, dry_run=dry_run)

    if dry_run:
        return {
            "dry_run": True,
            "torch_packages": torch_packages,
            "index_url": index_url,
            "attention_manifest": str(attention_manifest),
            "qualification": qualification,
        }

    verification_code = (
        "import json, torch, mslk, xformers; "
        "available=torch.cuda.is_available(); "
        "p=torch.cuda.get_device_properties(0) if available else None; "
        "print(json.dumps({'torch':torch.__version__,'torch_cuda':torch.version.cuda,"
        "'cuda_available':available,"
        "'gpu':torch.cuda.get_device_name(0) if available else None,"
        "'compute_capability':[p.major,p.minor] if p else None,"
        "'mslk':getattr(mslk,'__version__','unknown'),"
        "'xformers':getattr(xformers,'__version__','unknown')}))"
    )
    raw = _run_capture([str(python), "-c", verification_code], env=env).strip().splitlines()[-1]
    verification = json.loads(raw)
    expected_capability = _normalize_compute_capability(gpu.compute_capability)
    if not verification.get("cuda_available"):
        raise InstallError("PyTorch installed successfully, but CUDA is not available in the new venv.")
    actual_capability = ".".join(str(item) for item in verification["compute_capability"])
    if actual_capability != expected_capability:
        raise InstallError(
            f"Selected GPU compute capability changed during verification: expected {expected_capability}, "
            f"got {actual_capability}."
        )
    if str(verification.get("torch_cuda")) != str(choice.cuda_runtime):
        raise InstallError(
            f"PyTorch CUDA runtime mismatch: expected {choice.cuda_runtime}, "
            f"got {verification.get('torch_cuda')}."
        )

    verification["qualification"] = qualification
    if community_mode:
        if skip_gpu_smoke:
            verification["community_attention_probe"] = {
                "status": "skipped",
                "passed": None,
                "reason": "GPU smoke/probe was disabled by the installer option.",
            }
        else:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            probe_dir = root / "artifacts" / "hardware_qualification" / stamp
            probe_command = [
                str(python),
                "-m",
                "image_gen.tools.verify_attention_stack",
                "--community-compatibility-test",
                "--output-dir",
                str(probe_dir),
            ]
            print("+ " + subprocess.list2cmdline(probe_command))
            completed = subprocess.run(
                probe_command,
                cwd=root,
                env=env,
                check=False,
            )
            summary_path = probe_dir / "validation_summary.json"
            summary: dict[str, Any] = {}
            if summary_path.is_file():
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    summary = {}
            verification["community_attention_probe"] = {
                "status": "passed" if completed.returncode == 0 else "failed",
                "passed": completed.returncode == 0,
                "returncode": int(completed.returncode),
                "output_directory": str(probe_dir),
                "operator": summary.get("operator"),
                "case_count": len(summary.get("compatibility_matrix") or []),
                "failure_count": len(summary.get("failures") or []),
            }
    return verification


def _hardware_profile_fingerprint(
    gpu: GPUInfo,
    profile: dict[str, Any],
    choice: InstallChoice,
    verification: dict[str, Any],
) -> str:
    material = "|".join(
        [
            gpu.name.strip(),
            gpu.compute_capability,
            gpu.driver_version,
            str(verification.get("torch") or ""),
            str(verification.get("torch_cuda") or choice.cuda_runtime),
            str(verification.get("mslk") or ""),
            str(verification.get("xformers") or ""),
            str(profile.get("source_profile_id") or profile.get("id") or ""),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def save_shareable_hardware_profile(
    root: Path,
    gpu: GPUInfo,
    driver_cuda_version: str | None,
    profile: dict[str, Any],
    choice: InstallChoice,
    verification: dict[str, Any],
) -> dict[str, Any]:
    """Save a privacy-minimized qualification profile suitable for sharing."""

    qualification = _profile_qualification(profile)
    fingerprint = _hardware_profile_fingerprint(gpu, profile, choice, verification)
    capability_tag = gpu.compute_capability.replace(".", "")
    profile_id = f"nvidia-sm{capability_tag}-{fingerprint}"
    probe = dict(verification.get("community_attention_probe") or {})
    directory = root / "user_config" / "hardware_profiles"
    existing_opt_in = False
    latest_existing = directory / "latest.json"
    if latest_existing.is_file():
        try:
            previous = json.loads(latest_existing.read_text(encoding="utf-8"))
            existing_opt_in = bool(dict(previous.get("sharing") or {}).get("opt_in", False))
        except (OSError, json.JSONDecodeError, AttributeError):
            existing_opt_in = False
    payload = {
        "schema_version": 1,
        "profile_id": profile_id,
        "generated_at_utc": _utc_now(),
        "project": "IMAGE_GEN",
        "qualification": qualification,
        "reference_profile_id": str(
            profile.get("source_profile_id") or profile.get("id") or choice.profile_id
        ),
        "hardware": {
            "gpu_vendor": "NVIDIA",
            "gpu_name": gpu.name,
            "compute_capability": gpu.compute_capability,
            "vram_mib": gpu.memory_mib,
            "driver_version": gpu.driver_version,
            "driver_cuda_ceiling": driver_cuda_version,
            "platform": sys.platform,
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "runtime": {
            "torch_version": verification.get("torch"),
            "torch_cuda_runtime": verification.get("torch_cuda"),
            "mslk_version": verification.get("mslk"),
            "xformers_version": verification.get("xformers"),
            "cuda_available": bool(verification.get("cuda_available")),
            "attention_probe": {
                "status": probe.get("status"),
                "passed": probe.get("passed"),
                "operator": probe.get("operator"),
                "case_count": probe.get("case_count"),
                "failure_count": probe.get("failure_count"),
            },
        },
        "sharing": {
            "opt_in": existing_opt_in,
            "automatic_upload_available": False,
            "submission_status": "not_submitted",
        },
        "privacy": {
            "sanitized": True,
            "excluded_fields": [
                "gpu_uuid",
                "pci_bus_id",
                "project_root",
                "local_cuda_paths",
                "local_model_paths",
                "prompts",
                "generated_images",
            ],
        },
    }
    path = directory / f"{profile_id}.json"
    latest = directory / "latest.json"
    _write_json(path, payload)
    _write_json(latest, payload)
    return {
        "profile_id": profile_id,
        "path": str(path),
        "latest_path": str(latest),
        "qualification": qualification,
        "share_opt_in": existing_opt_in,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_user_lock(
    path: Path,
    python: Path,
    env: dict[str, str],
    profile: dict[str, Any],
    choice: InstallChoice,
    gpu: GPUInfo,
    *,
    dry_run: bool,
) -> dict[str, Any]:
    command = [str(python), "-m", "pip", "freeze", "--all"]
    print("+ " + subprocess.list2cmdline(command) + f" > {path}")
    if dry_run:
        return {
            "path": str(path),
            "generated": False,
            "dry_run": True,
            "profile_id": choice.profile_id,
        }

    frozen = _run_capture(command, env=env)
    package_lines = [line for line in frozen.splitlines() if line.strip()]
    header = [
        "# IMAGE_GEN machine-specific user lock.",
        "# Generated only after the full installation and validation completed successfully.",
        "# Do not commit this file as a universal project or developer lock.",
        f"# Generated UTC: {_utc_now()}",
        f"# Hardware profile: {choice.profile_id}",
        f"# Profile label: {profile.get('label') or choice.profile_label}",
        f"# GPU: {gpu.name}",
        f"# Compute capability: {gpu.compute_capability}",
        f"# PyTorch CUDA runtime: {choice.cuda_runtime}",
        "",
    ]
    payload = "\n".join(header + package_lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(path)
    return {
        "path": str(path),
        "generated": True,
        "sha256": _sha256_file(path),
        "package_line_count": len(package_lines),
        "profile_id": choice.profile_id,
    }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the Windows NVIDIA/CUDA environment and install an IMAGE_GEN "
            "PyTorch, Triton, custom MSLK, and custom xFormers stack. SM120 is the "
            "validated reference target; other NVIDIA architectures can use community qualification."
        )
    )
    parser.add_argument("--project-root", type=Path, default=_project_root())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gpu-index", type=int)
    parser.add_argument("--profile")
    parser.add_argument(
        "--cuda",
        help="Choose 'bundled', an installed toolkit version such as 12.8, or an exact toolkit path.",
    )
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--skip-gpu-smoke", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.project_root.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest
        else root / "requirements" / "hardware_profiles.json"
    )
    report_dir = root / "artifacts" / "install" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    report_path = report_dir / "install_report.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "started_at_utc": _utc_now(),
        "project_root": str(root),
        "manifest": str(manifest_path),
        "success": False,
        "dry_run": bool(args.dry_run),
        "scan_only": bool(args.scan_only),
    }
    backup: Path | None = None
    runtime_bat: Path | None = None
    runtime_bat_existed = False
    runtime_bat_previous: bytes | None = None
    user_lock: Path | None = None
    user_lock_existed = False
    user_lock_previous: bytes | None = None

    try:
        _check_host_requirements()
        manifest = load_hardware_manifest(manifest_path)
        if args.list_profiles:
            for profile in manifest["profiles"]:
                print(f"{profile['id']}: {profile.get('label', profile['id'])}")
            return 0

        gpus, driver_cuda_version, nvidia_smi = scan_nvidia()
        toolkits = discover_cuda_toolkits()
        report["nvidia_smi"] = nvidia_smi
        report["driver_cuda_version"] = driver_cuda_version
        report["gpus"] = [asdict(item) for item in gpus]
        report["cuda_toolkits"] = [asdict(item) for item in toolkits]

        if args.gpu_index is not None:
            selected_gpu = next((gpu for gpu in gpus if gpu.index == args.gpu_index), None)
            if selected_gpu is None:
                raise InstallError(f"GPU index {args.gpu_index} was not reported by nvidia-smi.")
        elif args.non_interactive:
            selected_gpu = gpus[0]
        else:
            selected_gpu = _select_menu(
                gpus,
                "GPU",
                lambda gpu: (
                    f"GPU {gpu.index}: {gpu.name} | compute {gpu.compute_capability} | "
                    f"driver {gpu.driver_version} | "
                    f"{gpu.memory_mib or 'unknown'} MiB"
                ),
            )

        validated_profiles = matching_profiles(manifest, selected_gpu, driver_cuda_version)
        community_profiles = community_matching_profiles(
            manifest, selected_gpu, driver_cuda_version
        )
        profiles = validated_profiles or community_profiles
        if args.profile:
            candidates = [*validated_profiles, *community_profiles]
            requested_matches = [
                item
                for item in candidates
                if str(item.get("id")) == args.profile
                or str(item.get("source_profile_id")) == args.profile
            ]
            if not requested_matches:
                raise InstallError(
                    f"Profile {args.profile} is not compatible with GPU compute capability "
                    f"{selected_gpu.compute_capability}, this Python, or the installed driver."
                )
            profiles = [requested_matches[0]]
        if not profiles:
            validated = sorted(
                {
                    capability
                    for item in manifest["profiles"]
                    for capability in item.get("compute_capabilities", [])
                }
            )
            raise InstallError(
                "No IMAGE_GEN NVIDIA package profile can be offered for compute capability "
                f"{selected_gpu.compute_capability}. Validated reference targets currently cover: "
                f"{', '.join(validated) or 'none'}. This usually means the driver, platform, "
                "or published package stack is incompatible rather than that the GPU is merely unvalidated."
            )

        if toolkits:
            print("\nDetected CUDA toolkits:")
            for toolkit in toolkits:
                print(
                    f"  CUDA {toolkit.version}: {toolkit.path} "
                    f"[{', '.join(toolkit.sources)}]"
                )
        else:
            print("\nNo local CUDA toolkit was detected. This is allowed for bundled runtimes.")

        choices = build_install_choices(profiles, toolkits)
        choice = _choose_install_choice(
            choices,
            requested_profile=None,
            requested_cuda=args.cuda,
            non_interactive=args.non_interactive,
        )
        profile = _find_profile(profiles, choice.profile_id)
        attention_contract = validate_profile_contract(root, profile)
        report["selected_gpu"] = asdict(selected_gpu)
        report["selected_choice"] = asdict(choice)
        qualification = _profile_qualification(profile)
        report["selected_profile"] = profile
        report["qualification"] = qualification
        report["attention_release_id"] = attention_contract.get("release_id")

        print("\nDetected environment")
        print("--------------------")
        print(f"GPU:                 {selected_gpu.name}")
        print(f"Compute capability:  {selected_gpu.compute_capability}")
        print(f"NVIDIA driver:       {selected_gpu.driver_version}")
        print(f"Driver CUDA ceiling: {driver_cuda_version or 'unknown'}")
        print(f"Install profile:     {choice.profile_id}")
        print(f"Qualification:       {qualification}")
        if qualification != "validated":
            print("Hardware status:     community/unverified; capability probes will be recorded")
        print(f"PyTorch CUDA runtime:{choice.cuda_runtime}")
        print(f"Local CUDA toolkit:  {choice.toolkit_path or 'not required'}")
        if toolkits:
            print("Installed toolkits:   " + ", ".join(item.version for item in toolkits))
        else:
            print("Installed toolkits:   none detected (allowed; PyTorch bundles its runtime)")

        _write_json(report_path, report)
        if args.scan_only:
            report["success"] = True
            report["completed_at_utc"] = _utc_now()
            _write_json(report_path, report)
            print(f"\nScan complete. Report: {report_path}")
            return 0

        runtime_bat = root / "user_config" / "runtime_environment.bat"
        runtime_bat_existed = runtime_bat.is_file()
        if runtime_bat_existed:
            runtime_bat_previous = runtime_bat.read_bytes()
        user_lock = root / "user_config" / "requirements-user-lock.txt"
        user_lock_existed = user_lock.is_file()
        if user_lock_existed:
            user_lock_previous = user_lock.read_bytes()
        if not args.dry_run:
            write_runtime_environment(runtime_bat, selected_gpu, profile, choice)
        python, backup = _create_venv(root, dry_run=args.dry_run)
        verification = _install_profile(
            root,
            python,
            profile,
            choice,
            selected_gpu,
            skip_gpu_smoke=args.skip_gpu_smoke,
            no_download=args.no_download,
            dry_run=args.dry_run,
        )
        report["verification"] = verification
        user_lock_record = generate_user_lock(
            user_lock,
            python,
            _environment_for_choice(root, selected_gpu, profile, choice),
            profile,
            choice,
            selected_gpu,
            dry_run=args.dry_run,
        )
        report["user_lock"] = user_lock_record
        if args.dry_run:
            saved_hardware_profile = {
                "generated": False,
                "dry_run": True,
                "qualification": qualification,
            }
        else:
            saved_hardware_profile = save_shareable_hardware_profile(
                root,
                selected_gpu,
                driver_cuda_version,
                profile,
                choice,
                verification,
            )
        report["saved_hardware_profile"] = saved_hardware_profile
        report["venv"] = str(root / ".venv")
        report["previous_venv_backup"] = str(backup) if backup else None
        report["runtime_environment"] = str(runtime_bat)
        report["success"] = True
        report["completed_at_utc"] = _utc_now()
        _write_json(report_path, report)
        if qualification == "validated":
            print("\nPASS: IMAGE_GEN environment installed with a validated reference profile.")
        else:
            print("\nPASS: IMAGE_GEN environment installed in community qualification mode.")
            probe = dict(verification.get("community_attention_probe") or {})
            print(
                "Attention probe: "
                + ("PASS" if probe.get("passed") is True else "UNVERIFIED/FAILED - Auto can fall back to SDPA")
            )
        print(f"Environment: {root / '.venv'}")
        print(f"Report:      {report_path}")
        if not args.dry_run:
            print(f"User lock:   {user_lock}")
            print(f"HW profile:  {saved_hardware_profile['path']}")
        print("Start with:  run_webui.bat")
        return 0
    except Exception as exc:
        if backup is not None and not args.dry_run:
            print("Installation failed. Restoring the previous .venv...")
            _restore_venv(root, backup)
        if runtime_bat is not None and not args.dry_run:
            if runtime_bat_existed and runtime_bat_previous is not None:
                runtime_bat.parent.mkdir(parents=True, exist_ok=True)
                runtime_bat.write_bytes(runtime_bat_previous)
            elif runtime_bat.exists():
                runtime_bat.unlink()
        if user_lock is not None and not args.dry_run:
            if user_lock_existed and user_lock_previous is not None:
                user_lock.parent.mkdir(parents=True, exist_ok=True)
                user_lock.write_bytes(user_lock_previous)
            elif user_lock.exists():
                user_lock.unlink()
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["completed_at_utc"] = _utc_now()
        _write_json(report_path, report)
        print(f"\nFAIL: {exc}")
        print(f"Report: {report_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
