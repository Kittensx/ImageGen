from __future__ import annotations

import json
from pathlib import Path

from scripts.setup import install_image_gen as installer


def _profile() -> dict:
    return {
        "id": "sm120",
        "label": "SM120",
        "status": "validated",
        "priority": 100,
        "gpu_vendor": "NVIDIA",
        "compute_capabilities": ["12.0"],
        "torch": {"cuda_runtime": "12.8"},
        "allow_bundled_cuda_runtime": True,
        "validated_local_cuda_toolkits": ["12.8"],
        "runtime_environment": {"MSLK_FMHA_POLICY": "blackwell_safe"},
    }


def test_parse_nvidia_smi_csv_supports_multiple_gpus() -> None:
    output = (
        "0, NVIDIA GeForce RTX 5070 Laptop GPU, 12.0, 596.13, 8151\n"
        "1, NVIDIA GeForce RTX 3080 Ti, 8.6, 596.13, 12288, GPU-ABC, 00000000:01:00.0\n"
    )
    gpus = installer.parse_nvidia_smi_csv(output)
    assert [gpu.index for gpu in gpus] == [0, 1]
    assert gpus[0].compute_capability == "12.0"
    assert gpus[1].memory_mib == 12288
    assert gpus[1].uuid == "GPU-ABC"
    assert gpus[1].pci_bus_id == "00000000:01:00.0"


def test_driver_cuda_version_is_parsed_from_nvidia_smi_header() -> None:
    output = "NVIDIA-SMI 596.13 Driver Version: 596.13 CUDA Version: 13.2"
    assert installer.parse_driver_cuda_version(output) == "13.2"


def test_matching_profiles_rejects_driver_below_runtime() -> None:
    gpu = installer.GPUInfo(0, "GPU", "12.0", "1", 8000)
    manifest = {"profiles": [_profile()]}
    assert installer.matching_profiles(manifest, gpu, "12.7", enforce_host=False) == []
    assert installer.matching_profiles(manifest, gpu, "13.2", enforce_host=False)


def test_choices_include_bundled_and_only_validated_installed_toolkits() -> None:
    toolkits = [
        installer.CudaToolkit("13.2", r"C:\CUDA\v13.2", None, ("test",)),
        installer.CudaToolkit("12.8", r"C:\CUDA\v12.8", None, ("test",)),
    ]
    choices = installer.build_install_choices([_profile()], toolkits)
    assert [choice.mode for choice in choices] == ["bundled", "local_toolkit"]
    assert choices[1].toolkit_version == "12.8"


def test_noninteractive_selection_prefers_bundled_runtime() -> None:
    choices = installer.build_install_choices(
        [_profile()],
        [installer.CudaToolkit("12.8", r"C:\CUDA\v12.8", None, ("test",))],
    )
    selected = installer._choose_install_choice(
        choices,
        requested_profile=None,
        requested_cuda=None,
        non_interactive=True,
    )
    assert selected.mode == "bundled"


def test_runtime_environment_records_selected_gpu_and_toolkit(tmp_path: Path) -> None:
    gpu = installer.GPUInfo(2, "GPU", "12.0", "1", 8000)
    choice = installer.InstallChoice(
        profile_id="sm120",
        profile_label="SM120",
        mode="local_toolkit",
        cuda_runtime="12.8",
        toolkit_path=r"C:\CUDA\v12.8",
        toolkit_version="12.8",
        description="test",
    )
    target = tmp_path / "runtime_environment.bat"
    installer.write_runtime_environment(target, gpu, _profile(), choice)
    text = target.read_text(encoding="utf-8")
    assert 'set "CUDA_VISIBLE_DEVICES=2"' in text
    assert 'set "CUDA_PATH=C:\\CUDA\\v12.8"' in text
    assert 'set "MSLK_FMHA_POLICY=blackwell_safe"' in text


def test_repository_profile_matches_published_attention_contract() -> None:
    root = Path(__file__).resolve().parents[2]
    profiles = json.loads(
        (root / "requirements" / "hardware_profiles.json").read_text(encoding="utf-8")
    )
    profile = profiles["profiles"][0]
    attention = json.loads(
        (root / profile["attention_manifest"]).read_text(encoding="utf-8")
    )
    compatibility = attention["compatibility"]
    assert profile["compute_capabilities"] == compatibility["compute_capabilities"]
    assert profile["torch"]["version"] == compatibility["torch_version"]
    assert profile["torch"]["cuda_runtime"] == compatibility["torch_cuda_runtime"]
    assert profile["triton_requirement"] == compatibility["triton_requirement"]
