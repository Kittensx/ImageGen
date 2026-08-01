from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence


LATENT_UPSCALERS = (
    "latent_bicubic",
    "latent_bilinear",
    "latent_nearest",
)


@dataclass(frozen=True)
class HiresRunSettings:
    positive_prompt: str
    negative_prompt: str
    steps: int
    cfg_scale: float
    seed: int
    width: int
    height: int
    batch_size: int
    batch_count: int
    unlimited: bool
    filename_pattern: str
    model_path: str
    sampler_name: str
    scheduler_name: str
    hires_positive_prompt: str
    hires_negative_prompt: str
    hires_scale: float
    hires_denoising_strength: float
    hires_steps: int
    hires_upscaler: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "width": self.width,
            "height": self.height,
            "batch_size": self.batch_size,
            "batch_count": self.batch_count,
            "unlimited": self.unlimited,
            "filename_pattern": self.filename_pattern,
            "model_path": self.model_path,
            "sampler_name": self.sampler_name,
            "scheduler_name": self.scheduler_name,
            "hires_positive_prompt": self.hires_positive_prompt,
            "hires_negative_prompt": self.hires_negative_prompt,
            "hires_scale": self.hires_scale,
            "hires_denoising_strength": self.hires_denoising_strength,
            "hires_steps": self.hires_steps,
            "hires_upscaler": self.hires_upscaler,
        }


def prompt_with_default(label: str, default: Any) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return str(default) if raw == "" else raw


def prompt_int(label: str, default: int, *, minimum: int = 1) -> int:
    while True:
        raw = prompt_with_default(label, default)
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        return value


def prompt_float(
    label: str,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    while True:
        raw = prompt_with_default(label, default)
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum}.")
            continue
        if maximum is not None and value > maximum:
            print(f"Value must not exceed {maximum}.")
            continue
        return value


def prompt_bool(label: str, default: bool = False) -> bool:
    default_text = "y" if default else "n"
    while True:
        raw = prompt_with_default(label, default_text).strip().lower()
        if raw in {"y", "yes", "1", "true", "on"}:
            return True
        if raw in {"n", "no", "0", "false", "off"}:
            return False
        print("Please enter y or n.")


def choose_latent_upscaler(default: str = "latent_bicubic") -> str:
    print("\n=== Built-in latent upscaler ===")
    for index, name in enumerate(LATENT_UPSCALERS, start=1):
        marker = " (default)" if name == default else ""
        print(f"{index}. {name}{marker}")
    default_index = LATENT_UPSCALERS.index(default) + 1
    while True:
        raw = input(f"Choose latent upscaler [{default_index}]: ").strip()
        if raw == "":
            return default
        try:
            index = int(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if 1 <= index <= len(LATENT_UPSCALERS):
            return LATENT_UPSCALERS[index - 1]
        print("Invalid selection.")


def build_base_payload(
    settings: HiresRunSettings,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "positive_prompt": settings.positive_prompt,
        "negative_prompt": settings.negative_prompt,
        "steps": settings.steps,
        "cfg_scale": settings.cfg_scale,
        "seed": seed,
        "width": settings.width,
        "height": settings.height,
        "batch_size": settings.batch_size,
        "batch_count": 1,
        "unlimited": False,
        "model_path": settings.model_path,
        "sampler_name": settings.sampler_name,
        "scheduler_name": settings.scheduler_name,
        "output_dir": str(output_dir),
        "output_prefix": f"lowres-{settings.filename_pattern}",
        "save_images": True,
        "hires_enabled": False,
    }


def build_hires_payload(
    settings: HiresRunSettings,
    *,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    payload = build_base_payload(settings, seed=seed, output_dir=output_dir)
    payload.update(
        {
            "output_prefix": f"hires-{settings.filename_pattern}",
            "hires_enabled": True,
            "hires_size_mode": "scale_from_base",
            "hires_scale": settings.hires_scale,
            "hires_steps": settings.hires_steps,
            "hires_denoising_strength": settings.hires_denoising_strength,
            "hires_upscaler": settings.hires_upscaler,
            "hires_prompt_parser_mode": "same_as_base",
            "hires_shortcut_profile_mode": "same_as_base",
            # Blank input means inherit. Storing the resolved base values makes
            # the replay manifest explicit and preserves the user's intent.
            "hires_positive_prompt": (
                settings.hires_positive_prompt
                if settings.hires_positive_prompt != ""
                else settings.positive_prompt
            ),
            "hires_negative_prompt": (
                settings.hires_negative_prompt
                if settings.hires_negative_prompt != ""
                else settings.negative_prompt
            ),
        }
    )
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    from image_gen.runtime_options import add_runtime_startup_arguments

    parser = argparse.ArgumentParser(
        description=(
            "Interactive paired low-resolution and latent-hires IMAGE_GEN runner."
        )
    )
    parser.add_argument("--project-root")
    parser.add_argument(
        "--output-root",
        help="Optional output root. Defaults to <txt2img output>/hires_run.",
    )
    parser.add_argument(
        "--latent-upscaler",
        choices=LATENT_UPSCALERS,
        help="Skip the interactive latent-upscaler prompt.",
    )
    add_runtime_startup_arguments(parser)
    return parser


def _collect_settings(context: Any, registry_system: Any, args: argparse.Namespace) -> HiresRunSettings:
    from modules.txt2img.cli_interactive import choose_from_registry
    from modules.txt2img.model_selector import ModelSelector

    print("\n=== Base generation settings ===")
    positive_prompt = input("Positive prompt []: ").strip()
    negative_prompt = input("Negative prompt []: ").strip()
    steps = prompt_int("Steps", 20)
    cfg_scale = prompt_float("CFG Scale", 7.0, minimum=0.0)

    while True:
        raw_seed = prompt_with_default("Seed", -1)
        try:
            seed = int(raw_seed)
            break
        except ValueError:
            print("Please enter a whole number. Use -1 for random.")

    width = prompt_int("Width", 640, minimum=8)
    height = prompt_int("Height", 960, minimum=8)
    batch_size = prompt_int("Batch size", 1)
    batch_count = prompt_int("Batch count", 1)
    unlimited = prompt_bool("Unlimited generation (y/n)", False)

    print(
        "Filename fields: {index:05d}, {seed}, {datetime}, {model}, "
        "{vae}, {lora}, {sampler}, {scheduler}, {width}, {height}"
    )
    filename_pattern = prompt_with_default("Filename pattern", "{index:05d}-{seed}")

    model_entry = ModelSelector(project_context=context).choose_model()
    live_sampler_map = registry_system.legacy_map("sampler")
    live_scheduler_map = registry_system.legacy_map("scheduler")
    sampler_entry = choose_from_registry("Sampler", live_sampler_map)
    scheduler_entry = choose_from_registry("Scheduler", live_scheduler_map)

    print("\n=== Hires settings ===")
    print("Press Enter for either hires prompt to keep the matching base prompt.")
    hires_positive_prompt = input("Hires positive prompt [inherit base]: ").strip()
    hires_negative_prompt = input("Hires negative prompt [inherit base]: ").strip()
    hires_scale = prompt_float("Hires scale", 1.5, minimum=1.01, maximum=8.0)
    hires_denoising_strength = prompt_float(
        "Hires denoising strength",
        0.4,
        minimum=0.01,
        maximum=1.0,
    )
    hires_steps = prompt_int("Hires steps", 20)
    hires_upscaler = args.latent_upscaler or choose_latent_upscaler()

    return HiresRunSettings(
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        width=width,
        height=height,
        batch_size=batch_size,
        batch_count=batch_count,
        unlimited=unlimited,
        filename_pattern=filename_pattern,
        model_path=model_entry.path,
        sampler_name=str(sampler_entry.get("name") or sampler_entry.get("label") or ""),
        scheduler_name=str(
            scheduler_entry.get("name") or scheduler_entry.get("label") or ""
        ),
        hires_positive_prompt=hires_positive_prompt,
        hires_negative_prompt=hires_negative_prompt,
        hires_scale=hires_scale,
        hires_denoising_strength=hires_denoising_strength,
        hires_steps=hires_steps,
        hires_upscaler=hires_upscaler,
    )


def _request_from_payload(
    payload: dict[str, Any],
    *,
    context: Any,
    runtime_startup_options: Any,
    live_sampler_map: dict[str, Any],
    live_scheduler_map: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    from image_gen.runtime.scheduler_settings import normalize_scheduler_payload
    from image_gen.runtime_options import (
        build_runtime_startup_status,
        runtime_request_settings,
    )
    from modules.txt2img.request_loader import load_request_payload, payload_to_generation_request

    merged = load_request_payload(
        base_payload=context.generation_defaults(),
        cli_overrides=payload,
    )
    merged, _resolution = normalize_scheduler_payload(merged)
    request, extras = payload_to_generation_request(merged)
    request.save_images = True
    extras.update(runtime_request_settings(runtime_startup_options))
    extras.update(
        {
            "live_sampler_map": live_sampler_map,
            "live_scheduler_map": live_scheduler_map,
            "runtime_startup_status": build_runtime_startup_status(
                runtime_startup_options,
                {"mslk_fmha": runtime_startup_options.mslk_fmha.to_dict()},
            ),
        }
    )
    return request, extras


def _saved_paths(result: Any) -> list[str]:
    paths = [str(record.image_path) for record in result.saved_records]
    if not paths:
        raise RuntimeError("Generation completed without saving an image.")
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    from image_gen.runtime_options import argv_for_primary_parser, bootstrap_runtime_startup

    raw_argv = list(argv if argv is not None else sys.argv[1:])
    parser = _build_parser()
    parser_argv = argv_for_primary_parser(raw_argv)
    args = parser.parse_args(parser_argv)
    args._runtime_argv = raw_argv

    from modules.project_context import ProjectContext

    context = ProjectContext.load(project_root=args.project_root)
    runtime_startup_options = bootstrap_runtime_startup(
        args,
        settings=context.generation_defaults(),
    )

    # Imports below this point may load Torch, model code, or attention modules.
    from image_gen.systems.registry import RuntimeRegistrySystem
    from modules.adapters.prompt_conditioning_adapter import PromptConditioningAdapter
    from modules.load_safetensors_model import LoadModel
    from modules.txt2img.seed_utils import iter_batch_base_seeds
    from modules.txt2img.txt2img_runner import Txt2ImgRunner

    registry_system = RuntimeRegistrySystem(project_context=context)
    settings = _collect_settings(context, registry_system, args)

    default_output_root = Path(context.txt2img_output_root) / "hires_run"
    output_root = Path(args.output_root).expanduser() if args.output_root else default_output_root
    if not output_root.is_absolute():
        output_root = context.resolve_project_path(output_root)
    session_dir = output_root / datetime.now().strftime("%Y%m%d-%H%M%S")
    lowres_dir = session_dir / "lowres"
    hires_dir = session_dir / "hires"
    lowres_dir.mkdir(parents=True, exist_ok=True)
    hires_dir.mkdir(parents=True, exist_ok=True)

    run_record: dict[str, Any] = {
        "schema_version": 1,
        "mode": "paired_lowres_and_hires",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(context.project_root),
        "output_root": str(session_dir),
        "runtime_startup_options": runtime_startup_options.to_dict(),
        "settings": settings.to_dict(),
        "note": (
            "The low-resolution artifact is generated first with the same base settings "
            "and seed. The hires job then repeats the deterministic base pass internally "
            "and performs the latent hires refinement while reusing the loaded model."
        ),
        "batches": [],
    }
    record_path = session_dir / "hires_run_manifest.json"
    _write_json(record_path, run_record)

    print("\n=== Paired output plan ===")
    print(f"Low-resolution artifacts: {lowres_dir}")
    print(f"Hires images:            {hires_dir}")
    print(f"Run manifest:            {record_path}")
    print(
        "The model remains loaded between the paired renders, but the base sampling "
        "pass is executed once for the saved low-resolution image and again inside "
        "the hires workflow."
    )

    model_loader = LoadModel(project_context=context)
    runner = Txt2ImgRunner(
        prompt_adapter_factory=lambda **_kwargs: PromptConditioningAdapter(),
        model_loader=model_loader,
        project_context=context,
        registry_system=registry_system,
    )
    live_sampler_map = registry_system.legacy_map("sampler")
    live_scheduler_map = registry_system.legacy_map("scheduler")

    seed_iterator = iter_batch_base_seeds(
        settings.seed,
        batch_size=settings.batch_size,
    )
    completed = 0
    try:
        while settings.unlimited or completed < settings.batch_count:
            batch_number = completed + 1
            batch_seed = next(seed_iterator)
            print(f"\n=== Paired batch {batch_number} ===")
            print(f"Base seed: {batch_seed}")

            base_payload = build_base_payload(
                settings,
                seed=batch_seed,
                output_dir=lowres_dir,
            )
            base_request, base_extras = _request_from_payload(
                base_payload,
                context=context,
                runtime_startup_options=runtime_startup_options,
                live_sampler_map=live_sampler_map,
                live_scheduler_map=live_scheduler_map,
            )
            runner.reset_runtime_state()
            print("\n--- Saving low-resolution artifact ---")
            base_result = runner.run_request(base_request, base_extras)
            base_paths = _saved_paths(base_result)

            hires_payload = build_hires_payload(
                settings,
                seed=batch_seed,
                output_dir=hires_dir,
            )
            hires_request, hires_extras = _request_from_payload(
                hires_payload,
                context=context,
                runtime_startup_options=runtime_startup_options,
                live_sampler_map=live_sampler_map,
                live_scheduler_map=live_scheduler_map,
            )
            runner.reset_runtime_state()
            print("\n--- Saving hires image ---")
            hires_result = runner.run_request(hires_request, hires_extras)
            hires_paths = _saved_paths(hires_result)

            batch_record = {
                "batch_number": batch_number,
                "base_seed": int(batch_seed),
                "lowres_run_id": base_result.run_id,
                "hires_run_id": hires_result.run_id,
                "lowres_paths": base_paths,
                "hires_paths": hires_paths,
                "lowres_generation_time_sec": base_result.generation_time_sec,
                "hires_generation_time_sec": hires_result.generation_time_sec,
                "hires_metadata": dict(
                    hires_result.pipeline_result.metadata.get("hires_fix") or {}
                ),
            }
            run_record["batches"].append(batch_record)
            _write_json(record_path, run_record)

            print("\nPaired batch complete.")
            for path in base_paths:
                print(f"Lowres: {path}")
            for path in hires_paths:
                print(f"Hires:  {path}")
            completed += 1
    except KeyboardInterrupt:
        print("\nCancelled by user.")
        run_record["cancelled"] = True
        run_record["completed_batches"] = completed
        _write_json(record_path, run_record)
        return 130

    run_record["completed_batches"] = completed
    run_record["completed_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(record_path, run_record)
    print(f"\nCompleted {completed} paired batch(es).")
    print(f"Results: {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
