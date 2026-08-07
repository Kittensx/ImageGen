from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from image_gen.runtime_options import (
    add_runtime_startup_arguments,
    argv_for_primary_parser,
    bootstrap_runtime_startup,
    build_runtime_startup_status,
    build_cuda_allocator_diagnostics,
    runtime_command_line_help_epilog,
    runtime_request_settings,
    runtime_replay_warnings,
)

from modules.project_context import (
    ProjectConfigurationError,
    ProjectContext,
    ProjectValidationError,
)


def _parse_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("Expected a JSON object.")
    return parsed


def _add_context_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-root",
        help="Override the IMAGE_GEN project root. Defaults to the package location.",
    )
    parser.add_argument(
        "--project-config",
        help=(
            "Override the canonical project config. Relative paths are resolved "
            "from --project-root."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="IMAGE_GEN txt2img runtime",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=runtime_command_line_help_epilog(),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a txt2img generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=runtime_command_line_help_epilog(),
    )
    _add_context_arguments(run_parser)
    add_runtime_startup_arguments(run_parser)
    run_parser.add_argument("--config", dest="config_path", help="Generation request YAML/JSON")
    run_parser.add_argument("--manifest", dest="manifest_path")
    run_parser.add_argument("--infotext", dest="infotext_path")
    run_parser.add_argument(
        "--effective-request-out",
        help=(
            "Write the fully merged, path-normalized request payload used by the "
            "CLI before generation starts. Intended for parity diagnostics."
        ),
    )

    run_parser.add_argument("--prompt", dest="positive_prompt")
    run_parser.add_argument("--negative-prompt", dest="negative_prompt")
    run_parser.add_argument("--steps", type=int)
    run_parser.add_argument("--cfg-scale", type=float)
    run_parser.add_argument("--seed", type=int)
    run_parser.add_argument("--width", type=int)
    run_parser.add_argument("--height", type=int)
    run_parser.add_argument("--batch-size", type=int)
    run_parser.add_argument("--batch-count", type=int)
    run_parser.add_argument(
        "--unlimited",
        action="store_true",
        default=None,
        help="Repeat batches until Ctrl+C. Random seed mode chooses a new base seed per batch.",
    )
    run_parser.add_argument("--model", dest="model_path")
    run_parser.add_argument("--sampler", dest="sampler_name")
    run_parser.add_argument("--scheduler", dest="scheduler_name")
    run_parser.add_argument("--output-dir")
    run_parser.add_argument(
        "--prefix",
        "--filename-pattern",
        dest="output_prefix",
        help=(
            "Output filename prefix or template. Example: "
            "{index:05d}-{seed}-{model}-{lora}"
        ),
    )
    run_parser.add_argument("--scheduler-kwargs", type=_parse_json_dict)
    run_parser.add_argument("--sampler-kwargs", type=_parse_json_dict)
    run_parser.add_argument("--prompt-parser", dest="prompt_parser_name")
    run_parser.add_argument("--prompt-parser-kwargs", type=_parse_json_dict)
    run_parser.add_argument("--prompt-shortcut-profile", dest="prompt_shortcut_profile_name")
    run_parser.add_argument("--prompt-parser-preset", dest="prompt_parser_preset_name")
    shadow_group = run_parser.add_mutually_exclusive_group()
    shadow_group.add_argument("--prompt-shadow-compare", dest="prompt_shadow_compare", action="store_true", default=None)
    shadow_group.add_argument("--no-prompt-shadow-compare", dest="prompt_shadow_compare", action="store_false")
    run_parser.add_argument("--hires-prompt-parser-mode", choices=("same_as_base", "explicit", "canonical_only"))
    run_parser.add_argument("--hires-prompt-parser", dest="hires_prompt_parser_name")
    run_parser.add_argument("--hires-prompt-parser-kwargs", type=_parse_json_dict)
    run_parser.add_argument("--hires-shortcut-profile-mode", choices=("same_as_base", "explicit", "canonical_only"))
    run_parser.add_argument("--hires-shortcut-profile", dest="hires_shortcut_profile_name")
    run_parser.add_argument("--hires-positive-prompt")
    run_parser.add_argument("--hires-negative-prompt")
    run_parser.add_argument("--hires-size-mode", choices=("same_as_base", "scale_from_base", "explicit_dimensions"))
    run_parser.add_argument("--hires-scale", type=float)
    run_parser.add_argument("--hires-width", type=int)
    run_parser.add_argument("--hires-height", type=int)
    hires_group = run_parser.add_mutually_exclusive_group()
    hires_group.add_argument(
        "--hires-fix",
        dest="hires_enabled",
        action="store_true",
        default=None,
        help="Enable the shared neural .pth hires second pass used by CLI and WebUI.",
    )
    hires_group.add_argument(
        "--no-hires-fix",
        dest="hires_enabled",
        action="store_false",
        help="Disable the hires second pass.",
    )
    run_parser.add_argument("--hires-steps", type=int)
    run_parser.add_argument("--hires-denoising-strength", type=float)
    run_parser.add_argument(
        "--hires-step-policy",
        choices=("a1111_fixed_steps_v1", "proportional_tail_v1"),
        help=(
            "Choose fixed executed-step hires semantics for new runs or the "
            "legacy proportional-tail policy for compatibility/replay."
        ),
    )
    run_parser.add_argument(
        "--hires-sampler",
        dest="hires_sampler_name",
        help="Optional second-pass sampler. Omit to inherit the base sampler.",
    )
    run_parser.add_argument(
        "--hires-scheduler",
        dest="hires_scheduler_name",
        help="Optional second-pass scheduler. Omit to inherit the base scheduler.",
    )
    run_parser.add_argument(
        "--hires-cfg-scale",
        type=float,
        help="Optional second-pass CFG scale. Omit to inherit the base CFG scale.",
    )
    run_parser.add_argument(
        "--hires-cfg-rescale",
        type=float,
        help="Optional second-pass CFG rescale. Omit to inherit the base CFG rescale.",
    )
    run_parser.add_argument(
        "--hires-strategy",
        choices=("pixel_neural",),
        help="Use a discovered pixel-neural .pth upscaler.",
    )
    run_parser.add_argument(
        "--hires-upscaler",
        help="Stable neural upscaler ID from the discovery catalog.",
    )
    run_parser.add_argument("--hires-tile-size", type=int)
    run_parser.add_argument("--hires-tile-overlap", type=int)
    run_parser.add_argument("--hires-tile-batch-size", type=int)
    run_parser.add_argument(
        "--hires-exact-resize-filter",
        choices=("nearest", "bilinear", "bicubic", "area"),
    )
    hires_pre_denoise_group = run_parser.add_mutually_exclusive_group()
    hires_pre_denoise_group.add_argument(
        "--hires-save-upscaled-pre-denoise",
        dest="hires_save_upscaled_pre_denoise",
        action="store_true",
        default=None,
        help="Save the exact post-upscale, pre-denoise image artifact.",
    )
    hires_pre_denoise_group.add_argument(
        "--no-hires-save-upscaled-pre-denoise",
        dest="hires_save_upscaled_pre_denoise",
        action="store_false",
    )
    hires_vae_roundtrip_group = run_parser.add_mutually_exclusive_group()
    hires_vae_roundtrip_group.add_argument(
        "--hires-save-vae-roundtrip",
        dest="hires_save_vae_roundtrip",
        action="store_true",
        default=None,
        help="Save the deterministic VAE encode/decode round-trip diagnostic artifact.",
    )
    hires_vae_roundtrip_group.add_argument(
        "--no-hires-save-vae-roundtrip",
        dest="hires_save_vae_roundtrip",
        action="store_false",
    )
    hires_lowres_group = run_parser.add_mutually_exclusive_group()
    hires_lowres_group.add_argument(
        "--hires-save-lowres",
        dest="hires_save_lowres",
        action="store_true",
        default=None,
        help="Save the exact base-pass image beside the final hires output.",
    )
    hires_lowres_group.add_argument(
        "--no-hires-save-lowres",
        dest="hires_save_lowres",
        action="store_false",
        help="Do not save an auxiliary base-pass image.",
    )
    run_parser.add_argument("--parser-kwargs", type=_parse_json_dict)
    interactive_group = run_parser.add_mutually_exclusive_group()
    interactive_group.add_argument(
        "--interactive",
        action="store_true",
        help="Prompt for the standard txt2img settings used by run.bat.",
    )
    interactive_group.add_argument(
        "--interactive-hires",
        action="store_true",
        help="Prompt for standard txt2img settings plus neural .pth hires controls.",
    )
    save_group = run_parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save",
        dest="save_images",
        action="store_true",
        help="Persist generated images and metadata sidecars.",
    )
    save_group.add_argument(
        "--no-save",
        dest="save_images",
        action="store_false",
        help="Keep generated images in memory only.",
    )
    run_parser.set_defaults(save_images=None)
    run_parser.add_argument("--no-txt", action="store_true")
    run_parser.add_argument("--no-json", action="store_true")
    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help=(
            "Print full structured runtime, seed, memory, preview, prompt, "
            "quality, and model diagnostic JSON to the console. Normal CLI "
            "runs keep these payloads in saved artifacts without flooding stdout."
        ),
    )

    run_parser.add_argument(
        "--console-memory",
        choices=("off", "compact", "json"),
        default=None,
        help=(
            "Choose memory output for human CLI runs: compact adds used/requested/"
            "available VRAM to the sampling line, off hides it, and json restores "
            "the machine-readable MEMORY_STATUS_JSON stream."
        ),
    )

    run_parser.add_argument(
        "--diagnostics",
        dest="diagnostic_verbosity",
        choices=("quiet", "normal", "verbose", "trace"),
        help="Set structured diagnostic console verbosity for this run.",
    )
    run_parser.add_argument(
        "--diagnostics-dir",
        help="Override the diagnostic artifact root for this run.",
    )
    run_parser.add_argument(
        "--export-diagnostic-events",
        action="store_true",
        help="Write successful-run events and timing summary under artifacts.",
    )
    run_parser.add_argument(
        "--tensor-summaries",
        action="store_true",
        help="Record tensor shape, dtype, device, and element count.",
    )
    run_parser.add_argument(
        "--tensor-statistics",
        action="store_true",
        help="Also record finite min/max/mean/std/norm statistics.",
    )
    run_parser.add_argument(
        "--sampler-trace",
        action="store_true",
        help="Enable the optional per-step sampler trace under artifacts.",
    )
    run_parser.add_argument("--no-failure-bundle", action="store_true")
    run_parser.add_argument("--no-progress", action="store_true")

    config_parser = subparsers.add_parser(
        "config",
        help="Inspect or validate the canonical project configuration",
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command", required=True)

    show_parser = config_subparsers.add_parser("show", help="Print effective configuration")
    _add_context_arguments(show_parser)
    show_parser.add_argument("--json", action="store_true", dest="as_json")

    validate_parser = config_subparsers.add_parser(
        "validate",
        help="Validate configured model, tokenizer, output, and support paths",
    )
    _add_context_arguments(validate_parser)
    validate_parser.add_argument("--model", dest="model_path")
    validate_parser.add_argument("--output-dir")
    validate_parser.add_argument("--no-output", action="store_true")
    validate_parser.add_argument("--json", action="store_true", dest="as_json")

    return parser


def build_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    diagnostic_overrides: dict[str, Any] = {}
    if getattr(args, "diagnostic_verbosity", None):
        diagnostic_overrides["verbosity"] = args.diagnostic_verbosity
    if getattr(args, "diagnostics_dir", None):
        diagnostic_overrides["artifacts_root"] = args.diagnostics_dir
    if getattr(args, "export_diagnostic_events", False):
        diagnostic_overrides["export_events"] = True
    if getattr(args, "tensor_summaries", False) or getattr(args, "tensor_statistics", False):
        diagnostic_overrides["tensor_summaries"] = True
    if getattr(args, "tensor_statistics", False):
        diagnostic_overrides["tensor_statistics"] = True
    if getattr(args, "sampler_trace", False):
        diagnostic_overrides["sampler_trace"] = {"enabled": True}
    if getattr(args, "no_failure_bundle", False):
        diagnostic_overrides["failure_bundles"] = False
    if getattr(args, "no_progress", False):
        diagnostic_overrides["progress"] = False

    overrides = {
        "positive_prompt": args.positive_prompt,
        "negative_prompt": args.negative_prompt,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "width": args.width,
        "height": args.height,
        "batch_size": args.batch_size,
        "batch_count": args.batch_count,
        "unlimited": args.unlimited,
        "model_path": args.model_path,
        "sampler_name": args.sampler_name,
        "scheduler_name": args.scheduler_name,
        "output_dir": args.output_dir,
        "output_prefix": args.output_prefix,
        "scheduler_kwargs": args.scheduler_kwargs,
        "sampler_kwargs": args.sampler_kwargs,
        "prompt_parser_name": args.prompt_parser_name,
        "prompt_parser_kwargs": args.prompt_parser_kwargs,
        "prompt_shortcut_profile_name": args.prompt_shortcut_profile_name,
        "prompt_parser_preset_name": args.prompt_parser_preset_name,
        "prompt_shadow_compare": args.prompt_shadow_compare,
        "hires_prompt_parser_mode": args.hires_prompt_parser_mode,
        "hires_prompt_parser_name": args.hires_prompt_parser_name,
        "hires_prompt_parser_kwargs": args.hires_prompt_parser_kwargs,
        "hires_shortcut_profile_mode": args.hires_shortcut_profile_mode,
        "hires_shortcut_profile_name": args.hires_shortcut_profile_name,
        "hires_positive_prompt": args.hires_positive_prompt,
        "hires_negative_prompt": args.hires_negative_prompt,
        "hires_size_mode": args.hires_size_mode,
        "hires_scale": args.hires_scale,
        "hires_width": args.hires_width,
        "hires_height": args.hires_height,
        "hires_enabled": args.hires_enabled,
        "hires_steps": args.hires_steps,
        "hires_denoising_strength": args.hires_denoising_strength,
        "hires_step_policy": args.hires_step_policy,
        "hires_sampler_name": args.hires_sampler_name,
        "hires_scheduler_name": args.hires_scheduler_name,
        "hires_cfg_scale": args.hires_cfg_scale,
        "hires_cfg_rescale": args.hires_cfg_rescale,
        "hires_strategy": args.hires_strategy,
        "hires_upscaler": args.hires_upscaler,
        "hires_tile_size": args.hires_tile_size,
        "hires_tile_overlap": args.hires_tile_overlap,
        "hires_tile_batch_size": args.hires_tile_batch_size,
        "hires_exact_resize_filter": args.hires_exact_resize_filter,
        "hires_save_upscaled_pre_denoise": args.hires_save_upscaled_pre_denoise,
        "hires_save_vae_roundtrip": args.hires_save_vae_roundtrip,
        "hires_save_lowres": args.hires_save_lowres,
        "parser_kwargs": args.parser_kwargs,
        "diagnostics": diagnostic_overrides or None,
        "save_images": args.save_images,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def _merge_interactive_overrides(
    base: dict[str, Any],
    interactive: dict[str, Any],
) -> dict[str, Any]:
    """Merge interactive values while preserving meaningful blank prompts.

    Most blank interactive fields mean "keep the configured default". A blank base
    negative prompt intentionally disables negative conditioning. Blank hires prompt
    values intentionally inherit the corresponding base prompt during preflight, so
    they must also replace any stale configured hires override.
    """
    explicit_blank_keys = {
        "negative_prompt",
        "hires_positive_prompt",
        "hires_negative_prompt",
    }
    merged = dict(base or {})
    for key, value in dict(interactive or {}).items():
        if value is None:
            continue
        if value == "" and key not in explicit_blank_keys:
            continue
        merged[key] = value
    return merged


def _build_prompt_adapter(*, request=None, extras=None, state=None):
    from modules.adapters.prompt_conditioning_adapter import PromptConditioningAdapter

    return PromptConditioningAdapter()




def _console_verbose_enabled(args: argparse.Namespace) -> bool:
    """Return whether full structured JSON diagnostics should be printed.

    ``--verbose`` is the user-facing switch. Existing ``--diagnostics verbose``
    and ``--diagnostics trace`` invocations also preserve their historical
    detailed console behavior for validation and troubleshooting tools.
    """

    diagnostic_verbosity = str(
        getattr(args, "diagnostic_verbosity", "") or ""
    ).strip().lower()
    return bool(getattr(args, "verbose", False)) or diagnostic_verbosity in {
        "verbose",
        "trace",
    }


def _emit_structured_console_json(
    prefix: str,
    payload: dict[str, Any],
    *,
    enabled: bool,
) -> None:
    if not enabled:
        return
    print(
        prefix + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        flush=True,
    )


def _emit_runtime_diagnostic(*, context, args) -> None:
    if not _console_verbose_enabled(args):
        return
    try:
        import torch  # type: ignore
        torch_version = getattr(torch, "__version__", "unknown")
        cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover
        torch_version = f"unavailable: {type(exc).__name__}: {exc}"
        cuda_available = False

    env_keys = [
        "VIRTUAL_ENV",
        "MSLK_FMHA_POLICY",
        "MSLK_FMHA_DEBUG",
        "MSLK_FMHA_BLOCK_N",
        "MSLK_FMHA_BLOCK_M",
        "MSLK_FMHA_NUM_WARPS",
        "MSLK_FMHA_NUM_STAGES",
        "PYTORCH_CUDA_ALLOC_CONF",
    ]
    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "project_root": str(context.project_root),
        "config_path": str(args.config_path) if getattr(args, "config_path", None) else None,
        "manifest_path": str(args.manifest_path) if getattr(args, "manifest_path", None) else None,
        "infotext_path": str(args.infotext_path) if getattr(args, "infotext_path", None) else None,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "env": {key: os.environ.get(key, "") for key in env_keys},
        "runtime_startup_options": (
            args.runtime_startup_options.to_dict()
            if getattr(args, "runtime_startup_options", None) is not None
            else None
        ),
        "cuda_allocator": build_cuda_allocator_diagnostics(
            getattr(args, "runtime_startup_options", None)
        ),
    }
    _emit_structured_console_json(
        "RUNTIME_DIAGNOSTIC_JSON: ",
        payload,
        enabled=True,
    )

def _load_context(args: argparse.Namespace) -> ProjectContext:
    return ProjectContext.load(
        project_root=getattr(args, "project_root", None),
        config_path=getattr(args, "project_config", None),
    )


def _print_effective_config(context: ProjectContext) -> None:
    effective = context.effective_config()
    print("IMAGE_GEN canonical project configuration")
    print(f"Project root: {effective['project_root']}")
    print(f"Config file:  {effective['config_path']}")
    print("\nPaths:")
    for name, value in effective["paths"].items():
        print(f"  {name}: {value}")
    print("\nGeneration defaults:")
    for name, value in effective["generation"].items():
        print(f"  {name}: {value}")
    print("\nRuntime defaults:")
    for name, value in effective["defaults"].items():
        print(f"  {name}: {value}")


def _run_config_command(args: argparse.Namespace) -> int:
    context = _load_context(args)
    if args.config_command == "show":
        if args.as_json:
            print(context.effective_config_json())
        else:
            _print_effective_config(context)
        return 0

    if args.config_command == "validate":
        report = context.validate(
            for_generation=True,
            model_path=args.model_path,
            output_dir=args.output_dir,
            require_output=not args.no_output,
        )
        if args.as_json:
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print(report.format_text())
        return 0 if report.is_valid else 2

    raise ValueError(f"Unsupported config command: {args.config_command}")


def _normalize_payload_paths(payload: dict[str, Any], context: ProjectContext) -> dict[str, Any]:
    normalized = dict(payload)
    for key in ("model_path", "vae_path", "output_dir"):
        value = normalized.get(key)
        if value is not None and str(value).strip():
            normalized[key] = str(context.resolve_project_path(str(value)))
    diagnostics = normalized.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("artifacts_root"):
        diagnostics = dict(diagnostics)
        diagnostics["artifacts_root"] = str(
            context.resolve_project_path(str(diagnostics["artifacts_root"]))
        )
        normalized["diagnostics"] = diagnostics
    lora_paths = normalized.get("lora_paths")
    if isinstance(lora_paths, list):
        normalized["lora_paths"] = [
            str(context.resolve_project_path(str(path))) for path in lora_paths
        ]
    return normalized


def _load_effective_generation_payload(
    *,
    args: argparse.Namespace,
    context: ProjectContext,
    cli_overrides: dict[str, Any],
) -> dict[str, Any]:
    """Merge a generation request before resolving its effective checkpoint.

    Model precedence is intentionally resolved after all sources are merged so a
    request-file model is not overwritten by defaults.model_path.  Explicit
    --model and interactive run.bat choices remain the highest-precedence input.
    """

    from modules.txt2img.model_selector import resolve_cli_model_path
    from modules.txt2img.request_loader import load_request_payload

    overrides = dict(cli_overrides or {})
    requested_override_model = overrides.get("model_path")
    interactive_model_selection = bool(
        getattr(args, "interactive", False)
        or getattr(args, "interactive_hires", False)
    )
    if interactive_model_selection or requested_override_model:
        resolved_override_model = resolve_cli_model_path(
            requested_override_model,
            interactive=interactive_model_selection,
            project_context=context,
        )
        if resolved_override_model:
            overrides["model_path"] = resolved_override_model

    payload = load_request_payload(
        config_path=getattr(args, "config_path", None),
        manifest_path=getattr(args, "manifest_path", None),
        infotext_path=getattr(args, "infotext_path", None),
        base_payload=context.generation_defaults(),
        cli_overrides=overrides,
    )

    effective_model_path = resolve_cli_model_path(
        str(payload.get("model_path") or ""),
        interactive=False,
        project_context=context,
    )
    if effective_model_path:
        payload["model_path"] = effective_model_path

    return _normalize_payload_paths(payload, context)


def _run_generation(args: argparse.Namespace) -> int:
    context = _load_context(args)
    runtime_startup_options = bootstrap_runtime_startup(
        args,
        settings=context.generation_defaults(),
    )

    # Runtime environment must be normalized before importing model, Torch, or
    # attention modules.  Later Phase 14K subphases populate the shared parser
    # hook with the concrete backend and memory switches.
    from modules.load_safetensors_model import LoadModel
    from image_gen.systems.registry import RuntimeRegistrySystem
    from modules.txt2img.cli_interactive import (
        build_hires_interactive_overrides,
        build_interactive_overrides,
        choose_from_registry,
    )
    from modules.txt2img.request_loader import payload_to_generation_request
    from modules.txt2img.txt2img_runner import Txt2ImgRunner

    _emit_runtime_diagnostic(context=context, args=args)
    cli_overrides = build_cli_overrides(args)

    # Build the descriptor registry once and share it with selection and runtime.
    registry_system = RuntimeRegistrySystem(project_context=context)
    live_sampler_map = registry_system.legacy_map("sampler")
    live_scheduler_map = registry_system.legacy_map("scheduler")

    if args.interactive or args.interactive_hires:
        interactive_overrides = (
            build_hires_interactive_overrides(context)
            if args.interactive_hires
            else build_interactive_overrides()
        )
        sampler_entry = choose_from_registry("Sampler", live_sampler_map)
        scheduler_entry = choose_from_registry("Scheduler", live_scheduler_map)
        interactive_overrides["sampler_name"] = (
            sampler_entry.get("name") or sampler_entry.get("label")
        )
        interactive_overrides["scheduler_name"] = (
            scheduler_entry.get("name") or scheduler_entry.get("label")
        )
        cli_overrides = _merge_interactive_overrides(
            cli_overrides,
            interactive_overrides,
        )

    payload = _load_effective_generation_payload(
        args=args,
        context=context,
        cli_overrides=cli_overrides,
    )
    if args.interactive or args.interactive_hires:
        existing_loras = payload.get("loras")
        if not isinstance(existing_loras, list) or not existing_loras:
            from modules.txt2img.lora_selector import choose_cli_loras_for_model

            selected_loras = choose_cli_loras_for_model(
                str(payload.get("model_path") or ""),
                project_context=context,
            )
            payload["loras"] = [dict(item) for item in selected_loras]
            payload["lora_paths"] = [
                str(item.get("path") or "")
                for item in selected_loras
                if str(item.get("path") or "").strip()
            ]
        else:
            print(f"Using {len(existing_loras)} LoRA selection(s) from the merged request.")
    from image_gen.runtime.scheduler_settings import normalize_scheduler_payload

    # Normalize the selected scheduler before emitting the effective request so
    # CLI/config and WebUI jobs persist the same canonical values.
    payload, _scheduler_resolution = normalize_scheduler_payload(payload)
    if getattr(args, "manifest_path", None):
        for warning in runtime_replay_warnings(
            payload.get("runtime_startup_options"),
            runtime_startup_options,
        ):
            print(f"WARNING: {warning}")
    if args.effective_request_out:
        effective_request_path = Path(args.effective_request_out).expanduser()
        if not effective_request_path.is_absolute():
            effective_request_path = context.resolve_project_path(effective_request_path)
        effective_request_path.parent.mkdir(parents=True, exist_ok=True)
        effective_request_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        print(f"EFFECTIVE_REQUEST_JSON: {effective_request_path.resolve()}")
    request, payload_extras = payload_to_generation_request(payload)
    console_verbose = _console_verbose_enabled(args)
    console_memory_mode = str(
        getattr(args, "console_memory", None)
        or os.environ.get("IMAGE_GEN_CONSOLE_MEMORY", "")
        or "compact"
    ).strip().lower()
    if console_memory_mode not in {"off", "compact", "json"}:
        console_memory_mode = "compact"
    if console_verbose:
        console_memory_mode = "json"
    payload_extras["_console_verbose"] = console_verbose
    payload_extras["_console_memory_mode"] = console_memory_mode
    payload_extras.update(runtime_request_settings(runtime_startup_options))
    payload_extras["runtime_startup_status"] = build_runtime_startup_status(
        runtime_startup_options,
        {"mslk_fmha": runtime_startup_options.mslk_fmha.to_dict()},
    )

    selected_model_request = str(payload_extras.get("model_path") or "")
    print(f"Selected model request: {selected_model_request}")

    if request.save_images:
        output_path = Path(request.output_dir or context.txt2img_output_root).expanduser()
        if not output_path.is_absolute():
            output_path = context.resolve_project_path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        request.output_dir = str(output_path.resolve())

    context.require_generation_ready(
        model_path=payload_extras.get("model_path"),
        output_dir=request.output_dir,
        require_output=bool(request.save_images),
    )

    model_loader = LoadModel(project_context=context)
    runner = Txt2ImgRunner(
        prompt_adapter_factory=_build_prompt_adapter,
        model_loader=model_loader,
        project_context=context,
        registry_system=registry_system,
    )

    extras = {
        "live_sampler_map": live_sampler_map,
        "live_scheduler_map": live_scheduler_map,
    }
    extras.update(payload_extras)

    from modules.txt2img.seed_utils import iter_batch_base_seeds, offset_seed

    batch_count = int(extras.pop("batch_count", 1) or 1)
    unlimited = bool(extras.pop("unlimited", False))
    if batch_count < 1:
        raise ValueError("batch_count must be at least 1")
    if request.batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    requested_seed = request.seed
    print(f"Requested seed: {requested_seed}")
    base_seed_iterator = iter_batch_base_seeds(
        requested_seed,
        batch_size=request.batch_size,
    )
    completed_batches = 0
    total_saved = 0

    try:
        while unlimited or completed_batches < batch_count:
            # Each batch-count iteration must start from a clean request-scoped
            # runtime state while still retaining any warmed model components.
            # Without this reset, mutable state from the first generated image can
            # leak into the next batch and prevent later batches from starting.
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
                prompt_semantic_pass_records=dict(request.prompt_semantic_pass_records or {}),
                prompt_semantic_recorded=dict(request.prompt_semantic_recorded or {}),
                prompt_semantic_replay_mode=str(request.prompt_semantic_replay_mode or "reconstruct"),
                region_pass_records=dict(request.region_pass_records or {}),
                region_recorded=dict(request.region_recorded or {}),
                region_replay_mode=str(request.region_replay_mode or "reconstruct"),
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
            mode_label = "unlimited" if unlimited else f"{batch_number}/{batch_count}"
            resolved_image_seeds = [
                offset_seed(int(batch_request.seed), index)
                for index in range(int(batch_request.batch_size))
            ]
            _emit_structured_console_json(
                "GENERATION_SEED_JSON: ",
                {
                    "batch_number": batch_number,
                    "base_seed": int(batch_request.seed),
                    "image_seeds": resolved_image_seeds,
                },
                enabled=console_verbose,
            )
            print(f"\n=== Starting batch {mode_label} ===")

            batch_extras = dict(extras)
            batch_extras.update(
                {
                    "batch_number": batch_number,
                    "batch_count": batch_count,
                    "unlimited": unlimited,
                    "generation_mode": "unlimited" if unlimited else "batch_count",
                }
            )
            result = runner.run_request(
                batch_request,
                batch_extras,
                save_txt=not args.no_txt,
                save_json=not args.no_json,
            )

            live_preview_summary = dict(
                result.pipeline_result.metadata.get("live_preview") or {}
            )
            _emit_structured_console_json(
                "LIVE_PREVIEW_SUMMARY_JSON: ",
                live_preview_summary,
                enabled=console_verbose,
            )

            output_quality_diagnostic = dict(
                result.pipeline_result.metadata.get("output_quality") or {}
            )
            _emit_structured_console_json(
                "OUTPUT_QUALITY_DIAGNOSTIC_JSON: ",
                output_quality_diagnostic,
                enabled=console_verbose,
            )

            prompt_parser_diagnostic = dict(
                result.pipeline_result.metadata.get("prompt_parser") or {}
            )
            _emit_structured_console_json(
                "PROMPT_PARSER_DIAGNOSTIC_JSON: ",
                prompt_parser_diagnostic,
                enabled=console_verbose,
            )

            model_diagnostic = dict(
                result.request_extras.get("model_provenance") or {}
            )
            model_diagnostic.setdefault("requested_path", selected_model_request)
            _emit_structured_console_json(
                "MODEL_DIAGNOSTIC_JSON: ",
                model_diagnostic,
                enabled=console_verbose,
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
                        f"requested batch base {expected_base_seed}, "
                        f"runtime used {result.request.seed}."
                    )
                expected_image_seeds = [
                    offset_seed(expected_base_seed, index)
                    for index in range(int(request.batch_size))
                ]
                if list(result.request.resolved_seeds or []) != expected_image_seeds:
                    raise RuntimeError(
                        "Fixed per-image seed sequence changed unexpectedly: "
                        f"expected {expected_image_seeds}, "
                        f"runtime used {result.request.resolved_seeds}."
                    )

            completed_batches += 1
            total_saved += len(saved_paths)
            print("=== txt2img batch complete ===")
            print(f"Run ID: {result.run_id}")
            print(f"Prompt: {result.request.positive_prompt}")
            print(f"Batch base seed: {result.request.seed}")
            print(f"Image seeds: {result.request.resolved_seeds}")
            print(f"Sampler: {result.request.sampler_name}")
            print(f"Scheduler: {result.request.scheduler_name}")
            if bool(getattr(result.request, "hires_enabled", False)):
                hires_info = dict(
                    result.pipeline_result.metadata.get("hires_fix") or {}
                )
                hires_dimensions = dict(hires_info.get("dimensions") or {})
                print("Hires fix: enabled")
                print(
                    "Hires output: "
                    f"{hires_dimensions.get('effective_width', '?')}x"
                    f"{hires_dimensions.get('effective_height', '?')}"
                )
                print(
                    "Hires pass: "
                    f"{hires_info.get('effective_second_pass_steps', '?')} effective step(s), "
                    f"strength {hires_info.get('denoising_strength', '?')}, "
                    f"{hires_info.get('upscaler', '?')}"
                )
            if model_diagnostic:
                print(f"Requested model: {model_diagnostic.get('requested_path', '')}")
                print(f"Resolved model:  {model_diagnostic.get('resolved_path', '')}")
                print(f"Loaded model:    {model_diagnostic.get('loaded_path', '')}")
                if model_diagnostic.get("sha256"):
                    print(f"Model SHA-256:   {model_diagnostic['sha256']}")
            print(f"Generation time (sec): {result.generation_time_sec:.3f}")
            if saved_paths:
                print(f"Saved images: {len(saved_paths)}")
                print(f"Output directory: {Path(saved_paths[0]).parent}")
                for record in result.saved_records:
                    seed_label = "unknown" if record.seed is None else str(record.seed)
                    print(f"  Image [seed {seed_label}]: {record.image_path}")
                    if record.txt_path:
                        print(f"  TXT:   {record.txt_path}")
                    if record.json_path:
                        print(f"  JSON:  {record.json_path}")
            else:
                print("Saving disabled; generation remained in memory only.")
    except KeyboardInterrupt:
        if not unlimited:
            raise
        print(
            f"\nUnlimited generation stopped after {completed_batches} "
            f"completed batch(es) and {total_saved} saved image(s)."
        )
        return 0

    print(
        f"\nCompleted {completed_batches} batch(es), "
        f"{completed_batches * request.batch_size} generated image(s), "
        f"and {total_saved} saved image(s)."
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    parser_argv = argv_for_primary_parser(raw_argv)
    args = parser.parse_args(parser_argv)
    args._runtime_argv = raw_argv

    try:
        if args.command == "config":
            return _run_config_command(args)
        if args.command == "run":
            return _run_generation(args)
        parser.error(f"Unsupported command: {args.command}")
    except Exception as exc:
        from image_gen.systems.diagnostics import PipelineStageError

        if isinstance(exc, PipelineStageError):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        if isinstance(exc, (ProjectConfigurationError, ProjectValidationError)):
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        raise
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
