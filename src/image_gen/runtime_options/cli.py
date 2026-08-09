from __future__ import annotations

from image_gen.program_metadata import PRODUCT_NAME

import argparse
import ctypes
import json
import os
import shlex
from ctypes import wintypes
from typing import Any, Mapping, MutableMapping, Sequence

from .contracts import RuntimeProfileSelection, RuntimeStartupOptions
from .cuda_allocator import (
    CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE,
    canonicalize_cuda_allocator_conf,
)
from .normalization import (
    apply_runtime_startup_environment,
    resolve_runtime_startup_options,
)
from .profiles import load_runtime_memory_profile


def _parse_optional_positive_int(value: str) -> str:
    """Preserve an explicitly blank MSLK value or validate a positive integer."""

    text = str(value)
    if text == "":
        return ""
    try:
        parsed = int(text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a positive integer or an explicitly empty value, received {value!r}."
        ) from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "Value must be greater than zero when supplied."
        )
    return str(parsed)


def _parse_non_negative_int(value: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            f"Expected a non-negative integer, received {value!r}."
        ) from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("Value must be non-negative.")
    return parsed


def _parse_cuda_alloc_conf(value: str) -> str:
    try:
        return canonicalize_cuda_allocator_conf(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _parse_mslk_policy(value: str) -> str:
    text = str(value)
    if text == "":
        return ""
    normalized = text.strip().lower()
    allowed = {"default", "auto", "blackwell_safe", "env", "off", "benchmark"}
    if normalized not in allowed:
        raise argparse.ArgumentTypeError(
            "MSLK FMHA policy must be blank or one of: "
            + ", ".join(sorted(allowed))
            + "."
        )
    return normalized


def _parse_mslk_debug(value: str) -> str:
    text = str(value)
    if text == "":
        return ""
    normalized = text.strip().lower()
    allowed = {"0", "1", "false", "true", "off", "on"}
    if normalized not in allowed:
        raise argparse.ArgumentTypeError(
            "MSLK FMHA debug must be blank or one of: 0, 1, false, true, off, on."
        )
    return normalized


RUNTIME_COMMAND_LINE_HELP_EPILOG = r"""
Runtime startup examples:
  Balanced WebUI:
    set "COMMANDLINE_ARGS=--xformers --medvram"

  Safe hires transition:
    set "COMMANDLINE_ARGS=--xformers --medvram --hires-memory-saver --no-preview-during-hires --pre-hires-cleanup"

  MSLK policy test (fresh process required):
    set "COMMANDLINE_ARGS=--xformers --medvram --mslk-fmha-policy env --mslk-fmha-block-n 32 --mslk-fmha-num-warps 2"

Built-in runtime profiles:
  auto, balanced, low-memory, maximum-memory-savings

References:
  docs/reference/RUNTIME_MEMORY_COMMAND_LINE.md
  docs/reference/RUNTIME_MEMORY_PROFILE_SCHEMA.md
  docs/reference/runtime_memory_profile.schema.json
""".strip()


def runtime_command_line_help_epilog() -> str:
    """Return the shared human-readable Phase 14K command help footer."""

    return RUNTIME_COMMAND_LINE_HELP_EPILOG


def add_runtime_startup_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared Phase 14K process-start options to one argument parser.

    Every CLI, WebUI, and resident-worker entry point calls this function so the
    accepted arguments and help text stay identical.
    """

    parser.add_argument(
        "--runtime-profile",
        dest="runtime_profile_selector",
        metavar="PROFILE_ID_OR_PATH",
        help=(
            "Load a Phase 14K runtime profile by built-in ID or JSON path. "
            "Built-ins: auto, balanced, low-memory, maximum-memory-savings. "
            "Environment and individual command-line options override profile values."
        ),
    )

    attention_group = parser.add_mutually_exclusive_group()
    attention_group.add_argument(
        "--xformers",
        dest="attention_backend",
        action="store_const",
        const="xformers",
        help=(
            "Use xformers memory-efficient attention. An explicit request fails "
            "before sampling if xformers cannot be activated and verified."
        ),
    )
    attention_group.add_argument(
        "--sdpa",
        dest="attention_backend",
        action="store_const",
        const="sdpa",
        help=(
            "Use PyTorch scaled-dot-product attention through Diffusers "
            "AttnProcessor2_0."
        ),
    )
    attention_group.add_argument(
        "--eager-attention",
        dest="attention_backend",
        action="store_const",
        const="eager",
        help="Use the compatibility eager attention processor.",
    )
    attention_group.add_argument(
        "--attention-backend",
        dest="attention_backend",
        choices=("auto", "default", "eager", "sdpa", "xformers"),
        help=(
            "Select the UNet attention backend. auto/default tries verified "
            "xformers, then SDPA, then eager attention."
        ),
    )

    parser.add_argument(
        "--attention-slicing",
        dest="attention_slicing",
        choices=("off", "auto", "max"),
        help=(
            "Configure Diffusers attention slicing. auto uses the framework-recommended "
            "slice and max uses the smallest supported slice. Slicing replaces incompatible "
            "xformers/SDPA processors and is intended as a memory fallback."
        ),
    )

    memory_selector = parser.add_mutually_exclusive_group()
    memory_selector.add_argument(
        "--highvram",
        dest="memory_policy",
        action="store_const",
        const="high_vram",
        help=(
            "Prefer speed and retain reusable components when measured VRAM "
            "headroom permits."
        ),
    )
    memory_selector.add_argument(
        "--medvram",
        dest="memory_policy",
        action="store_const",
        const="balanced",
        help="Use the balanced whole-component memory policy.",
    )
    memory_selector.add_argument(
        "--lowvram",
        dest="memory_policy",
        action="store_const",
        const="low_vram",
        help="Use sequential whole-component residency and strong stage cleanup.",
    )
    memory_selector.add_argument(
        "--memory-policy",
        dest="memory_policy",
        choices=("auto", "high_vram", "balanced", "low_vram", "cpu_fallback"),
        help="Select the normalized component-memory policy.",
    )

    memory_group = parser.add_argument_group("Component memory policy")
    memory_group.add_argument(
        "--vram-safety-margin-mb",
        dest="vram_safety_margin_mb",
        type=_parse_non_negative_int,
        metavar="INT",
        help=(
            "Reserve this many MiB when planning optional GPU residency "
            "(must be non-negative)."
        ),
    )
    for positive_option, negative_option, destination, label in (
        (
            "--retain-unet-between-jobs",
            "--no-retain-unet-between-jobs",
            "retain_unet_between_jobs",
            "UNet",
        ),
        (
            "--retain-vae-between-jobs",
            "--no-retain-vae-between-jobs",
            "retain_vae_between_jobs",
            "VAE",
        ),
        (
            "--retain-text-encoder-between-jobs",
            "--no-retain-text-encoder-between-jobs",
            "retain_text_encoder_between_jobs",
            "text encoder",
        ),
    ):
        retain_group = memory_group.add_mutually_exclusive_group()
        retain_group.add_argument(
            positive_option,
            dest=destination,
            action="store_true",
            default=None,
            help=(
                f"Retain the {label} on the configured retention device "
                "between jobs."
            ),
        )
        retain_group.add_argument(
            negative_option,
            dest=destination,
            action="store_false",
            default=None,
            help=f"Offload the {label} after the job when it is no longer required.",
        )

    vae_tiling_group = memory_group.add_mutually_exclusive_group()
    vae_tiling_group.add_argument(
        "--vae-tiling",
        dest="vae_tiling",
        action="store_true",
        default=None,
        help=(
            f"Use {PRODUCT_NAME}-owned overlap-add tiled VAE encode/decode. This does not "
            "call nonexistent Diffusers tiling methods on the custom LDM VAE."
        ),
    )
    vae_tiling_group.add_argument(
        "--no-vae-tiling",
        dest="vae_tiling",
        action="store_false",
        default=None,
        help=f"Disable {PRODUCT_NAME}-owned tiled VAE encode/decode.",
    )
    vae_slicing_group = memory_group.add_mutually_exclusive_group()
    vae_slicing_group.add_argument(
        "--vae-slicing",
        dest="vae_slicing",
        action="store_true",
        default=None,
        help="Process VAE batches one image at a time to reduce peak memory.",
    )
    vae_slicing_group.add_argument(
        "--no-vae-slicing",
        dest="vae_slicing",
        action="store_false",
        default=None,
        help="Process the full VAE batch together.",
    )
    memory_group.add_argument(
        "--vae-device",
        dest="vae_device",
        choices=("auto", "cuda", "cpu"),
        help=(
            "Select the VAE execution device. auto follows stage residency; cuda or cpu "
            "forces VAE stages to the selected device."
        ),
    )

    allocator_group = parser.add_argument_group("CUDA allocator controls")
    allocator_group.add_argument(
        "--cuda-alloc-conf",
        dest="cuda_alloc_conf",
        type=_parse_cuda_alloc_conf,
        metavar="VALUE",
        help=(
            "Set PYTORCH_CUDA_ALLOC_CONF before Torch initializes CUDA. VALUE uses "
            "PyTorch's comma-separated name:value syntax. Allocator tuning can reduce "
            "fragmentation but cannot satisfy one allocation larger than available VRAM."
        ),
    )
    allocator_group.add_argument(
        "--cuda-expandable-segments",
        dest="cuda_expandable_segments",
        action="store_true",
        default=None,
        help=(
            "Add or replace expandable_segments:True in PYTORCH_CUDA_ALLOC_CONF "
            "before CUDA initialization."
        ),
    )

    preview_group = parser.add_argument_group("Preview memory policy")
    preview_selector = preview_group.add_mutually_exclusive_group()
    preview_selector.add_argument(
        "--preview-policy",
        dest="preview_policy",
        choices=("normal", "suspend_on_pressure", "disable_during_hires", "disabled"),
        help=(
            "Control image-preview decoding for this process. normal preserves the configured "
            "preview behavior; suspend_on_pressure permits automatic VAE preview suspension "
            "when the active stage would violate the VRAM safety margin; "
            "disable_during_hires keeps base preview but disables image decoding for the hires "
            "second pass; disabled emits no step image decodes. CFG telemetry may continue."
        ),
    )
    preview_selector.add_argument(
        "--no-preview-during-hires",
        dest="preview_policy",
        action="store_const",
        const="disable_during_hires",
        help=(
            "Allow base-pass preview, then suspend image preview decoding and release queued "
            "preview work before the hires pass."
        ),
    )
    preview_selector.add_argument(
        "--disable-live-preview",
        dest="preview_policy",
        action="store_const",
        const="disabled",
        help=(
            "Disable step image decoding for the job while allowing non-image progress and CFG "
            "telemetry to continue."
        ),
    )

    hires_group = parser.add_argument_group("Hires memory behavior")
    hires_profile_group = hires_group.add_mutually_exclusive_group()
    hires_profile_group.add_argument(
        "--hires-memory-profile",
        dest="hires_memory_profile",
        choices=("inherit", "balanced", "low_vram", "maximum"),
        help=(
            "Select the component-residency profile used only by the hires "
            "transition and second denoising pass."
        ),
    )
    hires_profile_group.add_argument(
        "--hires-memory-saver",
        dest="hires_memory_profile",
        action="store_const",
        const="low_vram",
        help="Alias for --hires-memory-profile low_vram.",
    )
    hires_profile_group.add_argument(
        "--hires-force-lowvram",
        dest="hires_memory_profile",
        action="store_const",
        const="low_vram",
        help="Compatibility alias for --hires-memory-profile low_vram.",
    )
    pre_hires_cleanup_group = hires_group.add_mutually_exclusive_group()
    pre_hires_cleanup_group.add_argument(
        "--pre-hires-cleanup",
        dest="pre_hires_cleanup",
        action="store_true",
        default=None,
        help=(
            "Run an explicit reference release, component offload, garbage "
            "collection, and CUDA cache cleanup boundary before hires allocation."
        ),
    )
    pre_hires_cleanup_group.add_argument(
        "--no-pre-hires-cleanup",
        dest="pre_hires_cleanup",
        action="store_false",
        default=None,
        help=(
            "Do not request the optional pre-hires cleanup boundary. Low-VRAM "
            "and maximum hires profiles still enforce their required cleanup."
        ),
    )

    oom_group = parser.add_argument_group("Bounded CUDA OOM recovery")
    oom_toggle_group = oom_group.add_mutually_exclusive_group()
    oom_toggle_group.add_argument(
        "--retry-on-oom",
        dest="oom_retry_enabled",
        action="store_true",
        default=None,
        help=(
            "Enable bounded CUDA OOM recovery. If no explicit profile is selected, "
            "cleanup recovery is used. Sampling retries restart from an explicit "
            "saved boundary; partially mutated sampler state is never continued."
        ),
    )
    oom_toggle_group.add_argument(
        "--no-retry-on-oom",
        dest="oom_retry_enabled",
        action="store_false",
        default=None,
        help="Disable automatic CUDA OOM recovery retries.",
    )
    oom_group.add_argument(
        "--oom-retry-profile",
        dest="oom_retry_profile",
        choices=("cleanup", "low_vram", "maximum"),
        help=(
            "Select the bounded OOM fallback: cleanup retries after allocator cleanup; "
            "low_vram also suspends preview and forces low-VRAM residency; maximum also "
            "enables compatible attention slicing and VAE tiling/slicing. This selector "
            "may be combined with --retry-on-oom."
        ),
    )
    oom_group.add_argument(
        "--oom-retry-limit",
        dest="oom_retry_limit",
        type=_parse_non_negative_int,
        metavar="INT",
        help=(
            "Maximum automatic OOM retries across the whole job. The default is 1; "
            "each individual stage is still retried at most once. Use 0 to disable retries."
        ),
    )

    mslk_group = parser.add_argument_group(
        "MSLK/Triton process-start settings"
    )
    mslk_group.add_argument(
        "--mslk-fmha-policy",
        type=_parse_mslk_policy,
        metavar="VALUE",
        help=(
            "Select the immutable MSLK FMHA launch policy for this process. "
            "An explicitly empty value is preserved."
        ),
    )
    mslk_group.add_argument(
        "--mslk-fmha-debug",
        type=_parse_mslk_debug,
        metavar="VALUE",
        help=(
            "Enable or disable MSLK FMHA launch diagnostics for this process. "
            "An explicitly empty value is preserved."
        ),
    )
    for option, destination, description in (
        ("--mslk-fmha-block-n", "mslk_fmha_block_n", "BLOCK_N"),
        ("--mslk-fmha-block-m", "mslk_fmha_block_m", "BLOCK_M"),
        ("--mslk-fmha-num-warps", "mslk_fmha_num_warps", "num_warps"),
        ("--mslk-fmha-num-stages", "mslk_fmha_num_stages", "num_stages"),
    ):
        mslk_group.add_argument(
            option,
            dest=destination,
            type=_parse_optional_positive_int,
            metavar="INT",
            help=(
                f"Override MSLK Triton {description} at process start. "
                "Use an explicitly empty value to defer to the installed "
                "implementation."
            ),
        )
    mslk_group.add_argument(
        "--mslk-fmha-experimental-head-dims",
        dest="mslk_fmha_experimental_head_dims",
        help=(
            "Comma-separated process-start experimental Split-K dimensions. "
            "Production dispatch still requires a matching validation profile."
        ),
    )


def runtime_values_from_namespace(namespace: argparse.Namespace) -> dict[str, Any]:
    values: dict[str, Any] = {}
    runtime_profile_selector = getattr(namespace, "runtime_profile_selector", None)
    if runtime_profile_selector is not None:
        values["runtime_profile_selector"] = str(runtime_profile_selector)
    for key in (
        "attention_backend",
        "memory_policy",
        "vram_safety_margin_mb",
        "attention_slicing",
        "vae_tiling",
        "vae_slicing",
        "vae_device",
        "retain_unet_between_jobs",
        "retain_vae_between_jobs",
        "retain_text_encoder_between_jobs",
        "preview_policy",
        "hires_memory_profile",
        "pre_hires_cleanup",
        "oom_retry_limit",
    ):
        value = getattr(namespace, key, None)
        if value is not None:
            values[key] = value

    oom_retry_enabled = getattr(namespace, "oom_retry_enabled", None)
    oom_retry_profile = getattr(namespace, "oom_retry_profile", None)
    if oom_retry_enabled is False:
        values["oom_retry_profile"] = "disabled"
    elif oom_retry_profile is not None:
        values["oom_retry_profile"] = str(oom_retry_profile)
    elif oom_retry_enabled is True:
        values["oom_retry_profile"] = "cleanup"

    mslk_values: dict[str, Any] = {}
    for attribute, key in (
        ("mslk_fmha_policy", "policy"),
        ("mslk_fmha_debug", "debug"),
        ("mslk_fmha_block_n", "block_n"),
        ("mslk_fmha_block_m", "block_m"),
        ("mslk_fmha_num_warps", "num_warps"),
        ("mslk_fmha_num_stages", "num_stages"),
        ("mslk_fmha_experimental_head_dims", "experimental_head_dims"),
    ):
        value = getattr(namespace, attribute, None)
        if value is not None:
            mslk_values[key] = value
    if mslk_values:
        values["mslk_fmha"] = mslk_values

    allocator_options = getattr(namespace, "allocator_options", None)
    if isinstance(allocator_options, Mapping):
        values["allocator_options"] = dict(allocator_options)

    cuda_alloc_conf = getattr(namespace, "cuda_alloc_conf", None)
    if cuda_alloc_conf is not None:
        values.setdefault("allocator_options", {})[
            CUDA_ALLOCATOR_ENVIRONMENT_VARIABLE
        ] = str(cuda_alloc_conf)
    cuda_expandable_segments = getattr(namespace, "cuda_expandable_segments", None)
    if cuda_expandable_segments is not None:
        values["cuda_expandable_segments"] = bool(cuda_expandable_segments)
    return values


def build_runtime_startup_preparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    add_runtime_startup_arguments(parser)
    return parser


def split_commandline_args(value: str | None) -> list[str]:
    """Split the A1111-style ``COMMANDLINE_ARGS`` string without losing quotes."""

    command = str(value or "").strip()
    if not command:
        return []
    if os.name != "nt":
        # Used by tests and non-Windows development environments. The production
        # Windows path below follows CommandLineToArgvW semantics exactly.
        return shlex.split(command, posix=True)

    argc = ctypes.c_int()
    command_line_to_argv = ctypes.windll.shell32.CommandLineToArgvW
    command_line_to_argv.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_int)]
    command_line_to_argv.restype = ctypes.POINTER(wintypes.LPWSTR)
    argv_pointer = command_line_to_argv(command, ctypes.byref(argc))
    if not argv_pointer:
        raise OSError("CommandLineToArgvW could not parse COMMANDLINE_ARGS.")
    try:
        return [argv_pointer[index] for index in range(argc.value)]
    finally:
        ctypes.windll.kernel32.LocalFree(argv_pointer)


def _runtime_values_from_tokens(tokens: Sequence[str]) -> dict[str, Any]:
    parser = build_runtime_startup_preparser()
    namespace, _unknown = parser.parse_known_args(list(tokens))
    return runtime_values_from_namespace(namespace)


def _remove_first_token_sequence(
    values: Sequence[str], sequence: Sequence[str]
) -> list[str]:
    remaining = list(values)
    target = list(sequence)
    if not target or len(target) > len(remaining):
        return remaining
    for index in range(len(remaining) - len(target) + 1):
        if remaining[index : index + len(target)] == target:
            return remaining[:index] + remaining[index + len(target) :]
    return remaining


def argv_for_primary_parser(
    argv: Sequence[str],
    *,
    environment: Mapping[str, str] | None = None,
) -> list[str]:
    """Remove the launcher-injected COMMANDLINE_ARGS copy before full parsing.

    The shared runtime string is parsed independently with its own source label.
    Removing its duplicated argv sequence lets an explicit launcher argument
    override a shared mutually-exclusive option without argparse rejecting the
    two different sources as one conflicting source.
    """

    source_environment = environment if environment is not None else os.environ
    shared_tokens = split_commandline_args(source_environment.get("COMMANDLINE_ARGS"))
    return _remove_first_token_sequence(argv, shared_tokens)


def _runtime_option_definitions() -> dict[str, bool]:
    parser = build_runtime_startup_preparser()
    definitions: dict[str, bool] = {}
    for action in getattr(parser, "_actions", []):
        option_strings = list(getattr(action, "option_strings", []) or [])
        if not option_strings:
            continue
        takes_value = not isinstance(action, argparse._StoreConstAction)
        for option in option_strings:
            definitions[str(option)] = takes_value
    return definitions


def _unsupported_runtime_tokens(tokens: Sequence[str]) -> list[str]:
    definitions = _runtime_option_definitions()
    unsupported: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        option_token, separator, _inline_value = token.partition("=")
        takes_value = definitions.get(option_token)
        if takes_value is None:
            unsupported.append(option_token)
            index += 1
            continue
        if separator:
            index += 1
        else:
            index += 2 if takes_value and index + 1 < len(tokens) else 1
    return unsupported


def _validate_commandline_args_tokens(tokens: Sequence[str]) -> None:
    unsupported = _unsupported_runtime_tokens(tokens)
    if not unsupported:
        return
    joined = ", ".join(unsupported)
    raise ValueError(
        f"Unsupported startup option(s) in COMMANDLINE_ARGS: {joined}. "
        "These options are planned for later Phase 14K subphases and are not active yet."
    )


def _split_runtime_sources(
    argv: Sequence[str], environment: Mapping[str, str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    shared_tokens = split_commandline_args(environment.get("COMMANDLINE_ARGS"))
    _validate_commandline_args_tokens(shared_tokens)
    commandline_values = _runtime_values_from_tokens(shared_tokens)
    explicit_tokens = _remove_first_token_sequence(argv, shared_tokens)
    explicit_values = _runtime_values_from_tokens(explicit_tokens)
    return commandline_values, explicit_values


def _runtime_profile_from_sources(
    *,
    commandline_values: dict[str, Any],
    explicit_values: dict[str, Any],
    environment: Mapping[str, str],
) -> tuple[dict[str, Any] | None, RuntimeProfileSelection | None]:
    commandline_selector = commandline_values.pop("runtime_profile_selector", None)
    explicit_selector = explicit_values.pop("runtime_profile_selector", None)
    environment_selector = str(
        environment.get("IMAGE_GEN_RUNTIME_PROFILE", "") or ""
    ).strip()

    inherited_selection = str(
        environment.get("IMAGE_GEN_RUNTIME_PROFILE_SELECTION", "") or ""
    ).strip()
    inherited_startup_options = str(
        environment.get("IMAGE_GEN_RUNTIME_STARTUP_OPTIONS", "") or ""
    ).strip()

    def inherited_profile_selection() -> RuntimeProfileSelection | None:
        if not inherited_selection:
            return None
        try:
            selection_payload = json.loads(inherited_selection)
        except json.JSONDecodeError as exc:
            raise ValueError(
                "IMAGE_GEN_RUNTIME_PROFILE_SELECTION must contain a JSON object."
            ) from exc
        if not isinstance(selection_payload, dict):
            raise ValueError(
                "IMAGE_GEN_RUNTIME_PROFILE_SELECTION must contain a JSON object."
            )
        return RuntimeProfileSelection.from_mapping(selection_payload)

    selector: str | None = None
    selected_from = "default"
    if explicit_selector:
        selector = str(explicit_selector)
        selected_from = "cli"
    elif commandline_selector:
        selector = str(commandline_selector)
        selected_from = "commandline_args"
    elif inherited_selection and inherited_startup_options:
        return None, inherited_profile_selection()
    elif environment_selector:
        selector = environment_selector
        selected_from = "environment"
    elif inherited_selection:
        return None, inherited_profile_selection()

    if selector is None:
        return None, None
    profile = load_runtime_memory_profile(selector)
    return (
        profile.runtime_values(),
        profile.selection(selector=selector, selected_from=selected_from),
    )


def prebootstrap_runtime_startup(
    argv: Sequence[str] | None = None,
    *,
    environment: MutableMapping[str, str] | None = None,
) -> RuntimeStartupOptions:
    """Apply process-start options before importing Torch or attention modules."""

    target_environment = environment if environment is not None else os.environ
    raw_argv = list(argv or [])
    commandline_values, explicit_values = _split_runtime_sources(
        raw_argv, target_environment
    )
    runtime_profile, runtime_profile_selection = _runtime_profile_from_sources(
        commandline_values=commandline_values,
        explicit_values=explicit_values,
        environment=target_environment,
    )
    options = resolve_runtime_startup_options(
        explicit_cli=explicit_values,
        commandline_args=commandline_values,
        environment=target_environment,
        runtime_profile=runtime_profile,
        runtime_profile_selection=runtime_profile_selection,
    )
    apply_runtime_startup_environment(options, environment=target_environment)
    return options


def bootstrap_runtime_startup(
    namespace: argparse.Namespace,
    *,
    argv: Sequence[str] | None = None,
    commandline_values: Mapping[str, Any] | None = None,
    environment: MutableMapping[str, str] | None = None,
    saved_profile: Mapping[str, Any] | None = None,
    settings: Mapping[str, Any] | None = None,
    defaults: Mapping[str, Any] | None = None,
) -> RuntimeStartupOptions:
    target_environment = environment if environment is not None else os.environ
    raw_argv = list(
        argv
        if argv is not None
        else getattr(namespace, "_runtime_argv", []) or []
    )
    parsed_commandline, parsed_explicit = _split_runtime_sources(
        raw_argv, target_environment
    )
    if commandline_values is not None:
        parsed_commandline.update(dict(commandline_values))

    # Programmatic callers often construct a Namespace directly rather than
    # preserving argv. In that case, namespace values are explicit CLI values.
    namespace_values = runtime_values_from_namespace(namespace)
    if not raw_argv:
        parsed_explicit.update(namespace_values)
    elif namespace_values and not parsed_explicit:
        # Existing argument parsers may already have normalized an option before
        # a later Phase 14K subphase adds it to the lightweight pre-parser.
        parsed_explicit.update(namespace_values)

    runtime_profile, runtime_profile_selection = _runtime_profile_from_sources(
        commandline_values=parsed_commandline,
        explicit_values=parsed_explicit,
        environment=target_environment,
    )
    options = resolve_runtime_startup_options(
        explicit_cli=parsed_explicit,
        commandline_args=parsed_commandline,
        environment=target_environment,
        saved_profile=saved_profile,
        runtime_profile=runtime_profile,
        runtime_profile_selection=runtime_profile_selection,
        settings=settings,
        defaults=defaults,
    )
    apply_runtime_startup_environment(options, environment=target_environment)
    setattr(namespace, "runtime_startup_options", options)
    return options
