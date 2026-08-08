from .compatibility_matrix import run_xformers_layout_matrix
from .layout_capture import (
    AttentionLayoutCapture,
    attention_layout_capture_report,
    install_attention_layout_capture,
    resolve_capture_mode,
)
from .model_signature import (
    attention_module_record,
    build_model_attention_signature,
    is_attention_module,
    iter_attention_modules,
)
from .numerical_validation import compare_tensors
from .provider_registry import build_provider_registry, load_environment_contract
from .production_dispatch import (
    build_verified_xformers_processor,
    capability_production_dispatch_decision,
    get_execution_evidence,
    hardware_qualification_mode,
    prepare_mslk_process_environment,
    production_dispatch_decision,
    require_production_dispatch,
    require_verified_production_dispatch,
    reset_execution_evidence_for_testing,
    verified_layout_executor,
    verified_production_dispatch_decision,
)
from .reports import module_device_dtype, stable_hash
from .release_reproducibility import (
    load_release_manifest,
    require_release_compatible_stack,
    summarize_release_report,
    verify_release_stack,
)
from .runtime_selection import AUTOMATIC_BACKEND_ORDER, EXPLICIT_BACKENDS, backend_candidates

__all__ = [
    "AUTOMATIC_BACKEND_ORDER",
    "EXPLICIT_BACKENDS",
    "AttentionLayoutCapture",
    "attention_layout_capture_report",
    "attention_module_record",
    "backend_candidates",
    "build_model_attention_signature",
    "build_verified_xformers_processor",
    "capability_production_dispatch_decision",
    "build_provider_registry",
    "compare_tensors",
    "install_attention_layout_capture",
    "is_attention_module",
    "iter_attention_modules",
    "load_environment_contract",
    "module_device_dtype",
    "resolve_capture_mode",
    "run_xformers_layout_matrix",
    "stable_hash",
    "load_release_manifest",
    "require_release_compatible_stack",
    "summarize_release_report",
    "verify_release_stack",
    "get_execution_evidence",
    "hardware_qualification_mode",
    "prepare_mslk_process_environment",
    "production_dispatch_decision",
    "require_production_dispatch",
    "require_verified_production_dispatch",
    "reset_execution_evidence_for_testing",
    "verified_layout_executor",
    "verified_production_dispatch_decision",
]
