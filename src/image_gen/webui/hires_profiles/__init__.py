from .contracts import (
    HIRES_DEFAULT_ASSIGNMENTS_SCHEMA_VERSION,
    HIRES_PROFILE_SAVE_MANIFEST_VERSION,
    HIRES_PROFILE_SCHEMA_VERSION,
    HIRES_SETTING_DESCRIPTOR_VERSION,
    SUPPORTED_HIRES_DEFAULT_SCOPES,
    HiresDefaultAssignment,
    HiresProfile,
    HiresProfileSaveManifest,
    HiresProfileValidationError,
    HiresSettingDescriptor,
)
from .schema import HiresFieldPolicy, HiresProfileSchemaRegistry, humanize_identifier
from .service import HiresProfileService
from .builtins import (
    HIRES_AUTO_PROFILE_SERIES,
    HIRES_AUTO_PROFILE_VERSION,
    REQUIRED_HIRES_AUTO_FAMILIES,
    build_builtin_auto_profiles,
    builtin_auto_profile_id,
    builtin_auto_profile_name,
    builtin_auto_profile_values,
)
from .resolver import (
    AUTO_SELECT, BUILTIN_PIXEL_RESIZE_ID, BUILTIN_PIXEL_RESIZE_SHA256,
    HiresAutoResolver, HiresResolutionContext, HiresResolutionResult, builtin_resize_descriptor,
)

__all__ = [
    "HIRES_DEFAULT_ASSIGNMENTS_SCHEMA_VERSION",
    "HIRES_PROFILE_SAVE_MANIFEST_VERSION",
    "HIRES_PROFILE_SCHEMA_VERSION",
    "HIRES_SETTING_DESCRIPTOR_VERSION",
    "SUPPORTED_HIRES_DEFAULT_SCOPES",
    "HiresDefaultAssignment",
    "HiresFieldPolicy",
    "HiresProfile",
    "HiresProfileSaveManifest",
    "HiresProfileSchemaRegistry",
    "HiresProfileService",
    "HiresProfileValidationError",
    "HiresSettingDescriptor",
    "HIRES_AUTO_PROFILE_SERIES",
    "HIRES_AUTO_PROFILE_VERSION",
    "REQUIRED_HIRES_AUTO_FAMILIES",
    "build_builtin_auto_profiles",
    "builtin_auto_profile_id",
    "builtin_auto_profile_name",
    "builtin_auto_profile_values",
    "humanize_identifier",
    "AUTO_SELECT",
    "BUILTIN_PIXEL_RESIZE_ID",
    "BUILTIN_PIXEL_RESIZE_SHA256",
    "HiresAutoResolver",
    "HiresResolutionContext",
    "HiresResolutionResult",
    "builtin_resize_descriptor",
]
