"""Compatibility imports for the former LoRA-only CivitAI metadata service.

CivitAI enrichment is now implemented by :mod:`civitai_asset_metadata` and is
shared by checkpoints, LoRAs, VAEs, upscalers, textual inversions, and future
asset catalogs. Keep this module as a source-compatible shim for extensions and
older tests that imported the previous LoRA-specific names.
"""

from image_gen.webui.civitai_asset_metadata import (
    CIVITAI_ASSET_METADATA_SCHEMA_VERSION,
    CivitaiAssetMetadataService,
    CivitaiCredentialError,
    CivitaiMetadataError,
    CivitaiMetadataNotFound,
    CivitaiRequestError,
    read_civitai_api_key,
    resolve_civitai_key_path,
    sha256_file,
)

CIVITAI_LORA_METADATA_SCHEMA_VERSION = CIVITAI_ASSET_METADATA_SCHEMA_VERSION


class CivitaiLoraMetadataClient(CivitaiAssetMetadataService):
    """Deprecated LoRA-specific facade over the generic asset service."""

    def lookup_by_hashes(self, hashes):
        return super().lookup_by_hashes(hashes, asset_type="lora")


__all__ = [
    "CIVITAI_LORA_METADATA_SCHEMA_VERSION",
    "CivitaiLoraMetadataClient",
    "CivitaiCredentialError",
    "CivitaiMetadataError",
    "CivitaiMetadataNotFound",
    "CivitaiRequestError",
    "read_civitai_api_key",
    "resolve_civitai_key_path",
    "sha256_file",
]
