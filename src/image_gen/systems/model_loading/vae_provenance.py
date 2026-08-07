"""Compatibility re-export for the canonical VAE provenance contract.

The contract lives outside the model-loading package so runtime systems can inspect
an already-loaded VAE without importing the model loader implementation.
"""

from image_gen.contracts.vae_provenance import (
    VAE_PROVENANCE_CONTRACT_VERSION,
    attach_vae_provenance,
    normalize_vae_provenance,
    read_vae_provenance,
)

__all__ = [
    "VAE_PROVENANCE_CONTRACT_VERSION",
    "attach_vae_provenance",
    "normalize_vae_provenance",
    "read_vae_provenance",
]
