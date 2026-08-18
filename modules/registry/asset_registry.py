from __future__ import annotations

import os

from .database import RegistryDatabase
from .fingerprint import FileFingerprint
from .migrations import RegistryMigrations
from .stores import AssetStore, ComponentStore, DiagnosticsStore, EvidenceStore


class AssetRegistry(
    RegistryDatabase,
    RegistryMigrations,
    AssetStore,
    ComponentStore,
    EvidenceStore,
    DiagnosticsStore,
):
    """Compatibility facade for IMAGE_GEN's local SQLite asset registry.

    Public method names and signatures remain on ``AssetRegistry`` through the
    persistence-domain base classes.  Existing callers should continue to use
    this facade rather than depending directly on individual stores.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = os.path.dirname(db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        self.fingerprinter = FileFingerprint()
        self._init_db()
