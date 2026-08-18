from __future__ import annotations

# Compatibility imports intentionally remain reachable from this module. Existing tests and
# developer tooling have historically patched ``service.subprocess.run`` during qualification.
import subprocess
import sys
from pathlib import Path

from modules.project_context import ProjectContext
from modules.registry.asset_registry import AssetRegistry
from modules.registry.component_selection import ComponentSelectionService
from image_gen.systems.registry import RuntimeRegistrySystem

from .artifact_io import (
    _json_safe,
    _load_yaml_mapping,
    _pixel_sha256,
    _safe_folder_name,
    _sha256_file,
    _slug,
    _utc_now,
    _write_yaml,
)
from .catalog import QualificationCatalogMixin
from .execution import QualificationExecutionMixin
from .planning import QualificationPlanningMixin
from .profiles import DEFAULT_NEGATIVE_PROMPT, DEFAULT_TEST_PROMPT, QualificationProfilesMixin
from .reviews import QualificationReviewsMixin


class ComponentQualificationRunner(
    QualificationCatalogMixin,
    QualificationProfilesMixin,
    QualificationPlanningMixin,
    QualificationExecutionMixin,
    QualificationReviewsMixin,
):
    """Stable public facade for component/model qualification workflows.

    R02 decomposes discovery, profiles, planning, execution, and review responsibilities
    behind this class without changing its public method names or call signatures. CMD and
    future WebUI callers should continue importing and using ``ComponentQualificationRunner``.
    """

    def __init__(
        self,
        context: ProjectContext,
        *,
        registry: AssetRegistry | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        self.context = context
        self.registry = registry or AssetRegistry(str(Path(context.registry_db_path).resolve()))
        self.selection = ComponentSelectionService(context, registry=self.registry)
        self.runtime_registry = RuntimeRegistrySystem(project_context=context)
        self.python_executable = Path(python_executable or sys.executable).resolve()
