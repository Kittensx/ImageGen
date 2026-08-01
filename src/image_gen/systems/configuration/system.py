from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.project_context import ProjectContext


class ConfigurationSystem:
    """Expose the canonical project context without performing inference work."""

    def __init__(self, context: ProjectContext | None = None) -> None:
        self.context = context or ProjectContext.load()

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "ConfigurationSystem":
        return cls(ProjectContext.load(config_path=config_path))

    def generation_defaults(self) -> dict[str, Any]:
        return self.context.generation_defaults()

    def validate_generation(self, **kwargs: Any):
        return self.context.validate(for_generation=True, **kwargs)
