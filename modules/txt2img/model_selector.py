from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from modules.asset_discovery import resolve_nested_asset
from modules.project_context import ProjectContext


MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".pth"}


@dataclass
class ModelListEntry:
    index: int
    name: str
    path: str
    extension: str
    size_mb: float


class ModelSelector:
    """CLI/UI model discovery driven by the canonical project context."""

    def __init__(
        self,
        loader: Any | None = None,
        project_context: ProjectContext | None = None,
    ):
        inherited_context = getattr(loader, "context", None)
        self.context = project_context or inherited_context or ProjectContext.load()
        self.loader = loader

    def get_checkpoint_root(self) -> Path:
        return self.context.checkpoints_dir

    def scan_models(self) -> list[ModelListEntry]:
        root = self.get_checkpoint_root()
        if not root.exists():
            return []

        entries: list[ModelListEntry] = []
        paths = sorted(
            [
                p for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in MODEL_EXTENSIONS
            ],
            key=lambda p: p.name.lower(),
        )

        for idx, path in enumerate(paths, start=1):
            try:
                size_mb = path.stat().st_size / (1024 * 1024)
            except OSError:
                size_mb = 0.0

            entries.append(
                ModelListEntry(
                    index=idx,
                    name=path.stem,
                    path=str(path),
                    extension=path.suffix.lower(),
                    size_mb=round(size_mb, 2),
                )
            )
        return entries

    def print_model_list(self, entries: list[ModelListEntry] | None = None) -> None:
        entries = entries if entries is not None else self.scan_models()
        if not entries:
            print("No models found in checkpoint root.")
            return

        print("\n=== Available Models ===")
        for entry in entries:
            print(
                f"{entry.index}. {entry.name} "
                f"({entry.extension}, {entry.size_mb:.2f} MB)"
            )
            print(f"   {entry.path}")

    def choose_model(self, entries: list[ModelListEntry] | None = None) -> ModelListEntry:
        entries = entries if entries is not None else self.scan_models()
        if not entries:
            raise RuntimeError(
                f"No models found under: {self.get_checkpoint_root()}"
            )

        self.print_model_list(entries)

        while True:
            raw = input(f"Choose model [1-{len(entries)}]: ").strip()
            try:
                selection = int(raw)
            except ValueError:
                print("Please enter a number.")
                continue

            if 1 <= selection <= len(entries):
                return entries[selection - 1]

            print("Invalid selection.")

    def resolve_model_path(
        self,
        explicit_path: str | None = None,
        *,
        interactive: bool = False,
    ) -> str:
        """Resolve an explicit, interactive, or configured model path."""

        if explicit_path:
            direct = self.context.resolve_project_path(explicit_path)
            if direct.is_file():
                return str(direct)
            nested = resolve_nested_asset(
                self.get_checkpoint_root(),
                explicit_path,
                extensions=MODEL_EXTENSIONS,
            )
            return str(nested) if nested is not None else str(direct)

        if interactive:
            return self.choose_model().path

        if self.context.default_model_path is not None:
            configured = self.context.default_model_path
            if configured.is_file():
                return str(configured)
            nested = resolve_nested_asset(
                self.get_checkpoint_root(),
                configured.name,
                extensions=MODEL_EXTENSIONS,
            )
            if nested is not None:
                return str(nested)
            return str(configured)

        return ""


def resolve_cli_model_path(
    explicit_path: str | None = None,
    *,
    interactive: bool = False,
    loader: Any | None = None,
    project_context: ProjectContext | None = None,
) -> str:
    selector = ModelSelector(loader=loader, project_context=project_context)
    return selector.resolve_model_path(explicit_path, interactive=interactive)
