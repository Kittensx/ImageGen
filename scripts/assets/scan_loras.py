from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _project_root_from_args(value: str | None) -> Path:
    if value:
        return Path(value).expanduser().resolve()
    return next((p for p in Path(__file__).resolve().parents if (p / "scripts" / "resolve_python.bat").is_file()), Path.cwd())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Refresh and scan IMAGE_GEN LoRA assets.")
    parser.add_argument("--project-root", help="IMAGE_GEN project root. Defaults to the repository root.")
    parser.add_argument("--project-config", help="Optional project config override.")
    parser.add_argument("--mode", choices=("missing", "all"), default="missing")
    parser.add_argument("--as-json", action="store_true", help="Print the full scan payload as JSON.")
    args = parser.parse_args(argv)

    project_root = _project_root_from_args(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    src_root = project_root / "src"
    if src_root.is_dir() and str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    from modules.project_context import ProjectContext
    from image_gen.webui.catalog import WebUICatalog

    context = ProjectContext.load(project_root=str(project_root), config_path=args.project_config)
    catalog = WebUICatalog(context)
    payload = catalog.scan_loras(mode=args.mode)

    if args.as_json:
        print(json.dumps(payload, indent=2))
    else:
        summary = payload.get("scan") or {}
        catalog_info = payload.get("catalog") or {}
        print(f"LoRA catalog refresh complete.")
        print(f"Project root: {project_root}")
        print(f"Mode:         {summary.get('mode')}")
        print(f"Catalog cnt:  {catalog_info.get('count')}")
        print(f"Scanned:      {summary.get('scanned')}")
        print(f"Refreshed:    {summary.get('refreshed')}")
        print(f"Errors:       {summary.get('errors')}")
        print(f"Unsupported:  {summary.get('unsupported')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
