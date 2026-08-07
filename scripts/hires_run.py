from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    """Compatibility launcher for the active `.pth` interactive hires flow."""

    from modules.txt2img.cli import main as txt2img_main

    forwarded = list(argv if argv is not None else sys.argv[1:])
    return txt2img_main(["run", "--interactive-hires", "--save", *forwarded])


if __name__ == "__main__":
    raise SystemExit(main())
