from __future__ import annotations

from typing import NoReturn

RETIRED_HIRES_UPSCALER_IDS = frozenset(
    {
        "latent_nearest",
        "latent_bilinear",
        "latent_bicubic",
        "pixel_lanczos",
    }
)
RETIRED_HIRES_ARCHIVE = "src/image_gen/experimental/retired_hires_methods/"


class RetiredHiresMethodError(ValueError):
    """Raised when a removed Python-only hires method is requested."""


def is_retired_hires_method(value: object) -> bool:
    return str(value or "").strip().casefold() in RETIRED_HIRES_UPSCALER_IDS


def raise_retired_hires_method(value: object) -> NoReturn:
    selected = str(value or "<unspecified>").strip() or "<unspecified>"
    raise RetiredHiresMethodError(
        f"Hires method {selected!r} was retired from the active runtime. "
        "Choose a discovered neural .pth upscaler. The former implementation "
        f"is preserved under {RETIRED_HIRES_ARCHIVE}"
    )


__all__ = [
    "RETIRED_HIRES_ARCHIVE",
    "RETIRED_HIRES_UPSCALER_IDS",
    "RetiredHiresMethodError",
    "is_retired_hires_method",
    "raise_retired_hires_method",
]
