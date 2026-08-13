from __future__ import annotations

import re
import secrets
from collections.abc import Iterator

import torch


MAX_SEED = 2**31 - 1
SEED_SPACE = MAX_SEED + 1


_SEED_RANGE_RE = re.compile(
    r"^\s*(?:-1\s*,\s*)?\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*$"
)


def parse_seed_range_expression(seed: object) -> tuple[int, int] | None:
    """Parse the WebUI-compatible finite random-seed range syntax.

    Accepted forms are ``[5000,15000]`` and ``-1, [5000,15000]``.
    ``None`` is returned for ordinary integer/random seed values so callers can
    retain their existing fixed-seed behavior.
    """

    if seed is None:
        return None
    match = _SEED_RANGE_RE.match(str(seed).strip())
    if match is None:
        return None
    minimum, maximum = int(match.group(1)), int(match.group(2))
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    if minimum < 0 or maximum > MAX_SEED:
        raise ValueError(f"Seed range must stay between 0 and {MAX_SEED}.")
    return minimum, maximum


def resolve_seed(seed: int | None) -> int:
    """Resolve one concrete seed.

    ``None`` and negative values request a fresh random seed. Non-negative
    values are preserved, modulo the supported seed space.
    """
    if seed is None:
        return secrets.randbelow(SEED_SPACE)

    resolved = int(seed)
    if resolved < 0:
        return secrets.randbelow(SEED_SPACE)
    return resolved


def offset_seed(seed: int, offset: int) -> int:
    """Advance a seed deterministically while remaining in the seed space."""
    return int(seed) + int(offset)


def resolve_seed_sequence(seed: int | None, count: int) -> list[int]:
    """Return A1111-style per-image seeds for one batch.

    A batch starts from one resolved base seed and each image advances by one.
    This makes every image individually reproducible by its recorded seed.
    """
    count = int(count)
    if count < 1:
        raise ValueError("seed sequence count must be at least 1")
    base_seed = resolve_seed(seed)
    return [offset_seed(base_seed, index) for index in range(count)]


def iter_batch_base_seeds(
    seed: int | None,
    *,
    batch_size: int,
) -> Iterator[int]:
    """Yield base seeds for successive batch-count or unlimited runs.

    Random mode (``None`` or a negative seed) selects a fresh random base for
    every batch. Fixed mode continues the sequence by ``batch_size`` so no
    image seed is repeated across batches.
    """
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    random_mode = seed is None or int(seed) < 0
    batch_index = 0
    while True:
        if random_mode:
            yield resolve_seed(-1)
        else:
            yield offset_seed(int(seed), batch_index * batch_size)
        batch_index += 1


def create_torch_generator(
    seed: int,
    device: str | torch.device = "cpu",
) -> torch.Generator:
    """Create a device-local ``torch.Generator`` for deterministic sampling."""
    resolved_device = torch.device(device)
    generator = torch.Generator(device=resolved_device)
    generator.manual_seed(int(seed))
    return generator


def prepare_generation_seed(
    seed: int | None,
    device: str | torch.device = "cpu",
) -> tuple[int, torch.Generator]:
    """Resolve a seed and construct the matching deterministic generator."""
    resolved_seed = resolve_seed(seed)
    generator = create_torch_generator(resolved_seed, device=device)
    return resolved_seed, generator
