from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, MutableMapping

from modules.txt2img.seed_utils import MAX_SEED, parse_seed_range_expression, resolve_seed

@dataclass
class SeedPlan:
    mode: str
    seed: int | None = None
    minimum: int = 0
    maximum: int = MAX_SEED
    unique: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "unique": self.unique,
        }


def parse_seed_plan(
    seed_value: Any,
    *,
    mode: Any = None,
    range_min: Any = None,
    range_max: Any = None,
    unique: Any = True,
) -> SeedPlan:
    """Normalize WebUI seed syntax and batch seed preferences.

    Accepted seed text includes ``-1`` and ``[5000,1925048]`` or
    ``-1, [5000,1925048]``. Explicit ``batch_seed_mode`` wins when supplied.
    """

    requested_mode = str(mode or "").strip().lower().replace("-", "_")
    if requested_mode not in {"", "sequential", "random", "random_range"}:
        raise ValueError("batch_seed_mode must be sequential, random, or random_range.")

    text = "" if seed_value is None else str(seed_value).strip()
    seed_range = parse_seed_range_expression(text)
    if seed_range is not None:
        minimum, maximum = seed_range
        return SeedPlan("random_range", seed=-1, minimum=minimum, maximum=maximum, unique=bool(unique))

    def as_int(value: Any, default: int | None = None) -> int | None:
        if value in (None, ""):
            return default
        try:
            return int(float(str(value).strip()))
        except (TypeError, ValueError):
            return default

    seed = as_int(seed_value, None)
    # The seed box keeps its long-standing contract: an explicit -1 means
    # fresh random seeds. Batch strategy can choose other behavior, but it
    # must not silently reinterpret the seed-box random sentinel as a fixed
    # sequential starting point.
    seed_box_requests_random = text == "-1"
    minimum = as_int(range_min, 0)
    maximum = as_int(range_max, MAX_SEED)
    minimum = 0 if minimum is None else minimum
    maximum = MAX_SEED if maximum is None else maximum
    if minimum > maximum:
        minimum, maximum = maximum, minimum
    if minimum < 0 or maximum > MAX_SEED:
        raise ValueError(f"Seed range must stay between 0 and {MAX_SEED}.")

    if requested_mode == "random_range":
        return SeedPlan("random_range", seed=-1, minimum=minimum, maximum=maximum, unique=bool(unique))
    if requested_mode == "random":
        return SeedPlan("random", seed=-1, minimum=0, maximum=MAX_SEED, unique=False)
    if requested_mode == "sequential" and not seed_box_requests_random:
        return SeedPlan("sequential", seed=seed, minimum=minimum, maximum=maximum, unique=False)
    if seed is None or seed < 0:
        return SeedPlan("random", seed=-1, minimum=0, maximum=MAX_SEED, unique=False)
    return SeedPlan("sequential", seed=seed, minimum=minimum, maximum=maximum, unique=False)


class UniqueRangeCycle:
    """Sample a finite integer range without replacement until exhausted."""

    def __init__(self, minimum: int, maximum: int) -> None:
        self.minimum = int(minimum)
        self.maximum = int(maximum)
        self.size = self.maximum - self.minimum + 1
        if self.size < 1:
            raise ValueError("Random range must contain at least one integer.")
        self._remaining: list[int] = []
        self.cycles = 0

    def _refill(self) -> None:
        # Avoid materializing pathological full 31-bit seed ranges. For large
        # ranges use a bounded uniqueness set and rejection sampling instead.
        if self.size <= 1_000_000:
            self._remaining = list(range(self.minimum, self.maximum + 1))
            secrets.SystemRandom().shuffle(self._remaining)
            self.cycles += 1

    def iterator(self) -> Iterator[int]:
        if self.size <= 1_000_000:
            while True:
                if not self._remaining:
                    self._refill()
                yield self._remaining.pop()
        seen: set[int] = set()
        while True:
            if len(seen) >= self.size:
                seen.clear()
                self.cycles += 1
            value = self.minimum + secrets.randbelow(self.size)
            if value in seen:
                continue
            seen.add(value)
            yield value


def iter_seed_plan(
    plan: SeedPlan,
    *,
    start_index: int = 0,
    exclude: Iterator[int] | list[int] | set[int] | tuple[int, ...] | None = None,
) -> Iterator[int]:
    """Yield seeds for a plan, preserving finite-range uniqueness across resumes.

    ``start_index`` advances deterministic sequential plans. ``exclude`` is most
    useful for finite random ranges: seeds already consumed by a paused job are
    skipped until every value in the range has necessarily been used, at which
    point a new uniqueness cycle begins and duplicates are unavoidable.
    """

    start_index = max(0, int(start_index))
    if plan.mode == "random_range":
        if not plan.unique:
            size = plan.maximum - plan.minimum + 1
            while True:
                yield plan.minimum + secrets.randbelow(size)
        blocked = {
            int(value)
            for value in (exclude or ())
            if plan.minimum <= int(value) <= plan.maximum
        }
        if len(blocked) >= (plan.maximum - plan.minimum + 1):
            blocked.clear()
        for value in UniqueRangeCycle(plan.minimum, plan.maximum).iterator():
            if value in blocked:
                continue
            blocked.add(value)
            if len(blocked) >= (plan.maximum - plan.minimum + 1):
                # The next draw begins a fresh cycle because the requested range
                # can no longer provide a new value without reuse.
                blocked.clear()
            yield value
        return
    if plan.mode == "random":
        while True:
            yield resolve_seed(-1)
    start = resolve_seed(plan.seed if plan.seed is not None else -1)
    index = start_index
    while True:
        yield start + index
        index += 1


def _get_path(root: Mapping[str, Any], path: str) -> Any:
    current: Any = root
    for token in str(path).split("."):
        if not isinstance(current, Mapping):
            return None
        current = current.get(token)
    return current


def _set_path(root: MutableMapping[str, Any], path: str, value: Any) -> None:
    tokens = str(path).split(".")
    current: MutableMapping[str, Any] = root
    for token in tokens[:-1]:
        child = current.get(token)
        if not isinstance(child, MutableMapping):
            child = {}
            current[token] = child
        current = child
    current[tokens[-1]] = value


def _finite_number(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def normalize_parameter_ranges(value: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for raw_path, raw_spec in value.items():
        path = str(raw_path or "").strip()
        if not path or not isinstance(raw_spec, Mapping):
            continue
        minimum = _finite_number(raw_spec.get("min"))
        maximum = _finite_number(raw_spec.get("max"))
        lock_min = _finite_number(raw_spec.get("lock_min"))
        lock_max = _finite_number(raw_spec.get("lock_max"))
        if minimum is not None and maximum is not None and minimum > maximum:
            minimum, maximum = maximum, minimum
        if lock_min is not None and lock_max is not None and lock_min > lock_max:
            lock_min, lock_max = lock_max, lock_min
        output[path] = {
            "enabled": bool(raw_spec.get("enabled", minimum is not None and maximum is not None)),
            "min": minimum,
            "max": maximum,
            "integer": bool(raw_spec.get("integer", False)),
            "lock_min": lock_min,
            "lock_max": lock_max,
        }
    return output


def apply_parameter_ranges(request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one image request from stored randomization specs and clamps."""

    resolved = dict(request)
    specs = normalize_parameter_ranges(request.get("_random_ranges"))
    resolution: dict[str, Any] = {}
    for path, spec in specs.items():
        current = _get_path(resolved, path)
        value = _finite_number(current)
        minimum, maximum = spec.get("min"), spec.get("max")
        if spec.get("enabled") and minimum is not None and maximum is not None:
            if spec.get("integer"):
                low, high = int(math.ceil(minimum)), int(math.floor(maximum))
                if low > high:
                    raise ValueError(f"Random integer range for {path} contains no whole numbers.")
                value = low + secrets.randbelow(high - low + 1)
            else:
                # 53 bits produces a stable double-precision fraction without
                # depending on the process-global random module.
                fraction = secrets.randbits(53) / float(1 << 53)
                value = float(minimum) + (float(maximum) - float(minimum)) * fraction
        if value is None:
            continue
        lock_min, lock_max = spec.get("lock_min"), spec.get("lock_max")
        unclamped = value
        if lock_min is not None:
            value = max(value, float(lock_min))
        if lock_max is not None:
            value = min(value, float(lock_max))
        if spec.get("integer"):
            value = int(round(value))
        _set_path(resolved, path, value)
        resolution[path] = {
            "requested_spec": dict(spec),
            "resolved_value": value,
            "clamped": value != unclamped,
        }
    if resolution:
        resolved["_randomization_resolved"] = resolution
    return resolved, resolution
