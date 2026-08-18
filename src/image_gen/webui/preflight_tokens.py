from __future__ import annotations

import copy
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar("T")
DEFAULT_PREFLIGHT_TOKEN_TTL_SECONDS = 15 * 60


@dataclass(slots=True)
class _StoredPreflightToken(Generic[T]):
    token: str
    specification: T
    created_monotonic: float


class PreflightTokenStore(Generic[T]):
    """Thread-safe, expiring store for server-authoritative preflight specifications.

    The store deliberately separates lookup from discard. Existing replay/import/
    variation flows may revalidate a token more than once before a successful
    submission, and only discard it when that workflow reaches its terminal
    submission path.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_PREFLIGHT_TOKEN_TTL_SECONDS,
        missing_message: str,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        if ttl_seconds <= 0:
            raise ValueError("Preflight token TTL must be greater than zero.")
        self._ttl_seconds = float(ttl_seconds)
        self._missing_message = str(missing_message)
        self._clock = clock
        self._token_factory = token_factory or (lambda: uuid.uuid4().hex)
        self._items: dict[str, _StoredPreflightToken[T]] = {}
        self._lock = threading.RLock()

    def cleanup(self) -> int:
        cutoff = self._clock() - self._ttl_seconds
        removed = 0
        with self._lock:
            stale = [
                token
                for token, stored in self._items.items()
                if stored.created_monotonic < cutoff
            ]
            for token in stale:
                self._items.pop(token, None)
                removed += 1
        return removed

    def issue(self, specification: T) -> str:
        self.cleanup()
        token = str(self._token_factory() or "").strip()
        if not token:
            raise RuntimeError("Preflight token factory returned an empty token.")
        with self._lock:
            self._items[token] = _StoredPreflightToken(
                token=token,
                specification=copy.deepcopy(specification),
                created_monotonic=self._clock(),
            )
        return token

    def get(self, token: str) -> T:
        self.cleanup()
        key = str(token or "")
        with self._lock:
            stored = self._items.get(key)
            if stored is None:
                raise ValueError(self._missing_message)
            return copy.deepcopy(stored.specification)

    def discard(self, token: str) -> None:
        with self._lock:
            self._items.pop(str(token or ""), None)

    def __len__(self) -> int:
        self.cleanup()
        with self._lock:
            return len(self._items)


__all__ = ["DEFAULT_PREFLIGHT_TOKEN_TTL_SECONDS", "PreflightTokenStore"]
