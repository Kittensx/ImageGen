from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import struct

from modules.state_dict_mapper import StateDictMapper


COMPONENT_SNAPSHOT_VERSION = "component-content-v1"


@dataclass(frozen=True)
class ComponentTensorSnapshot:
    key: str
    dtype: str
    shape: tuple[int, ...]
    byte_count: int
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "byte_count": self.byte_count,
            "payload_sha256": self.payload_sha256,
        }


@dataclass(frozen=True)
class ComponentSnapshot:
    component_role: str
    source_prefixes: tuple[str, ...]
    tensor_count: int
    total_bytes: int
    component_sha256: str
    structure_sha256: str
    dtype_summary: dict[str, int]
    tensors: tuple[ComponentTensorSnapshot, ...]
    snapshot_version: str = COMPONENT_SNAPSHOT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "component_role": self.component_role,
            "source_prefixes": list(self.source_prefixes),
            "tensor_count": self.tensor_count,
            "total_bytes": self.total_bytes,
            "component_sha256": self.component_sha256,
            "structure_sha256": self.structure_sha256,
            "dtype_summary": dict(self.dtype_summary),
            "tensors": [item.to_dict() for item in self.tensors],
        }


@dataclass(frozen=True)
class _TensorHeader:
    source_key: str
    relative_key: str
    dtype: str
    shape: tuple[int, ...]
    data_start: int
    data_end: int
    source_prefix: str

    @property
    def byte_count(self) -> int:
        return self.data_end - self.data_start


class SafetensorsComponentSnapshotter:
    """Compute deterministic component fingerprints directly from safetensors bytes.

    Component identity is content based. The component role/name is metadata only and
    is intentionally not part of ``component_sha256``. Outer checkpoint packaging
    prefixes are stripped through :class:`StateDictMapper` so an embedded component
    can match an equivalent standalone state dict when their relative tensor schema
    and payloads are identical.

    The implementation streams tensor payload bytes from disk and never moves model
    weights to CUDA or materializes the whole checkpoint in RAM.
    """

    def __init__(self, mapper: StateDictMapper | None = None, chunk_size: int = 4 * 1024 * 1024):
        self.mapper = mapper or StateDictMapper()
        self.chunk_size = max(64 * 1024, int(chunk_size))

    def discover_checkpoint_roles(
        self,
        path: str | Path,
        *,
        architecture: str | None,
        include_extras: bool = False,
    ) -> tuple[str, ...]:
        """Return component roles discoverable from the Safetensors header only.

        This is intentionally payload-free so registry completeness can be checked
        cheaply before deciding whether a multi-gigabyte checkpoint must be hashed
        again. Component routing uses the same :class:`StateDictMapper` rules as
        ``snapshot_checkpoint``.
        """
        file_path = Path(path).expanduser().resolve()
        header, _data_base = self._read_header(file_path)
        roles: set[str] = set()
        for source_key, entry in header.items():
            if source_key == "__metadata__" or not isinstance(entry, dict):
                continue
            component_role, _relative_key = self.mapper.route_key(
                source_key,
                architecture=architecture,
            )
            if component_role == "extras" and not include_extras:
                continue
            roles.add(component_role)
        return tuple(sorted(roles))

    def snapshot_checkpoint(
        self,
        path: str | Path,
        *,
        architecture: str | None,
        include_extras: bool = False,
        include_roles: set[str] | frozenset[str] | None = None,
    ) -> dict[str, ComponentSnapshot]:
        file_path = Path(path).expanduser().resolve()
        header, data_base = self._read_header(file_path)
        grouped: dict[str, list[_TensorHeader]] = defaultdict(list)

        for source_key, entry in header.items():
            if source_key == "__metadata__":
                continue
            if not isinstance(entry, dict):
                continue
            component_role, relative_key = self.mapper.route_key(
                source_key,
                architecture=architecture,
            )
            if component_role == "extras" and not include_extras:
                continue
            if include_roles is not None and component_role not in include_roles:
                continue
            data_offsets = entry.get("data_offsets")
            if not isinstance(data_offsets, list) or len(data_offsets) != 2:
                raise ValueError(f"Invalid safetensors data_offsets for {source_key!r}.")
            start, end = (int(data_offsets[0]), int(data_offsets[1]))
            if start < 0 or end < start:
                raise ValueError(f"Invalid safetensors byte range for {source_key!r}: {data_offsets!r}")
            grouped[component_role].append(
                _TensorHeader(
                    source_key=source_key,
                    relative_key=relative_key,
                    dtype=str(entry.get("dtype") or ""),
                    shape=tuple(int(value) for value in (entry.get("shape") or [])),
                    data_start=data_base + start,
                    data_end=data_base + end,
                    source_prefix=self._source_prefix(source_key, relative_key),
                )
            )

        snapshots: dict[str, ComponentSnapshot] = {}
        with file_path.open("rb") as handle:
            for role, tensors in sorted(grouped.items()):
                snapshots[role] = self._hash_component(handle, role=role, tensors=tensors)
        return snapshots

    def snapshot_standalone_component(
        self,
        path: str | Path,
        *,
        component_role: str = "standalone",
    ) -> ComponentSnapshot:
        """Fingerprint a standalone safetensors state dict without trusting its filename.

        All tensor keys are treated as component-relative keys. ``component_role`` is
        stored only as metadata and does not change the resulting content hash.
        """
        file_path = Path(path).expanduser().resolve()
        header, data_base = self._read_header(file_path)
        tensors: list[_TensorHeader] = []
        for source_key, entry in header.items():
            if source_key == "__metadata__":
                continue
            if not isinstance(entry, dict):
                continue
            data_offsets = entry.get("data_offsets")
            if not isinstance(data_offsets, list) or len(data_offsets) != 2:
                raise ValueError(f"Invalid safetensors data_offsets for {source_key!r}.")
            start, end = (int(data_offsets[0]), int(data_offsets[1]))
            tensors.append(
                _TensorHeader(
                    source_key=source_key,
                    relative_key=source_key,
                    dtype=str(entry.get("dtype") or ""),
                    shape=tuple(int(value) for value in (entry.get("shape") or [])),
                    data_start=data_base + start,
                    data_end=data_base + end,
                    source_prefix="",
                )
            )
        with file_path.open("rb") as handle:
            return self._hash_component(handle, role=component_role, tensors=tensors)

    def _hash_component(
        self,
        handle,
        *,
        role: str,
        tensors: Iterable[_TensorHeader],
    ) -> ComponentSnapshot:
        ordered = sorted(tensors, key=lambda item: item.relative_key)
        content_hasher = hashlib.sha256()
        structure_hasher = hashlib.sha256()
        content_hasher.update((COMPONENT_SNAPSHOT_VERSION + "\n").encode("utf-8"))
        structure_hasher.update((COMPONENT_SNAPSHOT_VERSION + "\n").encode("utf-8"))

        dtype_counter: Counter[str] = Counter()
        tensor_snapshots: list[ComponentTensorSnapshot] = []
        prefixes: set[str] = set()
        total_bytes = 0

        for tensor in ordered:
            descriptor = self._tensor_descriptor(tensor)
            content_hasher.update(descriptor)
            structure_hasher.update(descriptor)
            dtype_counter[tensor.dtype] += 1
            prefixes.add(tensor.source_prefix)
            total_bytes += tensor.byte_count

            payload_hasher = hashlib.sha256()
            handle.seek(tensor.data_start)
            remaining = tensor.byte_count
            while remaining:
                chunk = handle.read(min(self.chunk_size, remaining))
                if not chunk:
                    raise IOError(
                        f"Unexpected EOF while hashing component tensor {tensor.source_key!r}."
                    )
                payload_hasher.update(chunk)
                content_hasher.update(chunk)
                remaining -= len(chunk)

            tensor_snapshots.append(
                ComponentTensorSnapshot(
                    key=tensor.relative_key,
                    dtype=tensor.dtype,
                    shape=tensor.shape,
                    byte_count=tensor.byte_count,
                    payload_sha256=payload_hasher.hexdigest(),
                )
            )

        return ComponentSnapshot(
            component_role=role,
            source_prefixes=tuple(sorted(prefix for prefix in prefixes if prefix)),
            tensor_count=len(tensor_snapshots),
            total_bytes=total_bytes,
            component_sha256=content_hasher.hexdigest(),
            structure_sha256=structure_hasher.hexdigest(),
            dtype_summary=dict(sorted(dtype_counter.items())),
            tensors=tuple(tensor_snapshots),
        )

    @staticmethod
    def _tensor_descriptor(tensor: _TensorHeader) -> bytes:
        payload = {
            "key": tensor.relative_key,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "byte_count": tensor.byte_count,
        }
        return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    @staticmethod
    def _source_prefix(source_key: str, relative_key: str) -> str:
        if source_key == relative_key:
            return ""
        if relative_key and source_key.endswith(relative_key):
            return source_key[: -len(relative_key)]
        return ""

    @staticmethod
    def _read_header(path: Path) -> tuple[dict[str, Any], int]:
        if path.suffix.lower() != ".safetensors":
            raise ValueError(f"Component snapshots require .safetensors, got: {path.name}")
        with path.open("rb") as handle:
            raw_length = handle.read(8)
            if len(raw_length) != 8:
                raise ValueError(f"Invalid safetensors header length in {path}.")
            header_length = struct.unpack("<Q", raw_length)[0]
            header_bytes = handle.read(header_length)
            if len(header_bytes) != header_length:
                raise ValueError(f"Truncated safetensors header in {path}.")
        header = json.loads(header_bytes.decode("utf-8"))
        if not isinstance(header, dict):
            raise ValueError(f"Invalid safetensors header object in {path}.")
        return header, 8 + header_length
