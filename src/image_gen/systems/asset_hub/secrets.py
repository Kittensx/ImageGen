from __future__ import annotations

import ctypes
import os
from ctypes import wintypes
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SecretStatus:
    provider_id: str
    configured: bool
    source: str = "none"
    persistent: bool = False
    persistent_available: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "providerId": self.provider_id,
            "configured": self.configured,
            "source": self.source,
            "persistent": self.persistent,
            "persistentAvailable": self.persistent_available,
        }


class SecretStore(Protocol):
    def status(self, provider_id: str) -> SecretStatus: ...
    def set(self, provider_id: str, secret: str, *, persistent: bool) -> None: ...
    def get(self, provider_id: str) -> str | None: ...
    def delete(self, provider_id: str) -> None: ...


class PersistentSecretBackend(Protocol):
    @property
    def available(self) -> bool: ...
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...


class _UnavailableBackend:
    @property
    def available(self) -> bool:
        return False

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str) -> None:
        raise RuntimeError("No OS-backed credential store is available on this system.")

    def delete(self, key: str) -> None:
        return None


class _KeyringBackend:
    service_name = "IMAGE_GEN Asset Hub"

    def __init__(self) -> None:
        try:
            import keyring  # type: ignore
        except Exception:
            self._keyring = None
        else:
            self._keyring = keyring

    @property
    def available(self) -> bool:
        return self._keyring is not None

    def get(self, key: str) -> str | None:
        if self._keyring is None:
            return None
        try:
            return self._keyring.get_password(self.service_name, key)
        except Exception:
            return None

    def set(self, key: str, value: str) -> None:
        if self._keyring is None:
            raise RuntimeError("No keyring backend is available.")
        self._keyring.set_password(self.service_name, key, value)

    def delete(self, key: str) -> None:
        if self._keyring is None:
            return
        try:
            self._keyring.delete_password(self.service_name, key)
        except Exception:
            pass


class _WindowsCredentialBackend:
    """Tiny stdlib-only Windows Credential Manager adapter.

    Secrets are stored as generic credentials and are never mirrored into IMAGE_GEN
    JSON/YAML settings. This backend is intentionally inactive on non-Windows hosts.
    """

    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    ERROR_NOT_FOUND = 1168
    _PREFIX = "IMAGE_GEN/AssetHub/"

    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", wintypes.LPVOID),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    def __init__(self) -> None:
        self._advapi = None
        if os.name != "nt":
            return
        try:
            advapi = ctypes.WinDLL("Advapi32.dll", use_last_error=True)
            advapi.CredReadW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(ctypes.POINTER(self._CREDENTIALW))]
            advapi.CredReadW.restype = wintypes.BOOL
            advapi.CredWriteW.argtypes = [ctypes.POINTER(self._CREDENTIALW), wintypes.DWORD]
            advapi.CredWriteW.restype = wintypes.BOOL
            advapi.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
            advapi.CredDeleteW.restype = wintypes.BOOL
            advapi.CredFree.argtypes = [wintypes.LPVOID]
            advapi.CredFree.restype = None
        except Exception:
            return
        self._advapi = advapi

    @property
    def available(self) -> bool:
        return self._advapi is not None

    def _target(self, key: str) -> str:
        return f"{self._PREFIX}{key}"

    def get(self, key: str) -> str | None:
        if self._advapi is None:
            return None
        pointer = ctypes.POINTER(self._CREDENTIALW)()
        ok = self._advapi.CredReadW(self._target(key), self.CRED_TYPE_GENERIC, 0, ctypes.byref(pointer))
        if not ok:
            return None
        try:
            credential = pointer.contents
            if not credential.CredentialBlob or not credential.CredentialBlobSize:
                return None
            raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
            return raw.decode("utf-16-le").rstrip("\x00") or None
        finally:
            self._advapi.CredFree(pointer)

    def set(self, key: str, value: str) -> None:
        if self._advapi is None:
            raise RuntimeError("Windows Credential Manager is unavailable.")
        raw = value.encode("utf-16-le")
        if len(raw) > 2560:
            raise ValueError("Credential exceeds the Windows Credential Manager size limit.")
        blob = (ctypes.c_ubyte * len(raw)).from_buffer_copy(raw)
        credential = self._CREDENTIALW()
        credential.Type = self.CRED_TYPE_GENERIC
        credential.TargetName = self._target(key)
        credential.CredentialBlobSize = len(raw)
        credential.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
        credential.Persist = self.CRED_PERSIST_LOCAL_MACHINE
        credential.UserName = "IMAGE_GEN"
        if not self._advapi.CredWriteW(ctypes.byref(credential), 0):
            raise OSError(ctypes.get_last_error(), "Windows Credential Manager rejected the credential.")

    def delete(self, key: str) -> None:
        if self._advapi is None:
            return
        if not self._advapi.CredDeleteW(self._target(key), self.CRED_TYPE_GENERIC, 0):
            error = ctypes.get_last_error()
            if error != self.ERROR_NOT_FOUND:
                raise OSError(error, "Windows Credential Manager could not delete the credential.")


def _default_persistent_backend() -> PersistentSecretBackend:
    windows = _WindowsCredentialBackend()
    if windows.available:
        return windows
    keyring = _KeyringBackend()
    if keyring.available:
        return keyring
    return _UnavailableBackend()


class AssetHubSecretStore:
    ENVIRONMENT_KEYS = {"civitai": "CIVITAI_API_TOKEN"}

    def __init__(self, *, persistent_backend: PersistentSecretBackend | None = None) -> None:
        self._session: dict[str, str] = {}
        self._persistent = persistent_backend or _default_persistent_backend()

    @staticmethod
    def _provider(provider_id: str) -> str:
        token = str(provider_id or "").strip().casefold()
        if not token:
            raise ValueError("provider_id is required")
        return token

    @staticmethod
    def _secret(value: str) -> str:
        secret = str(value or "").strip()
        if not secret or any(ch.isspace() for ch in secret):
            raise ValueError("Provider token must be a non-empty single-line value without whitespace.")
        if len(secret) > 4096:
            raise ValueError("Provider token is too large.")
        return secret

    def status(self, provider_id: str) -> SecretStatus:
        provider = self._provider(provider_id)
        if provider in self._session:
            return SecretStatus(provider, True, "session", False, self._persistent.available)
        env_key = self.ENVIRONMENT_KEYS.get(provider, "")
        if env_key and str(os.environ.get(env_key) or "").strip():
            return SecretStatus(provider, True, "environment", True, self._persistent.available)
        if self._persistent.get(provider):
            return SecretStatus(provider, True, "os_store", True, self._persistent.available)
        return SecretStatus(provider, False, "none", False, self._persistent.available)

    def get(self, provider_id: str) -> str | None:
        provider = self._provider(provider_id)
        if provider in self._session:
            return self._session[provider]
        env_key = self.ENVIRONMENT_KEYS.get(provider, "")
        if env_key:
            value = str(os.environ.get(env_key) or "").strip()
            if value:
                return value
        return self._persistent.get(provider)

    def set(self, provider_id: str, secret: str, *, persistent: bool) -> None:
        provider = self._provider(provider_id)
        value = self._secret(secret)
        if persistent:
            if not self._persistent.available:
                raise RuntimeError("Persistent credential storage is not available on this system; use session-only authentication or CIVITAI_API_TOKEN.")
            self._persistent.set(provider, value)
            self._session.pop(provider, None)
        else:
            self._session[provider] = value

    def delete(self, provider_id: str) -> None:
        provider = self._provider(provider_id)
        self._session.pop(provider, None)
        if self._persistent.available:
            self._persistent.delete(provider)
