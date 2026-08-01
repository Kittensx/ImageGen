from __future__ import annotations


class PluginRegistryError(RuntimeError):
    """Base error for plugin discovery, validation, or construction failures."""


class PluginDescriptorError(PluginRegistryError, ValueError):
    """Raised when a plugin descriptor is missing or malformed."""


class DuplicatePluginIdentityError(PluginRegistryError, ValueError):
    """Raised when two plugins claim the same ID, name, label, or alias."""


class PluginDiscoveryError(PluginRegistryError):
    """Raised when a candidate plugin module cannot be discovered or imported."""


class PluginCompatibilityError(PluginRegistryError, ValueError):
    """Raised when a sampler and scheduler capability declaration conflict."""


class PluginInstantiationError(PluginRegistryError, TypeError):
    """Raised when a validated plugin adapter cannot be instantiated."""
