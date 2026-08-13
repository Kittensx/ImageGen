const providers = new Map();

function normalizeCapabilityId(value) {
  return String(value || "").trim().toLowerCase();
}

export function registerComponentCapability(capabilityId, provider) {
  const id = normalizeCapabilityId(capabilityId);
  if (!id) throw new TypeError("Capability ID is required.");
  if (!provider || typeof provider !== "object") throw new TypeError(`Capability '${id}' requires a provider object.`);
  if (providers.has(id)) throw new Error(`Component capability '${id}' is already registered.`);
  const registered = Object.freeze({ id, version: Number(provider.version || 1), ...provider });
  providers.set(id, registered);
  return registered;
}

export function componentCapability(capabilityId) {
  return providers.get(normalizeCapabilityId(capabilityId)) || null;
}

export function requireComponentCapabilities(capabilityIds = []) {
  const resolved = {};
  for (const rawId of Array.isArray(capabilityIds) ? capabilityIds : []) {
    const id = normalizeCapabilityId(rawId);
    if (!id) continue;
    const provider = componentCapability(id);
    if (!provider) throw new Error(`Required component capability '${id}' is unavailable.`);
    resolved[id] = provider;
  }
  return Object.freeze(resolved);
}

export function componentCapabilityCatalog() {
  return [...providers.values()].map((provider) => ({
    id: provider.id,
    version: provider.version,
    description: String(provider.description || ""),
  }));
}
