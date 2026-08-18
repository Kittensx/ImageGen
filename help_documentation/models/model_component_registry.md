---
title: Model / Component Registry
summary: Inspect fingerprinted model components, their current locations, compatibility evidence, and user exclusions.
category: Models
audience: user
status: current
keywords:
- registry
- components
- models
- fingerprints
- compatibility
- validation
- exclusions
- disconnected drives
related:
- setup/advanced_models_component_composition
featured: false
media: []
---

# Model / Component Registry

The Model / Component Registry is the WebUI inspection surface for IMAGE_GEN's component-native model database.

IMAGE_GEN treats a model file location and a component identity as different things. A checkpoint may move to another drive, disappear temporarily, or be copied to a new location without changing the identity of the component tensors that were previously scanned from it.

## Open the registry browser

Enable **Advanced Models**, then select **Registry Browser** from the Advanced Models controls.

The browser has three areas:

1. **Registered models** shows known checkpoint file locations and their current accessibility state.
2. **Fingerprint components** shows deduplicated component identities keyed by their full component SHA-256.
3. **Component evidence** shows the selected fingerprint's known source locations, user policy, compatibility validation evidence, and analytical/provenance relationship evidence.

## Accessible and unavailable locations

The registry preserves historical source locations instead of deleting them just because a drive is disconnected.

Typical states include:

- **available** - the registered source is currently accessible;
- **missing** - the registered file is not currently present;
- **inaccessible** - the configured location cannot currently be reached;
- **moved_relinked** - IMAGE_GEN found the same exact file content at another location and relinked the content identity;
- **archived** - the location has been retained as historical registry information.

Use **Accessible sources only** to hide unavailable source occurrences while browsing. Hiding a location does not delete its registry history.

## Scan the model library

Use **Refresh Registry** in the browser or **Scan Model Library** in Advanced Models to scan the configured model library again.

The scan checks reachable configured roots, updates currently accessible locations, discovers new files, and reconciles exact file hashes with existing registry entries. An unreachable configured root is reported as unavailable instead of causing the entire scan to fail.

If you copy an already-known checkpoint from a disconnected drive to a reachable local drive, IMAGE_GEN can recognize the same full checkpoint SHA-256 and associate the new location with the existing content/component identities. The filename is not used as proof of identity.

## Component identity and source paths

The component fingerprint is the identity. A component can have several sources, including:

- an embedded digital component inside a checkpoint;
- a standalone VAE or text encoder;
- a physical/materialized component;
- a reconstructed export.

Source paths are occurrence evidence only. Moving a source does not create a new component identity when the fingerprinted component content is unchanged.

## Compatibility status

Structural eligibility is not the same as proven runtime compatibility. The registry keeps the following dimensions separate:

- **Structural eligibility** - the architecture provider considers the role/interface structurally plausible.
- **Compatibility validation** - an exact base/candidate fingerprint combination has passed or failed one or more validation stages.
- **User policy** - the user has chosen to disable a fingerprint globally or exclude it for a specific base fingerprint.
- **Analytical relationships** - exact overlap or other relationship evidence. Analytical similarity/overlap does not automatically mean runtime compatibility.

Advanced Models may show these status labels:

- **Untested** - structurally eligible, but no qualifying compatibility result is recorded.
- **Validated** - one or more compatibility stages have passed.
- **Validation failed** - a recorded blocking validation failure applies to the selected base/component combination.
- **Disabled for this base** - the user excluded this candidate only for the selected base fingerprint.
- **Globally disabled** - the user disabled this exact fingerprint for normal/automatic selection everywhere.

Disabled candidates remain visible for inspection but cannot be selected by normal Advanced Models resolution.

## Global disable and per-base exclusion

Select a component in the Registry Browser to manage policy.

**Disable globally** prevents the exact fingerprint from normal or automatic selection until re-enabled.

**Exclude for selected base** prevents the candidate only when the currently selected Advanced Models base fingerprint is active. The same candidate remains eligible with other base fingerprints unless separately excluded.

These actions do not modify the original checkpoint blueprint, component bytes, source files, or analytical relationship evidence.

## Validation evidence

Compatibility validation is staged. A weaker successful stage does not prove a stronger stage.

The current evidence model supports:

1. structural compatibility;
2. component hydration;
3. runtime interface compatibility;
4. conditioning/forward-pass compatibility;
5. real generation success;
6. deterministic parity/quality evidence where applicable.

Failures such as a transient out-of-memory condition are recorded as advisory evidence rather than automatically becoming a permanent compatibility blacklist. A blocking incompatibility must be explicitly categorized as blocking evidence.

The Registry Browser also provides **Clear validation evidence**. This removes compatibility-test records for that component but does not remove user policy or analytical/provenance relationships.

## Analytical and provenance relationships

The relationship section is intentionally separate from compatibility evidence. Exact block/tensor overlap, recorded provenance, future similarity analysis, and future derivation analysis describe relationships between fingerprinted components. They do not silently enable or disable a component.

This separation allows IMAGE_GEN to preserve exact relationship evidence while independently tracking whether a particular component combination is usable at runtime.
