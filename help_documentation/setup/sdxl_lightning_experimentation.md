---
title: SDXL-Lightning Recommendations and Experimental Overrides
summary: How IMAGE_GEN exposes opt-in SDXL-Lightning step/CFG recommendations while keeping fields editable and sampler/scheduler choices unrestricted.
category: Setup
audience: user
status: current
keywords:
- sdxl lightning
- cfg
- steps
- cfg lab
- experimental overrides
- recommended settings
related:
- setup/sdxl_support
featured: false
media: []
external_links: []
---

# SDXL-Lightning Recommendations and Experimental Overrides

SDXL-Lightning checkpoints are distilled around small inference step counts. IMAGE_GEN exposes those model recommendations without turning the profile into an allowlist.

For a 4-step Lightning checkpoint, the normal starting point is:

```text
Steps: 4
CFG: 1.0

[ ] Auto-use recommended steps (4)
[ ] Auto-use recommended CFG (1) + recommended CFG Lab preset
```

The recommendation fields remain editable at all times.

## Checked recommendations are opt-in effective values

The checkboxes have a concrete meaning:

- **Auto-use recommended steps** checked: IMAGE_GEN uses the model's recommended step count for generation.
- **Auto-use recommended CFG** checked: IMAGE_GEN uses the model's recommended CFG for generation.
- Clearing either checkbox makes the value currently typed into that field authoritative.

The fields are never disabled merely because a recommendation is enabled.

If the visible value differs while its recommendation remains checked, IMAGE_GEN shows a warning instead of failing.

Example:

```text
Visible Steps: 20
[ ] Auto-use recommended steps (4)

Effective generation Steps: 4
```

and:

```text
Visible CFG: 5.5
[ ] Auto-use recommended CFG (1)

Effective generation CFG: 1.0
```

The warning tells you to uncheck the recommendation if you want the custom value to be used.

## SDXL Lightning Recommended CFG Lab preset

Lightning profiles now advertise a built-in CFG Lab preset:

```text
SDXL Lightning Recommended
```

When a Lightning profile becomes active with **Auto-use recommended CFG** enabled, or when that checkbox is enabled, IMAGE_GEN selects and applies this preset once.

The preset is deliberately conservative:

- legacy/flat guidance;
- no early CFG boost;
- no late CFG taper;
- no early absolute CFG floor;
- no CFG rescale;
- no hidden replacement of the base CFG field.

This keeps the model-recommended CFG `1.0` genuinely flat instead of allowing stale CFG Lab shaping values to amplify or taper it unexpectedly.

After the preset is applied, CFG Lab remains fully editable. Selecting another preset is allowed and does not re-lock the advanced controls.

## Classic / Flat is now a true identity preset

**Classic / Flat** no longer writes CFG `7.0` into the base CFG box.

It means:

> Use the current base CFG value unchanged across generation.

For example:

```text
CFG field: 1.0
Preset: Classic / Flat
```

produces flat requested/effective CFG `1.0` unless another explicit system such as a prompt CFG directive changes the requested curve.

Likewise:

```text
CFG field: 5.5
Preset: Classic / Flat
```

keeps CFG `5.5` flat.

## Corrected Low-CFG presets

The built-in Low-CFG presets no longer assume a normal SDXL CFG around `5-7` by using an absolute `6.2`/`6.8` early floor.

They now preserve the base CFG field and apply relative shaping.

**Low-CFG Safe** provides a small early composition boost without tapering below the selected base CFG.

**Low-CFG Strong Composition** provides a stronger early boost, again without an automatic late drop below the selected base.

If you intentionally want CFG to fall below the base value late in the run, use **Soft Detail Taper**. That behavior is explicit rather than hidden inside the Safe preset.

## Warnings are not failures

IMAGE_GEN may warn when:

- Steps differ from the model recommendation;
- CFG differs from the model recommendation;
- sampler differs from the recommendation;
- scheduler differs from the recommendation;
- a scheduler setting needs a safe normalization for a short distilled schedule.

Sampler and scheduler recommendations are never enforced by the SDXL model profile.

If you select KES, A1111 compatibility, another scheduler, or another registered sampler, the model profile may warn but it does not replace that selection or reject the generation.

Actual runtime errors remain possible for genuinely invalid or unavailable runtime resources. Those errors are separate from model recommendations.

## Short-step scheduler safety

Very short Lightning runs can expose scheduler settings that were authored for longer schedules. IMAGE_GEN safely normalizes bounded operations where possible.

For example:

```text
Requested steps: 4
Requested in-place blend tail: 5
Effective blend tail: 4
```

That normalization is surfaced as a warning instead of becoming an out-of-range scheduler crash.

## Experimenting with custom Steps or CFG

To intentionally use custom values:

```text
[ ] Auto-use recommended steps
[ ] Auto-use recommended CFG
```

Then the normal fields become the effective generation values.

Example:

```text
Steps: 20
CFG: 5.5
Sampler: kes
Scheduler: a1111_compatibility
```

IMAGE_GEN may still show model-recommendation warnings, but **Generation is allowed**.

## IMAGE_GEN policy

The policy is:

```text
Model profile
    -> describes the model
    -> provides recommended values
    -> can apply Steps/CFG only when the user explicitly checks those recommendations
    -> may select a recommended advanced CFG Lab preset as a convenience
    -> warns when choices differ
    -> never enforces sampler/scheduler allowlists
    -> never turns a recommendation mismatch into a hard failure
```
