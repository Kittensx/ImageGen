# Prompt Conditioning + Negative Conditioning Contract

## Overview

This document defines the **prompt conditioning contract** for the
txt2img pipeline.

The goal is to ensure all components---parser, adapter, pipeline, and
sampler---agree on:

-   what conditioning data looks like
-   how positive and negative prompts are handled
-   how stepwise prompt scheduling works
-   how conditioning is passed into the model

This is a **Phase 1 (SD1-style)** contract and intentionally keeps
things simple and stable.

------------------------------------------------------------------------

## Core Concept

The pipeline operates on a unified object:

``` python
ConditioningOutput
```

This object is produced by the **PromptConditioningAdapter** and
consumed by:

-   pipeline
-   sampler
-   stepwise conditioning resolver (optional)

------------------------------------------------------------------------

## ConditioningOutput Structure

### Required Fields

These fields must always be present:

``` python
cond: torch.Tensor
uncond: torch.Tensor
```

-   `cond` → positive prompt conditioning
-   `uncond` → negative prompt conditioning

These are passed directly into the model (e.g. UNet).

------------------------------------------------------------------------

### Optional Fields

``` python
pooled_cond: torch.Tensor | None
pooled_uncond: torch.Tensor | None
prompt_schedules: dict[str, Any]
extra: dict[str, Any]
```

-   `pooled_*` → reserved for future models (e.g. SDXL)
-   `prompt_schedules` → parsed scheduling data from prompt parser
-   `extra` → extension space for additional features

------------------------------------------------------------------------

## Negative Prompt Contract

Negative conditioning is always present.

Rules:

-   If user provides no negative prompt → use empty string `""`
-   `uncond` must always exist
-   Samplers should never need to handle "missing negative conditioning"

------------------------------------------------------------------------

## Conditioning Type (Phase 1)

For this phase, conditioning is strictly:

``` python
torch.Tensor
```

Not supported yet:

-   dict-based conditioning
-   multi-encoder conditioning
-   SDXL-style conditioning structures

These may be added later without breaking the contract.

------------------------------------------------------------------------

## Prompt Encoding Contract

The system expects a model that exposes:

``` python
get_learned_conditioning(texts: list[str]) -> torch.Tensor
```

This function is responsible for:

-   tokenizing text
-   encoding into conditioning tensors

The adapter will call this function to produce `cond` and `uncond`.

------------------------------------------------------------------------

## Stepwise Prompt Scheduling

Stepwise prompt changes are optional.

### Resolver Contract

If stepwise scheduling is used:

``` python
conditioning.extra["resolver"]
```

must exist and implement:

``` python
resolve(step_index: int) -> tuple[torch.Tensor, torch.Tensor]
```

### Behavior

-   If resolver exists → sampler can update conditioning per step
-   If not → static conditioning is used for entire generation

------------------------------------------------------------------------

## Canonical Step Conditioning Function

All samplers should use:

``` python
resolve_step_conditioning(conditioning, step_index, latents, state)
```

This function:

-   returns correct `(cond, uncond)` for current step
-   handles resolver logic automatically
-   ensures dtype/device alignment

------------------------------------------------------------------------

## Sampler Expectations

Samplers may assume:

-   `conditioning.cond` exists
-   `conditioning.uncond` exists
-   `conditioning.extra["resolver"]` may exist

Samplers should:

-   use static conditioning OR
-   call step resolver per step

------------------------------------------------------------------------

## Model Invocation Contract

The current pipeline uses:

``` python
unet(
    sample=latents,
    timestep=sigma,
    encoder_hidden_states=cond
)
```

This means:

-   conditioning is passed via `encoder_hidden_states`
-   conditioning must match expected UNet shape

------------------------------------------------------------------------

## Design Principles

### 1. Separation of Responsibilities

-   Parser → builds prompt schedules
-   Adapter → converts prompts into conditioning tensors
-   Pipeline → passes conditioning forward
-   Sampler → consumes conditioning

------------------------------------------------------------------------

### 2. Always Produce cond + uncond

Even if:

-   negative prompt is empty
-   scheduling is not used

------------------------------------------------------------------------

### 3. Keep Contract Narrow

This phase intentionally avoids:

-   SDXL complexity
-   multi-conditioning dicts
-   image conditioning
-   controlnet-style conditioning

------------------------------------------------------------------------

### 4. Future Compatibility

The contract is designed to expand later to support:

-   dict-based conditioning
-   pooled embeddings
-   multiple encoders
-   advanced prompt blending

------------------------------------------------------------------------

## Summary

### Required

-   `cond: torch.Tensor`
-   `uncond: torch.Tensor`

### Optional

-   `extra["resolver"]` for stepwise scheduling
-   pooled conditioning (future use)

### Key Rule

> Every generation must have valid positive (`cond`) and negative
> (`uncond`) conditioning tensors.

------------------------------------------------------------------------

## Next Steps (Future Work)

Planned extensions:

-   dict-based conditioning support
-   SDXL-style conditioning
-   multi-encoder pipelines
-   conditioning-aware schedulers

------------------------------------------------------------------------

## Status

This contract reflects the current implementation and should be treated
as the **authoritative reference** for conditioning behavior in the
txt2img pipeline.
