---
title: Model Loading and Runtime Reuse
summary: Understand model preflight, component residency, cache reuse, CPU fallback, and unload behavior during generation.
category: Generation
audience: user
status: current
keywords:
- generation
- model loading
- preflight
- residency
- cache
- cpu fallback
- unload
related:
- setup/sdxl_support
- setup/sd3_sd35_support
- setup/advanced_models_component_composition
- generation/persistent_queue
featured: false
media: []
external_links: []
---

# Model Loading and Runtime Reuse

IMAGE_GEN validates the selected model and required runtime assets before generation begins. The exact checks depend on the model family and the selected generation configuration.

## Model preflight

Before the generation pipeline is assembled, IMAGE_GEN can apply model-family runtime profiles and verify architecture-specific requirements. This includes SD2.x, SDXL, SD3.x, and Advanced Models/component compositions where those paths are supported.

A preflight failure stops the request before sampling rather than continuing with a partially compatible runtime configuration.

## Reusing loaded model components

The runtime can keep hydrated model components available between requests so the same model does not always need to be loaded from storage again. Reuse depends on the active memory and retention settings, the selected model, and whether the existing cached components are compatible with the next request.

Changing to a different model or incompatible composition can require different components to be loaded.

## CPU-first and staged residency

Some memory policies intentionally keep cached components on CPU until the generation stage needs them. This is especially important for lower-VRAM execution and for model families or Advanced Models compositions that are designed to promote only the active working set to the GPU.

A component being cached does not necessarily mean that every cached component is currently GPU-resident.

## GPU fallback

When a CUDA-preferred runtime cannot use CUDA, supported paths may fall back to CPU and report the reason. A configuration that explicitly requires CUDA should fail instead of silently switching devices.

## Unloading the model cache

Clearing the model cache releases IMAGE_GEN's references to cached checkpoint components. When requested, movable components are first transferred to CPU and CUDA allocator caches are then released where possible.

Operating-system GPU monitors may still show memory associated with the CUDA process context or with other applications after an unload. IMAGE_GEN's runtime memory diagnostics are the better source for determining whether IMAGE_GEN-owned tensor allocations were released.

## Troubleshooting

If a model does not become generation-ready:

* Confirm the selected model path still exists and is readable.
* Review any architecture or runtime-asset preflight message.
* For SDXL or SD3.x, verify that the required runtime support files are installed.
* For Advanced Models, verify that the selected component composition is complete and compatible.
* If GPU memory is constrained, use a staged or lower-memory policy rather than assuming every cached component must remain on CUDA.
* If a previous model appears to remain resident unexpectedly, use the model unload/cache-clear control and review the reported component devices and memory telemetry.
