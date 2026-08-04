# SuperHybrid REGION in IMAGE_GEN

IMAGE_GEN implements SuperHybrid REGION through a native model-output backend. It does not import the A1111 or Forge processing pipeline, replace the CFG denoiser, or monkey-patch attention.

## Basic syntax

Split an image horizontally:

```text
scene REGION{house@0,0.5,0,1 | whale@0.5,1,0,1}
```

Use automatic horizontal or vertical tiles:

```text
scene REGION{house | whale}:H
scene REGION{sky | mountains | ground}:V
```

Use proportional tiles:

```text
scene REGION{house | whale}:H:0.35,0.65
```

Coordinates are normalized when they are between `0` and `1`. Pixel coordinates are also accepted when they fit the requested generation dimensions.

## Branch settings

A branch may include a weight and an easing curve:

```text
scene REGION{house@0,0.5,0,1*1.2~ease-in | whale@0.5,1,0,1*0.8~sine-in-out}
```

Block directives include:

```text
*base=<prompt>
mode=overlay|common
backend=latent
start=<0..1>
stop=<0..1>
blur=<0..1>
base_ratio=<0..1>
canvas=<base64 PNG>
```

`mode=overlay` conditions a region from its branch text. `mode=common` conditions it from the base prompt followed by the branch text.

`base_ratio` defaults to `0.2`, matching the SuperHybrid latent backend. It mixes some base-prompt conditioning into the isolated regional branch to reduce overburning.

## Overlap policy

The SuperHybrid parser option `region_overlap_policy` accepts:

- `additive`: source-compatible regional delta accumulation. This is the default.
- `normalize`: scales overlapping masks so their total influence does not exceed one.
- `priority`: later declared regions replace earlier regions within overlaps.

## Runtime behavior

For every active logical sampling step, IMAGE_GEN:

1. Evaluates the unconditional branch.
2. Evaluates the base positive branch.
3. Evaluates each active regional branch sequentially.
4. Blends regional conditional outputs through latent-space masks.
5. Applies the canonical CFG or CFG Lab schedule once.
6. Continues through the selected sampler.

Sequential regional evaluation is deliberate. It avoids keeping all regional UNet predictions resident simultaneously and is the safest initial behavior for low-VRAM GPUs.

Supported sampler paths:

- KES
- Simple Euler
- DPM++ 2M

Base and hires passes build independent masks from their actual generation dimensions and logical step counts.

## Replay

Saved manifests include a separate REGION contract for the base and hires passes. The contract records:

- Parser and contract versions
- Base and regional prompts
- Coordinates and units
- Weights, curves, activation windows, blur, mode, and base ratio
- Canvas digest and dimensions
- Region semantic fingerprints
- Image dimensions and logical step count
- Overlap policy
- SHA-256 contract fingerprint

Exact replay verifies the complete contract. Prompt, geometry, parser semantics, dimensions, step count, or policy changes unlock replay and reconstruct the plan.

For multi-image batches, each saved image receives a projected one-slot REGION contract while the complete original batch contract remains available in output details.

## Current limitations

- REGION is positive-conditioning only. There is no region-specific negative prompt yet.
- A single canonical CFG schedule still applies to the whole latent batch.
- Canvas data must currently be embedded as base64 in the prompt contract; the standalone SuperHybrid region-builder canvas store is not yet wired into the WebUI.
- The backend performs additional sequential UNet calls: one regional call per active region per model evaluation.
- The first release targets the current SD1.x and SD2.x single-text-encoder pipeline. SDXL, Flux, SD3, and T5 conditioning require separate model-family contracts.
- An optimized Diffusers attention-processor backend is intentionally deferred until the model-output backend has passed GPU acceptance testing.
