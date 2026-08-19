---
title: SD3.x Hires Quality Guide
summary: Choose SD3 and SD3.5 hires settings that preserve a strong low-resolution composition while adding detail at a larger output size.
category: Generation
audience: user
status: current
keywords:
- sd3 hires
- sd3.5 hires
- hires fix
- flow match
- denoise
- neural upscaler
- bicubic
- upscale
- quality
related:
- setup/sd3_sd35_support
- generation/generation_pipeline_stages
- generation/replay_and_preflight
featured: true
media: []
external_links: []
---

# SD3.x Hires Quality Guide

SD3.x hires is designed for workflows where the base image is easier or safer to compose at a smaller resolution and the final image should be produced at a larger resolution without starting the whole composition again from noise.

A representative qualified workflow is:

```text
Base:          640 x 960
Hires target:  960 x 1440
Base steps:    20
Base CFG:      5.0
Hires steps:   20 active refinement steps
Sampler:       flow_euler
Scheduler:     flow_match_euler
Strategy:      pixel_neural
Correction:    bicubic for the qualified x4 -> 1.5x comparison
```

The base resolution is not a quality mistake. On hardware that cannot comfortably generate the larger canvas directly, or when a model becomes less compositionally reliable at larger starting dimensions, a smaller composition-safe base followed by hires is an appropriate workflow.

## Recommended starting point

For SD3.x and SD3.5, start with the architecture-qualified Flow pair:

```text
Sampler:    flow_euler
Scheduler:  flow_match_euler
```

Keep the fixed-active-step hires policy when you want the Hires Steps value to mean approximately the number of second-pass model evaluations:

```text
Step policy: a1111_fixed_steps_v1
Hires steps: 20
```

For a preservation-oriented neural upscale, begin around:

```text
Denoise: 0.20 to 0.30
```

We've directly validated `0.20` as a strong preservation-oriented result. The current Auto value of `0.30` sits between the qualified `0.20` and `0.45` cases and is intended as a balanced starting point.

## What denoise changes

Denoise is the most important user-facing control for how much the SD3 second pass is allowed to reinterpret the neural upscale.

### Very low denoise

A value around `0.01` preserves the incoming image extremely closely, but it is mainly useful as a diagnostic or near-no-op refinement reference.

With fixed-active-step semantics, very low strengths can require a very large internal schedule. In the HA6 reference case, 20 active steps at `0.01` required an internal 2000-step schedule. This is inefficient and normally unnecessary.

Avoid using extremely low denoise as a routine "maximum quality" setting. It mostly prevents the second pass from doing meaningful work.

### Preservation / balanced range

A value around `0.20` is a good preservation-oriented starting point for the qualified SD3.5 Medium portrait case.

Values around `0.20` to `0.30` are appropriate when the goal is:

- keep the original face, pose, framing, and large shapes;
- let the neural upscaler provide most of the spatial detail;
- allow SD3 to clean and integrate the enlarged image without substantially redrawing it.

### Stronger refinement

The corrected Flow-Match path also produced a high-quality result at `0.45`, but the image was visibly more reinterpreted. Fine dress patterns, facial presentation, hair detail, and background structure changed more than in the `0.20` case.

Use values around `0.40` to `0.45` when some redraw is acceptable or desirable. Treat values above the qualified range as experimental until they have been tested for the selected model and image type.

## Do not use denoise to compensate for a broken-looking image

Older SD3.x hires builds could become badly over-contrasted and smeared as denoise increased. That behavior was caused by incorrect Flow-Match image-conditioning noise preparation, not by an inherent requirement to keep SD3 denoise near zero.

Current builds use the Flow-Match forward-noise form:

```text
noisy latent = source latent * (1 - sigma) + noise * sigma
```

Do not compensate for an outdated or broken runtime by forcing denoise to `0.01`.

## Upscaler and target correction

The qualified HA6 reference used a neural x4 RRDB/ESRGAN-style upscaler even though the requested final enlargement was only 1.5x.

That combination is valid. The test showed that a native x4 neural result can be reduced to the requested 1.5x output and still retain strong detail.

For the tested photographic case:

```text
Bicubic correction -> slightly crisper
Area correction     -> slightly smoother
```

Bicubic is therefore the preferred starting correction filter for the qualified SD3 x4 -> 1.5x workflow.

Do not assume that an x4 upscaler is automatically wrong for a 1.5x output. Native scale, model character, correction filter, and final refinement all matter.

Upscaler models are safe to experiment with. When comparing them, keep the prompt, seed, base image, target dimensions, denoise, sampler, scheduler, and correction filter unchanged so the upscaler is the only meaningful variable.

## Settings that are safe to experiment with

These are reasonable user-controlled experiments when changed one at a time:

- **Denoise** — changes preservation versus redraw. Start around `0.20` to `0.30`.
- **Neural upscaler model** — different models may favor sharpness, texture, illustration, or photographic detail.
- **Target scale / target dimensions** — change the final output size while keeping a composition-safe base.
- **Area versus Bicubic correction** — Area is smoother; Bicubic was slightly sharper in the qualified photographic comparison.
- **Hires steps** — experiment after establishing a good denoise value. Keep 20 as the reference while comparing other variables.
- **Hires CFG** — modest experimentation is acceptable, but CFG is not a replacement for denoise or an image-sharpening control.

Change one major variable at a time when evaluating image quality.

## Settings to treat cautiously

### Sampler and scheduler

`flow_euler` + `flow_match_euler` is the qualified SD3.x pair. Other sampler/scheduler combinations should be considered experimental unless they explicitly support the SD3 Flow-Match denoising domain.

Do not select a conventional non-Flow scheduler simply because it works for SD1.x, SD2.x, or SDXL.

### Step policy

Use `a1111_fixed_steps_v1` when you expect Hires Steps to represent active second-pass refinement work.

The proportional-tail policy has different semantics. For example, an earlier 20-step / 0.40 test produced only 8 active refinement steps. It is not inherently invalid, but it should not be used when you expect "20 hires steps" to mean 20 active model evaluations.

### Extremely low denoise

Very low values can create unnecessarily large internal schedules under fixed-active-step semantics. Use them for diagnosis or deliberate near-preservation, not as the default quality strategy.

### High denoise

`0.45` is now technically valid and produced a clean result in the corrected HA6 reference, but it also caused substantially more redraw than `0.20`. Higher values are increasingly likely to change identity, texture, clothing detail, background elements, and other fine structure.

### CFG and CFG rescale

Do not raise CFG simply to make an image look sharper. Excessive guidance can change contrast, saturation, and prompt pressure rather than recover real detail.

`CFG Rescale` is a normalized value and must remain between `0.0` and `1.0`. A normal SD3 reference value is `0.0` unless a particular workflow has been separately qualified.

## Dimensions and SD3 spatial alignment

SD3's VAE and transformer impose a stricter spatial requirement than the traditional multiple-of-8 assumption used by many Stable Diffusion workflows.

IMAGE_GEN now applies architecture-aware SD3 spatial alignment automatically. SD3.x uses an 8-pixel VAE scale and a 2-cell transformer patch size, so its internal working canvas is aligned upward to a 16-pixel grid. The requested output size is preserved: IMAGE_GEN generates/refines on the slightly larger compatible canvas and center-crops only the alignment padding at the end.

For example, a 360x360 request runs internally at 368x368 and returns 360x360. A hires target such as 721x719 runs internally at 736x720 and returns exactly 721x719. You do not need to round SD3.x dimensions manually.

Examples:

```text
Requested 640 x 960   -> internal 640 x 960   -> output 640 x 960
Requested 960 x 1440  -> internal 960 x 1440  -> output 960 x 1440
Requested 360 x 360   -> internal 368 x 368   -> output 360 x 360
Requested 721 x 719   -> internal 736 x 720   -> output 721 x 719
```

The alignment expansion is internal only. IMAGE_GEN does not stretch the padded canvas back down; it center-crops the alignment border so the requested pixel scale is preserved.

## What the qualified pipeline should look like

For neural SD3 hires, quality should progress approximately like this:

```text
composition-safe base image
        -> neural upscale
        -> target-size correction
        -> SD3 VAE encode
        -> Flow-Match image conditioning
        -> second-pass SD3 refinement
        -> final output
```

A small amount of softness can be introduced by a VAE encode/decode round trip. In HA6 this loss was measurable but modest. The catastrophic quality loss seen before the Flow-Match correction was not normal VAE behavior.

## If the result looks wrong

If a final hires image is unexpectedly soft, over-contrasted, or heavily altered:

1. Return to `flow_euler` + `flow_match_euler`.
2. Use `a1111_fixed_steps_v1` and 20 hires steps.
3. Set denoise near `0.20`.
4. Use the same positive and negative prompts as the base pass while diagnosing.
5. Use the known neural upscaler and Bicubic target correction.
6. Replay with the same seed before changing another setting.

Once that reference looks correct, change one setting at a time.

## Practical rule of thumb

For ordinary SD3.x photographic hires:

```text
Want to preserve the base closely?  Start near 0.20.
Want a balanced cleanup/refine?      Start near 0.30.
Want meaningful redraw?              Try 0.40 to 0.45.
Want almost no redraw?                Very low denoise works, but is inefficient and usually unnecessary.
```

The right denoise value is an artistic choice after the runtime math is correct. Higher denoise is no longer automatically "wrong"; it simply gives SD3 more authority to reinterpret the enlarged image.
