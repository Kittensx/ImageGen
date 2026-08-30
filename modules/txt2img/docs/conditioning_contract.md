# Prompt Conditioning Contract

## Status

This document reflects the PPSR-07 conditioning runtime. Prompt parsing is model-neutral; model-family runtimes explicitly declare which conditioning channels are safe to compose.

## Core output

`PromptConditioningAdapter` produces `ConditioningOutput` with:

```python
cond: torch.Tensor
uncond: torch.Tensor
pooled_cond: torch.Tensor | None
pooled_uncond: torch.Tensor | None
prompt_schedules: dict
extra: dict
```

Positive and negative conditioning are always present. An empty negative prompt is encoded as `""` rather than represented by a missing tensor.

## Semantic parser boundary

PPSR compiles prompt syntax into a model-neutral conditioning plan before text encoding. Structural controls such as:

```text
{}
::
:::
!
!!
[a:b:n]
[a|b]
```

must not reach CLIP/T5 unless explicitly escaped as literal user text.

The semantic hierarchy is resolved in this order:

```text
member encoder outputs
  -> group-local composition
  -> sequence / relationship-local composition
  -> outer / AND composition
```

Every scope normalizes only its own local weights.

## Runtime capability contract

IMAGE_GEN-owned text-conditioning runtimes expose:

```python
semantic_conditioning_capabilities() -> SemanticConditioningCapabilities
```

Contract version:

```text
image-gen-semantic-conditioning-capabilities-v1
```

The declaration includes:

```text
architecture
runtime_name
output_kind
composable_fields
required_fields
supports_group_conditioning
supports_sequence_conditioning
supports_temporal_conditioning
supports_pooled_conditioning
unsupported_structured_fields
t5_policy
safe_flatten_supported
```

The parser never infers these capabilities from filenames.

## Qualified family channels

### SD1.x / local CLIP

```text
cross_attention
```

Tensor conditioning is composed with the PPSR hierarchy.

### SD2.x / OpenCLIP

```text
cross_attention
```

The qualified 77-token / 1024-wide OpenCLIP runtime uses the same PPSR hierarchy.

### SDXL

```text
cross_attention
pooled
```

Both channels are composed with the same semantic member weights. Token conditioning and pooled conditioning from different semantic members must never be mixed by index.

### SD3 / SD3.5

```text
cross_attention   # combined CLIP-L/G + T5 sequence
pooled            # combined CLIP-L/G pooled projection
```

When T5 is disabled, its zero replacement sequence remains zero after semantic composition. When T5 is enabled, each branch uses the same branch text and the same PPSR weight as the CLIP channels.

## Structured BREAK contract

Structured runtimes must opt into forced `BREAK` through an explicit
`encode_chunk_break_conditioning(segments, full_prompt=...)` model-family hook.
The generic parser does not infer pooled-vector reduction.

For SDXL, each BREAK segment becomes a native 77-position CLIP chunk in the
cross-attention sequence, while `pooled` is encoded once from the complete
lowered branch.

For SD3/SD3.5, BREAK is applied to the CLIP-L/G fixed-context contribution only.
T5, when enabled, is encoded once from the complete lowered branch; when disabled,
only one zero-T5 replacement sequence is appended. `pooled` is also encoded once
from the complete lowered branch. This keeps CLIP chunk semantics from being
projected onto T5.

Because SD3 BREAK can make positive and unconditional sequence lengths differ,
ordinary SD3 CFG uses sequential U/P transformer evaluations when the sequence
lengths differ, while retaining the existing concatenated fast path when they
match.

## Structured payload validation

If a runtime returns a dict, every `required_field` must be present. A structured field not declared in either `composable_fields` or `unsupported_structured_fields` is rejected rather than silently dropped. This forces new model-family conditioning fields to receive an explicit composition policy.

## Safe fallback

If a runtime explicitly declares that it cannot preserve a required group, sequence, or temporal operation, PPSR does not leak control punctuation and does not pretend full support.

The runtime adaptation layer:

1. renders a punctuation-safe flat representation;
2. resolves temporal controls to a stable punctuation-free concept set when necessary;
3. encodes one flat branch;
4. records `model_family_safe_flatten` and the degradation reason.

Unknown third-party/custom runtimes without a capability declaration retain legacy compatibility behavior; IMAGE_GEN does not guess their architecture from names or dimensions.

## Step resolver

`conditioning.extra["resolver"]` implements:

```python
resolve(step_index) -> (cond, uncond)
```

For SDXL/SD3, `conditioning.extra["pooled_resolver"]` resolves the same parsed schedules and semantic weights for pooled conditioning.

Temporal schedule boundaries are resolved against the active pass step count, including hires passes.

## Weight rule

Typed weights are local to their grammar scope. Example:

```text
{cat:2,dog:1}:0.5 AND bird:1.5
```

resolves as:

```text
group outer share = 0.5 / (0.5 + 1.5) = 0.25
cat inside group  = 2/3
 dog inside group = 1/3
bird outer share  = 1.5 / 2.0 = 0.75

final cat  = 0.25 * 2/3 = 1/6
final dog  = 0.25 * 1/3 = 1/12
final bird = 3/4
```

PPSR-07 tests these effective final contributions through the actual hierarchical resolver, not only through parser metadata.

## Key invariants

```text
{standing} ~= standing
```

on every qualified model family.

For a multi-member group, all required composable conditioning channels use the same normalized semantic weights.

No branch expansion may gain additional influence merely because it produced more encoder texts.
