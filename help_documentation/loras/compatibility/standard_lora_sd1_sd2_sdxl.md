# Standard LoRA Runtime Compatibility

IMAGE_GEN classifies a LoRA by both its base-model family and its adapter tensor format. A matching SD1.x, SD2.x, or SDXL label is not enough by itself; the file must also use a runtime-qualified adapter representation and target only components supported by that family.

## Standard formats handled by the standard loader

The standard Diffusers/PEFT loader path is registered as:

```text
image_gen.adapter_loader.standard_diffusers.v1
```

It handles the conventional representations classified as:

```text
standard_kohya_lora
standard_diffusers_peft_lora
standard_lora_up_down
```

Conventional linear and convolutional LoRA targets can use this path when their converted keys map cleanly to supported components.

LyCORIS LoHa, LoKr, and algorithm-specific LoCon variants are not silently routed through the standard loader. They retain separate support states and require their own qualified runtime paths.

## Family and component targets

### SD1.x

The standard adapter layer supports:

```text
UNet
text encoder 1
UNet + text encoder 1
linear targets
convolutional targets admitted by the standard conversion path
```

`text_encoder_2` is not a valid SD1.x standard target.

### SD2.x

The standard adapter layer independently recognizes:

```text
UNet
text encoder 1
UNet + text encoder 1
linear targets
convolutional targets admitted by the standard conversion path
```

The LoRA loader does not apply SD1-specific text-encoder shape assumptions to SD2 adapter mapping.

Important: the current IMAGE_GEN base-model generation capability still blocks SD2 checkpoints until the OpenCLIP tokenizer/text-encoder conditioning path is implemented and qualified. Adapter compatibility and whole-model generation availability are separate checks.

### SDXL

The standard adapter layer supports explicit component targets for:

```text
UNet
text encoder 1 / TE1
text encoder 2 / TE2
combinations of UNet + TE1 + TE2
```

A dual-text-encoder adapter is not considered verified unless every expected text-encoder target is actually registered by the loader.

Important: the current IMAGE_GEN base-model generation capability still blocks SDXL checkpoints until the dual-tokenizer/dual-text-encoder conditioning path, pooled conditioning, time IDs, and SDXL UNet call contract are implemented and qualified. TE2 adapter-loader readiness does not by itself enable SDXL image generation.

## Weight behavior

IMAGE_GEN treats the LoRA weight as the user multiplier applied after the format loader's native rank/alpha normalization.

Runtime diagnostics record:

```text
source rank/dimension
source alpha when present
loader-native normalization ownership
requested user weight
effective user multiplier
final effective scale
```

For the standard Diffusers path, rank/alpha normalization remains owned by Diffusers/PEFT rather than being reimplemented independently by IMAGE_GEN.

Weights of `0` are retained as explicit stack weights. Negative weights are passed through when accepted by the active adapter backend.

## Stacking and unloading

Standard adapters can remain resident while the active stack changes. IMAGE_GEN can:

```text
activate one adapter
stack multiple adapters with independent weights
disable one while another remains active
reactivate a resident adapter
disable the full stack and return components to adapter-free behavior
```

Reweighting a resident adapter updates runtime diagnostics to the current multiplier rather than reporting the original load-time weight.

## Unsupported standard extensions

A file may use conventional PEFT LoRA tensors plus an extension such as a DoRA magnitude vector. IMAGE_GEN inventories that extension separately from the base LoRA format.

`dora_magnitude` is currently treated as not runtime-qualified. Such an adapter is reported as partial/blocked rather than being accepted merely because the installed PEFT version can parse its keys.

## Failure diagnostics

Standard-loader failures include bounded context for:

```text
loader ID
adapter format
model family
expected target scopes
failed target
mapping summary
small examples of unmapped keys
verification state
```

Unknown adapter keys are not silently discarded. A missing expected component, including SDXL TE2, is a verification failure.
