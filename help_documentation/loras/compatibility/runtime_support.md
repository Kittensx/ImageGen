# LoRA Runtime Compatibility

IMAGE_GEN inspects Safetensors adapters before generation so it can distinguish the **base-model family** from the **adapter format**. These are separate compatibility checks: an adapter can target the correct checkpoint family and still require a loader that is not available in the current build.

## What the LoRA details show

The LoRA workspace reports four separate pieces of technical status:

- **Family** — the detected Stable Diffusion architecture family, such as SD1.x, SD2.x, or SDXL.
- **Format** — the adapter representation detected from Safetensors metadata and tensor keys.
- **Targets** — the components the adapter is expected to modify, such as UNet, text encoder, text encoder 2, linear layers, or convolution layers.
- **Runtime Support** — whether the current IMAGE_GEN build has a qualified loader for that format, family, and target combination.

Technical file evidence takes priority when it conflicts with provider or sidecar family metadata. Provider metadata is still retained as supporting evidence rather than silently replacing what the file itself indicates.

## Support states

- **Supported** — the selected loader is qualified for the detected format and targets.
- **Supported with warning** — the adapter can be loaded, but some information such as the family or target scope could not be fully proven.
- **Partial** — only part of the adapter's detected targets are currently qualified. IMAGE_GEN blocks it by default rather than silently applying only part of the file.
- **Unsupported** — the file is a valid adapter, but this build does not provide a qualified loader for its format.
- **Misclassified** — the file looks technically like a checkpoint/full model rather than a LoRA adapter.
- **Invalid** — the file could not be inspected as a valid supported Safetensors adapter file.

## LyCORIS formats in Phase 1

Phase 1 can identify LyCORIS variants such as **LoHa** and **LoKr**, but it does not add those runtime algorithms. A valid LoHa file can therefore show a compatible SD1.x family while also showing **Unsupported in this build**. That is different from a model-family mismatch.

There is no generic fallback loader for unsupported formats. This avoids late conversion failures and prevents IMAGE_GEN from guessing at an incompatible loading path.

## Checkpoint-like files in the LoRA library

If inspection finds checkpoint-style tensor groups and no recognized adapter parameter groups, the file is marked as a likely full model/checkpoint and is not loadable as a LoRA. Generation is rejected during adapter preflight before checkpoint hydration.

If a download provider labels such a file as a LoRA, Asset Hub treats the technical mismatch as a review/reclassification condition instead of silently committing it as a normal LoRA.

## Rescanning older LoRA metadata

Older scan-cache records keep useful hashes and known technical fields. Records created before the Phase 1 inspection contract are eligible for a bounded technical rescan so IMAGE_GEN can populate the newer Format, Targets, and Runtime Support fields without treating an old `Unknown` value as permanent.
