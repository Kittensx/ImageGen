---
title: Asset Classification and Installation
summary: How verified files are inspected, classified, routed, and atomically installed.
category: Asset Hub
audience: user
status: current
keywords:
- classification
- installation
- lora
- checkpoint
- upscaler
- vae
related:
- asset_hub/installation/conflicts_and_quarantine
- asset_hub/provenance/asset_provenance
- asset_hub/downloads/resume_and_verification
featured: false
media: []
external_links: []
---

# Asset Classification and Installation

After a download is verified, Asset Hub creates an **install plan** before changing a live model directory.

The plan records the provider file identity, verified staged artifact, proposed asset kind, proposed destination, classification method, conflict policy, warnings, and whether review is required.

## Automatic classification

IMAGE_GEN reuses its existing inspectors instead of trusting a provider label alone.

* **LoRA:** safetensors metadata and tensor keys are inspected with IMAGE_GEN's LoRA inspector.
* **Checkpoint:** safetensors checkpoints are inspected with IMAGE_GEN's checkpoint inspector. Installing a checkpoint does not activate or unload the current generation model.
* **Upscaler:** `.pth` files use the existing safe ESRGAN/RealESRGAN classifier and upscaler discovery cache.
* **VAE:** automatic routing currently requires a recognizable safetensors VAE key layout.
* **Textual inversion:** automatic routing currently requires a safely readable safetensors payload.
* **Unknown/unsafe formats:** IMAGE_GEN quarantines rather than guesses.

## Configured destinations

Final destinations come only from IMAGE_GEN's `ProjectContext`. Asset Hub does not invent paths from the working directory.

## Atomic installation

Before committing a live file, IMAGE_GEN rechecks the staged hash. It copies the payload to a temporary file inside the destination directory, flushes it, verifies the copied SHA-256, and then uses an atomic replace/rename operation.

After the file is committed, IMAGE_GEN writes provenance, registers the final SHA-256 in the existing asset registry, and refreshes the existing catalog for that asset type.
