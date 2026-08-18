---
title: Replay and Preflight Validation
summary: How IMAGE_GEN validates recorded generation settings before replaying one or more outputs.
category: Generation
audience: user
status: current
keywords:
- replay
- preflight
- batch replay
- variation matrix
- validation
related:
- models/asset_catalog
- generation/persistent_queue
featured: false
media: []
external_links: []
---

# Replay and Preflight Validation

IMAGE_GEN validates replay requests before they are submitted to the generation queue. This prevents a recorded output from being silently rerun with missing models, unsupported settings, or a request that has changed since it was reviewed.

## What preflight checks

The exact checks depend on the workflow, but replay preflight can validate items such as:

- the recorded checkpoint and VAE
- sampler and scheduler selections
- prompt parser and prompt-profile information
- LoRA and other prompt assets
- recorded generation dimensions and seed behavior
- hires and Advanced Models information when present
- requested remaps or overrides

The WebUI presents validation errors and warnings before queue submission.

## Preflight tokens

After successful validation, IMAGE_GEN issues a short-lived server-side preflight token. The token refers to a protected copy of the specification that was validated; the client does not get authority to replace that specification after validation.

Preflight tokens expire after approximately 15 minutes. If a token expires, run preflight again before submitting the job.

A successfully submitted workflow discards its token. This applies consistently to single replay, batch replay, batch import, and variation-matrix submission.

## Single-output replay

Single replay reconstructs a generation request from one recorded output, validates the available assets and supported settings, then submits the revalidated request to the normal generation queue.

## Batch replay and imported queues

Batch workflows perform the same server-authoritative validation for multiple jobs. Invalid jobs remain visible for review; workflows that support "queue valid only" can submit the valid subset without silently treating invalid rows as successful.

## Variation Matrix

Variation Matrix preflight validates the generated plan before queue submission. Its preflight token protects the exact plan that was reviewed from being replaced between validation and submission.
