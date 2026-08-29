# Experimental ImageGen Features

This page lists capabilities that are **implemented in the current source** but should not yet be treated as fully qualified or final. Experimental does not mean “planned”; these paths exist and can be tested, but their behavior, compatibility boundaries, or user experience may still receive corrective changes during alpha development.

## Asset Browser / Asset Hub Downloads

**Status: Experimental — active bug testing**

The Asset Browser is implemented for provider-backed model discovery, with Civitai as the first provider. Current code includes independent search sessions, local discovery/index filtering, preview staging, model/version/file inspection, download queue controls, transfer resume/recovery, verification, classification, automatic finalization into safe library destinations, quarantine handling, gallery caching, saved assets, and download history.

The full checkpoint/LoRA download lifecycle is still receiving active bug testing, especially around:

- provider paging and paused/resumed searches;
- preview retrieval and staging;
- model version and file selection;
- concurrent download limits and bandwidth controls;
- interrupted-transfer recovery and partial-file handling;
- post-download classification;
- automatic library installation;
- library reconciliation/provenance; and
- restart/recovery edge cases.

Keep backups of important model libraries and verify newly installed assets before relying on unattended asset management.

## Prompt Parser — Grouping and Attribute Binding

**Status: Experimental — active A/B image testing**

The current alpha intentionally exposes two grouping behaviors side by side:

```text
{...}   existing branch-average grouping control
⦃...⦄   experimental shared-context cohesive grouping
```

The cohesive-group candidate keeps all members present in the shared encoder context while locally reinforcing one member per weighted branch. This is being compared against the existing grouping behavior rather than being declared the replacement in advance.

The same experiment introduces two modifier/target bindings:

```text
modifier^target   target-only binding
modifier*target   target + structural-descendant binding
```

`^` does not propagate to descendants and acts as a barrier to an inherited `*` modifier. `*` establishes a structural-descendant scope; an explicit child `^` or `*` blocks the ancestor binding at that child, and child `*` begins a new inherited subtree scope.

Prompt Inspector can expose the selected grouping algorithm, members/local weights, binding operator/scope, inheritance barriers, and encoder-visible lowering. Escaped experimental symbols remain literal text.

These semantics are being judged by real image behavior, not parser structure alone. Qualification should compare multiple same-seed outputs for:

- whether modifiers remain attached to the intended target;
- unwanted color or concept leakage;
- preservation of surrounding context;
- composition stability; and
- diversity versus over-constraining the prompt.

Until those comparisons establish a preferred behavior, the existing grouping control, cohesive-group candidate, and both binding operators should all be treated as **experimental rather than final language guarantees**.

## SD3 / SD3.5 LoRA Architecture Groundwork

**Status: Unverified — not currently claimed as supported**

ImageGen contains architecture-aware standard-adapter mapping for SD3-family transformer and text-encoder targets. That groundwork is not equivalent to qualified SD3/SD3.5 LoRA support.

A suitable real SD3/SD3.5 LoRA has not yet been available for controlled end-to-end testing, so ImageGen currently makes no public claim that SD3-family LoRAs are supported. Future qualification may require creating controlled LoRA test assets so the runtime can be exercised against known training/output expectations.

## Canvas Expansion

**Status: Available — alpha / intermediate workflow**

Canvas Expansion is usable for enlarging a canvas while protecting the source composition, but continuation quality can still vary with model, prompt, geometry, denoising, and edge context. It remains an intermediate workflow rather than a general Img2Img replacement.

## Neural Hires

**Status: Available — alpha**

The pixel-neural Hires pipeline is active, but upscaler compatibility, very large targets, memory pressure, and architecture-specific secondary-pass combinations remain qualification-sensitive.
