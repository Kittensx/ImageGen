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

## Prompt Parser — Brace Grouping

**Status: Experimental — behavior under revision**

The current prompt parser recognizes brace-group syntax and records typed group-local weights, but the intended semantic behavior is still being revised.

The design goal is for a brace group to bind its contained concepts more closely as a related semantic unit than ordinary comma-separated concepts. Current testing found that the implemented grouping behavior does not yet reproduce that intent reliably enough to call grouping finished.

Other recently updated prompt-parser behaviors — including relationship/owner scopes, structural terminators, typed numeric interpretation, schedules/alternates, nested scope handling, semantic inspection, and replay records — can be tested independently of the unfinished grouping correction.

Additional binding syntax is also being evaluated, but it is not documented here as a current feature until its behavior has been implemented and tested.

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
