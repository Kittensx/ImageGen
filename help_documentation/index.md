---
title: IMAGE_GEN Help Documentation
summary: Browse public IMAGE_GEN help by feature, workflow, and troubleshooting topic.
category: General
audience: user
status: current
keywords:
- help
- documentation
- guides
- search
related:
- home/help_center
- workspace/workspace_manager
- asset_hub/index
- theme_manager/overview
featured: true
media: []
external_links:
- label: IMAGE_GEN public repository
  href: https://github.com/Kittensx/ImageGen
---

# IMAGE_GEN Help Documentation

This folder contains public, user-facing IMAGE_GEN help. It is intentionally separate from the private/internal `docs/` development workspace.

## Categories

* [Asset Hub](asset_hub/index.md)
* [Stable Diffusion XL Runtime Support](setup/sdxl_support.md)
* [Stable Diffusion 3 and SD3.5 Support](setup/sd3_sd35_support.md)
* [Advanced Models and Component Composition](setup/advanced_models_component_composition.md)
* [Model / Component Registry](models/model_component_registry.md)
* [Model and Asset Catalog](models/asset_catalog.md)
* [Replay and Preflight Validation](generation/replay_and_preflight.md)
* [Theme Manager](theme_manager/overview.md)
* [Persistent Generation Queue](generation/persistent_queue.md)
* [Model Loading and Runtime Reuse](generation/model_loading_and_runtime_reuse.md)
* [Generation Pipeline Stages](generation/generation_pipeline_stages.md)
* [Prompt Tools Frontend](generation/prompt_tools_frontend.md)

More categories are added as their public workflows stabilize.


## Prompt parser numeric interpretation

Prompt Inspector now labels structured prompt numbers by their actual semantic role, such as attention weight, group member weight, absolute schedule step, fractional/percent boundary, legacy inferred step, or quantity. The same number can have different meanings in different grammar positions; ImageGen no longer treats integer-versus-decimal spelling as a universal rule.

For developer/parser qualification, `run.bat parser-test` remains model-free and now prints a readable input-prompt -> typed-meaning -> conditioning-plan preview.

## Prompt schedules, alternates, and nesting

Standard schedule forms such as `[cat:dog:2]`, `[cat:dog:0.5]`, and `[cat:dog:50%]` are now compiled into concrete per-step conditioning before text encoding. Standard alternates such as `[cat|dog]` use a deterministic one-based step cycle.

These temporal forms can be nested inside structured groups and relationships. For example, `{[red:blue:2] hair, green eyes}` keeps a two-member semantic group while only the first member changes from red to blue. Base and hires passes resolve fractional/percentage boundaries against their own active step counts. Escaped bracket syntax remains literal.

The model-free `run.bat parser-test` report includes `temporal_translation.txt`, which shows the resolved schedule/alternate behavior step by step for manual parser review.

## Deep grouped parent scope

Grouping changes ownership as well as local composition when a structured sequence is placed to the left of `:::`.

```text
{lake:sky:clouds}:::X
```

means the complete equal/local-weight `lake -> sky -> clouds` composition acts as the parent of `X`. Without the outer braces:

```text
lake:sky:clouds:::X
```

the legacy sequence still contributes locally, but its terminal member `clouds` is the attachment owner for `X`.

Groups are also recognized recursively in nested relation text. For example `{pink, blue, yellow} sky` remains a three-member color group when used as a relation child rather than becoming literal brace text.

The model-free `run.bat parser-test` output includes `parent_scope_translation.txt`, which shows the grouped and ungrouped forms side by side with their encoder-visible branches.

## Effective prompt weights across model families

Structured prompt weights are local to the scope where they are written. ImageGen now validates the final effect of those weights across its SD1.x, SD2.x, SDXL, and SD3/SD3.5 conditioning paths.

For example:

```text
{cat:2,dog:1}:0.5 AND bird:1.5
```

means the `{cat,dog}` group receives one quarter of the outer composition and `bird` receives three quarters. Inside the group, `cat` receives two thirds and `dog` one third. The final effective contributions are therefore:

```text
cat  = 1/6
dog  = 1/12
bird = 3/4
```

The parser-only `run.bat parser-test` output now includes `effective_weight_translation.txt`, which shows local weights and final effective contributions for several nested examples. This review does not load a model checkpoint or generate an image.


## Prompt semantic replay and inspection

New prompt metadata records the semantic PromptIR and conditioning plan with stable semantic/structure digests. Exact replay uses the recorded semantic structure and verifies its digest instead of reinterpreting punctuation or re-guessing numeric meaning.

Prompt Inspector includes a Semantic Structure view for groups, owner relationships, schedules, fallbacks, encoder-visible text, and static effective-final weights. Temporal weights are labeled dynamic by step.

The normal `run.bat parser-test` gate remains model-free. Real-image cutover qualification is opt-in through `testing\test_validations\qualification\generation\ppsr08_prompt_parser_image_qualification.bat` and writes timestamped `images/`, `requests/`, and `logs/` evidence plus a contact sheet.
