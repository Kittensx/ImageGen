---
title: Theme Colors and Contrast
summary: Understand semantic theme colors, readability diagnostics, and optional contrast
  warnings.
category: Theme Manager
audience: user
status: current
keywords:
- theme
- contrast
- text
- muted
- surface
- color
related:
- theme_manager/overview
- theme_manager/local_theme_packages
- workspace/workspace_manager
featured: false
media: []
external_links: []
---

# Theme Colors and Contrast

Theme Manager measures text colors against the surfaces on which they may appear. Low-contrast themes are allowed, but IMAGE_GEN warns by default before you save or apply combinations that may make text difficult or impossible to read.

## Recommended text contrast

For normal readability, primary text and secondary/muted text should each have a contrast ratio of at least **4.5:1** against:

- the primary surface;
- the secondary surface;
- the component/card surface.

A text color that is exactly the same as one of those backgrounds is especially likely to be unreadable. Theme Manager reports it prominently, but does not forbid it.

When a relationship falls below the recommendation, the live preview and Save action remain available. With warnings enabled, IMAGE_GEN asks whether you are sure before saving or applying the theme. Declining the prompt returns you to the editor without changing your selected colors.

## Warning preference

The **Warn before saving or applying low-contrast themes** option is enabled by default. You can turn it off if low contrast is intentional. The preference persists across restarts. Contrast diagnostics still appear in Theme Manager even when confirmation prompts are disabled.

## UI contrast guidance

Borders, accents, focus treatments, and button text target at least **3:1** contrast where applicable. TM-02 reports these lower-contrast UI relationships as recommendations rather than rewriting your chosen colors automatically.

## Why IMAGE_GEN does not auto-correct colors

Theme Manager preserves intentional color choices. It reports potentially unreadable combinations and asks for confirmation by default rather than silently changing or rejecting your colors.
