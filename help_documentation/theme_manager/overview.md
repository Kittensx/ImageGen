---
title: Theme Manager
summary: Customize IMAGE_GEN using semantic colors, surfaces, text roles, and local
  theme packages.
category: Theme Manager
audience: user
status: current
keywords:
- theme
- appearance
- colors
- surface
- shell
related:
- theme_manager/colors_and_contrast
- theme_manager/local_theme_packages
- home/help_center
featured: true
media: []
external_links: []
---

# Theme Manager

Theme Manager controls IMAGE_GEN's shared appearance. TM-02 expands the original accent and surface palette into semantic appearance roles so the same theme can be applied consistently across the WebUI without editing individual CSS selectors.

## Appearance roles

The editor distinguishes several jobs that colors perform:

- **Accent** — primary emphasis and interactive identity.
- **Primary surface** — the main application/page background.
- **Secondary surface** — recessed or supporting surfaces.
- **Component surface** — cards, panels, and component shells.
- **Component border** — the outline of cards and shells.
- **Component accent** — emphasis inside component shells.
- **Primary text** — normal high-emphasis text.
- **Secondary text** — muted descriptions, hints, metadata, and explanatory text.

These are semantic roles. A theme changes the role once and compatible components consume that role consistently.

## Safe editing

Theme Manager previews changes before they are saved. Text contrast is measured against the surfaces where text can render. Low-contrast combinations are allowed, but IMAGE_GEN warns before Save/Apply by default; that confirmation can be disabled in the Theme Manager GUI. See [Colors and contrast](colors_and_contrast.md).

## Local packages

Theme packages can be imported from a local ZIP-compatible package. Importing validates and installs the package but does not activate it automatically. See [Local theme packages](local_theme_packages.md).
