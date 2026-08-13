---
title: Local Theme Packages
summary: Import, validate, activate, disable, and remove offline theme packages.
category: Theme Manager
audience: user
status: current
keywords:
- theme package
- import
- local
- library
- appearance
related:
- theme_manager/overview
- theme_manager/colors_and_contrast
- home/help_center
featured: false
media: []
external_links: []
---

# Local Theme Packages

TM-02 supports offline local theme packages. A package is a ZIP-compatible archive containing a `manifest.json`, semantic token files, and optional visual assets.

## Import workflow

1. Open Theme Manager.
2. Choose **Import theme package**.
3. Select the local package archive.
4. IMAGE_GEN validates the manifest, capabilities, package paths, and visual assets, and records text-contrast readability warnings.
5. A valid package is copied into IMAGE_GEN's external theme library and appears in the local package list.
6. The package remains disabled until you explicitly choose **Use**. If it has low-contrast text and warnings are enabled, IMAGE_GEN asks for confirmation before activation.

Simply opening or importing a package never activates it automatically.

## Security rules

Local theme packages are treated as visual data, not executable extensions. Executable/script content such as JavaScript, Python, shell/batch files, DLLs, and executables is rejected. Path traversal, symbolic links, and unsafe SVG content are rejected as well.

Optional scoped CSS is accepted only when the package declares the corresponding capability. Ordinary card, shell, text, border, and accent customization should use semantic theme tokens rather than arbitrary CSS.

## Storage

Installed packages, activation state, cache files, and previews live in relocatable theme storage outside the IMAGE_GEN source tree. Removing a package does not delete your separate custom/legacy palette.

## Recovery

If an active package is missing or becomes corrupt, IMAGE_GEN records a diagnostic, disables the broken package, and falls back to the user's lower-layer/custom palette so the WebUI can continue starting normally.
