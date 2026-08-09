# ImageGen Feature Documentation

The `features/` folder is the user-facing feature layer for ImageGen.

It is intentionally separate from the chronological changelog and from internal engineering phase documents.

## Files

- [CURRENT.md](CURRENT.md) — what users can do in the current release.
- [NEW.md](NEW.md) — the most important recent user-facing additions and improvements.
- [LIMITATIONS.md](LIMITATIONS.md) — current unsupported areas, qualification boundaries, and alpha caveats.
- [UPCOMING.md](UPCOMING.md) — planned and near-future product work that is not yet supported.

## Documentation Rule

A feature belongs in **CURRENT** only after the release runtime actually implements it.

A feature belongs in **UPCOMING** while it is still a phase plan, research program, design target, or unfinished integration. Planned work should not be presented as if it is already available.

**NEW** is a short-lived highlight page, not a permanent history. Once an addition is no longer recent, its final supported behavior stays in **CURRENT**, while its original dated history remains under `changelog/`.
