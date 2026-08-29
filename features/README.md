# ImageGen Feature Documentation

The `features/` folder is the user-facing feature layer for ImageGen.

It is intentionally separate from the chronological changelog and from internal engineering phase documents.

## Files

- [CURRENT.md](CURRENT.md) — what users can do in the current source, including clearly labeled alpha/experimental capabilities.
- [NEW.md](NEW.md) — the most important recent user-facing additions and improvements.
- [LIMITATIONS.md](LIMITATIONS.md) — current unsupported areas, qualification boundaries, and alpha caveats.
- [EXPERIMENTAL.md](EXPERIMENTAL.md) — implemented capabilities that remain under active bug testing or qualification.
- [UPCOMING.md](UPCOMING.md) — planned and near-future product work that is not yet supported.

## Status Vocabulary

ImageGen documentation uses the following public status language:

- **Available** — implemented and part of the current supported runtime path.
- **Available — alpha** — implemented and usable, but still subject to normal alpha change and qualification limits.
- **Experimental** — implemented and available for testing, but active bug testing or behavioral qualification is still underway.
- **Unverified / qualification pending** — architecture or runtime groundwork exists, but ImageGen does not yet claim end-to-end support because representative real-world validation is missing.
- **Planned** — not implemented as a current user capability.

## Documentation Rule

A feature belongs in **CURRENT** only after the release runtime actually implements it. An implemented feature may also appear in **EXPERIMENTAL** when it is still undergoing active testing or qualification.

A feature belongs in **UPCOMING** while it is still a plan, research program, design target, or unfinished integration. Planned work should not be presented as if it is already available.

**NEW** is a short-lived highlight page, not a permanent history. Once an addition is no longer recent, its final supported behavior stays in **CURRENT**, while its original dated history remains under `changelog/`.
