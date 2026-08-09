# ImageGen Changelog

ImageGen keeps chronological change history in this folder instead of appending an ever-growing changelog to the root README.

## Current Entry

- [2026-08-08 — Current Alpha Development Snapshot](2026-08-08.md)

## Changelog Policy

Use one file per meaningful public update or release snapshot.

Recommended filename format:

```text
YYYY-MM-DD.md
```

If multiple public releases occur on the same date, add a short release suffix rather than combining unrelated releases into one oversized file.

Examples:

```text
2026-08-08.md
2026-08-15.md
2026-08-15-hotfix.md
```

Each changelog should describe **what changed in that update**. It should not become the permanent feature reference.

Use the documentation layers like this:

```text
README.md
    installation + first run + concise support table

features/CURRENT.md
    what the current program can do

features/NEW.md
    recent user-facing highlights

features/UPCOMING.md
    planned work not yet available

features/LIMITATIONS.md
    current boundaries and unsupported areas

changelog/YYYY-MM-DD.md
    chronological changes for one update
```

## Writing Rules

- Put newest entries first in this index.
- Separate new features, improvements, fixes, compatibility changes, and known limitations when useful.
- Do not copy the complete current feature inventory into every changelog.
- Do not describe a planned feature as released.
- Prefer user-visible behavior over internal phase numbers in public changelogs.
- Link to `features/CURRENT.md` when readers need the full current-state description.
- Keep internal validation/test-only work out of public changelogs unless it changes a user-visible guarantee.

A reusable starting point is available in [_TEMPLATE.md](_TEMPLATE.md).
