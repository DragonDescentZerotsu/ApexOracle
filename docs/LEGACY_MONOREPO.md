# Legacy monorepo recovery

The public repository was converted in place rather than replaced by a second repository.

## Frozen recovery points

- source commit: `2f29dee9cf6b7750425414f66c1a2d67998cb87f`
- branch: `legacy-monorepo`
- annotated tag: `legacy-monorepo-snapshot-2026-08-10`
- tag object: `9da37c783ef238f53a9100cb4056bed2fb9ee16e`

The frozen tree contains 1,671 tracked files and historical copies of Core, DLM-Pretraining, MDLM, Generation,
PepLink, data, notebooks, and project-specific outputs. Those copies were removed from the active super-repo because
their canonical implementations and licenses are maintained elsewhere.

Inspect a historical file without changing the current checkout:

```bash
git show legacy-monorepo-snapshot-2026-08-10:ApexOracle/Readme.md
```

Create a detached recovery worktree:

```bash
git worktree add --detach /tmp/apexoracle-legacy \
  legacy-monorepo-snapshot-2026-08-10
```

The branch and tag preserve source recovery, but they also retain the historical repository size. The active
super-repo must not reintroduce those copied sources or binary assets.
