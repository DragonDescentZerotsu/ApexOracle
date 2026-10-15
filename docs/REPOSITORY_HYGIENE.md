# Repository hygiene and anti-bloat policy

The ApexOracle super-repository is an orchestration layer, not a storage
location for datasets, model checkpoints, embeddings, raw predictions, caches
or generated experiment directories. Large scientific assets belong on
Hugging Face or Zenodo and are referenced through immutable manifests.

## Automatic gate

Run:

```bash
python scripts/check_repository_bloat.py
```

The policy in `manifests/repository_size_policy.json` enforces:

- per-repository file-count and total-byte ceilings;
- a 1 MiB default maximum for any new tracked file;
- exact-path exceptions only for existing, documented large assets;
- rejection of checkpoint/cache suffixes and generated cache/build paths;
- rejection of same-repository exact duplicates at or above 1 KiB;
- rejection of cross-tree exact duplicate content at or above 20 KiB;
- an allowlist of expected top-level source directories for every tree;
- reporting of nonignored untracked files, top-level distribution, the five
  largest files, source files at or above 500 lines, and 80% soft-limit alerts.

An exception must name one exact path, a narrow maximum size and a scientific
reason. A directory-wide wildcard is not an acceptable exception. Raising a
repository ceiling requires documenting the reason and the new measured baseline.

Adding a new top-level directory also requires a deliberate policy update. In
normal work, new Core evidence belongs below an existing `experiments/<topic>/`,
new reusable implementation belongs below the module's existing package, and
new orchestration belongs below the existing `scripts/` categories.

## Paper-data capsule boundary

The paper-data capsule must remain external to Git. Git may contain only:

- schemas and compact manifests;
- deterministic exporters;
- split IDs or compact aggregate tables below the policy limit;
- checksums, row counts, expected metrics and download instructions.

The following remain external even when public:

- model-ready tables containing the full paper rows;
- sample-level predictions if they exceed the compact-file limit;
- genome/text/molecule embeddings;
- inference and training checkpoints;
- optimizer state, W&B data, raw generation attempts and caches.

The intended archival layout is one version series under the existing Zenodo
concept DOI `10.5281/zenodo.15612047`, with `tables/`, `splits/`,
`predictions/`, `manifests/` and a README. This is not a second Zenodo project.
The Git manifest records the concept DOI, exact version DOI and per-file hashes;
it does not mirror the payload.

## Working-tree rules

- Stage explicit paths; do not use `git add -A` for release work.
- Never copy a submodule implementation into the root repository.
- Prefer one parameterized exporter to multiple experiment-specific scripts.
- Generated test caches may exist locally but are ignored and must never be
  treated as scientific artifacts.
- Before every default-branch update run the release-tree, module-lock,
  repository-bloat and test gates.

## Local working documentation

Agent instructions, refactoring plans, manuscript response drafts and local working
records are excluded through each repository's `.gitignore`. Public documentation
contains installation, usage, scientific results and reproducibility contracts.
Local backups belong under `.local/`, which is excluded from version control.
The size-policy checker also rejects agent instructions and refactoring plans if
they are force-added to a release tree.
