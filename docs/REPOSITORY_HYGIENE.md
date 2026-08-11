# Repository hygiene and anti-bloat policy

The ApexOracle super-repository is an orchestration layer, not a storage
location for datasets, model checkpoints, embeddings, raw predictions, caches
or generated experiment directories. Large scientific assets belong on
Hugging Face or Zenodo and are referenced through immutable manifests.

## Audited baseline

The full six-tree audit on 2026-08-11 found:

| Repository | Tracked files | Tracked bytes | Main size driver |
| --- | ---: | ---: | --- |
| super-repo root | 55 | 1,953,345 | 1.69 MB hash-fixed README banner plus release/capsule orchestration |
| Core | 431 | 13,229,116 | 4.65 MB record-level PepLink reviewer table and compact reviewer evidence |
| DLM-Pretraining | 68 | 171,562 | source code |
| MDLM | 224 | 2,451,393 | tokenizer and compact ledgers |
| Evo2 | 79 | 46,379,792 | unchanged upstream notebook/example data |
| Generation | 119 | 602,935 | source code and configs |

No same-repository byte-identical files of at least 1 KiB and no cross-tree
byte-identical files of at least 20 KiB were found. Nine smaller cross-tree
groups are reported but are intentional: two license copies and the small
upstream runtime files that allow DLM-Pretraining and downstream MDLM to remain
independently installable repositories. They must not be replaced by an
implicit cross-submodule import.

The layout review also verified that all 16 tracked Core experiment directories
have their own README, all root orchestration scripts are referenced from
documentation/tests, and no active tree contains a nonignored untracked file.
Core concentrates 173 tracked files under `experiments/`; these are organized
by scientific question and mostly comprise compact reviewer evidence, not a
second implementation tree. MDLM keeps its callable package under
`src/apexoracle_mdlm/`, with migration evidence isolated under
`reproducibility/`. Generation keeps the two ApexOracle launchers under
`scripts/reproduce/`; the remaining root runtime and shell scripts are preserved
upstream interfaces. Evo2's size is dominated by unchanged upstream
notebook/example assets, each covered by an exact-path size exception.

The local root `.git` directory is larger than the active source tree because it
preserves the public legacy-monorepo branch and recovery tag. That history is
deliberate provenance and must not be rewritten merely to reduce a local clone.

## Current growth signals

The hard gate still passes. The checker now also emits non-failing alerts at
80% of a ceiling so growth is reviewed before a repository becomes difficult to
maintain:

| Repository | Signal | Decision |
| --- | --- | --- |
| super-repo root | 55/60 files (91.7%) | No duplicate payload is present. The increase is six compact benchmark/release/registry manifests and tests; keep new logic parameterized and do not add one script per new asset or Zenodo version. |
| Core | 431/500 files (86.2%) | Reuse an existing experiment directory for the same scientific question; full row tables stay external. |
| MDLM | 224/275 files (81.5%) | New guidance/scoring variants must be profiles of existing libraries, not copied trainers. |
| Evo2 | 46,379,792/52,428,800 bytes (88.5%) | Do not add further notebook/data payloads; ApexOracle additions remain source-only. |

Source files at or above 500 lines are also reported as review candidates. This
is not an automatic refactor instruction: several are upstream numerical
kernels or parity-frozen scientific runners. Split one only when a stable
responsibility boundary and behavior-preserving tests exist. File movement for
appearance alone is not worth invalidating a checked scientific path.

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
repository ceiling requires updating this document with the reason and the
new measured baseline.

Adding a new top-level directory also requires a deliberate policy update. In
normal work, new Core evidence belongs below an existing `experiments/<topic>/`,
new reusable implementation belongs below the module's existing package, and
new orchestration belongs below the existing `scripts/` categories.

## Public trees versus local producer workspaces

The six trees under `modules/` are the publication boundary and must remain
clean. The older producer workspaces may retain ignored datasets, checkpoints,
outputs, wheels and caches needed for scientific audit. In particular, the
downstream MDLM producer remains a large asset workspace by design. Those
ignored assets are not evidence of public-repository bloat and must not be
moved into a submodule merely to make the producer directory look smaller.

Any unfinished local Core reviewer work remains separate from the clean public
Core gitlink until it is complete. Local `dist/`, `build/` and `*.egg-info/` products are disposable
packaging outputs, never scientific source. The public Core and Generation
ignore rules keep regenerated paths out of normal Git status, while the
anti-bloat checker still rejects them if force-added as tracked content.

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
