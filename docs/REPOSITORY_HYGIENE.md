# Repository hygiene and anti-bloat policy

The ApexOracle super-repository is an orchestration layer, not a storage
location for datasets, model checkpoints, embeddings, raw predictions, caches
or generated experiment directories. Large scientific assets belong on
Hugging Face or Zenodo and are referenced through immutable manifests.

## Audited baseline

The post-strain-mapping audit found:

| Repository | Tracked files | Tracked bytes | Main size driver |
| --- | ---: | ---: | --- |
| super-repo root | 36 | 1,790,228 | 1.69 MB hash-fixed README banner plus release-policy files |
| Core | 425 | 13,020,436 | 4.65 MB record-level PepLink reviewer table |
| DLM-Pretraining | 68 | 171,562 | source code |
| MDLM | 224 | 2,450,056 | tokenizer and compact ledgers |
| Evo2 | 79 | 46,379,792 | unchanged upstream notebook/example data |
| Generation | 119 | 602,507 | source code and configs |

No byte-identical files of at least 20 KiB were found across the six active
trees. The local root `.git` directory is larger than the active source tree
because it preserves the public legacy-monorepo branch and recovery tag. That
history is deliberate provenance and must not be rewritten merely to reduce a
local clone.

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
- rejection of exact duplicate content at or above 20 KiB;
- reporting of nonignored untracked files and the five largest files per tree.

An exception must name one exact path, a narrow maximum size and a scientific
reason. A directory-wide wildcard is not an acceptable exception. Raising a
repository ceiling requires updating this document with the reason and the
new measured baseline.

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

The intended archival layout is a single versioned Zenodo capsule with
`tables/`, `splits/`, `predictions/`, `manifests/` and a README. The Git manifest
records the DOI/revision and per-file hashes; it does not mirror the payload.

## Working-tree rules

- Stage explicit paths; do not use `git add -A` for release work.
- Never copy a submodule implementation into the root repository.
- Prefer one parameterized exporter to multiple experiment-specific scripts.
- Generated test caches may exist locally but are ignored and must never be
  treated as scientific artifacts.
- Before every default-branch update run the release-tree, module-lock,
  repository-bloat and test gates.
