# Release provenance

This document separates four identifiers that must not be conflated:

1. the commit pinned by a super-repository release;
2. the commit at which scientific behavior was validated;
3. a source-recovery ref;
4. an external model or data revision.

## Current documentation closure

`v0.2.1` is a documentation-only patch over the validated `v0.2.0` release. Core, MDLM, and Generation advance to
commits that close stale plans and status records. DLM-Pretraining and Evo2 are unchanged. No model code, checkpoint,
condition tensor, protocol, or reference prediction is changed by this patch.

| Module | Default branch | `v0.2.1` lock | Validated scientific implementation |
| --- | --- | --- | --- |
| ApexOracle-Core | `main` | `3b6db2b70bb565cdb4df43e7ba4aafb6e52ca3cc` | `8c1def518ac148a878c14f4a39876db59649d43c` (`v0.1.0`) |
| ApexOracle-DLM-Pretraining | `main` | `362ffccac79bdd638a4e913c4f17df613da18f36` | same (`v0.1.0`) |
| ApexOracle-MDLM | `master` | `7c0bbd31f2bd9b1cc00c0a153d6987b96a037b6c` | `c9d17c7f6f091234aaaebf5f08dbe23542f980c1` |
| ApexOracle-Evo2 | `main` | `2184211acda07b0d5ca865067174ac42f530ad04` | same (`v0.6.0-apexoracle.1`) |
| ApexOracle-Generation | `main` | `2d648ce61da134faa211ad9fe7f1442ef8a40c60` | `80d9a2cf9b0921f29e4a44edf5557ccac39f5af9` |

The machine-readable current locks are authoritative in `manifests/modules.lock.yaml`. `v0.2.0` remains the first
complete scientific release and pins Core `8c1def5`, DLM-Pretraining `362ffcc`, MDLM `c9d17c7`, Evo2 `2184211`,
and Generation `80d9a2c`.

## Source recovery

| Scope | Public recovery ref | Resolved source commit |
| --- | --- | --- |
| Historical ApexOracle monorepo | branch `legacy-monorepo` and annotated tag `legacy-monorepo-snapshot-2026-08-10` | `2f29dee9cf6b7750425414f66c1a2d67998cb87f` |
| Core pre-refactor source | annotated tag `legacy-code-snapshot-2026-07-17` | `a68707c23699e7d79b9b0096f106d845459fd9fd` |
| DLM-Pretraining original producer | annotated tag `legacy-code-snapshot-2026-08-10` | `fda167cf5fb90ac57952482fb5c0e605b188c105` |
| MDLM author legacy source | annotated tag `legacy-code-snapshot-2026-08-09` | `79eed10cac8d5feb446be886eee0c5b356b23b06` |
| Evo2 fork base | upstream base `53f195997257c56c00e5ef8d33a54f5baad143a6` | same |
| Generation author legacy source | annotated tag `legacy-code-snapshot-2026-08-10` | `2368c25ce831c187e5b2699b85a6ae1a4cdca31a` |

Recovery refs preserve source, not ignored checkpoints, datasets, caches, W&B runs, or raw outputs. The expanded
GitHub Release archive is the canonical way to download one source-only tree containing all five fixed modules;
GitHub's automatically generated source ZIP does not expand submodules.

## Public external assets

| Purpose | Repository | Immutable revision | Integrity record |
| --- | --- | --- | --- |
| DLM molecule embedding runtime | `Kiria-Nozan/ApexOracle` | `77694f08c1d0664fdb24c5a7bab130c8a3bc2eda` | `manifests/model_weights.yaml` |
| Core MIC single-member quickstart | `Kiria-Nozan/ApexOracle-Core` | `459026cf4ae4e4e38ce5d2cae16ee3871d0a81df` | model manifest plus `manifests/model_weights.yaml` and `manifests/data_assets.yaml` |
| Compact BAA-3170 generation smoke | `Kiria-Nozan/ApexOracle-Generation` | `2fb1aa08187eaa359263be6c12c8a41868d8959c` | Hub `manifest.json` plus both root asset manifests |

The compact generation release contains three inference-only checkpoints and two condition tensors. Its purpose is
runtime verification. A complete SELFIES from the smoke is not evidence of deterministic generation, candidate
yield, antimicrobial activity, or experimental validation.

## Intentionally external or incomplete scopes

- The public MIC quickstart is one ensemble member; paper metrics use the frozen seven-member ensemble.
- The public generation bundle contains BAA-3170 only; the full paper condition bank is not redistributed here.
- Full curated training datasets, private assay records, caches, raw outputs, and training checkpoints are not Git
  objects in the super-repository.
- There is currently no ApexOracle Zenodo record or software DOI; citation metadata must not imply otherwise.

## Documentation audit

The 2026-08-10 final audit checked public default branches, release tags, recovery refs, module locks, Hugging Face
revisions, per-file quickstart hashes, clone/install boundaries, and source-archive policy. It corrected stale
"pending" states in Core, MDLM, and Generation, and moved the already released quickstart condition tensors out of
the pending data list. Validation commands are recorded in `docs/RELEASE_STATUS.md` and each owning module.
