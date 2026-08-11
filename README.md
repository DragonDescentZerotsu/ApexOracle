# ApexOracle

[![Embedding data DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15612048.svg)](https://doi.org/10.5281/zenodo.15612048)

ApexOracle is the lightweight entry point for the modular ApexOracle release. Scientific implementations remain in
independently versioned Git submodules; this repository owns module locks, environment guidance, asset manifests,
bootstrap checks, and cross-module quickstarts.

The complete public release is available from
[GitHub Releases](https://github.com/DragonDescentZerotsu/ApexOracle/releases). `v0.2.0` freezes the validated
scientific implementation and first expanded five-module archive; `v0.2.1` is a documentation-only closure that
updates module plans, asset manifests, recovery records, and the expanded archive without changing model behavior.
`v0.2.2` corrects the previously omitted paper-listed Zenodo embedding dataset and its integrity metadata; scientific
implementations and protocols are unchanged.

## Current release status

The in-place conversion from the historical monorepo is complete. All five validated modules are locked:

| Module | Role | Locked commit |
| --- | --- | --- |
| [ApexOracle-Core](https://github.com/DragonDescentZerotsu/ApexOracle-Core) | prediction, training/evaluation contracts, and reproducibility workflows | `1949350` |
| [ApexOracle-DLM-Pretraining](https://github.com/DragonDescentZerotsu/ApexOracle-DLM-Pretraining) | collaborator-developed DLM + 209-descriptor MTR producer | `362ffcc` |
| [ApexOracle-MDLM](https://github.com/DragonDescentZerotsu/ApexOracle-MDLM) | downstream embedding, guidance heads, and candidate scoring | `7c0bbd3` |
| [ApexOracle-Evo2](https://github.com/DragonDescentZerotsu/ApexOracle-Evo2) | record-aware genome embedding extraction | `2184211` |
| [ApexOracle-Generation](https://github.com/DragonDescentZerotsu/ApexOracle-Generation) | guided discrete diffusion and remasking | `2d648ce` |

Every gitlink is pinned by its full commit in `manifests/modules.lock.yaml`. The public MIC quickstart uses an
inference-only single member for a runnable smoke; paper metrics continue to require the frozen ensemble. Exact
release locks, scientific implementation commits, recovery refs, and public asset revisions are separated in
`docs/RELEASE_PROVENANCE.md`.

## Data and archived embeddings

The paper's Code availability statement links Zenodo record
[`15612048`](https://doi.org/10.5281/zenodo.15612048), published as *URGE precomputed embeddings*. It contains
`Genome_embs.tar.gz` and `Text_Description.tar.gz` under CC BY 4.0. This is the DOI for the precomputed embedding
dataset used by ApexOracle, not a DOI for the GitHub software snapshot. Exact sizes and both Zenodo MD5 and independently
verified SHA-256 checksums are recorded in `manifests/data_assets.yaml`.

## Clone

```bash
git clone --recurse-submodules https://github.com/DragonDescentZerotsu/ApexOracle.git
cd ApexOracle
python scripts/check_module_locks.py
```

If the repository was cloned without submodules:

```bash
./scripts/bootstrap.sh
```

Build a single source-only archive containing all five fixed modules:

```bash
python scripts/build_source_archive.py --output ApexOracle-source.tar.gz
```

The builder also writes JSON provenance and a SHA-256 sidecar. It does not include Git metadata, model weights,
datasets, embeddings, caches, or raw outputs.

```bash
python scripts/check_source_archive.py ApexOracle-source.tar.gz
```

Each module retains its own environment and license. See `environments/README.md`, `NOTICE`, and the license inside each
submodule. Model weights and datasets are never stored as Git objects; released and pending assets are recorded under
`manifests/`.

## Layout

```text
ApexOracle/
├── modules/
│   ├── core/             # prediction, evaluation, and reproducibility contracts
│   ├── dlm_pretrain/     # ready collaborator DLM + MTR producer
│   ├── mdlm/             # ready
│   ├── evo2/             # ready
│   └── generation/       # ready
├── quickstarts/
├── environments/
├── manifests/
├── scripts/
└── docs/
```

The pre-conversion tree is preserved at branch `legacy-monorepo` and annotated tag
`legacy-monorepo-snapshot-2026-08-10`.
