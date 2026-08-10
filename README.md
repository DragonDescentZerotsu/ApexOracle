# ApexOracle

ApexOracle is the lightweight entry point for the modular ApexOracle release. Scientific implementations remain in
independently versioned Git submodules; this repository owns module locks, environment guidance, asset manifests,
bootstrap checks, and cross-module quickstarts.

## Current release status

The in-place conversion from the historical monorepo has started. Three validated modules are currently locked:

| Module | Role | Locked commit |
| --- | --- | --- |
| [ApexOracle-MDLM](https://github.com/DragonDescentZerotsu/ApexOracle-MDLM) | downstream embedding, guidance heads, and candidate scoring | `c9d17c7` |
| [ApexOracle-Evo2](https://github.com/DragonDescentZerotsu/ApexOracle-Evo2) | record-aware genome embedding extraction | `2184211` |
| [ApexOracle-Generation](https://github.com/DragonDescentZerotsu/ApexOracle-Generation) | guided discrete diffusion and remasking | `de6c1e5` |

Core and DLM-Pretraining remain explicitly `pending` in `manifests/modules.lock.yaml`; no floating or placeholder
gitlinks are published for them. This repository is therefore a conversion candidate, not yet the final paper release.

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

Each module retains its own environment and license. See `environments/README.md`, `NOTICE`, and the license inside each
submodule. Model weights and datasets are never stored as Git objects; released and pending assets are recorded under
`manifests/`.

## Planned final layout

```text
ApexOracle/
├── modules/
│   ├── core/             # existing Synergy repository, later renamed ApexOracle-Core
│   ├── dlm_pretrain/     # collaborator DLM + MTR producer
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
