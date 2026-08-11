# Reproducibility scope and model-asset policy

ApexOracle separates four reproducibility levels because they have very
different storage and compute requirements.

| Level | Public contract | Current state |
| --- | --- | --- |
| Source reproducibility | Fixed commits for all five modules, environments, tests, expanded source archive | Released |
| Functional inference reproducibility | Small, real, inference-only assets for MIC prediction and guided generation | Released |
| Paper-result reproducibility | Frozen sample-level predictions, split membership, checkpoint registry, metric and figure recomputation | Classification and post-paper fixed MIC reconstruction released; historical 2025 MIC and synergy remain incomplete |
| Full model rerun/retraining | Every ensemble binary, training dataset, accelerator environment and training run | Not the default public contract |

## Why every historical checkpoint is not uploaded

The paper MIC hierarchy uses 21 members (`3 groups × 7 members`). The
classification experiments include a 30-member strict zero-shot family and two
150-member fivefold families. The synergy evaluation uses another 21 members.
Many source checkpoints include a repeated multi-gigabyte backbone, optimizer
state and training-only payload.

Uploading all of those source checkpoints would require hundreds of gigabytes
and would still not, by itself, identify the correct data rows, fold membership,
preprocessing code or metric calculation. The release therefore uses:

1. representative inference-only weights for runnable public quickstarts;
2. frozen sample-level ensemble predictions for reported numerical results;
3. exact split and sample identifiers;
4. scripts that recompute metrics, statistics, tables and figures;
5. a registry linking every historical member to role, fold/group/member,
   source SHA-256, code revision and prediction artifact;
6. optional inference-only ensemble exports when a complete model rerun is
   scientifically necessary and redistribution is practical.

Optimizer states, W&B runs, caches and duplicated backbones are not public
release requirements. A result reproduced from frozen predictions should be
described as **paper-result recomputation**, not as a fresh model inference or
full retraining reproduction.

## Storage locations

- GitHub: source, manifests, quickstarts and recomputation scripts;
- Hugging Face: runnable inference-only model assets;
- Zenodo: one version series under the existing ApexOracle concept DOI
  `10.5281/zenodo.15612047`, containing embedding data plus progressively added
  paper-data and prediction payloads suitable for DOI-based archival;
- retained author archive: full historical training checkpoints and raw runs.

No weight or dataset should be described as public until its immutable URI,
revision, size, checksum, license/redistribution decision and smoke test are
recorded in the root manifests.
