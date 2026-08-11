# Paper-data capsule plan

The machine-readable status is
`manifests/paper_data_capsule_plan.json`. This document records the decisions
needed to avoid both overclaiming reproducibility and growing the Git tree.

## What can be frozen exactly

- The revised Fig. 1b classification folds are deterministic
  `KFold(n_splits=5, shuffle=True, random_state=42)`. The strict-zero-shot,
  fine-tuned 10-member and matched Chemprop sample-level predictions are
  public and hash-frozen in Zenodo v2.0.0.
- The 2026 fixed strain-wise MIC reconstruction has an exact seed-0 split, all
  21 trained members and 86,358 sample-level ensemble predictions. Its
  privacy-minimized external capsule is public in Zenodo v3.0.0 as a post-paper
  reconstruction.
- The standardized strain mapping is already public in Core.

## What cannot be called the exact historical split

The original strain-wise MIC and synergy drivers constructed groups through
unordered Python `set` operations. Their independent 2025 processes did not
record `PYTHONHASHSEED`.

- For strain-wise MIC, the public seed-0 manifest is a deterministic
  legacy-codepath candidate; archived row counts remain authoritative, but
  exact 2025 strain membership was not recovered.
- For synergy, the seed-0 manifest matches every archived fold count and the
  2,732-row eligible set, but it is still a high-confidence candidate rather
  than proof of exact 2025 membership.
- Species-wise and phylum-wise memberships require a separate lineage audit
  before they enter the capsule.

The public capsule must use explicit labels such as `historical-counts-only`,
`deterministic-candidate` and `postpaper-reconstruction`. It must never replace
those distinctions with a generic claim that all original splits are exact.

## Model-ready data boundary

The current MIC table combines DBAASP-derived and in-house records. Before
external release it must be partitioned by source and private/public status;
private assay rows must not be exposed merely because they were present in an
author workstation table. The small-molecule merge and synergy source likewise
need source-specific redistribution records.

The release unit will be a new version of the existing ApexOracle Zenodo
record, not a second independent Zenodo project. The stable concept DOI is
`10.5281/zenodo.15612047`; the paper-listed first version remains immutable at
`10.5281/zenodo.15612048`. Each substantial file update receives a new version
DOI while staying linked under that one concept DOI.

The added paper-data payload uses this layout:

```text
paper-data/
├── tables/
├── splits/
├── predictions/
├── manifests/
│   ├── assets.json
│   └── SHA256SUMS
└── README.md
```

Git stores the exporter, schema, DOI/revision, row counts, hashes and metric
recomputation code. The full tables and predictions are uploaded exactly once
to the external record. In particular, the 18–19 MB MIC prediction tables and
multi-megabyte classification outputs must not be copied into Core and then
again into the super-repository.

## Classification capsule release

The classification payload is assembled by:

```bash
python scripts/build_classification_capsule.py \
  --source-root /path/to/preserved/core-workspace \
  --output /path/outside/git/apexoracle_fig1b_classification_reproduction_v1
```

The builder validates every source file against
`manifests/classification_capsule_sources.json`, emits exact eligible split
membership and nine normalized prediction tables, removes unnecessary SMILES
and workspace-only columns, and independently recomputes all pooled and
fold-mean AUPRC/AUROC values. The included checker uses only the Python standard
library. Archives use fixed file modes, timestamps and root names; two clean
builds must be byte-identical.

The released asset is:

- version DOI: `10.5281/zenodo.21882300`;
- concept DOI: `10.5281/zenodo.15612047`;
- archive size: 1,317,912 bytes;
- MD5: `f663551b545de70277b5e665d2a6dab5`;
- SHA-256: `6d053c68ef21afd37d0c7bb76d555c55073513db3785238ace0a7055ea203f68`;
- release manifest: `manifests/zenodo_release_21882300.json`.

Both the authenticated draft download and the unauthenticated public download
matched the released SHA-256. Internal `SHA256SUMS`, AUPRC/AUROC recomputation,
exact fold membership, normalized schemas and the no-absolute-path check also
passed. No prediction table is duplicated into Git.

## Fixed MIC reconstruction capsule

The post-paper fixed strain-wise reconstruction is assembled by:

```bash
python scripts/build_mic_reconstruction_capsule.py \
  --source-root /path/to/fixed_strain_retrain \
  --output /path/outside/git/apexoracle_fixed_mic_reconstruction_v1 \
  --version-doi VERSION_DOI
```

The capsule contains all 21 normalized member prediction tables, their
86,358-row ensemble, frozen metrics, the molecule-cluster bootstrap summary,
the exact fixed membership, and a member registry. The included standard-library
checker verifies every file hash, reconstructs each seven-member mean, and
recomputes R², Spearman and Pearson values.

This capsule is explicitly a **post-paper reconstruction** using the frozen
`PYTHONHASHSEED=0` deterministic legacy-codepath candidate. It is not presented
as the unrecovered membership used by the 2025 checkpoints. The normalized
tables omit molecule structures, token sequences, exact MIC values, source-row
identifiers, embeddings, checkpoints, optimizer state, and private source
tables. They retain normalized labels, predictions, hashed molecule identity,
strain IDs, condition route and a `MIC <= 16 µM` boolean so the reported
metrics remain independently computable.

The released asset is:

- version DOI: `10.5281/zenodo.21883545`;
- concept DOI: `10.5281/zenodo.15612047`;
- archive size: 40,177,188 bytes;
- MD5: `bbf7e3a1ab36b1bc029163a9e8d3ad30`;
- SHA-256: `25e74abde1f01be57e83b22f6bd1633634284e74257d71f3c71864f7c4b9eebc`;
- release manifest: `manifests/zenodo_release_21883545.json`.

Two independent pre-release builds were byte-identical before DOI insertion.
After the version DOI was reserved, the final authenticated draft and
unauthenticated public downloads matched the final archive hashes. A fresh
public-download extraction passed all 30 internal hashes, reconstructed every
seven-member ensemble mean, recomputed all 48 metric rows, and passed the
author-path and excluded-column scan.

For the pooled 86,358 measurements, the reconstructed ensemble has R²
`0.463848`. The train-seen and train-unseen subsets contain 60,086 and 26,272
measurements with R² `0.567177` and `0.094245`, respectively; train-unseen
Spearman/Pearson correlations are `0.406976/0.412980`. These values describe
the released post-paper reconstruction and must not be substituted for a claim
about the unrecovered 2025 membership.

## Execution order

1. **Completed:** release the exact classification fold/prediction capsule as
   Zenodo v2.0.0 and verify all AUPRC/AUROC values from freshly downloaded
   files.
2. **Completed:** release and independently verify the fixed strain-wise MIC
   reconstruction in Zenodo v3.0.0 with an explicit post-paper label.
3. Audit species/phylum split lineage before adding their predictions.
4. Replay the synergy checkpoint family to create sample-level predictions, or
   state that only fold-level metrics can currently be recomputed.
5. Release source-partitioned model-ready tables only after their data and
   redistribution status is recorded.
