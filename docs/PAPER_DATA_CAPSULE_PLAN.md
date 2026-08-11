# Paper-data capsule plan

The machine-readable status is
`manifests/paper_data_capsule_plan.json`. This document records the decisions
needed to avoid both overclaiming reproducibility and growing the Git tree.

## What can be frozen exactly

- The revised Fig. 1b classification folds are deterministic
  `KFold(n_splits=5, shuffle=True, random_state=42)`. The strict-zero-shot,
  fine-tuned 10-member and matched Chemprop sample-level predictions are
  locally complete and hash-frozen in the plan manifest.
- The 2026 fixed strain-wise MIC reconstruction has an exact seed-0 split, all
  21 trained members and 86,358 sample-level ensemble predictions. It is ready
  for an external capsule as a post-paper reconstruction.
- The standardized strain mapping is already public in Core.

## What cannot be called the exact historical split

The original strain-wise MIC and synergy drivers constructed groups through
unordered Python `set` operations. Their independent 2025 processes did not
record `PYTHONHASHSEED`.

- For strain-wise MIC, the public seed-0 manifest is a deterministic
  legacy-codepath candidate; archived row counts remain authoritative, but
  exact 2025 strain membership was not recovered.
- For synergy, the seed-0 manifest matches every archived fold count and the
  2,732-row eligible cohort, but it is still a high-confidence candidate rather
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

The release unit will be one Zenodo capsule:

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

## Execution order

1. Publish the exact classification fold/prediction capsule and verify all
   AUPRC/AUROC values from downloaded files.
2. Publish the fixed strain-wise MIC reconstruction with an explicit
   post-paper label and independently recompute the reported metrics.
3. Audit species/phylum split lineage before adding their predictions.
4. Replay the synergy checkpoint family to create sample-level predictions, or
   state that only fold-level metrics can currently be recomputed.
5. Release source-partitioned model-ready tables only after their data and
   redistribution status is recorded.
