# Reviewer response draft: code, data, and reproducibility

> Internal working draft synchronized to public assets on 2026-08-11. All
> code/data release items below are closed. The separately tracked new-strain
> study is not part of this code-release ledger.

## Consolidated question ledger

| Reviewer | Question or requested material | Current action and evidence | Status |
| --- | --- | --- | --- |
| Reviewer 1, reproducibility | Report compute requirements and runtimes. | Both public quickstarts were rerun from empty caches. MIC computation took 7.27 s with 5.77 GiB peak RSS; one-sample 256-step H100 generation took 40.34 s with 12,281 MiB peak GPU memory. Downloads are reported separately in `manifests/quickstart_benchmarks_2026-08-11.json`. | **DONE** |
| Reviewer 1, code; Reviewer 2 and Reviewer 3, code availability | The repository was incomplete and lacked Evo-2 extraction, Me-LLaMA extraction, fusion, MIC/classification/synergy heads, and guided generation. | Existing `DragonDescentZerotsu/ApexOracle` was converted in place into a super-repository that pins five public modules. The missing implementations now have public module entry points and immutable gitlinks. | **DONE** |
| Reviewer 1; Reviewer 2 and Reviewer 3 | Provide click-and-run MIC prediction and guided-generation quickstarts. | Root `quickstarts/README.md` provides a CPU MIC inference example and a compact 256-step BAA-3170 generation smoke using immutable Hugging Face revisions and hashes. | **DONE** |
| Reviewer 4, minor point 2 | Release the full training-set layout, processed tables, standardized strain mapping, small-molecule and synergy data, frozen splits, preprocessing, and strain-description texts. | Zenodo releases the paper-listed genome embeddings and strain descriptions. The compact Core strain mapping is public at immutable commit `8751c80`. Zenodo v2.0.0 (`10.5281/zenodo.21882300`) releases exact classification folds and predictions; v3.0.0 (`10.5281/zenodo.21883545`) adds the post-paper fixed MIC reconstruction; v4.0.0 (`10.5281/zenodo.21883954`) adds the complete 21-member synergy replay and checker; v5.0.0 (`10.5281/zenodo.21891064`) adds source-partitioned public model-ready MIC, classification and synergy tables. All 15,718 private in-house MIC rows are excluded. | **DONE** |
| Reviewer 4, data availability | Which strains were used for Evo-2, and where are strain traits? | The Evo-2 module provides the extraction CLI; Zenodo record `15612048` contains genome embeddings and strain-description assets. Core now publishes a 563-row paper genome list with species labels, conservative source identifiers, current filename-matched FASTA identities, embedding hashes, and separate MIC/classification/synergy usage flags. | **DONE** |
| Implied by “reproducing the main results” | Must every historical checkpoint be uploaded? | No. Representative inference-only weights support executable quickstarts. Classification, the post-paper fixed MIC reconstruction and the high-confidence synergy replay use frozen per-sample predictions, split membership and metric scripts. `manifests/paper_result_registry.json` links every released result family and explicitly marks historical hashes that were never recorded. | **DONE; POLICY AND REGISTRY PUBLIC** |

Reviewer 2 and Reviewer 3 repeated the same repository assessment and
quickstart request. They should receive equivalent substantive answers rather
than two different technical claims.

## Completed release facts that may be stated now

- The existing public ApexOracle repository was converted in place; it was not
  replaced by a second super-repository.
- The root release pins five public modules: Core, DLM-Pretraining, MDLM, Evo2,
  and Generation. Exact commits are recorded in
  `manifests/modules.lock.yaml`.
- The previously missing functional areas now have public owners:
  - Evo-2 genome embedding extraction: `modules/evo2`;
  - Me-LLaMA strain-text embedding extraction: `modules/core`;
  - cross-attention fusion and MIC/classification/synergy heads: `modules/core`;
  - downstream DLM embedding, guidance heads, and candidate scoring:
    `modules/mdlm`;
  - guided discrete diffusion and paper sampling presets:
    `modules/generation`.
- The public MIC quickstart downloads one inference-only ensemble member and a
  condition tensor, then reproduces the documented CPU prediction.
- The generation quickstart downloads a compact BAA-3170 bundle and runs the
  complete 256-step sampling path. It is a runtime smoke, not an activity claim.
- Hugging Face assets are fixed by immutable revision and SHA-256 in the root
  manifests.
- Zenodo record `10.5281/zenodo.15612048` contains the paper-listed genome and
  strain-description/embedding archives.
- Zenodo v2.0.0 (`10.5281/zenodo.21882300`) adds the exact Fig. 1b
  classification folds, nine sample-level prediction tables, frozen reporting
  metrics, and an independent standard-library metric checker. The public
  archive passed SHA-256, internal-manifest, schema, metric and absolute-path
  audits.
- Zenodo v3.0.0 (`10.5281/zenodo.21883545`) adds the fixed strain-wise MIC
  reconstruction: 21 normalized member tables, an 86,358-row ensemble, exact
  reconstruction membership, frozen metrics and an independent checker.
  Public-download hashes, every ensemble mean and all 48 metric rows were
  verified. This is a post-paper reconstruction, not the unrecovered 2025 MIC
  checkpoint membership.
- Zenodo v4.0.0 (`10.5281/zenodo.21883954`) adds 2,371 sample-level synergy
  predictions from the complete 3-fold × 7-member checkpoint family and an
  independent checker. All 22 checkpoint hashes were recomputed, a second GPU
  replay produced byte-identical CSVs, and each fold AUROC/AUPRC reproduces the
  archived value to four decimals. The split remains labeled a high-confidence
  seed-0 candidate rather than proven exact 2025 membership.
- Zenodo v5.0.0 (`10.5281/zenodo.21891064`) adds 105,237 DBAASP-derived MIC
  rows, 49,330 classification rows and 4,285 synergy source rows. All 15,718
  private in-house MIC rows are excluded.
- The canonical all-peptide MIC candidate scorer is also public in Zenodo
  v2.0.0 under the non-misleading filename
  `mic_candidate_scorer_all_peptide_non_pad_t1e-3_epoch13.pth`. It is a
  fixed-epsilon (`t=1e-3`) post-generation scorer, not exact clean `t=0`, not
  the Generation sampler checkpoint, and not a validated general
  small-molecule scorer.
- Core publishes the 563-row paper Evo-2 genome list at immutable commit
  `1ab309c`. It identifies the genomes used by paper MIC, classification and
  synergy tasks (563/2/100 respectively), with source labels and FASTA/embedding
  hashes. It includes no sequences, tensors, assay labels or private rows.
- The source release is MIT licensed at the orchestration and ApexOracle-owned
  code layers; third-party module notices and data licenses remain separate.

## Boundaries that must remain explicit

The release work is complete, but the response must retain these scientific
boundaries:

1. exact 2025 MIC membership and mathematically
   proven 2025 synergy membership were not recovered;
2. missing historical checkpoint hashes are recorded as `not_recorded`, not inferred;
3. the released MIC reconstruction is post-paper, and the synergy split is a
   high-confidence candidate;
4. quickstart timing is a single measurement on the documented host, not a
   throughput guarantee;
5. CC BY 4.0 does not relicense underlying database records, and private
   in-house assay rows are not public.

The release tree is protected by an automated anti-bloat gate. The Fig. 1b
classification, fixed MIC reconstruction and synergy replay payloads are
archived once in Zenodo v2.0.0/v3.0.0/v4.0.0 under the existing concept DOI and
referenced by hashes; they are not duplicated across Git or module
repositories. The model-ready tables are likewise archived once in Zenodo
v5.0.0 rather than copied into Git.

The complete historical training checkpoints are not a release gate. MIC alone
uses 21 large source checkpoints, classification contains hundreds of members,
and synergy uses another 21. Uploading all optimizer-bearing binaries would add
hundreds of gigabytes without being necessary to recompute the paper's tables
and figures.

## Master English response for Reviewers 1–3

**Reviewer comment.** The repository is incomplete, the main results cannot be
reproduced from the available code, and an end-to-end quickstart should cover
MIC prediction and guided generation. Compute requirements and runtimes should
also be reported.

**Draft response.** We thank the reviewer for identifying the incomplete code
release. We have replaced the earlier incomplete repository tree by converting
the existing public ApexOracle repository in place into a modular release. The
root repository now pins five independently installable public modules at
immutable commits: ApexOracle-Core, ApexOracle-DLM-Pretraining,
ApexOracle-MDLM, ApexOracle-Evo2, and ApexOracle-Generation. Together these
modules provide the previously missing Evo-2 genome-embedding pipeline,
Me-LLaMA strain-text embedding pipeline, multimodal cross-attention fusion,
MIC-regression, antibiotic-classification and synergy heads, downstream
guidance/scoring utilities, and the guided discrete-diffusion sampler.

We also added two public end-to-end examples. The MIC quickstart downloads an
inference-only checkpoint member and a fixed known-strain condition, verifies
their SHA-256 values, and runs prediction on CPU. The guided-generation
quickstart downloads a compact fixed-revision runtime bundle and executes the
complete 256-step strain-conditioned sampling path. The root manifests record
immutable module commits, model/data revisions, file sizes, checksums, license
boundaries, and the scope of each released asset. Because GitHub's automatic
source archives do not expand submodules, we additionally provide a validated
expanded source archive with the release.

The runnable quickstarts and reproduction of the reported paper values have
different storage requirements. We therefore release representative
inference-only weights for functional verification. We have released the exact
Fig. 1b classification folds, sample-level predictions, frozen metrics, and an
independent metric checker in Zenodo v2.0.0. Zenodo v3.0.0 adds a separately
labeled post-paper fixed MIC reconstruction with all 21 member prediction
tables, exact reconstruction membership and metric recomputation. The
v4.0.0 release adds the complete 21-member synergy replay with sample-level
predictions, checkpoint provenance and metric recomputation under an explicit
high-confidence-candidate split boundary. This avoids
requiring reviewers to download hundreds of gigabytes of duplicated optimizer
and backbone state while preserving direct recomputation of the reported
results. On our documented AMD EPYC 9534/H100 host, the public MIC computation
completed in 7.27 s with 5.77 GiB peak resident memory, and the one-sample
256-step generation computation completed in 40.34 s with 12,281 MiB peak GPU
memory. Empty-cache download time is reported separately in the public
machine-readable benchmark manifest.

## Reviewer 4 English response: training data and strain mapping

**Reviewer comment.** Where can readers find the full training set, including
strain descriptions? Please provide the standardized strain mapping, model
tables, splits and preprocessing. Which strains were processed by Evo-2?

**Draft response.** We thank the reviewer for this helpful suggestion. We have
reorganized the public release so that source code, large model assets and
paper data are separately versioned but connected by immutable manifests. The
paper-listed genome embeddings and strain-description/text-embedding archives
are available from Zenodo record 15612048. The Evo-2 and Me-LLaMA extraction
entry points are now public in ApexOracle-Evo2 and ApexOracle-Core,
respectively. ApexOracle-Core also publishes the exact 563-row paper genome
list, including species labels, conservative source identifiers, current
filename-matched FASTA identities, embedding SHA-256 values, and separate
MIC/classification/synergy usage flags. The list does not expose sequences,
embedding tensors, assay labels, molecule structures or private rows.

We have also added a machine-readable strain mapping that connects each source
strain label in the paper MIC workflow to its canonical runtime condition key,
species, genome/text or text-only route, and embedding filenames. It contains
1,766 unique source strain labels and 1,769 condition routes, corresponding to
92,322 routed MIC records before token-length filtering, without exposing MIC
labels, molecule structures, embedding tensors or private assay rows. The
exact classification folds, nine sample-level prediction tables, frozen
metrics and an independent checker are now public in Zenodo v2.0.0
(`10.5281/zenodo.21882300`) under the same concept DOI. Zenodo v3.0.0
(`10.5281/zenodo.21883545`) adds the post-paper fixed MIC reconstruction with
21 member tables, an 86,358-row ensemble, exact reconstruction membership and
an independent metric checker. Zenodo v4.0.0 (`10.5281/zenodo.21883954`) adds
the complete 21-member synergy replay, 2,371 sample-level predictions and an
independent checker; the seed-0 split matches archived counts and metrics but is
explicitly labeled a high-confidence candidate. Zenodo v5.0.0
(`10.5281/zenodo.21891064`) adds the source-partitioned public model-ready
tables: 105,237 DBAASP-derived MIC rows, the frozen 49,330-row classification
table and 4,285-row synergy source table. All 15,718 private in-house MIC rows
are excluded. The compilation includes source and non-relicensing notices and
is linked from the root asset manifest.
We explicitly distinguish
recovered historical membership from post-paper deterministic reconstruction
rather than representing the latter as the original 2025 split.

## Final pre-submission checks

- [x] Fresh MIC quickstart: record platform, command, wall time and peak RAM.
- [x] Fresh generation quickstart: record GPU, wall time and peak VRAM.
- [x] Publish and hash the compact strain mapping from Core.
- [x] Build and independently verify the exact classification capsule locally.
- [x] Publish the classification capsule in Zenodo v2.0.0 under existing
      concept DOI `10.5281/zenodo.15612047` and verify an unauthenticated public
      download.
- [x] Publish source-partitioned public model-ready data in Zenodo v5.0.0;
      exclude all private in-house MIC rows.
- [x] Publish and publicly verify the fixed MIC reconstruction in Zenodo
      v3.0.0 with the post-paper historical boundary.
- [x] Publish and publicly verify the high-confidence synergy replay in Zenodo
      v4.0.0 with the unproven-exact-membership boundary.
- [x] Add the explicit paper-result/checkpoint-status registry and metric/
      plotting entry-point documentation.
- [x] Publish and hash the 563-row paper Evo-2 genome list.
- [x] Replace every code/data future-tense promise above with a verified public link.
- [x] Confirm that README, Code Availability, Data Availability, Zenodo and
      Hugging Face links refer to immutable records or the stable Zenodo concept DOI.
