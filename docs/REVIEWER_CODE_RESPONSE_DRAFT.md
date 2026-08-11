# Reviewer response draft: code, data, and reproducibility

> Internal working draft. Do not paste into the response letter until every
> item marked **OPEN** below has either been released or removed from the
> corresponding claim.

## Consolidated question ledger

| Reviewer | Question or requested material | Current action and evidence | Status |
| --- | --- | --- | --- |
| Reviewer 1, reproducibility | Report compute requirements and runtimes. | Public requirements boundary added in `docs/COMPUTE_REQUIREMENTS.md`; exact wall time and peak RAM/VRAM still need fresh-run capture. | **OPEN** |
| Reviewer 1, code; Reviewer 2 and Reviewer 3, code availability | The repository was incomplete and lacked Evo-2 extraction, Me-LLaMA extraction, fusion, MIC/classification/synergy heads, and guided generation. | Existing `DragonDescentZerotsu/ApexOracle` was converted in place into a super-repository that pins five public modules. The missing implementations now have public module entry points and immutable gitlinks. | **DONE** |
| Reviewer 1; Reviewer 2 and Reviewer 3 | Provide click-and-run MIC prediction and guided-generation quickstarts. | Root `quickstarts/README.md` provides a CPU MIC inference example and a compact 256-step BAA-3170 generation smoke using immutable Hugging Face revisions and hashes. | **DONE** |
| Reviewer 4, minor point 2 | Release the full training-set layout, processed tables, standardized strain mapping, small-molecule and synergy data, frozen splits, preprocessing, and strain-description texts. | Zenodo already releases the paper-listed genome embeddings and strain descriptions. The compact Core strain mapping is public at immutable commit `8751c80`. Model-ready MIC/activity/synergy tables and complete frozen split/prediction capsules remain to be released. | **OPEN (mapping done)** |
| Reviewer 4, data availability | Which strains were used for Evo-2, and where are strain traits? | The Evo-2 module now provides the extraction CLI; Zenodo record `15612048` contains genome embeddings and strain-description assets. A standalone paper-cohort genome/source list remains to be added to the data capsule. | **OPEN** |
| Implied by “reproducing the main results” | Must every historical checkpoint be uploaded? | No. Representative inference-only weights support executable quickstarts. Paper-number reproduction will use frozen per-sample predictions, split membership, metric scripts, and a complete checkpoint registry. Raw optimizer-bearing checkpoints remain external. | **POLICY FIXED; CAPSULE OPEN** |

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
- The source release is MIT licensed at the orchestration and ApexOracle-owned
  code layers; third-party module notices and data licenses remain separate.

## Claims that must remain pending

Do not yet say that a reader can regenerate every reported paper value solely
from public assets. The remaining result-level release items are:

1. model-ready public tables and their redistribution notes;
2. complete paper split membership;
3. frozen per-sample predictions for the reported ensembles;
4. metric/figure recomputation commands and expected hashes;
5. a complete checkpoint registry for all paper members, including role,
   group/fold/member, source hash, code commit, and prediction hash;
6. measured wall time and peak RAM/VRAM for the two public quickstarts;
7. a standalone Evo-2 paper-cohort genome/source manifest.

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
inference-only weights for functional verification and are releasing the paper
evaluation capsule as frozen sample-level predictions, split membership,
checkpoint provenance, and metric/figure recomputation scripts. This avoids
requiring reviewers to download hundreds of gigabytes of duplicated optimizer
and backbone state while preserving direct recomputation of the reported
results. We have also added a compute-requirements table and will insert the
measured wall time and peak RAM/VRAM from fresh runs before submission.

> Before use: change “are releasing” to “have released” only after the paper
> evaluation capsule is public, and replace the last sentence with exact
> measured values.

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
respectively.

We have also added a machine-readable strain mapping that connects each source
strain label in the paper MIC workflow to its canonical runtime condition key,
species, genome/text or text-only route, and embedding filenames. It contains
1,766 unique source strain labels and 1,769 condition routes, corresponding to
92,322 routed MIC records before token-length filtering, without exposing MIC
labels, molecule structures, embedding tensors or private assay rows. The
remaining model-ready tables, complete split membership, sample-level paper
predictions, and the standalone Evo-2 genome/source list will be released in a
versioned paper-data capsule and linked from the root asset manifest.

> Before use: the final sentence must be converted to present tense and linked
> to a stable DOI/revision, or retained explicitly as an unresolved action.

## Final pre-submission checks

- [ ] Fresh MIC quickstart: record platform, command, wall time and peak RAM.
- [ ] Fresh generation quickstart: record GPU, wall time and peak VRAM.
- [x] Publish and hash the compact strain mapping from Core.
- [ ] Publish model-ready data allowed for redistribution.
- [ ] Publish exact split membership and frozen paper predictions.
- [ ] Add checkpoint registry and metric/figure recomputation README.
- [ ] Publish the Evo-2 genome/source manifest.
- [ ] Replace every future-tense promise above with a verified public link.
- [ ] Confirm that README, Code Availability, Data Availability, Zenodo and
      Hugging Face links all refer to immutable records.
