# Release status

## Virus-extension development branch

The non-release `virus-extension` branch keeps the five paper-release modules and adds a sixth active module,
`ApexOracle-Virus`, for viral-corpus and DRAVP target/genome preparation. Its current machine-readable lock is
`manifests/modules.lock.yaml`. This branch does not retroactively change the five-module `v0.2.3` release or its
scientific claims. All current-branch module-lock, archive, anti-bloat and recursive-clone gates cover six modules.

## Verified facts

- The existing `DragonDescentZerotsu/ApexOracle` repository was converted in place; no second super-repository was
  created.
- The legacy tree is recoverable from branch `legacy-monorepo` and annotated tag
  `legacy-monorepo-snapshot-2026-08-10`.
- `ApexOracle-DLM-Pretraining` is locked to `362ffccac79bdd638a4e913c4f17df613da18f36`; its original source
  recovery tag, 56-file manifest, source contracts, remote CI, fresh clone, and H100 joint-objective train/save/load
  smoke have passed.
- `ApexOracle-MDLM` scientific implementation was validated at
  `c9d17c7f6f091234aaaebf5f08dbe23542f980c1`; the current documentation-closure lock is
  `548f65cbc9f40fb38ae26cbb6d9c6b004bfb4e36`. Maintenance through `26e414b` refreshed generated lineage outputs;
  `548f65c` adds the generic peptide-inventory workflow and peptide-table I/O/provenance fixes. All 127 tests and the
  179-asset ledger stale check pass.
- The released `ApexOracle-Evo2` default branch remains locked to
  `2184211acda07b0d5ca865067174ac42f530ad04`; its CPU contracts, package archives, remote CI, and Evo2-40B extraction
  runtime smoke have passed. On the non-release branch `codex/fix-multi-contig-windowing`, the super-repository instead
  locks Evo2 to `5c7bc1890b727a740924328d0845fadc8de2c9d6`. That commit identifies new artifacts with the
  `per_record_zero_based_v1` contract and guards the per-contig start reset against the historical cross-record global
  counter bug. Its 10 CPU tests and package build pass; the 40B runtime smoke has not been rerun for this branch.
- The non-release branch `codex/claude-agents-links` continues from the fixed multi-contig line and makes every
  repository guidance file available to Claude Code through a colocated relative `CLAUDE.md -> AGENTS.md` Git
  symlink. Seven of seven `AGENTS.md` locations are covered across the super-repository and five modules; the two
  Core links already existed at its unchanged `bbaaedf` lock. DLM-Pretraining, MDLM, Evo2, and Generation advance
  only by one symlink commit to `6a7cc4b`, `ea8d9c0`, `d60301b`, and `bfff9b9`, respectively. No model, protocol,
  checkpoint, data, or default branch changes are included.
- `ApexOracle-Generation` scientific implementation was validated at
  `80d9a2cf9b0921f29e4a44edf5557ccac39f5af9`; the current documentation-closure lock is
  `706e06fe8ab6e2f71bffa330c7d9acb309200df9`. The maintenance delta only tightens ignored workspace artifacts and
  corrects two README links to existing guidance-evaluation scripts. Compact BAA-3170
  inference-only assets are fixed at Hugging Face revision `2fb1aa08187eaa359263be6c12c8a41868d8959c` and have passed
  empty-cache hashes plus a real 256-step H100 smoke.
- The original `Synergy` repository was renamed in place to public `ApexOracle-Core`; its scientific release `v0.1.0`
  is `8c1def518ac148a878c14f4a39876db59649d43c`; the `v0.2.3` documentation-closure lock is
  `1973d2d3cc6b27202a3960c363c207dd030f74e7`, and post-release `main` is
  `bbaaedf6030c7606c31db35f285041a715db9249`; scientific additions through `bc4aa31` contain the strain mapping,
  paper genome list and verified synergy checkpoint replay, `23be273` tightens workspace hygiene, and `bbaaedf`
  publishes the Providencia ATCC 29914 screening/generation capsule. The earlier 217-test
  release gate, wheel/sdist, public-history audit and fresh Hugging Face MIC inference passed; the genome-list batch
  passed 211 tests with 4 external-asset skips; the synergy replay closure passed 215 tests with 4 skips.
- The root repository contains no model weights, embeddings, datasets, or raw experiment outputs after conversion.
- Core publishes a compact paper strain mapping at commit `8751c80`: 1,766 unique source labels, 1,769 condition
  routes and 92,322 routed MIC records before length filtering. It contains condition keys and filenames, not MIC
  labels, molecule structures, embedding tensors or private assay rows.
- Core publishes a 563-row paper Evo-2 genome list at commit `1ab309c`: MIC/classification/synergy use 563/2/100
  genomes respectively. Each row records a species label, conservative source identifier, current filename-matched
  FASTA identity and embedding SHA-256. The list contains no sequence or tensor payload and does not represent the
  current FASTA files as proven byte-identical original producer inputs.
- The manuscript-listed Zenodo dataset record `15612048` is public under CC BY 4.0 and contains
  `Genome_embs.tar.gz` plus `Text_Description.tar.gz`. It is an external data record, so no archive is copied into Git.
- Zenodo v2.0.0, DOI `10.5281/zenodo.21882300`, is public under the same concept DOI
  `10.5281/zenodo.15612047`. It retains both embedding archives and adds the exact Fig. 1b classification capsule,
  the canonical fixed-`t=1e-3` all-peptide MIC candidate scorer, and a machine-readable release manifest. The old
  misleading checkpoint filename is absent from the public file list.
- Zenodo v3.0.0, DOI `10.5281/zenodo.21883545`, remains in that same concept DOI series and retains every v2 file.
  It adds the 40,177,188-byte fixed strain-wise MIC reconstruction archive and a v3 release manifest. The archive
  contains 21 normalized member tables, an 86,358-row ensemble, exact reconstruction membership, frozen metrics and
  a standard-library checker; it is explicitly a post-paper reconstruction rather than the unrecovered membership
  used by the 2025 checkpoints.
- Zenodo v4.0.0, DOI `10.5281/zenodo.21883954`, retains all earlier files and adds the complete synergy replay:
  3 folds × 7 members, 2,371 token-filtered predictions, candidate split, checkpoint/prediction hashes and a
  standard-library AUROC/AUPRC checker. Every replayed fold metric rounds to the archived log value; the split is
  therefore a high-confidence historical candidate but remains explicitly unproven as exact 2025 membership.
- Zenodo v5.0.0, DOI `10.5281/zenodo.21891064`, retains all earlier files and adds the source-partitioned public
  model-ready data capsule. It contains 105,237 DBAASP-derived MIC rows, 49,330 classification rows, 4,285 synergy
  source rows, the compact strain mapping and the 563-row paper genome list. All 15,718 private in-house MIC rows
  are excluded. The paper-level 121,265 MIC count precedes tokenization; the released model-ready source contains
  120,955 rows after the documented exclusion of 310 structures over 1,024 tokens.
- The six active trees now have a CI-enforced anti-bloat policy: exact duplicates at or above 20 KiB, unallowlisted
  files above 1 MiB, checkpoint/cache suffixes, generated build/cache paths and per-repository count/byte-limit
  violations fail `scripts/check_repository_bloat.py`. The current audit found no >=20 KiB exact duplicates.
- The 2026-08-11 filesystem review upgraded that gate to schema v2. Same-repository duplicates now fail from 1 KiB,
  unexpected top-level directories fail, and top-level distribution, long-source review candidates and 80% soft-limit
  alerts are visible without changing scientific implementations. All six public trees have zero nonignored untracked
  files and zero same-repository duplicates; the only smaller cross-tree duplicates are self-contained license/upstream
  runtime files needed by independently installable modules.
- `manifests/paper_data_capsule_plan.json` distinguishes exact classification folds, the exact 2026 fixed MIC
  reconstruction, unrecovered 2025 MIC membership, and the count-matched but unproven synergy seed-0 candidate.
  Full tables and predictions remain external-capsule assets rather than Git payloads. The public model-ready
  partition is released in Zenodo v5.0.0.
- The exact Fig. 1b classification capsule is public in Zenodo v2.0.0: 1,317,912 bytes, MD5
  `f663551b545de70277b5e665d2a6dab5`, SHA-256
  `6d053c68ef21afd37d0c7bb76d555c55073513db3785238ace0a7055ea203f68`. Authenticated draft and unauthenticated
  public downloads, internal `SHA256SUMS`, fixed folds, normalized schemas, AUPRC/AUROC recomputation and the
  no-absolute-path gate passed.
- The fixed MIC reconstruction capsule is public in Zenodo v3.0.0: 40,177,188 bytes, MD5
  `bbf7e3a1ab36b1bc029163a9e8d3ad30`, SHA-256
  `25e74abde1f01be57e83b22f6bd1633634284e74257d71f3c71864f7c4b9eebc`. Authenticated draft and unauthenticated
  public downloads matched; a fresh public extraction passed 30 internal hashes, all 21-member ensemble means,
  all 48 recomputed metric rows and the excluded-path/column scan.
- The synergy replay capsule is public in Zenodo v4.0.0: 205,983 bytes, MD5
  `08407d97ab8aee3ea6130e47452aaefb`, SHA-256
  `a40ec811b179782ffd9d2429c2d3d262df0149c3594a286d0f0c666d9c58d70c`. All 22 source checkpoint hashes were
  recomputed, a second GPU replay produced byte-identical prediction CSVs, and authenticated draft plus public
  download checks passed all seven internal hashes, seven-member means and three fold metrics.
- The public model-ready capsule is released in Zenodo v5.0.0: 3,743,537 bytes, MD5
  `e403f6836dd2dccfd2eb8b62addbaad1`, SHA-256
  `ae0c76febd4e0b4d43fd68c8bf3ddfa27fc2251011f88c5f693d9aa27be95901`. Two deterministic builds were
  byte-identical; authenticated draft and unauthenticated public downloads matched; internal hashes, row counts,
  private-row exclusion and the author-path scan passed.
- Fresh public quickstart measurements are complete. On the recorded host, MIC computation took 7.27 s with
  5.77 GiB peak RSS; the one-sample 256-step H100 generation computation took 40.34 s with 12,281 MiB peak GPU
  memory. Download times are reported separately in `manifests/quickstart_benchmarks_2026-08-11.json`.
- The 2026-08-11 public-link audit returned HTTP 200 for the root commit, all five locked module commits, the three
  immutable Hugging Face model revisions, the Hugging Face pretraining dataset, Zenodo v1--v5, and the stable Zenodo
  concept DOI. The concept DOI resolves to the current v5 record as intended; version DOIs remain immutable.
- The 2026-08-11 local Core-worktree transition did not change any committed super-repo or module content. The
  long-lived super-repo checkout now leaves `modules/core` deinitialized while preserving gitlink
  `bbaaedf6030c7606c31db35f285041a715db9249`; daily Core development remains in the single original Synergy
  worktree. A disposable public `--recurse-submodules` clone at root commit
  `7b009091f9f1605b669393ecba8d361979e397fe` checked out all five locked commits, passed 17/17 root tests, passed
  `scripts/check_module_locks.py`, and produced the expected five-module source-archive plan. The temporary clone was
  then moved to the system trash. In the deliberately deinitialized long-lived checkout, three Core-content-dependent
  tests are expected not to constitute a release gate; future full validation must use a disposable recursive clone.
- The 2026-08-11 Providencia maintenance advances Core/MDLM locks to `bbaaedf`/`548f65c`. On the combined current
  Core tree, the MIC quickstart plus Providencia focused tests passed 8/8 and the complete suite passed 221 with four
  external-asset skips and 14 existing warnings. MDLM passed 127/127. The public quickstart scripts, immutable HF
  revisions and expected MIC output are unchanged. A remote recursive fresh clone checked out all five new locks;
  module-lock, release-tree and anti-bloat checks, 17 root tests and both Core MIC quickstart tests passed.

## Module gates

The five paper module repositories and the Virus extension repository are public and pinned to immutable commits.
No active module uses a floating placeholder.

## Final release gates

- all active gitlinks match the lock manifest;
- MIC prediction and compact guided generation run from fixed public assets; full paper condition-bank distribution
  remains separate;
- weights and data have stable URIs, revisions, SHA-256 values, and redistribution decisions;
- license/NOTICE, secret, large-file, and broken-link audits pass;
- a full-source archive expands all fixed submodules.

Canonical archive builder is `scripts/build_source_archive.py`. It validates all gitlinks against the lock manifest,
expands root and module `git archive` streams, and emits a deterministic tarball plus JSON/SHA-256 sidecars.

Release `v0.2.0` publishes `ApexOracle-source-v0.2.0.tar.gz`: 932 files, 36,553,607 bytes, SHA-256
`895c682ba6ede090dd28e3b5d64f1995c014779c6447ae6abcf593bad78b4fdd`. Two independent builds were byte-identical;
`scripts/check_source_archive.py` passed. The same release follows an empty-cache download and fresh recursive-clone
H100 smoke of the compact generation assets.

Release `v0.2.1` is a documentation-only closure: it updates Core/MDLM/Generation status records, moves the already
released quickstart conditions out of the pending data list, records per-file checkpoint/condition hashes, and emits
a new expanded source archive. It does not change scientific implementation, public weight revisions, or protocols.
`ApexOracle-source-v0.2.1.tar.gz` contains 933 files, is 36,563,257 bytes, and has SHA-256
`eba6138903dada6806a212c287327999538196d8282678e6cc9a19b4337cd4f2`; two independent builds were byte-identical
and `scripts/check_source_archive.py` passed. Exact version-to-commit distinctions are recorded in
`docs/RELEASE_PROVENANCE.md`.

Release `v0.2.2` corrects the erroneous Zenodo omission in `v0.2.1`, records the paper-listed embedding dataset DOI,
and adds verified MD5/SHA-256 values for both Zenodo archives. This is a documentation/manifest correction only;
scientific implementations, public model revisions, and protocols remain unchanged.
The attached expanded source archive contains 933 files and is 36,561,986 bytes; two independent builds were
byte-identical, archive validation passed, and its SHA-256 is
`4123c4c65ec60a1282ffe913fccb479db549b317aedbede317a9532096a86235`.

Release `v0.2.3` corrects the downstream reporting/candidate scorer terminology from historical `clean_non_pad` to
canonical `fixed_epsilon_non_pad`. The producer is fixed at `t=1e-3`; it is not exact clean `t=0`. The 9.17 GB
checkpoint was renamed locally without binary conversion, retaining SHA-256
`c0d7c2be49ef179a25a19dcd9c54c592c282b6961e51aff60e95fabc13786802`. Generation sampling still uses its
separate random-time noisy checkpoint. The canonical scorer is now public in Zenodo v2.0.0 with the same bytes and
SHA-256; scientific protocols are unchanged.
