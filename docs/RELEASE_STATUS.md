# Release status

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
  `931e3dc09bfc2501809c03dbd016741406950f5f`.
- `ApexOracle-Evo2` is locked to `2184211acda07b0d5ca865067174ac42f530ad04`; its CPU contracts, package
  archives, remote CI, and Evo2-40B extraction runtime smoke have passed.
- `ApexOracle-Generation` scientific implementation was validated at
  `80d9a2cf9b0921f29e4a44edf5557ccac39f5af9`; the current documentation-closure lock is
  `67b593e1a623af3af80c64e263bde527d73d89ef`. Compact BAA-3170
  inference-only assets are fixed at Hugging Face revision `2fb1aa08187eaa359263be6c12c8a41868d8959c` and have passed
  empty-cache hashes plus a real 256-step H100 smoke.
- The original `Synergy` repository was renamed in place to public `ApexOracle-Core`; its scientific release `v0.1.0`
  is `8c1def518ac148a878c14f4a39876db59649d43c`; the `v0.2.3` documentation-closure lock is
  `1973d2d3cc6b27202a3960c363c207dd030f74e7`, and post-release `main` is `8751c80cb86c3382a9fc3c8689666e992c0ee7a9`
  for the strain-mapping exporter/manifest. The earlier 217-test release gate, wheel/sdist, public-history audit and
  fresh Hugging Face MIC inference passed; the mapping batch independently passed 208 tests with 4 external-asset
  skips.
- The root repository contains no model weights, embeddings, datasets, or raw experiment outputs after conversion.
- Core publishes a compact paper strain mapping at commit `8751c80`: 1,766 unique source labels, 1,769 condition
  routes and 92,322 routed MIC records before length filtering. It contains condition keys and filenames, not MIC
  labels, molecule structures, embedding tensors or private assay rows.
- The manuscript-listed Zenodo dataset record `15612048` is public under CC BY 4.0 and contains
  `Genome_embs.tar.gz` plus `Text_Description.tar.gz`. It is an external data record, so no archive is copied into Git.
- The six active trees now have a CI-enforced anti-bloat policy: exact duplicates at or above 20 KiB, unallowlisted
  files above 1 MiB, checkpoint/cache suffixes, generated build/cache paths and per-repository count/byte-limit
  violations fail `scripts/check_repository_bloat.py`. The current audit found no >=20 KiB exact duplicates.
- `manifests/paper_data_capsule_plan.json` distinguishes exact classification folds, the exact 2026 fixed MIC
  reconstruction, unrecovered 2025 MIC membership, and the count-matched but unproven synergy seed-0 candidate.
  Full tables and predictions remain external-capsule assets rather than Git payloads.

## Module gates

All five module repositories are public and pinned to immutable commits. No module uses a floating placeholder.

## Final release gates

- all five gitlinks match the lock manifest;
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
separate random-time noisy checkpoint; public model revisions and scientific protocols are unchanged.
