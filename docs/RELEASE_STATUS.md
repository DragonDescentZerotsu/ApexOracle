# Release status

## Verified facts

- The existing `DragonDescentZerotsu/ApexOracle` repository is being converted in place; no second super-repository
  is being created.
- The legacy tree is recoverable from branch `legacy-monorepo` and annotated tag
  `legacy-monorepo-snapshot-2026-08-10`.
- `ApexOracle-DLM-Pretraining` is locked to `362ffccac79bdd638a4e913c4f17df613da18f36`; its original source
  recovery tag, 56-file manifest, source contracts, remote CI, fresh clone, and H100 joint-objective train/save/load
  smoke have passed.
- `ApexOracle-MDLM` is locked to `c9d17c7f6f091234aaaebf5f08dbe23542f980c1`.
- `ApexOracle-Evo2` is locked to `2184211acda07b0d5ca865067174ac42f530ad04`; its CPU contracts, package
  archives, remote CI, and Evo2-40B extraction runtime smoke have passed.
- `ApexOracle-Generation` is locked to `de6c1e590c25b2ce36b4ce5c42c5a4fa0dcc7705`.
- The original `Synergy` repository was renamed in place to public `ApexOracle-Core`, released as `v0.1.0`, and is
  locked to `8c1def518ac148a878c14f4a39876db59649d43c`. Its 217-test local release gate, wheel/sdist, public-history audit,
  and fresh Hugging Face MIC inference have passed.
- The root repository contains no model weights, embeddings, datasets, or raw experiment outputs after conversion.

## Module gates

All five module repositories are public and pinned to immutable commits. No module uses a floating placeholder.

## Final release gates

- all five gitlinks match the lock manifest;
- MIC prediction runs from a fresh recursive clone; the compact guided-generation asset quickstart remains pending;
- weights and data have stable URIs, revisions, SHA-256 values, and redistribution decisions;
- license/NOTICE, secret, large-file, and broken-link audits pass;
- a full-source archive expands all fixed submodules.
