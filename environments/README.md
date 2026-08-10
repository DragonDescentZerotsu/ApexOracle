# Environment policy

ApexOracle does not force the five scientific modules into one Python environment. Use each submodule's documented
environment:

- prediction/scoring: ApexOracle-Core plus ApexOracle-MDLM;
- guided generation: ApexOracle-Generation plus ApexOracle-MDLM and Core-owned condition assets;
- DLM pretraining: ApexOracle-DLM-Pretraining;
- genome extraction: ApexOracle-Evo2.

Each owning module now records the environment used for its fresh-clone or runtime gate. The super-repo intentionally
does not duplicate those profiles or claim that one environment can run all five modules. Root scripts must never
silently depend on an author's pre-existing conda environment.

The root CI validates only orchestration contracts under `tests/`. It does not recursively collect each submodule's
scientific tests; those remain owned by the module's documented environment and CI.
