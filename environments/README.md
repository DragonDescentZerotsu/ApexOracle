# Environment policy

ApexOracle does not force the five scientific modules into one Python environment. Use each submodule's documented
environment:

- prediction/scoring: ApexOracle-Core plus ApexOracle-MDLM;
- guided generation: ApexOracle-Generation plus ApexOracle-MDLM and Core-owned condition assets;
- DLM pretraining: ApexOracle-DLM-Pretraining;
- genome extraction: ApexOracle-Evo2.

Versioned environment profiles will be added only after their owning modules pass fresh-clone installation. Root
scripts must never silently depend on an author's pre-existing conda environment.

The root CI validates only orchestration contracts under `tests/`. It does not recursively collect each submodule's
scientific tests; those remain owned by the module's documented environment and CI.
