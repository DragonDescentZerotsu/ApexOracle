# Quickstarts

## Genome embedding extraction

Install the fixed Evo2 module in its documented environment, then inspect the deterministic extraction plan without
loading model weights:

```bash
python -m pip install -e modules/evo2
apexoracle-evo2-extract --input path/to/genomes --output-dir embeddings --plan-only
```

Remove `--plan-only` to run extraction with the module's default Evo2-40B profile. File inputs and flat directories of
FASTA files are supported; tensors and JSON provenance manifests are written together. See
`modules/evo2/README.md` for model, layer, window, device, and batching options.

## End-to-end prediction and generation

Two root quickstarts are planned:

1. predict MIC for one public molecule and known strain using fixed Core/MDLM commits;
2. generate a small target-strain candidate batch using fixed Generation/MDLM/Core commits.

They are not published as executable wrappers yet because Core and its public asset contract are still pending. This
directory intentionally contains no placeholder command that could be mistaken for a validated end-to-end release.
