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

## MIC prediction

Install Core and the Hugging Face downloader, then fetch the immutable
single-member quickstart assets:

```bash
python -m pip install "./modules/core[inference]" huggingface_hub
huggingface-cli download Kiria-Nozan/ApexOracle-Core \
  apexoracle_mic_strain_group0_member0_inference.pth \
  example_text_only_dbaasp_2136_atcc_4965.pt \
  --revision 459026cf4ae4e4e38ce5d2cae16ee3871d0a81df \
  --local-dir assets/core-mic

apexoracle-predict-mic \
  --checkpoint assets/core-mic/apexoracle_mic_strain_group0_member0_inference.pth \
  --input assets/core-mic/example_text_only_dbaasp_2136_atcc_4965.pt \
  --device cpu \
  --verify-checkpoint-sha256 007c8f34cbc2d5fb4d26b0b5261cd697a4d84d85551b03eb29be492153589341 \
  --verify-input-sha256 48527beb21111ac45107d2f61bd8262109dd63e4e950ae6d89bc35f63b569f3c
```

The verified output is `11.79631996 µM`. This single member demonstrates the
released inference path; it is not the seven-member paper ensemble and is not
a prospective activity claim. Full schema and provenance details are in
`modules/core/docs/MIC_QUICKSTART.md`.

## Guided generation

Download the fixed compact BAA-3170 bundle (about 4.06 GB):

```bash
huggingface-cli download Kiria-Nozan/ApexOracle-Generation \
  --revision 2fb1aa08187eaa359263be6c12c8a41868d8959c \
  --local-dir assets/generation-baa3170

python modules/generation/scripts/reproduce/run_paper_mic_peptide.py \
  --mdlm-root modules/mdlm \
  --core-root modules/core \
  --asset-root assets/generation-baa3170 \
  --output-dir runs/generation-smoke \
  --strain BAA-3170 \
  --device 0 \
  --smoke \
  --check-asset-hashes
```

This retains all 256 sampling steps, `15/15` guidance, and the frozen remasking
schedule while generating one sample. It validates runtime and output
contracts; it is not deterministic and does not establish activity or yield.
