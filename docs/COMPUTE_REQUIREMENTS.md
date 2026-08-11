# Compute requirements and runtime reporting

This page distinguishes verified execution requirements from measurements that
still need to be captured on a fresh run. It must not be used to infer
whole-paper training cost from a quickstart smoke.

| Workflow | Released assets | Minimum public path | Verified hardware boundary | Runtime measurements |
| --- | ---: | --- | --- | --- |
| MIC prediction quickstart | 2.875 GB checkpoint + 1.76 MB condition | CPU inference | Public CPU quickstart reproduces the documented prediction | Wall time and peak RAM: pending fresh-run capture |
| Guided-generation quickstart | 4.059 GB bundle | CUDA GPU, one sample, 256 steps | Completed on one NVIDIA H100 80 GB | Wall time and peak VRAM: pending fresh-run capture |
| Evo-2 paper embedding producer | Model weights are external; paper embeddings are downloadable from Zenodo | Paper profile uses Evo2-40B | The released extraction path passed a real Evo2-40B GPU smoke; upstream documentation requires Hopper-class execution for the 40B profile | Full 567-genome runtime and peak VRAM: not reconstructed |
| DLM + MTR pretraining | Full training data/model run is separate from downstream inference | Multi-GPU training environment | Released producer passed a synthetic train/save/load smoke on H100 | Full historical pretraining runtime: not reconstructed |
| Molecule DLM embedding | 389 MB safetensors model | CPU or CUDA through the Hugging Face runtime | Fresh-cache padded GPU inference passed | Per-batch wall time and peak memory: pending standardized benchmark |

The public quickstarts intentionally use precomputed pathogen conditions. A
reader can therefore test MIC inference or generation without first running
Evo2-40B or Me-LLaMA over the paper corpus.

## Measurement protocol still required

Before the reviewer response is finalized, run each public quickstart from an
empty asset cache and record:

- hardware model and available RAM/VRAM;
- operating system, Python, PyTorch and CUDA versions;
- exact command and immutable asset revision;
- download time separately from model load and computation time;
- peak resident RAM and, for CUDA, peak allocated/reserved VRAM;
- output hash or documented stochastic completion criterion.

Until those measurements are present, the reviewer response should say that
compute requirements have been documented but should not claim that complete
runtime reporting is finished.
