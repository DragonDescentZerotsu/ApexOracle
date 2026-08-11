# Compute requirements and runtime reporting

This page distinguishes verified execution requirements from fresh quickstart
measurements. It must not be used to infer whole-paper training cost from a
quickstart smoke. Machine-readable measurements and exact asset revisions are
frozen in `manifests/quickstart_benchmarks_2026-08-11.json`.

| Workflow | Released assets | Minimum public path | Verified hardware boundary | Runtime measurements |
| --- | ---: | --- | --- | --- |
| MIC prediction quickstart | 2.875 GB checkpoint + 1.76 MB condition | CPU inference | Public CPU quickstart reproduces `11.79631996 µM` | Empty-cache download: 32.26 s and 2,205,452 KiB peak RSS; computation: 7.27 s and 6,053,272 KiB (5.77 GiB) peak RSS |
| Guided-generation quickstart | 4.059 GB bundle | CUDA GPU, one sample, 256 steps | Completed on one NVIDIA H100 PCIe 80 GB | Empty-cache download: 46.22 s and 2,853,784 KiB peak RSS; 256-step computation: 40.34 s, 6,070,044 KiB peak RSS and 12,281 MiB peak GPU memory |
| Evo-2 paper embedding producer | Model weights are external; paper embeddings are downloadable from Zenodo | Paper profile uses Evo2-40B | The released extraction path passed a real Evo2-40B GPU smoke; upstream documentation requires Hopper-class execution for the 40B profile | Full 567-genome runtime and peak VRAM: not reconstructed |
| DLM + MTR pretraining | Full training data/model run is separate from downstream inference | Multi-GPU training environment | Released producer passed a synthetic train/save/load smoke on H100 | Full historical pretraining runtime: not reconstructed |
| Molecule DLM embedding | 389 MB safetensors model | CPU or CUDA through the Hugging Face runtime | Fresh-cache padded GPU inference passed | Per-batch wall time and peak memory: pending standardized benchmark |

The public quickstarts intentionally use precomputed pathogen conditions. A
reader can therefore test MIC inference or generation without first running
Evo2-40B or Me-LLaMA over the paper corpus.

## Measurement protocol used

Both public quickstarts were run from empty asset caches on 2026-08-11. The
host used one AMD EPYC 9534 CPU (64 physical cores, 128 logical CPUs), 755 GiB
installed RAM and, for generation, one NVIDIA H100 PCIe with 81,559 MiB memory.
The software stack was Linux 5.15/glibc 2.35, Python 3.12.7, PyTorch
2.6.0+cu126 and CUDA runtime 12.6. We recorded:

- hardware model and available RAM/VRAM;
- operating system, Python, PyTorch and CUDA versions;
- exact command and immutable asset revision;
- download time separately from model load and computation time;
- peak resident RAM and, for CUDA, peak allocated/reserved VRAM;
- output hash or documented stochastic completion criterion.

Download and computation are reported separately. These are single-run
operational measurements on the stated host, not performance guarantees. The
MIC output SHA-256 and Generation run-manifest SHA-256 are recorded in the
machine-readable benchmark manifest.
