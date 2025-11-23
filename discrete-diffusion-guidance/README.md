# Simple Guidance Mechanisms for Discrete Diffusion Models

[![arXiv](https://img.shields.io/badge/arXiv-2412.10193-red.svg)](https://arxiv.org/abs/2412.10193)
[![deploy](https://img.shields.io/badge/Blog%20%20-8A2BE2)](https://discrete-diffusion-guidance.github.io/)
[![deploy](https://img.shields.io/badge/Huggingface%20-UDLM%20-blue)](https://huggingface.co/collections/kuleshov-group/udlm-675e63ab42bc757093099e1b)

<p align="center">
    <img src="https://discrete-diffusion-guidance.github.io/static/images/udlm.gif" alt="graphical abstract" width="450"/>
</p>

This repository contains code for reproducing experiments in the paper [Simple Guidance Mechanisms for Discrete Diffusion Models](https://arxiv.org/abs/2412.10193)

We also share [trained models](https://huggingface.co/collections/kuleshov-group/udlm-675e63ab42bc757093099e1b) on HuggingFace 🤗 and support intergration with these models.
See the "[Using HuggingFace Models" section](#using-huggingface-models) below.

## Code Organization
<a name="code-organization"></a>
1. ```main.py```: Routines for training (language models and classifiers)
2. ```noise_schedule.py```: Noise schedules
3. ```diffusion.py```: Forward/reverse diffusion
    - Absorbing state / uniform noise diffusion
    - AR
4. ```dataloader.py```: Dataloaders
   - For Discretized CIFAR10 and the Species10 datasets we use custom dataset classes defined in ```custom_datasets/```
5. ```utils.py```: LR scheduler, logging, `fsspec` handling
6. ```models/```: Denoising network architectures.
7. ```configs/```: Config files for datasets/denoising networks/noise schedules/LR schedules
8. ```scripts/```: Shell scripts for training/evaluation
9. ```guidance_eval/```: Guidance evaluation scripts


### Implemented Decoding Mechanisms
<a name="implemented-decoding"></a>
In [`diffusion.py`](./diffusion.py),
we define baseline and proposed decoding mechanisms for guidance.
These decoding schemes can be controlled via the hydra config with the `guidance` field.
For example, to use the proposed D-CFG guidance mechanism,
set `guidance=cfg` in the config file and optionally set the `guidance.gamma` parameter to control the strength of the guidance signal.

The implemented decoding methods are as follows:
- AR (Baseline):
   - Standard decoding (i.e., no-guidance); set `guidance=null`
   - Classifier-free guidance (D-CFG); set `guidance=cfg`
   - Classifier-based guidance using [FUDGE](https://arxiv.org/abs/2104.05218) (set `guidance=fudge`) and using [PPLM](https://arxiv.org/abs/1912.02164) (set `guidance=pplm`)
- Diffusion:
  - Standard decoding (i.e., no guidance); set `guidance=null`
  - Classifier-free guidance (D-CFG); set `guidance=cfg`
  - Classifier-based guidance (D-CBG); set `guidance=cbg`
  - Classifier-based (baseline) method of [NOS](https://arxiv.org/abs/2305.20009); set `guidance=nos`

### Implemented Generative Models
<a name="implemented-models"></a>
The three modeling parameterizations
we explore in this work are:
1. Autoregressive (AR) Models
2. Masked Diffusion Language Models (MDLM)
3. Uniform Diffusion Language Models (UDLM)

The `config` files can be used
to specify which of these parameterizations to use.
Below we detail which config parameters correspond to which model.

**AR**
```bash
diffusion="absorbing_state"  # AR models can be thought of as a special case of abosrbing state diffusion models
parameterization="ar"
T=0  # N/A for AR models, this is a placeholder
time_conditioning=False  # AR models are not conditioned on time
zero_recon_loss=False  # N/A for this model
```

**MDLM**
```bash
diffusion="absorbing_state"
parameterization="subs"  # See MDLM paper for details: https://arxiv.org/abs/2406.07524
T=0  # Indicates continuous-time, e.g. T --> infinity
time_conditioning=False  # MDLM not conditioned on time
zero_recon_loss=False  # N/A for this model
```

**UDLM**
```bash
diffusion="uniform"
parameterization="d3pm"  # Indicates that we explicitly compute KL on posteriors
T=0  # Indicates continuous-time, e.g. T --> infinity
time_conditioning=True  # UDLM is conditioned on time
zero_recon_loss=True  # In continuous time, recon loss evaluates to zero
```

## Getting started in this repository
<a name="getting-started"></a>

To get started, create a conda environment containing the required dependencies.

```bash
conda env create -f requirements.yaml
conda activate discdiff
```

Create the following directories to store saved models and slurm logs:
```bash
mkdir outputs
mkdir watch_folder
```

We rely on `wandb` integration
to log experiments and eval curves.

## Reproducing Experiments
<a name="reproducing-experiments"></a>

Below, we describe the steps required for reproducing the experiments in the paper.
Throughout, the main entry point for running experiments is the [`main.py`](./main.py) script.
We also provide sample `slurm` scripts for launching pre-training and evaluation experiments in the [`scrips/`](./scripts) directory.


### Language Modeling Experiments
<a name="lm_training"></a>
To reproduce the language modeling results, please refer to the following shell scripts in the [`scripts/`](./scripts) directory:
- Species10: [`train_ten_species_guidance.sh`](./scripts/train_ten_species_guidance.sh)
- QM9: [`train_qm9_no-guidance.sh`](./scripts/train_qm9_no-guidance.sh)
- CIFAR10: [`train_cifar10_unet_guidance.sh`](./scripts/train_cifar10_unet_guidance.sh)
- text8: [`train_text8.sh`](./scripts/train_text8.sh)
- Amazon Polarity: [`train_amazon_polarity.sh`](./scripts/train_amazon_polarity.sh)
- LM1B: [`train_lm1b.sh`](./scripts/train_lm1b.sh)

Each script contains a comment detailing the usage.
For example, to train either an AR,
MDLM, or UDLM model on the `text8` dataset, use the following command:
```bash
cd scripts/
MODEL=<ar|mdlm|udlm>
sbatch \
  --export=ALL,MODEL=${MODEL} \
  --job-name=train_text8_${MODEL} \
  train_text8.sh
```
### Guidance Training
<a name="guidance-training"></a>
#### Classifier-Free
<a name="guidance-training-cfg"></a>
For classifier-free guidance we require training models
that can condition on the class label
to model conditional distributions,
and we randomly mask out the signal,
replacing it with a dummy value of `num_claseses + 1`, to simulate an unconditional model.
Refer to the shell scripts with the `_guidance` suffix
to train these models for CIFAR10,
QM9, and Species10 datasets.
For QM9, we have two experiments,
one where we condition on the drug-likeness
(`qed`)
of the molecules and another
where we condition on the ring counts (`ring_count`).

#### Classifier-Based
<a name="guidance-training-cbg"></a>
For classifier-based guidance,
we need to train a classifier on the noisy latent samples.
Refer to the following shell scripts
to train these classifiers:
- [FUDGE](https://arxiv.org/abs/2104.05218) (AR guidance): [`train_qm9_fudge_classifier.sh`](./scripts/train_qm9_fudge_classifier.sh)
- D-CBG (diffusion guidance): [`train_qm9_classifier.sh`](./scripts/train_qm9_classifier.sh)

##### PPLM / NOS baselines
An alternative classifier-based guidance mechanism to D-CBG is that of [PPLM](https://arxiv.org/abs/1912.02164)
(which was adapted for diffusion models in [NOS](https://arxiv.org/abs/2305.20009)).
To train these classifiers,
refer to the following shell script:
[`train_qm9_pplm_classifier.sh`](./scripts/train_qm9_pplm_classifier.sh)
(for both PPLM and NOS classifiers).

### Guidance Evaluation
<a name="guidance-eval"></a>
To evaluate guidance mechanisms, we load trained models
(and classifiers, if applicable)
and generate some number of samples
for which we compute "quality" metrics
(e.g., validity/novelty in the QM9 experiments)
and control label satisfaction (e.g., mean value of novel generated molecules for the property of interest in the QM9 experiments).

The scripts for these evaluations can be found in the [`guidance_eval/`](./guidance_eval) directory.
To run these evaluations, please refer to the following shell scripts:
- QM9: [`eval_qm9_guidance.sh`](./guidance_eval/eval_qm9_guidance.sh)
- Species10: [`eval_ten_species_guidance.sh`](./guidance_eval/eval_ten_species_guidance.sh)
  - For this dataset, we also evaluate the accuracy of a HyenaDNA classifier on correctly classifying generated sequences.
    This model can be trained using [`train_ten_species_eval_classifier.sh`](./scripts/train_ten_species_eval_classifier.sh).
    - To see how this trained evaluation classifier performs on the validation set of the original data use this notebook [`eval_hyenadna_classifier.ipynb`](./notebooks/eval_hyenadna_classifier.ipynb).

In the paper,
we performed an extensive hyperparameter sweep for our proposed guidance mechanisms and for baselines.
The shell scripts can be used
to reproduce these experiments,
e.g., for the D-CFG experiments on QM9:
```bash
export MODEL=<ar|mdlm|udlm>
export PROP=<qed|ring_count>
export GUIDANCE=cfg
for GAMMA in $(seq 1 5); do
    sbatch \
      --export=ALL,MODEL=${MODEL},PROP=${PROP},GUIDANCE=${GUIDANCE},GAMMA=${GAMMA} \
      --job-name=eval_qm9_${GUIDANCE}_${PROP}_${MODEL}_GAMMA-${GAMMA} \
      eval_qm9_guidance.sh
done
```

Once each evaluation run is complete,
a `.csv` file
containing the results is saved in the run directory of the trained generative model.

## Using HuggingFace Models
<a name="hf_models"></a>
We provide pre-trained models on HuggingFace 🤗:
- UDLM trained on LM1B: [kuleshov-group/udlm-lm1b](https://huggingface.co/kuleshov-group/udlm-lm1b)
- UDLM trained on QM9: [kuleshov-group/udlm-qm9](https://huggingface.co/kuleshov-group/udlm-qm9)
  - Note: this model was trained without guidance and can be used with classifier-free guidance.

Please see the README pages for these models on HuggingFace or our paper for more details about the training of these models.

To use these models, you can load them using the HuggingFace API, e.g.,
```python
from transformers import AutoModelForMaskedLM

model = AutoModelForMaskedLM.from_pretrained("kuleshov-group/udlm-lm1b")
```

To use these models in our repository, set the following `config` parameters:
```bash
backbone="hf_dit"
model="hf"
model.pretrained_model_name_or_path="kuleshov-group/udlm-lm1b"  # or "kuleshov-group/udlm-qm9"
```

## Acknowledgements
<a name="acknowledgements"></a>
This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm),
which in used [SEDD](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion).
Our code implementation of D-CBG is adapted from Nisonoff et al.'s [repo](https://github.com/hnisonoff/discrete_guidance). 

## Citation
<a name="citation"></a>
```
@article{
    schiff2024discreteguidance,
    title={Simple Guidance Mechanisms for Discrete Diffusion Models},
    author={Schiff, Yair and Sahoo, Subham Sekhar and Phung, Hao and Wang, Guanghan and Boshar, Sam and Dalla-torre, Hugo and de Almeida, Bernardo P and Rush, Alexander and Pierrot, Thomas and Kuleshov, Volodymyr},
    journal={arXiv preprint arXiv:2412.10193},
    year={2024}
}
```
