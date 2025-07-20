### Train diffusion language model (DLM) on molecular SELFIES
To train the DLM with the multitask descriptor regression loss on our curated SELFIES data, please use the following command:
```
python main.py \
  model=small \
  data=SELFIES \
  wandb.name=dlm-mtr \
  parameterization=subs \
  model.length=1024 \
  sampling.steps=1000
```

### Software version
```
torch: 2.4.1+cu118 
datasets: 2.18.0 
einops: 0.7.0 
fsspec: 2024.2.0 
git-lfs: 1.6 
h5py: 3.10.0 
hydra-core: 1.3.2 
ipdb: 0.13.13 
lightning: 2.2.1 
notebook: 7.1.1 
nvitop: 1.3.2 
omegaconf: 2.3.0 
packaging: 23.2 
pandas: 2.2.1 
rich: 13.7.1 
seaborn: 0.13.2 
scikit-learn: 1.4.0 
transformers: 4.38.2 
triton: 2.1.0 
wandb: 0.13.5 
flash-attn: 2.6.3 
```

### Acknowledgements
This repository was built off of [MDLM](https://github.com/kuleshov-group/mdlm).


