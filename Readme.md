![ApexOracle](./assets/hf.png)
<p align="center">
<h1 align="center"><strong>Predicting and generating antibiotics against future pathogens with ApexOracle</strong></h1>
  <p align="center">
    <a href='https://scholar.google.com/citations?user=uL97fK8AAAAJ' target='_blank'>Tianang Leng </a><sup><img src="assets/upenn.png" align="center" width=2.7% >&#8224;</sup>&emsp;
    <a href='https://scholar.google.com/citations?hl=en&user=-_X99PYAAAAJ&view_op=list_works&sortby=pubdate' target='_blank'>Fangping Wan </a><sup><img src="assets/upenn.png" align="center" width=2.7% >&#8224;</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=N-Z6jh8AAAAJ&hl=en' target='_blank'>Marcelo Der Torossian Torres </a><sup><img src="assets/upenn.png" align="center" width=2.7% ></sup>&emsp;
    <a href='https://delafuentelab.seas.upenn.edu/principal-investigator/' target='_blank'>Cesar de la Fuente </a><sup><img src="assets/upenn.png" align="center" width=2.7% ></sup>&emsp;
    <br>
    <sup><img src="assets/upenn.png" align="center" width=2.7% ></sup> University of Pennsylvania
    <br>
    <sup>&#8224;</sup>: Equal contribution
    <br>

  <p align="center">
    <a href='https://www.alphaxiv.org/overview/2507.07862v1'>
      <img src='https://img.shields.io/badge/alphaXiv Blog-2507.07862-A42C25?style=flat&logo=arXiv&logoColor=A42C25'></a>
    <a href='https://arxiv.org/pdf/2507.07862'>
      <img src='https://img.shields.io/badge/Paper-PDF-pink?style=flat&logo=arXiv&logoColor=pink'></a>
    <a href='https://huggingface.co/Kiria-Nozan/ApexOracle'>
      <img src='https://img.shields.io/badge/HuggingFace-Model-orange?logo=huggingface&style=flat'></a>  
    <a href='https://huggingface.co/datasets/Kiria-Nozan/ApexOracle'>
      <img src='https://img.shields.io/badge/HuggingFace-Dataset-green?logo=huggingface&style=flat'></a>
  </p>

  </p>
</p>

----
## Content Direction
- Quick start implementation for **molecule embedding extraction** with our Diffusion Language Model (DLM) can be found at [HuggingFace](https://huggingface.co/Kiria-Nozan/ApexOracle) 🤗
- Code for pre-training our DLM can be found in folder [DLM_pretrain](./DLM_pretrain)
- Flexible and accurate transforming between peptides and SMILES/SELFIES: [PepLink✨](./PepLink)
- To train and reproduce the results in the paper, please check folder [ApexOracle](./ApexOracle)
- To do inference with our pre-trained DLM and predict MICs of peptides, please check folder [mdlm](./mdlm)
- To do guided generation of peptides against specific pathogens, please check folder [discrete-diffusion-guidance](./discrete-diffusion-guidance)
---
## Detailed File Direction
- Use [mdlm/temp_save_milk_embedding.py](./mdlm/temp_save_milk_embedding.py) to extract molecule embeddings without huggingface.
- Use [mdlm/temp_judge_generated_mols_MIC.py](./mdlm/temp_judge_generated_mols_MIC.py) to predict MICs of given or generated molecules. Results will be saved as a .csv file.
- Use [mdlm/temp_judge_mol_mic_with_fig.py](./mdlm/temp_judge_mol_mic_with_fig.py) to predict MICs of given or generated molecules. Molecuels with MIC lower than 15 $\mu mol~l^{-1}$ will be saved as .png files.
- Use [discrete-diffusion-guidance/main.py](./discrete-diffusion-guidance/main.py) to do guided generation of peptides against specific pathogens. Change the parameters in [discrete-diffusion-guidance/configs/config.yaml](./discrete-diffusion-guidance/configs/config.yaml) to change the generation settings.

## Citation
If you find this repo helpful, please cite:
```Tex
@article{leng2025predicting,
  title={Predicting and generating antibiotics against future pathogens with ApexOracle},
  author={Leng, Tianang and Wan, Fangping and Torres, Marcelo Der Torossian and de la Fuente-Nunez, Cesar},
  journal={arXiv preprint arXiv:2507.07862},
  year={2025}
}
```
