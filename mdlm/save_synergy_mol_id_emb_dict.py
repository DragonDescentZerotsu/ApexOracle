from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import AgglomerativeClustering
from Bio import Phylo
from triton.language import bfloat16
from scipy.stats import pearsonr, spearmanr
import json
import itertools
import logging

import hydra
from hydra import compose, initialize

import models
from collections import OrderedDict
import noise_schedule

import torch.nn.functional as F
import selfies as sf

# current_directory = Path(__file__).parent
current_directory = Path('/data2/tianang/projects/Synergy')

with initialize(config_path="configs"):
    config = compose(config_name="config")

class mol_emb_mdlm(nn.Module):
    def __init__(self, config, vocab_size, ckpt_path, mask_index):
        super(mol_emb_mdlm, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.mask_index = mask_index
        self.ckpt_path = ckpt_path
        self.parameterization = self.config.parameterization
        self.time_conditioning = self.config.time_conditioning
        self.backbone = self.load_DIT()  # hidden_size = 768
        # print(self.bert.config.max_position_embeddings)
        self.noise = noise_schedule.get_noise(self.config)

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == 'ar'
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def _sample_t(self, n, device):
        sampling_eps = 1e-3
        _eps_t = torch.rand(n, device=device)  # 因为是要做性质预测了这里
        t = (1 - sampling_eps) * _eps_t + sampling_eps
        return t * 0

    def _forward(self, x, sigma):
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = self.backbone.vocab_embed(x)
            c = F.silu(self.backbone.sigma_map(sigma))
            rotary_cos_sin = self.backbone.rotary_emb(x)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i in range(len(self.backbone.blocks)):
                    x = self.backbone.blocks[i](x, rotary_cos_sin, c, seqlens=None)

        return x

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        xt = torch.where(move_indices, self.mask_index, x)
        return xt

    def forward(self, input_ids, attention_mask=None):
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, dsigma = self.noise(t)
        unet_conditioning = sigma[:, None]
        move_chance = 1 - torch.exp(-sigma[:, None])
        xt = self.q_xt(input_ids, move_chance)
        outputs = self._forward(xt, unet_conditioning)
        return outputs

    def load_DIT(self):
        backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        lightning_ckpt = torch.load(self.ckpt_path, map_location='cpu')
        state_dict = lightning_ckpt['state_dict']

        new_sd = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                new_key = k[len('backbone.'):]
            else:
                new_key = k
            new_sd[new_key] = v

        backbone.load_state_dict(new_sd, strict=False)

        return backbone


if __name__ == '__main__':

    device = 'cuda:0'

    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    pooling_method = 'cls_wo_pad' # cls_wo_pad / mean_w_pad / cls_w_pad / mean_wo_pad / cls_wo_pad_eval / mean_wo_pad_eval

    Evo_strain_FICI_data_path = current_directory / 'DataPrepare' / 'Data' / 'synergistic_pairs_Evo.csv'  # 'DBAASP_id_bact_name_SMILES_MIC_Evo.csv'
    all_Evo_MIC_data = pd.read_csv(Evo_strain_FICI_data_path).values

    # Evo_strain_MIC_data_path = current_directory / 'DataPrepare' / 'Data' / 'small_molecule' / 'processed' / 'small_molecule_Evo_binary_data_SELFIES.csv'
    # SM_Evo_binary_data = pd.read_csv(Evo_strain_MIC_data_path)

    # Pep_ids = all_Evo_MIC_data['DBAASP_id'].values
    # Pep_input_ids = all_Evo_MIC_data['SMILES'].values
    #
    # SM_ids = SM_Evo_binary_data['DBAASP_id'].values
    # SM_input_ids = SM_Evo_binary_data['SMILES'].values

    mol_dict = {}
    for mol_id_1, mol_id_2, _, mol_smiles_1, mol_smiles_2, _ in tqdm(all_Evo_MIC_data, desc='filtering peptides'):
        if mol_id_1 not in mol_dict.keys():
            # input_ids_list = ast.literal_eval(Pep_input_id)
            # if len(input_ids_list) <= 512:
            mol_selfies = sf.encoder(mol_smiles_1)
            input_ids = tokenizer(  # 这样默认就是有 special tokens 的
                            mol_selfies.replace('][', '] ['),
                            return_tensors='pt',
                            padding=False,
                            truncation=False,
                        )['input_ids'].squeeze(0)
            if tokenizer.unk_token_id in input_ids:
                print(f'unknown token id in {mol_id_1}')
                continue
            if len(input_ids) > 1024:
                print(f'more than 1024 tokens in {mol_id_1}')
                continue
            mol_dict[mol_id_1] = input_ids
        if mol_id_2 not in mol_dict.keys():
            # input_ids_list = ast.literal_eval(Pep_input_id)
            # if len(input_ids_list) <= 512:
            mol_selfies = sf.encoder(mol_smiles_2)
            input_ids = tokenizer(  # 这样默认就是有 special tokens 的
                            mol_selfies.replace('][', '] ['),
                            return_tensors='pt',
                            padding=False,
                            truncation=False,
                        )['input_ids'].squeeze(0)
            if tokenizer.unk_token_id in input_ids:
                print(f'unknown token id in {mol_id_2}')
                continue
            if len(input_ids) > 1024:
                print(f'more than 1024 tokens in {mol_id_2}')
                continue
            mol_dict[mol_id_2] = input_ids

    # SM_dict = {}
    # for SM_id, SM_input_id in tqdm(zip(SM_ids, SM_input_ids), desc='filtering small molecules'):
    #     if SM_id not in SM_dict.keys():
    #         SM_dict[SM_id] = torch.from_numpy(np.array(ast.literal_eval(SM_input_id)))



    # DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/1-255000-fine-tune.ckpt'
    DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/last_reg_v1.ckpt'
    mdlm_model = mol_emb_mdlm(config, len(tokenizer.get_vocab()), DIT_ckpt_path, tokenizer.mask_token_id)
    mdlm_model.to(device)
    mdlm_model.eval()


    mol_emb ={}
    for Pep_id, Pep_input_id in tqdm(mol_dict.items(), desc='computing peptide embeddings:'):
        Pep_input_id = Pep_input_id.to(device)
        with torch.amp.autocast('cuda', enabled=True):
            if pooling_method == 'cls_wo_pad':
                mol_emb[Pep_id] = mdlm_model(Pep_input_id.unsqueeze(0))[:, 0, :].detach().cpu()
            if pooling_method == 'cls_wo_pad_eval':
                mdlm_model.eval()
                mol_emb[Pep_id] = mdlm_model(Pep_input_id.unsqueeze(0))[:, 0, :].detach().cpu()
            elif pooling_method == 'mean_w_pad':
                Pep_input_id = Pep_input_id.unsqueeze(0)
                seq_len = Pep_input_id.shape[-1]
                paddings = torch.ones([1, 1024]) * tokenizer.pad_token_id
                paddings[0, :seq_len] = Pep_input_id[0, :seq_len]
                output = mdlm_model(paddings.long().to(device))[0, :seq_len, :].detach().cpu()
                mol_emb[Pep_id] = torch.mean(output, dim=0)
            elif pooling_method == 'cls_w_pad':
                Pep_input_id = Pep_input_id.unsqueeze(0)
                seq_len = Pep_input_id.shape[-1]
                paddings = torch.ones([1, 1024]) * tokenizer.pad_token_id
                paddings[0, :seq_len] = Pep_input_id[0, :seq_len]
                mol_emb[Pep_id] = mdlm_model(paddings.long().to(device))[:, 0, :].detach().cpu()
            elif pooling_method == 'mean_wo_pad':
                Pep_input_id = Pep_input_id.unsqueeze(0)
                output = mdlm_model(Pep_input_id.long().to(device))[0].detach().cpu()
                mol_emb[Pep_id] = torch.mean(output, dim=0)
            elif pooling_method == 'mean_wo_pad_eval':
                mdlm_model.eval()
                Pep_input_id = Pep_input_id.unsqueeze(0)
                output = mdlm_model(Pep_input_id.long().to(device))[0].detach().cpu()
                mol_emb[Pep_id] = torch.mean(output, dim=0)


    # SM_emb = {}
    # for SM_id, SM_input_id in tqdm(SM_dict.items(), desc='computing small molecule embeddings'):
    #     SM_input_id = SM_input_id.to(device)
    #     with torch.amp.autocast('cuda', enabled=True):
    #         if pooling_method == 'cls_wo_pad':
    #             SM_emb[SM_id] = mdlm_model(SM_input_id.unsqueeze(0))[:, 0, :].detach().cpu()
    #         if pooling_method == 'cls_wo_pad_eval':
    #             mdlm_model.eval()
    #             SM_emb[SM_id] = mdlm_model(SM_input_id.unsqueeze(0))[:, 0, :].detach().cpu()
    #         elif pooling_method == 'mean_w_pad':
    #             SM_input_id = SM_input_id.unsqueeze(0)
    #             seq_len = SM_input_id.shape[-1]
    #             paddings = torch.ones([1, 1024]) * tokenizer.pad_token_id
    #             paddings[0, :seq_len] = SM_input_id[0, :seq_len]
    #             output = mdlm_model(paddings.long().to(device))[0, :seq_len, :].detach().cpu()
    #             SM_emb[SM_id] = torch.mean(output, dim=0)
    #         elif pooling_method == 'cls_w_pad':
    #             SM_input_id = SM_input_id.unsqueeze(0)
    #             seq_len = SM_input_id.shape[-1]
    #             paddings = torch.ones([1, 1024]) * tokenizer.pad_token_id
    #             paddings[0, :seq_len] = SM_input_id[0, :seq_len]
    #             SM_emb[SM_id] = mdlm_model(paddings.long().to(device))[:, 0, :].detach().cpu()
    #         elif pooling_method == 'mean_wo_pad':
    #             SM_input_id = SM_input_id.unsqueeze(0)
    #             output = mdlm_model(SM_input_id.long().to(device))[0].detach().cpu()
    #             SM_emb[SM_id] = torch.mean(output, dim=0)
    #         elif pooling_method == 'mean_wo_pad_eval':
    #             mdlm_model.eval()
    #             SM_input_id = SM_input_id.unsqueeze(0)
    #             output = mdlm_model(SM_input_id.long().to(device))[0].detach().cpu()
    #             SM_emb[SM_id] = torch.mean(output, dim=0)

    torch.save(mol_emb, current_directory / 'DataPrepare' / 'Data' / f'synergy_mol_emb_dict_{pooling_method}.pt')
    # torch.save(SM_emb, current_directory / 'DataPrepare' / 'Data' / f'SM_emb_dict_{pooling_method}.pt')