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
import ast


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
        _eps_t = torch.rand(n, device=device) * 0  # 因为是要做性质预测了这里
        t = (1 - sampling_eps) * _eps_t + sampling_eps
        return t * 0

    def _forward(self, x, sigma, attnmask):  # TODO: non pad 不一样的地方
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            x = self.backbone.vocab_embed(x)
            c = F.silu(self.backbone.sigma_map(sigma))
            rotary_cos_sin = self.backbone.rotary_emb(x)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for i in range(len(self.backbone.blocks)):
                    x = self.backbone.blocks[i](x, rotary_cos_sin, c, seqlens=None, attnmask=attnmask)  # TODO: non pad 不一样的地方

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
        outputs = self._forward(xt, unet_conditioning, attnmask = attention_mask)  # TODO: non pad 不一样的地方
        return outputs

    def load_DIT(self):
        backbone = models.dit.DIT_non_pad(self.config, vocab_size=self.vocab_size)  # TODO: non pad 不一样的地方
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

input_ids = torch.from_numpy(np.array([1, 163, 163, 60, 15, 163, 163, 60, 15, 163, 158, 163, 114, 163, 171, 158, 172, 15, 117, 163, 163, 163, 158, 77, 15, 163, 158, 158, 163, 114, 163, 171, 158, 60, 145, 117, 51, 163, 114, 163, 171, 158, 60, 145, 17, 15, 163, 114, 163, 171, 158, 172, 15, 25, 163, 163, 15, 163, 163, 163, 163, 114, 163, 171, 158, 172, 15, 117, 163, 163, 163, 158, 77, 15, 163, 158, 158, 163, 114, 163, 171, 158, 172,]))
input_ids_padded = torch.from_numpy(np.array([1, 163, 163, 60, 15, 163, 163, 60, 15, 163, 158, 163, 114, 163, 171, 158, 172, 15, 117, 163, 163, 163, 158, 77, 15, 163, 158, 158, 163, 114, 163, 171, 158, 60, 145, 117, 51, 163, 114, 163, 171, 158, 60, 145, 17, 15, 163, 114, 163, 171, 158, 172, 15, 25, 163, 163, 15, 163, 163, 163, 163, 114, 163, 171, 158, 172, 15, 117, 163, 163, 163, 158, 77, 15, 163, 158, 158, 163, 114, 163, 171, 158, 172,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3]))
input_ids = input_ids.unsqueeze(0)
input_ids_padded = input_ids_padded.unsqueeze(0)
attnmask = torch.ones_like(input_ids).to(torch.bool)
attnmask_padded = input_ids_padded != 3

model_name = "ibm-research/materials.selfies-ted"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = 'cuda:3'

# DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/1-255000-fine-tune.ckpt'
DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/best_3.ckpt'
mdlm_model = mol_emb_mdlm(config, len(tokenizer.get_vocab()), DIT_ckpt_path, tokenizer.mask_token_id)
mdlm_model.to(device)
mdlm_model.eval()

input_ids = input_ids.to(device)
attnmask = attnmask.to(device)

input_ids_padded = input_ids_padded.to(device)
attnmask_padded = attnmask_padded.to(device)

outputs = mdlm_model(input_ids=input_ids, attention_mask=attnmask)
outputs_padded = mdlm_model(input_ids=input_ids_padded, attention_mask=attnmask_padded)

print(0)

"""
111
222    yes no
333

"""

ffn1 = nn.Linear(768, 256)
ffn2 = nn.Linear(256, 1)

output = ffn1(ffn2(input))