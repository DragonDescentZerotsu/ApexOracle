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
from hydra import compose, initialize, initialize_config_dir
import models
from collections import OrderedDict
import noise_schedule

import torch.nn.functional as F
import ast
from omegaconf import DictConfig, ListConfig
from huggingface_hub import PyTorchModelHubMixin

# current_directory = Path(__file__).parent
current_directory = Path('/data2/tianang/projects/Synergy')

with initialize_config_dir(config_dir="/data2/tianang/projects/mdlm/configs"):
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


class MolEmbDLM(mol_emb_mdlm, PyTorchModelHubMixin):
    """
    继承你的原模型 + Hub Mixin。
    不重写任何方法，使用默认 _save_pretrained/_from_pretrained。
    """
    pass



def build_hf_config(hydra_cfg, tokenizer):
    return {
        "model_type": "mol_emb_raw",            # 仅标识，不影响逻辑
        "vocab_size": len(tokenizer.get_vocab()),
        "hidden_size": hydra_cfg.model.hidden_size,
        "n_blocks": hydra_cfg.model.n_blocks,
        "n_heads": hydra_cfg.model.n_heads,
        "max_position_embeddings": hydra_cfg.model.length,
        "parameterization": hydra_cfg.parameterization,
        "time_conditioning": hydra_cfg.time_conditioning,
        "noise_schedule_type": hydra_cfg.noise.type,
        "sigma_min": hydra_cfg.noise.sigma_min,
        "sigma_max": hydra_cfg.noise.sigma_max,
        "mask_index": tokenizer.mask_token_id,
        "tokenizer_name_or_path": hydra_cfg.data.tokenizer_name_or_path
    }

if __name__ == '__main__':
    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/1-255000-fine-tune.ckpt'
    model = MolEmbDLM(config, len(tokenizer.get_vocab()), DIT_ckpt_path, tokenizer.mask_token_id)

    hf_config = build_hf_config(config, tokenizer)

    EXPORT_DIR = "/data2/tianang/projects/mdlm/huggingface/huggingface_model"
    model.save_pretrained(EXPORT_DIR, config=hf_config)  # 生成：pytorch_model.bin + config.json
    tokenizer.save_pretrained(EXPORT_DIR)



