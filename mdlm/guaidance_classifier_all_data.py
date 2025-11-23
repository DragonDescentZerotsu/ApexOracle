from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
torch.set_float32_matmul_precision("medium")
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
from collections import Counter
import itertools
import logging
from datasets import load_from_disk
from torch.utils.data import DataLoader
import hydra
from hydra import compose, initialize
import models
from collections import OrderedDict
import noise_schedule

import torch.nn.functional as F
import ast

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from diffusion import Diffusion

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
        _eps_t = torch.rand(n, device=device)  # * 0
        t = (1 - sampling_eps) * _eps_t + sampling_eps
        return t

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

class ClsHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        hidden_dim_1 = 384,
        hidden_dim_2 = 128,
        num_targets = 19,
        pooler_dropout: float=0.2,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        # x = torch.sigmoid(x)
        return x

class Pep_SM_Classifier(L.LightningModule):
    def __init__(self, config, vocab_size, backbone_ckpt_path, mask_index, lr, num_epochs, pos_weight):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = mol_emb_mdlm(config, vocab_size, backbone_ckpt_path, mask_index)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.ClsHead = ClsHead(input_dim=768, num_targets=1)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, x):
        x = self.backbone(x)
        x = x[:, 0, :]
        x = self.ClsHead(x)
        return x

    def training_step(self, batch, batch_idx):
        labels = batch['labels'].float()
        pred = self(batch['input_ids'])
        loss = self.criterion(pred.squeeze(), labels)
        acc = ((torch.sigmoid(pred)>0.5).squeeze().int() == labels).float().mean()
        self.log("global_step", self.global_step, prog_bar=True)

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch['labels'].float()
        pred = self(batch['input_ids'])
        loss = self.criterion(pred.squeeze(), labels)
        acc = ((torch.sigmoid(pred)>0.5).squeeze().int() == labels).float().mean()

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.num_epochs,
            eta_min=1e-9
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',     # 调度器更新频率：'step' 或 'epoch'
                'frequency': 1,          # 每 1 个 interval 更新一次
                # 'monitor': 'val_loss',  # 如果是 ReduceLROnPlateau，则需要 monitor
            }
        }


if __name__ == '__main__':
    ckpt_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch-{epoch}-step-{step}-train_loss-{train_loss:.3f}",
        monitor="train_loss",
        mode="min",
        every_n_train_steps=1000,
        save_top_k=-1,
        save_last=True,
        verbose=True,
    )

    wandb_logger = WandbLogger(
        project="Pep_SM_Classification",  # W&B 中的项目名
        name="run1",  # 本次运行的显示名称
        save_dir="wandb_logs",  # 本地日志保存路径
        log_model=False,  # 是否将 checkpoint 作为 artifact 上传
        offline=False  # 是否离线模式，不实时上传
    )

    max_epochs = 10

    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/last_reg_v2.ckpt'
    # model = Pep_SM_Classifier(config, len(tokenizer.get_vocab()), DIT_ckpt_path, tokenizer.mask_token_id, 1e-5, max_epochs)

    dataset_path = '/data2/tianang/projects/Synergy/DataPrepare/MDLM/Data/hf_pep_SM_cls_1024'
    dataset = load_from_disk(dataset_path)
    splits = dataset.train_test_split(test_size=0.01, seed=42)
    train_ds = splits["train"]
    val_ds = splits["test"]

    # labels = train_ds["labels"]

    # Counter 统计
    # print(f' counting pos neg ratio')
    # counter = Counter(labels)
    # num_neg, num_pos = counter[0], counter[1]

    # num_neg, num_pos = 0, 0
    # for _label in tqdm(labels, desc=f' counting pos neg ratio', unit='line'):
    #     if _label == 1:
    #         num_pos += 1
    #     else:
    #         num_neg += 1

    pos_rate = 0.125 #num_pos / (num_neg + num_pos)
    neg_rate = 0.875 #num_neg / (num_neg + num_pos)

    print(f' pos_rate: {pos_rate}\n neg_rate: {neg_rate}')
    pos_weight = torch.tensor(neg_rate / pos_rate)

    DIT_ckpt_path = '/data2/tianang/projects/mdlm/Checkpoints_fangping/last_reg_v2.ckpt'
    model = Pep_SM_Classifier(config, len(tokenizer.get_vocab()), DIT_ckpt_path, tokenizer.mask_token_id, 1e-5,
                              max_epochs, pos_weight)

    # print(train_ds[0])
    train_ds.set_format(type="torch", columns=["mol_ids", "input_ids", "labels"])
    val_ds.set_format(type="torch", columns=["mol_ids", "input_ids", "labels"])

    # 4. 创建 DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=300,
        shuffle=True,
        num_workers=30,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=300,
        shuffle=False,
        num_workers=30,
    )

    trainer = L.Trainer(
        default_root_dir=str(current_directory / "Checkpoints" / "lg_outputs"),  # 所有相对路径都会基于 outputs/
        callbacks=[ckpt_cb],
        logger=wandb_logger,
        accelerator='cuda',
        strategy="ddp",
        devices=3,
        max_epochs=max_epochs,
        precision='bf16',
    )

    trainer.fit(model, train_loader, val_loader,
                ckpt_path='/data2/tianang/projects/mdlm/checkpoints/epoch-epoch=0-step-step=87000-train_loss-train_loss=0.051.ckpt'
                )

