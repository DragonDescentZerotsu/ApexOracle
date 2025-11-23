'''
这个脚本的作用是：
把每一个生成额序列的分子图上都标注一个预测的 MIC
'''

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import MolDrawOptions, MolToImage
from sympy.codegen.ast import continue_
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import PIL
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
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter, FuncFormatter
import ast
import selfies as sf
import shutil
from PIL import ImageDraw, ImageFont
from smiles_to_peptide import smiles_to_pepseq

from guaidance_regressor_all_data import FirstTokenAttention_genome, RegressionHead, load_all_genome_embeddings, load_text_wo_genome_embeddings
from aa_seq_to_smiles import *

current_directory = Path('/data2/tianang/projects/Synergy')

with initialize(config_path="configs"):
    config = compose(config_name="config")

class mol_emb_mdlm(nn.Module):
    def __init__(self, config, vocab_size, ckpt_path):
        super(mol_emb_mdlm, self).__init__()
        self.config = config
        self.vocab_size = vocab_size
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
        _eps_t = torch.rand(n, device=device)
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


    def forward(self, input_ids, attention_mask=None):
        t = self._sample_t(input_ids.shape[0], input_ids.device)
        sigma, dsigma = self.noise(t)
        unet_conditioning = sigma[:, None]
        outputs = self._forward(input_ids, unet_conditioning)
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

class mol_emb_mdlm_no_weights(mol_emb_mdlm):
    """
    重写 load_DIT 的部分消去对路径的依赖
    """
    def __init__(self, config, vocab_size):
        super().__init__(config, vocab_size, ckpt_path=None)

    def load_DIT(self):
        backbone = models.dit.DIT(self.config, vocab_size=self.vocab_size)
        return backbone


class MIC_regressor(nn.Module):
    """
    融合了 mdlm 和所有的 genome text emb 的部分
    """

    def __init__(self, config, ckpt_path, device):
        super(MIC_regressor, self).__init__()
        self.config = config
        # self.strain_cond = strain_cond
        self.ckpt_path = ckpt_path
        self.device = device
        self.mdlm_model: nn.Module = mol_emb_mdlm_no_weights(config, len(tokenizer.get_vocab()))
        self.co_cross_attn_genome = FirstTokenAttention_genome(self.mdlm_model.config.model.hidden_size, 8192, 4, 0.1)
        self.co_cross_attn_text = FirstTokenAttention_genome(self.mdlm_model.config.model.hidden_size, 4096, 4, 0.1)
        self.reg_head = RegressionHead(8192 + 4096, (8192 + 4096) // 4, 128, 1, 0.2)
        self.learnable_embedding_weight = nn.Parameter(torch.randn(1, 8192, device=device))
        self.load_pretrained_weight()

        self.ATCC_genome_emb_dict, self.ATCC_text_emb_dict, self.text_only_emb_dict = self.load_genome_test_embedding()

    def load_genome_test_embedding(self):
        ATCC_genome_emb_dict = load_all_genome_embeddings(
            Path(current_directory/'DataPrepare'/'Data'/'Genome_embs'),
            1e14,
            'cpu',
            'ATCC genome embedding')
        ATCC_text_emb_dict = load_all_genome_embeddings(
            Path(current_directory/'DataPrepare'/'Data'/'Text_Description'/'ATCC'/'embeddings'),
            1,
            'cpu',
            'ATCC text embedding')
        text_only_emb_dict = load_text_wo_genome_embeddings(
            Path(current_directory/'DataPrepare'/'Data'/'Text_Description'/'wo_ATCC'/'embeddings'),
            1,
            'cpu',
            'text only embedding')
        return ATCC_genome_emb_dict, ATCC_text_emb_dict, text_only_emb_dict

    def load_pretrained_weight(self):
        """
        在使用 forward 之前进行，load 自己要用的权重
        """
        regressor_ckpt_path = self.ckpt_path
        checkpoint = torch.load(regressor_ckpt_path, map_location=self.device)
        # new_sd = OrderedDict()
        # for k, v in checkpoint['mdlm_model_state_dict'].items():
        #     if k.startswith('backbone.'):
        #         new_key = k[len('backbone.'):]
        #     else:
        #         new_key = k
        #     new_sd[new_key] = v
        self.mdlm_model.load_state_dict(checkpoint['mdlm_model_state_dict'])
        self.reg_head.load_state_dict(checkpoint['re_head_state_dict'])
        self.co_cross_attn_genome.load_state_dict(checkpoint['co_cross_attn_genome'])
        self.co_cross_attn_text.load_state_dict(checkpoint['co_cross_attn_text'])
        self.learnable_embedding_weight = checkpoint['learnable_embedding_weight']

    def forward(self, input_ids: torch.Tensor, strain_cond):
        output = self.mdlm_model(input_ids)
        mol_cls_embedding = output[:, 0, :]
        if strain_cond in self.ATCC_genome_emb_dict.keys():
            padded_genome_embeddings = self.ATCC_genome_emb_dict[strain_cond].to(device)
            padded_text_embeddings = self.ATCC_text_emb_dict[strain_cond].to(device)
            padded_genome_embeddings = padded_genome_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
            genome_attn_masks = torch.ones(padded_genome_embeddings.shape[0], padded_genome_embeddings.shape[1]).to(device)
        else:
            padded_genome_embeddings = None
            padded_text_embeddings = self.text_only_emb_dict[strain_cond].to(device)
        padded_text_embeddings = padded_text_embeddings[None, ...].expand(mol_cls_embedding.shape[0], -1, -1)
        text_attn_masks = torch.ones(padded_text_embeddings.shape[0], padded_text_embeddings.shape[1]).to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            if padded_genome_embeddings is not None:
                mol_cls_embedding_genome, _ = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
            else:
                padded_genome_embeddings, _ = self.learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1)
                mol_cls_embedding_genome, _ = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
            mol_cls_embedding_text, _ = self.co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
            mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
            reg_logits = self.reg_head(mol_cls_embedding)

        return reg_logits

aa_smiles_dict = get_aa_smiles_dict('/data2/tianang/projects/Synergy/DataPrepare/Data/all_aa_smiles_new_handcrafted.csv')
aa_smiles_dict['B'] = 'N[C@@H](CCCNC(N)=N)C(=O)O'

def get_mic(file_names, mic_regressor, tokenizer, strain, target_MIC, guidance_method, target_length):
    SELFIES_strs = []
    for file_name in file_names:
        # file_strain = file_name.split('.txt')[0].split('_')[1]
        # file_target_MIC = file_name.split('.txt')[0].split('_')[3]
        # try:
        #     file_guidance_method = file_name.split('.txt')[0].split('_')[-1]
        #     file_target_length = file_name.split('.txt')[0].split('_')[5]
        # except:
        #     continue
        # if strain == file_strain and file_target_MIC == target_MIC and file_guidance_method == guidance_method and file_target_length == target_length:
        if strain in file_name:
            with open(generate_mol_save_dir/file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()  # 返回所有行的列表，每行末尾包含 '\n'
            # 如果想去掉末尾的换行符：
            SELFIES_strs.extend([line.rstrip("\n") for line in lines])

    pep_seqs_linear = []
    SELFIES_strs_linear = []
    for selfies_str in SELFIES_strs:
        _, pep_seq = smiles_to_pepseq(sf.decoder(selfies_str))
        if pep_seq is not None and 'cyclo' not in pep_seq:
            AMP_peptide = Peptide(pep_seq.replace('R', 'B'), aa_smiles_dict, 0, [],
                                  [], [])
            if not AMP_peptide.noise_data_flag:
                smiles_str = Chem.MolToSmiles(AMP_peptide.ncTerminus_modified_mols[0])
            else:
                print(f' can not transform to peptide: {selfies_str}')
                continue
            SELFIES_strs_linear.append(sf.encoder(smiles_str))
            pep_seqs_linear.append(pep_seq)


    input_ids = tokenizer(
        [s.replace('][', '] [') for s in SELFIES_strs_linear],
        return_tensors='pt',
        padding=True,
        truncation=False,
        add_special_tokens=True
    )['input_ids'].squeeze().to(device)

    mics = []
    i=0
    step_size = 1  # 1 / 100
    while i<len(input_ids):

        batch_input_ids = input_ids[i:i+step_size]
        if step_size == 1:
            batch_input_ids = batch_input_ids[batch_input_ids!=tokenizer.pad_token_id].unsqueeze(0)
        mic_logits = mic_regressor(batch_input_ids, strain)
        mic_logits = mic_logits.squeeze().detach()
        mic = 10 ** (-mic_logits) * 10

        if step_size == 1:
            mic = mic.cpu().to(torch.float)
        mics.append(mic)

        i += step_size

    if step_size != 1:
        mics = torch.concat(mics, dim = 0)
    else:
        mics = torch.from_numpy(np.array(mics))

    return mics, SELFIES_strs_linear, file_name

def has_only_allowed_atoms(mol: Chem.Mol) -> bool:
    """检查分子中是否只含 ALLOWED_ATOMS。"""
    ALLOWED_ATOMS = {'C', 'N', 'O', 'S', 'P', 'H', 'Cl', 'F'}
    for atom in mol.GetAtoms():
        if atom.GetSymbol() not in ALLOWED_ATOMS:
            return False
    return True

def is_peptide_simple(mol,
                      PEPTIDE_PATTERN,
                      avg_heavy_atoms: float = 9.0) -> bool:
    """
    简化判定一个 SMILES 是否为多肽：
      1. 只含 ALLOWED_ATOMS；
      2. 估算残基数 ≈ round(heavy_atoms / avg_heavy_atoms)；
      3. 匹配肽键数 n_bonds；
      4. 返回 n_bonds > residues - 1。
    """
    if mol is None:
        return False

    # 1) 元素检查
    # if not has_only_allowed_atoms(mol):
    #     return False

    # 2) 计算 heavy atom 数、估算残基数
    heavy_atoms = mol.GetNumAtoms()
    residues = max(1, int(round(heavy_atoms / avg_heavy_atoms)))

    # 3) 匹配肽键
    n_bonds = len(mol.GetSubstructMatches(PEPTIDE_PATTERN))

    # 4) 判定
    return n_bonds > residues - 1

if __name__ == '__main__':
    # try:
    pil_fonts_dir = Path(PIL.__file__).parent / "fonts"
    font_path = pil_fonts_dir / "DejaVuSans.ttf"
    font = ImageFont.truetype(str(font_path), 48)
    # except IOError:
    #     font = ImageFont.load_default()

    show_log2 = True

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_non_pad_clean/noise_guidance_best_R2_all_peptide_epoch_13.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_pad_no_mask/noise_guidance_best_R2_all_peptide_epoch_100.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/guidance_best_R2_all_peptide_epoch_12.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/noise_guidance_all_peptide_epoch_100_of_100.pth'

    # 先读取 SELFIES 的文件
    guidance_method = 'noise'  # clean / noise
    strain_show_names = ['A. baumannii ATCC 19606']
    strains = ['BAA-3170']
    # length = '320'
    lengths = [0]
    # lengths = list(range(340, 416 + 1, 4))
    target_MICs = ['1']  # '1000' 表示 unconditional generation
    # generate_mol_save_dir = Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES_w_mic-new')
    # generate_mol_save_dir = Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES_w_mic_corrected')
    generate_mol_save_dir = Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES_w_mic')
    file_names = [file.name for file in generate_mol_save_dir.rglob('*') if file.is_file()]

    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mic_regressor = MIC_regressor(config, ckpt_path, device)
    mic_regressor.to(device)
    mic_regressor.eval()

    ax_labels = ['Small', 'Unconditional', 'Large']

    # 用于匹配肽键 N–C(=O)
    PEPTIDE_PATTERN = Chem.MolFromSmarts('[CX4][NX3][CX3](=O)[CX4]')
    # 平均每个残基重原子数（不含隐式氢）
    AVG_HEAVY_ATOMS_PER_RESIDUE = 9.0

    for length in lengths:
        length = str(length)
        for strain, strain_show_name in zip(strains, strain_show_names):
            mic_list = []
            for target_MIC in target_MICs:
                mics, SELFIES_strs, file_name = get_mic(file_names, mic_regressor, tokenizer, strain, target_MIC, guidance_method, length)

                SMILES_strs = []  # [sf.decoder(sf_str) for sf_str in SELFIES_strs]
                for sf_str in SELFIES_strs:
                    try:
                        SMILES_strs.append(sf.decoder(sf_str))
                    except:
                        continue
                file_strain = file_name.split('.txt')[0].split('_')[1]
                file_target_MIC = file_name.split('.txt')[0].split('_')[3]
                file_guidance_method = file_name.split('.txt')[0].split('_')[-1]
                file_target_length = file_name.split('.txt')[0].split('_')[5]

                img_save_dir = Path(f'/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_img_w_mic/strain_{strain}/MIC_{target_MIC}_smi2pep2smi')
                if img_save_dir.exists():
                    shutil.rmtree(img_save_dir)
                img_save_dir.mkdir(parents=True, exist_ok=True)

                qualified_SELFIES_strs = []

                for i, (mic, smiles_str) in tqdm(enumerate(zip(mics, SMILES_strs)), total = len(mics), desc=' Creating img with predicted MIC'):
                    # if mic > 15:
                    #     continue

                    ####   严格检查是不是 canonical peptide
                    _, pep_seq = smiles_to_pepseq(smiles_str)
                    if pep_seq is None or 'X' in pep_seq:
                        continue

                    mol = Chem.MolFromSmiles(smiles_str)

                    # 简单判断是不是 peptide
                    # if not is_peptide_simple(mol, PEPTIDE_PATTERN, avg_heavy_atoms=AVG_HEAVY_ATOMS_PER_RESIDUE):
                    #     continue

                    try:
                        rdDepictor.SetPreferCoordGen(True)
                        rdDepictor.Compute2DCoords(mol)

                        opts = MolDrawOptions()
                        opts.reduceOverlap = True

                        try:
                            img = Draw.MolToImage(mol, size=(1500, 1500), options=opts)
                        except Exception as e:
                            print(e)
                            continue
                        draw = ImageDraw.Draw(img)
                        text = f"MIC: {mic:.2f}\nSeq: {pep_seq.replace('B', 'R`')}"
                        x, y = 20, 20
                        # 半透明背景框（可选）
                        bbox = draw.textbbox((x, y), text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                        draw.rectangle(
                            [x - 5, y - 5, x + text_w + 5, y + text_h + 5],
                            fill=(255, 255, 255, 200)
                        )
                        # 写字
                        draw.text((x, y), text, fill="black", font=font)
                    except ValueError as e:
                        print(f"Warning: failed drawing molecule #{i}: {e!r}")
                        continue

                    try:
                        qualified_SELFIES_strs.append(sf.encoder(smiles_str))
                        img.save(img_save_dir / f"mol_{i}_mic_{mic:.2f}.png")
                    except Exception as e:
                        continue

            # img_save_dir.mkdir(parents=True, exist_ok=True)
            selfies_save_dir = Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES_w_mic')
            selfies_save_dir.mkdir(parents=True, exist_ok=True)
            with open(selfies_save_dir/f'smi2pep2smi.txt', 'w') as f:
                f.write('\n'.join(qualified_SELFIES_strs))




