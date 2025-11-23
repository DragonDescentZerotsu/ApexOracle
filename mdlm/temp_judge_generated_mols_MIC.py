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
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter, FuncFormatter
import ast
import seaborn as sns
import selfies as sf
from smiles_to_peptide import smiles_to_pepseq
from guaidance_regressor_all_data import FirstTokenAttention_genome, RegressionHead, load_all_genome_embeddings, load_text_wo_genome_embeddings

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
        self.mdlm_model.eval()
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

def get_mic(file_names, mic_regressor, tokenizer, strain, target_MIC=None, guidance_method=None, target_length=None):
    for file_name in file_names:
        file_strain = file_name.split('.txt')[0].split('_')[1]
        # file_target_MIC = file_name.split('.txt')[0].split('_')[3]
        # try:
            # file_guidance_method = file_name.split('.txt')[0].split('_')[-1]
            # file_target_length = file_name.split('.txt')[0].split('_')[5]
        # except:
        #     continue
        # if file_name == 'strain_BS111_MIC_1_length_256_noise.txt':
        #     print(file_name)
        if strain == file_strain:# and file_target_MIC == target_MIC and file_guidance_method == guidance_method and file_target_length == target_length:
            with open(generate_mol_save_dir/file_name, "r", encoding="utf-8") as f:
                lines = f.readlines()  # 返回所有行的列表，每行末尾包含 '\n'
            # 如果想去掉末尾的换行符：
            SELFIES_strs = [line.rstrip("\n") for line in lines]
            break

    print('tokenizing selfies')
    input_ids = tokenizer(
        [s.replace('][', '] [') for s in SELFIES_strs],
        return_tensors='pt',
        padding=True,
        truncation=False,
        add_special_tokens=True
    )['input_ids'].squeeze().to(device)
    print('tokenizing selfies done')

    mics = []
    i=0
    step_size = 1  # 1 / 100
    pbar = tqdm(total=len(input_ids), desc='calculating MICs')
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
        pbar.update(step_size)

    pbar.close()

    if step_size != 1:
        mics = torch.concat(mics, dim = 0)
    else:
        mics = torch.from_numpy(np.array(mics))

    return mics, SELFIES_strs, file_name

if __name__ == '__main__':

    show_log2 = True
    save_csv = True  # TODO: Set to False to disable CSV saving

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_non_pad_clean/noise_guidance_best_R2_all_peptide_epoch_13.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_pad_no_mask/noise_guidance_best_R2_all_peptide_epoch_100.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/guidance_best_R2_all_peptide_epoch_12.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/noise_guidance_all_peptide_epoch_100_of_100.pth'

    # 先读取 SELFIES 的文件
    guidance_method = 'noise'  # clean / noise
    strain_show_names = ['ATCC 19606', 'ATCC 11775', 'ATCC 13883', 'PA01', 'PA14', 'ATCC 12600', 'ATCC BAA-1556', 'ATCC 700802', 'ATCC 4356', 'ATCC 15697', 'ATCC 15700', 'ATCC 23272', 'BAA-999']  # TODO: Apex strain list: ['19606', '11775', '13883', '47085', '#002', '12600', 'BAA-1556', '700802']
    strains = ['19606', '11775', '13883', '47085', '#002', '12600', 'BAA-1556', '700802', '4356', '15697', '15700', '23272', 'BAA-999']  #['BAA-999', '15700', '15697', '23272', '4356']
    length = '368'
    # target_MICs = ['1', '1000']  # '1000' 表示 unconditional generation
    generate_mol_save_dir = Path('/data2/tianang/projects/mdlm/temp_data/milk')  # TODO: 每一个 strain 保存一个要screen的 SELFIES 的文件 ####### 每一个strain都要有对应的文件！
    # generate_mol_save_dir = Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES-new-test')
    file_names = [file.name for file in generate_mol_save_dir.iterdir()]  # TODO: 这里保存的 files 必须是 SELFIES 的, temp.ipynb 里面有 peptide 转 SELFIES 的代码

    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mic_regressor = MIC_regressor(config, ckpt_path, device)
    mic_regressor.to(device)
    mic_regressor.eval()

    # Single violin color
    violin_color = "#B49EDE"

    # Data collection for CSV export
    if save_csv:
        all_strain_data = {}  # {strain: {selfies_str: mic_value}}
        all_selfies_set = set()  # To collect all unique SELFIES

    for strain, strain_show_name in zip(strains, strain_show_names):
        mic, SELFIES_strs, file_name = get_mic(file_names, mic_regressor, tokenizer, strain)

        print(mic)
        print(f' predicted mean MIC: {mic.mean()}')
        print(f' predicted median MIC: {torch.median(mic)}')

        top_k = 1

        min_values, _ = torch.topk(mic, top_k, largest=False)

        print(f'{top_k} min mics: {min_values}')
        print(f' mean: {min_values.mean()}')
        print(f' median: {min_values.median()}')

        # Collect data for CSV export
        if save_csv:
            strain_mic_dict = {}
            for selfies_str, mic_value in zip(SELFIES_strs, mic):
                strain_mic_dict[selfies_str] = mic_value.item()
                all_selfies_set.add(selfies_str)
            all_strain_data[strain] = strain_mic_dict

        # Prepare data for violin plot
        if show_log2:
            mic_values = torch.log2(mic).detach().cpu().to(torch.float32).numpy()
        else:
            mic_values = mic.detach().cpu().to(torch.float32).numpy()

        fig, ax = plt.subplots(figsize=(5, 5))

        # Create violin plot with single data
        parts = ax.violinplot(
            [mic_values],
            positions=[0],
            showmeans=False,
            showmedians=True,
            widths=0.5
        )

        # Customize violin plot appearance
        for pc in parts['bodies']:
            pc.set_facecolor(violin_color)
            pc.set_alpha(0.7)
            pc.set_edgecolor('none')

        # Style the quartile lines
        if 'cmedians' in parts:
            parts['cmedians'].set_edgecolor('#6B4FA8')
            parts['cmedians'].set_linewidth(2)
        if 'cbars' in parts:
            parts['cbars'].set_edgecolor('#6B4FA8')
            parts['cbars'].set_linewidth(1.5)
        if 'cmaxes' in parts:
            parts['cmaxes'].set_edgecolor('#6B4FA8')
            parts['cmaxes'].set_linewidth(1.5)
        if 'cmins' in parts:
            parts['cmins'].set_edgecolor('#6B4FA8')
            parts['cmins'].set_linewidth(1.5)

        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=1.6)

        def inv_log2(x, pos):
            orig = 2 ** x
            return int(orig)

        # Apply formatter to y-axis
        if show_log2:
            ax.yaxis.set_major_formatter(FuncFormatter(inv_log2))

        # if show_log2:
        #     ax.set_yscale("log", base=2)  # 2 为底的对数坐标
        #     ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.set_axisbelow(True)
        ax.set_title(f"Molecule MIC distribution\nagainst {strain_show_name}", fontsize=14)
        ax.set_xlabel("")
        y_label = ax.set_ylabel("log 2 scale MIC value (µmol)")
        y_label.set_fontsize(11)

        # Hide x-axis labels and ticks for single violin plot
        ax.set_xticks([])
        ax.set_xticklabels([])

        sns.despine(fig=fig, ax=ax,
                    top=True, right=True, bottom=True, left=True)

        ax.tick_params(axis='both', which='both', length=0)

        plt.tight_layout()

        # Save figure
        fig_save_dir = Path('/data2/tianang/projects/mdlm/temp_data/milk_violin_plots')
        fig_save_dir.mkdir(parents=True, exist_ok=True)
        save_path = fig_save_dir / f'strain_{strain}_MIC_distribution.pdf'
        plt.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
        print(f"Figure saved to: {save_path}")

        plt.show()

    # Create and save CSV file with all strain data
    if save_csv:
        print("\nConverting SELFIES to peptide sequences...")
        # Create a mapping from SELFIES to peptide sequence
        selfies_to_peptide = {}

        for selfies_str in tqdm(all_selfies_set, desc="Converting SELFIES to peptide"):
            try:
                smiles_str = sf.decoder(selfies_str)
                _, pep_seq = smiles_to_pepseq(smiles_str)

                # Only include valid peptide sequences (no X or None)
                if pep_seq is not None and 'X' not in pep_seq:
                    selfies_to_peptide[selfies_str] = pep_seq
            except Exception as e:
                # Skip molecules that cannot be converted
                continue

        # Build DataFrame: rows = peptide sequences, columns = strains
        print("\nBuilding DataFrame...")
        data_rows = []

        for selfies_str, pep_seq in selfies_to_peptide.items():
            row_data = {'Peptide_Sequence': pep_seq}
            for strain in strains:
                if strain in all_strain_data and selfies_str in all_strain_data[strain]:
                    row_data[strain] = all_strain_data[strain][selfies_str]
                else:
                    row_data[strain] = np.nan  # Missing values
            data_rows.append(row_data)

        df = pd.DataFrame(data_rows)

        # Save CSV
        csv_save_dir = Path('/data2/tianang/projects/mdlm/temp_data')
        csv_save_dir.mkdir(parents=True, exist_ok=True)
        csv_save_path = csv_save_dir / 'milk_all_strains_mic_predictions.csv'  # TODO: change to your file name
        df.to_csv(csv_save_path, index=False)

        print(f"\nCSV file saved to: {csv_save_path}")
        print(f"Total peptide sequences: {len(df)}")
        print(f"Total strains: {len(strains)}")

        # fig, ax = plt.subplots()
        # ax.violinplot(mic_list)
        # ax.set_xticks(list(range(1, len(mic_list) + 1)))
        # ax.set_xticklabels(ax_labels[:len(mic_list)])
        # ax.set_title(f'{strain_show_name} generated molecule MIC, {guidance_method} guidance')
        # ax.set_ylabel('log 2 scale MIC value (µmol)')
        #
        # # 4. 如果你绘的是 log2 转换后的值，就不需要再转；否则可以用对数坐标轴
        #
        #
        # # ax.set_yscale('log', base=2)
        # # ax.yaxis.set_major_locator(LogLocator(base=2, numticks=10))
        # # ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        #
        # def inv_log2(x, pos):
        #     orig = 2 ** x
        #     # 如果数值足够大，改成科学记数法或保留整数
        #     # if orig >= 1000:
        #     #     return f"{orig:.1e}"
        #     # else:
        #     #     return f"{orig:.0f}"
        #     return orig
        #
        #
        # # 3) 应用到 y 轴主刻度
        # if show_log2:
        #     ax.yaxis.set_major_formatter(FuncFormatter(inv_log2))

        # plt.show()



