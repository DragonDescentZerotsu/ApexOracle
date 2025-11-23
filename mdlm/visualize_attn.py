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
from Bio import Phylo, SeqIO
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
from matplotlib.colors import LinearSegmentedColormap
from guaidance_regressor_all_data import FirstTokenAttention_genome, RegressionHead, load_all_genome_embeddings, load_text_wo_genome_embeddings
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors  # 仅用于颜色处理

import selfies as sf
from IPython.core.display import display, HTML
import matplotlib

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
        self.cls_head = RegressionHead(8192 + 4096, (8192 + 4096) // 4, 128, 1, 0.2)
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
        self.cls_head.load_state_dict(checkpoint['cls_head_state_dict'])
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
                mol_cls_embedding_genome, attn_genome = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
            else:
                padded_genome_embeddings = self.learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1)
                mol_cls_embedding_genome, attn_genome = self.co_cross_attn_genome(mol_cls_embedding, padded_genome_embeddings, 1 - genome_attn_masks)
            mol_cls_embedding_text, attn_text = self.co_cross_attn_text(mol_cls_embedding, padded_text_embeddings, 1 - text_attn_masks)
            mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
            reg_logits = self.reg_head(mol_cls_embedding)
            cls_output = torch.sigmoid(self.cls_head(mol_cls_embedding))

        return reg_logits, cls_output, attn_genome, attn_text

def get_mic(file_path, mic_regressor, tokenizer, strain):
    # for file_name in file_names:
    #     file_strain = file_name.split('.txt')[0].split('_')[1]
    #     file_target_MIC = file_name.split('.txt')[0].split('_')[3]
    #     try:
    #         file_guidance_method = file_name.split('.txt')[0].split('_')[-1]
    #         file_target_length = file_name.split('.txt')[0].split('_')[5]
    #     except:
    #         continue
    #     if strain == file_strain and file_target_MIC == target_MIC and file_guidance_method == guidance_method and file_target_length == target_length:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()  # 返回所有行的列表，每行末尾包含 '\n'
    # 如果想去掉末尾的换行符：
    SELFIES_strs = [sf.encoder(line.rstrip("\n")) for line in lines]

    input_ids = tokenizer(
        [s.replace('][', '] [') for s in SELFIES_strs],
        return_tensors='pt',
        padding=True,
        truncation=False,
        add_special_tokens=True
    )['input_ids'].to(device)

    mics = []
    cls_outputs = []
    i=0
    step_size = 1  # 1 / 100
    while i<len(input_ids):

        batch_input_ids = input_ids[i:i+step_size]
        if step_size == 1:
            batch_input_ids = batch_input_ids[batch_input_ids!=tokenizer.pad_token_id].unsqueeze(0)
        mic_logits, cls_output, attn_genome, attn_text = mic_regressor(batch_input_ids, strain)
        mic_logits = mic_logits.squeeze().detach()
        cls_output = cls_output.squeeze().detach()
        mic = 10 ** (-mic_logits) * 10

        if step_size == 1:
            mic = mic.cpu().to(torch.float)
        mics.append(mic)

        if step_size == 1:
            cls_output = cls_output.cpu().to(torch.float)
        cls_outputs.append(cls_output)

        i += step_size

    if step_size != 1:
        mics = torch.concat(mics, dim = 0)
    else:
        mics = torch.from_numpy(np.array(mics))

    if step_size != 1:
        cls_outputs = torch.concat(cls_outputs, dim = 0)
    else:
        cls_outputs = torch.from_numpy(np.array(cls_outputs))

    return mics, cls_outputs, attn_genome, attn_text

if __name__ == '__main__':

    show_log2 = True

    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_non_pad_clean/noise_guidance_best_R2_all_peptide_epoch_13.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/all_AMP_SM_data_train/MDLM_MTR_fix_cls_wo_pad/all_data_best_R2_epoch_13.pth'  # 这个权重在所有的 MIC data 上训练过
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor_pad_no_mask/noise_guidance_best_R2_all_peptide_epoch_100.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/guidance_best_R2_all_peptide_epoch_12.pth'
    # ckpt_path = '/data2/tianang/projects/Synergy/Checkpoints/genome_text_learnable_emb/guidance_regressor/noise_guidance_all_peptide_epoch_100_of_100.pth'

    # 先读取 SELFIES 的文件
    # guidance_method = 'noise'  # clean / noise
    # strain_show_names = [r'P. aeruginosa BAA-3197']
    strains = ['BAA-3170']
    # length = '232'
    # target_MICs = ['1', '1000']  # '1000' 表示 unconditional generation
    # generate_mol_save_dir = Path('/data2/tianang/projects/discrete-diffusion-guidance/outputs/generated_mol_SELFIES')
    file_path = current_directory / 'DataPrepare' / 'Data' / 'mol_visualize' / 'Colistin.txt'

    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    mic_regressor = MIC_regressor(config, ckpt_path, device)
    mic_regressor.to(device)
    mic_regressor.eval()

    ax_labels = ['Guided', 'Unconditional']

    colors = ["#F7CFE1", "#B49EDE"]  # , "#759ECD"]
    custom_cmap = LinearSegmentedColormap.from_list("custom_gradient", colors, N=2)

    fractions = np.linspace(0, 1, len(ax_labels))
    palette = [custom_cmap(f) for f in fractions]

    for strain in strains:
        mic_list = []
        # for target_MIC in target_MICs:
        mic, cls_outputs, attn_genome, attn_text = get_mic(file_path, mic_regressor, tokenizer, strain)

        print(f'mic:{mic}')
        print(f'cls:{cls_outputs}')

        # attn_genome = attn_genome[0]  # 形状 (num_heads, tgt_len, src_len)

        # 可视化每个头
        for head in range(attn_genome.size(0)):
            arr = attn_genome[head].detach().cpu().numpy().flatten()  # (128,)
            plt.figure(figsize=(50, 2))
            plt.bar(range(len(arr)), arr, width=0.8)
            plt.title(f"Head {head} – Attention Weights (Bar Plot)")
            plt.xlabel("Position Index")
            plt.ylabel("Attention Weight")
            plt.ylim(0, arr.max() * 1.1)
            plt.show()

        # k=4
        # top_vals, top_idx = torch.topk(attn_genome[0][0], k)
        # print(f' top genome indexs:{top_idx+1}')
        # print(f' top genome vals:{top_vals}')

        big_attn_indexes = torch.where(attn_genome[0][0] > 0.05)[0].detach().cpu().numpy()
        print(f' big attn scores: {attn_genome[0][0][attn_genome[0][0] > 0.05].detach().cpu().numpy()}')
        print(f' corresponding index: {big_attn_indexes}')

        big_attn_gene_ranges = []
        for big_attn_index in big_attn_indexes:
            big_attn_gene_ranges.append([big_attn_index * 10000, big_attn_index * 10000 + 11000])

        annotation_found_flag = False
        genome_annotation_folder_path = current_directory / 'DataPrepare' / 'Data' / 'Genome_annotation'
        for annotation_file in genome_annotation_folder_path.iterdir():
            if strain in annotation_file.name:
                annotation_found_flag = True
                break

        if not annotation_found_flag:
            print(f'genome annotation for {strain} not found')
            exit(1)

        attented_products = []
        attented_genes = []
        attented_locs = []
        contig_lengths = [0]
        current_contig_id = 0
        for seq_record in SeqIO.parse(annotation_file, "genbank"):
            config_id = int(seq_record.id.split('_')[-1])
            print(f' config_id:{seq_record.id}')
            if config_id != current_contig_id:
                current_contig_id = config_id
                contig_lengths.append(len(seq_record.seq))

            for feature in seq_record.features:
                if feature.type == 'CDS':
                    for big_attn_gene_range in big_attn_gene_ranges:
                        if config_id > 1:
                            big_attn_gene_range_low_bound, big_attn_gene_range_high_bound = big_attn_gene_range[
                                                                                                0] - np.array(
                                contig_lengths)[:-1].sum(), big_attn_gene_range[1] - np.array(contig_lengths)[:-1].sum()
                        else:
                            big_attn_gene_range_low_bound, big_attn_gene_range_high_bound = big_attn_gene_range[0], \
                            big_attn_gene_range[1]
                        if int(feature.location.start) > big_attn_gene_range[0] and int(feature.location.end) < \
                                big_attn_gene_range[1]:
                            # print(f"\n index range:{big_attn_gene_range}")
                            # print(f" locus_tag: {feature.qualifiers.get('locus_tag')}")
                            # print(f" product: {feature.qualifiers.get('product')}")

                            products = feature.qualifiers.get('product')  # feature.qualifiers.get('gene / product')
                            if products is not None:
                                for product in products:
                                    if 'hypothetical' not in product:
                                        attented_products.append(product)
                            genes = feature.qualifiers.get('gene')  # feature.qualifiers.get('gene / product')
                            if genes is not None:
                                for gene in genes:
                                    if 'hypothetical' not in gene:
                                        attented_genes.append(gene)
                            if products is not None:
                                attented_locs.append([seq_record.id, int(feature.location.start),int(feature.location.end)])
                            else:
                                attented_genes.append('None')
        print(f" attended products:\n{attented_products}")
        print(f" attended genes:\n{attented_genes}")
        print(f" attended locations:\n{attented_locs}")

        # attn_text = attn_text[0]  # 形状 (num_heads, tgt_len, src_len)

        # 可视化每个头
        for head in range(attn_text.size(0)):
            plt.figure(figsize=(50, 2))
            plt.title(f"Head {head}")
            plt.imshow(attn_text[head].detach().cpu().numpy(), aspect='auto')
            plt.xlabel("Source Position")
            plt.ylabel("Target Position")
            plt.colorbar(label="Attention Weight")
            plt.show()

        k = 10  # 要取最大的 10 个
        top_vals, top_idx = torch.topk(attn_text[0][0], k)  # 默认 largest=True, dim=0

        for text_file_path in (
                current_directory / 'DataPrepare' / 'Data' / 'Text_Description' / 'ATCC' / 'Text').iterdir():
            if strains[0] in text_file_path.name:
                break

        with open(text_file_path, 'r') as f:
            text = f.read()
            file_name = text_file_path.name.split('.')[0]
            text = text.replace(file_name.replace('_', ' ').replace('subsp', 'subsp.'), 'This strain')
            text = text.replace(file_name.replace('_', ' '), 'This strain')  # 把所有的 strain 的名字都用 This strain 代替
            # ATCC_ID = file_name.replace('_', ' ').split('ATCC')[-1].strip()

    tokenizer = AutoTokenizer.from_pretrained("YBXL/Med-LLaMA3-8B")
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    ids_list = input_ids.tolist()
    input_text = tokenizer.convert_ids_to_tokens(ids_list)
    for word in input_text:
        if 'Ġ' in word:
            word = word.replace('Ġ', ' ')

    attn_text = attn_text[0][0].detach().cpu()
    scores_np = attn_text.numpy()
    min_v, max_v = scores_np.min(), scores_np.max()
    norm_scores = (scores_np - min_v) / (max_v - min_v + 1e-8)  # 防止除零

    # 2. 定义一个小函数：根据在 [0,1] 中的注意力值，输出对应的背景色（这里用 matplotlib 的 colormap）
    cmap = matplotlib.cm.get_cmap('Reds')  # 选择一种红色渐变


    def score_to_hex(score):
        """把 0~1 的 float 映射到一个 HEX 颜色，比如 #FFCCCC."""
        rgba = cmap(score)  # 返回 RGBA 四元组
        rgb = tuple(int(x * 255) for x in rgba[:3])
        return '#%02x%02x%02x' % rgb  # 转为十六进制字符串


    # 3. 生成 HTML 片段
    html_pieces = []
    for token, ns in zip(input_text, norm_scores):
        color = score_to_hex(ns)
        # 把 token 包裹在一个 <span> 里，用行内样式指定背景色
        cleaned_token = token.replace("Ġ", " ").replace("ĊĊ", "<br><br>")
        html_pieces.append(
            f'<span style="background-color:{color}; padding:2px; margin:1px; border-radius:2px;">{cleaned_token}</span>'
        )

    html_str = ''.join(html_pieces)  # 拼成一句 HTML
    display(HTML(html_str))





