import itertools

import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path
import argparse
import json
from scipy.stats import pearsonr, spearmanr
import logging
import selfies as sf
from sklearn.metrics import roc_auc_score, average_precision_score
# -------------------------
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from multiprocessing import get_context
# -------------------------

def get_embedded_genome_IDs(folder_path):
    """
    检查哪些 genome ID 的genome已经被转成 Evo2 的 embedding 了
    :param folder_path: 保存 Genome 的 Evo2 embed 的文件夹路径
    :return: 不带 ATCC 的纯 ID list  | e.g. ['25332', '11060', ‘BAA-252', ...]
    """
    stored_genome_IDs = []
    genome_ID_to_species_first_name_dict = {}
    files = [f.name for f in folder_path.iterdir() if f.is_file()]
    for file_name in files:
        file_name = file_name.split('.')[0]
        file_name_temp = file_name.split('ATCC')[-1]
        components = file_name_temp.split('_')[1:]
        if len(components) == 2:
            ATCC_ID = '-'.join(components)
            stored_genome_IDs.append(ATCC_ID)  # 组装成形如 ‘BAA-252' 或者 'MYA-730'
        else:
            ATCC_ID = components[0]
            stored_genome_IDs.append(ATCC_ID)  # 就是普通的 '25922'

        genome_ID_to_species_first_name_dict[ATCC_ID] = file_name.split('_')[0]

    return stored_genome_IDs, genome_ID_to_species_first_name_dict

def get_original_strain_name_with_genome_embedding(Evo_MIC_count_file_path, embedded_genome_IDs):
    with open(Evo_MIC_count_file_path, 'r', encoding='utf-8') as f:
        strain_count_data = json.load(f)  # 解析 JSON 文件

    origin_to_standard_name_map_list_handcrafted = []  # [(original_name, standard_name (ATCC ID)), (Staphylococcus aureus ATCC 25923, 25923)...]
    origin_to_standard_name_map_list_DBAASP_original = []
    for name, count in strain_count_data.items():

        # 先处理手动标记的 strain
        if '*' in name:
            original_name, standard_name = name.split('*')
            if 'ATCC' in standard_name:
                standard_name = standard_name.split('ATCC')[-1].strip()
            else:
                # 包含那些没有 ATCC 但是单独下载了 Genome 数据的
                standard_name = standard_name.strip()
            origin_to_standard_name_map_list_handcrafted.append((original_name.strip(), standard_name))

        # 如果没有手动标记，那就只处理原始 strain 中就有 ATCC ID 的那些
        else:
            if 'ATCC' in name:
                original_name = name
                ATCC_id = name.split('ATCC')[-1].strip()
                if 'BAA' in name:
                    ATCC_id = ATCC_id.replace(" ", "-")
                if 'MY' in name:
                    ATCC_id = ATCC_id.replace(" ", "")
                if 'MAY' in name:
                    ATCC_id = ATCC_id.replace("MAY", "MYA")
                if 'D' in name:
                    ATCC_id = ATCC_id.split("D")[0]
                if 'T' in name:
                    ATCC_id = ATCC_id.split("T")[0]
                if 's' in name:
                    ATCC_id = ATCC_id.split("s")[0]
                if " " in name:
                    ATCC_id = ATCC_id.split(" ")[0]

                origin_to_standard_name_map_list_DBAASP_original.append((original_name.strip(), ATCC_id))

    origin_to_standard_name_map_list = np.array(origin_to_standard_name_map_list_handcrafted + origin_to_standard_name_map_list_DBAASP_original)

    original_names_with_genome_embedding_handcrafted = []  # 提取出那些有对应 Evo2 embedding 的 DBAASP 中的完整 strain name
    for line_idx, (original_name, standard_name) in enumerate(origin_to_standard_name_map_list_handcrafted):
        # 检查这些 ATCC ID 是不是已经在有 Evo2 embedding 的 strain 里
        if standard_name in embedded_genome_IDs:
            original_names_with_genome_embedding_handcrafted.append(original_name)

    original_names_with_genome_embedding_DBAASP_original = []  # 提取出那些有对应 Evo2 embedding 的 DBAASP 中的完整 strain name
    for line_idx, (original_name, standard_name) in enumerate(origin_to_standard_name_map_list_DBAASP_original):
        # 检查这些 ATCC ID 是不是已经在有 Evo2 embedding 的 strain 里
        if standard_name in embedded_genome_IDs:
            original_names_with_genome_embedding_DBAASP_original.append(original_name)

    return original_names_with_genome_embedding_handcrafted, original_names_with_genome_embedding_DBAASP_original, dict(origin_to_standard_name_map_list)

def load_all_genome_embeddings(embeddings_folder_path, scale, device, desc_str):
    """
    返回一个 genome ID 到 Evo2 embedding 字典
    :param embeddings_folder_path: 保存 Genome 的 Evo2 embed 的文件夹路径
    :param scale: Evo2 的 embedding 量级大概在 1e-15 左右，和模型参数 1e-2 左右的量级差太多了，所以需要缩放匹配
    :param device: 提前将所有的 Evo2 embedding 载入到显存之中，减少加载时间
    :return: dict  e.g. {'25922': torch.tensor([...], dtype=torch.bfloat16), ...}
    """
    file_paths = [embeddings_folder_path / f.name for f in embeddings_folder_path.iterdir() if f.is_file()]
    embeddings_dict = {}
    for file_path in tqdm(file_paths, desc=f' loading {desc_str} embeddings ... '):
        embedding = torch.load(file_path).to(device)
        file_name = file_path.name.split('.')[0]
        if 'ATCC' in file_name:
            file_name = file_name.split('ATCC')[-1]
            components = file_name.split('_')[1:]
            if len(components) == 2:
                ID = '-'.join(components)
            else:
                ID = components[0]
        else:
            # 自己下载的情况
            ID = file_name
        embeddings_dict[ID] = embedding * scale

    return embeddings_dict

def load_text_wo_genome_embeddings(embeddings_folder_path, scale, device, desc_str):
    """
    返回一个 genome ID 到 Evo2 embedding 字典
    :param embeddings_folder_path: 保存 Genome 的 Evo2 embed 的文件夹路径
    :param scale: Evo2 的 embedding 量级大概在 1e-15 左右，和模型参数 1e-2 左右的量级差太多了，所以需要缩放匹配
    :param device: 提前将所有的 Evo2 embedding 载入到显存之中，减少加载时间
    :return: dict  e.g. {'25922': torch.tensor([...], dtype=torch.bfloat16), ...}
    """
    file_paths = [embeddings_folder_path / f.name for f in embeddings_folder_path.iterdir() if f.is_file()]
    embeddings_dict = {}
    for file_path in tqdm(file_paths, desc=f' loading {desc_str} embeddings ... '):
        embedding = torch.load(file_path).to(device)
        file_name = file_path.name.split('.pt')[0]
        strain_name = file_name.replace('～', ' ').replace('^', '/')
        embeddings_dict[strain_name] = embedding * scale

    return embeddings_dict

# 自定义 PyTorch Dataset
class SMILESDataset_with_genome_and_text(Dataset):
    def __init__(self, dataframe, tokenizer, embeddings_dict, text_embeddings_dict, set_desc:str, mol_emb_dict, max_length=512):
        self.dataframe = dataframe
        self.original_length = len(self.dataframe)
        self.tokenizer = tokenizer
        self.embeddings_dict = embeddings_dict
        self.text_embeddings_dict = text_embeddings_dict
        self.max_length = max_length
        self.mol_emb_dict = mol_emb_dict
        # self.SM_emb_dict = SM_emb_dict
        self.target_columns = 'FICI'
        # self.remove_long_smiles()
        # print(f'\n {set_desc}:\n original length: {self.original_length}\n after SMILES length limitation length: {len(self.dataframe)}')
        logger.info(f'\n {set_desc}:\n original length: {self.original_length}\n after SMILES length limitation length: {len(self.dataframe)}')

    def tokenize_smiles(self, smiles):
        # 对单个 SMILES 进行 tokenize，返回 input_ids 和 attention_mask（去除 batch 维度）
        tokenized = self.tokenizer(sf.encoder(smiles).replace('][', '] ['), return_tensors='pt', padding=False, truncation=False)
        input_ids = tokenized['input_ids'].squeeze(0)
        attn_mask = tokenized['attention_mask'].squeeze(0)
        return input_ids, attn_mask

    def remove_long_smiles(self):
        # self.dataframe = self.dataframe[self.dataframe['SMILES'].apply(lambda x: len(self.tokenizer(x, return_tensors='pt', padding=False, truncation=False)['input_ids'].squeeze(0)) <= self.max_length)]
        # self.dataframe = self.dataframe.reset_index(drop=True)  # 重置索引

        # 对 SMILES 列进行 tokenize，并拆分为两列
        tokenized_cols = self.dataframe['AMP_smiles'].apply(
            lambda x: pd.Series(self.tokenize_smiles(x), index=['input_ids_1', 'attn_mask_1'])
        )

        # 将新的两列拼接到原 dataframe 中
        self.dataframe = pd.concat([self.dataframe, tokenized_cols], axis=1)

        tokenized_cols = self.dataframe['antibiotic_smiles'].apply(
            lambda x: pd.Series(self.tokenize_smiles(x), index=['input_ids_2', 'attn_mask_2'])
        )

        self.dataframe = pd.concat([self.dataframe, tokenized_cols], axis=1)

        # 根据 input_ids 长度进行过滤，确保 token 长度不超过 max_length
        self.dataframe = self.dataframe[self.dataframe['input_ids_1'].apply(len) <= self.max_length]
        self.dataframe = self.dataframe[self.dataframe['input_ids_2'].apply(len) <= self.max_length]
        self.dataframe = self.dataframe.reset_index(drop=True)

        # 删除原来的 SMILES 列
        self.dataframe.drop(columns=['AMP_smiles'], inplace=True)
        self.dataframe.drop(columns=['antibiotic_smiles'], inplace=True)

        # self.dataframe.to_csv('/home/tianang/Projects/Synergy/DataPrepare/Data/DBAASP_id_SMILES_bact_MICs_512_limit.csv', index=False)
        # print(f'new data file saved to /home/tianang/Projects/Synergy/DataPrepare/Data/DBAASP_id_SMILES_bact_MICs_512_limit.csv')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # smiles = self.dataframe.iloc[idx]['SMILES']
        # DBAASP_id = self.dataframe.iloc[idx]['DBAASP_id']
        # target_columns = self.dataframe.columns.tolist()[2:]
        mol_id_1 = self.dataframe.iloc[idx]['DBAASP_id']
        mol_id_2 = self.dataframe.iloc[idx]['antibio_id_or_name']
        mol_emb_1 = self.mol_emb_dict[mol_id_1]
        mol_emb_2 = self.mol_emb_dict[mol_id_2]
        strain_name = self.dataframe.iloc[idx]['strain_name']
        target = self.dataframe.iloc[idx][self.target_columns]
        if target<0.5:
            target = 1.0
        else:
            target = 0.0
        # inputs = self.tokenizer(smiles, return_tensors='pt', padding=False, truncation=False)  #, max_length=self.max_length)
        # inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # 去掉 batch 维度
        return {
            # 'input_ids_1': self.dataframe.iloc[idx]['input_ids_1'],
            # 'attention_mask_1': self.dataframe.iloc[idx]['attn_mask_1'],
            # 'input_ids_2': self.dataframe.iloc[idx]['input_ids_2'],
            # 'attention_mask_2': self.dataframe.iloc[idx]['attn_mask_2'],
            'label': torch.tensor(target, dtype=torch.float),
            'genome_embedding': self.embeddings_dict[strain_name],
            'text_embedding': self.text_embeddings_dict[strain_name],
            'strain_name': strain_name,
            'mol_emb_1': mol_emb_1.squeeze(),
            'mol_emb_2': mol_emb_2.squeeze()
        }

class SMILESDataset_with_text_only(SMILESDataset_with_genome_and_text):
    def __init__(self, dataframe, tokenizer, text_embeddings_dict, set_desc: str, mol_emb_dict, max_length=512):
        # 调用父类的 __init__ 方法时，可以将 embeddings_dict 传入一个 None 或者空字典（如果父类内部没有用到的话）
        super().__init__(dataframe, tokenizer, embeddings_dict=None, text_embeddings_dict=text_embeddings_dict,
                         set_desc=set_desc, mol_emb_dict=mol_emb_dict, max_length=max_length)
        # 如果父类中对 self.embeddings_dict 有特殊处理，可以在这里重置或忽略它

    def __getitem__(self, idx):
        mol_id_1 = self.dataframe.iloc[idx]['DBAASP_id']
        mol_id_2 = self.dataframe.iloc[idx]['antibio_id_or_name']
        mol_emb_1 = self.mol_emb_dict[mol_id_1]
        mol_emb_2 = self.mol_emb_dict[mol_id_2]
        strain_name = self.dataframe.iloc[idx]['strain_name']
        target = self.dataframe.iloc[idx][self.target_columns]
        if target<0.5:
            target = 1.0
        else:
            target = 0.0
        return {
            # 'input_ids_1': self.dataframe.iloc[idx]['input_ids_1'],
            # 'attention_mask_1': self.dataframe.iloc[idx]['attn_mask_1'],
            # 'input_ids_2': self.dataframe.iloc[idx]['input_ids_2'],
            # 'attention_mask_2': self.dataframe.iloc[idx]['attn_mask_2'],
            'label': torch.tensor(target, dtype=torch.float),
            'text_embedding': self.text_embeddings_dict[strain_name],
            'strain_name': strain_name,
            'mol_emb_1': mol_emb_1.squeeze(),
            'mol_emb_2': mol_emb_2.squeeze()
        }

def collate_fn(batch):
    """
    这里把一个batch中所有的label都转换成 log 计算之后的
    """
    # input_ids = []
    # attention_mask = []
    # for item in batch:
    #     input_ids.extend([item['input_ids_1'], item['input_ids_2']])
    #     attention_mask.extend([item['attention_mask_1'], item['attention_mask_2']])
    labels = [item['label'] for item in batch]
    genome_embeddings = []
    text_embeddings = []
    for item in batch:
        genome_embeddings.extend([item['genome_embedding'], item['genome_embedding']])
        text_embeddings.extend([item['text_embedding'], item['text_embedding']])
    strain_names = [item['strain_name'] for item in batch]

    mol_emb = []
    for item in batch:
        mol_emb.extend([item['mol_emb_1'], item['mol_emb_2']])
    # mol_emb_1 = [item['mol_emb_1'] for item in batch]
    # mol_emb_2 = [item['mol_emb_2'] for item in batch]

    mol_emb = torch.stack(mol_emb)

    max_genome_length = 0
    for genome_embedding in genome_embeddings:
        if len(genome_embedding) > max_genome_length:
            max_genome_length = len(genome_embedding)

    padded_genome_embeddings = []
    genome_attn_masks = []
    for genome_embedding in genome_embeddings:
        L, D = genome_embedding.shape
        genome_attn_mask = torch.zeros(max_genome_length, device=genome_embedding.device, dtype=torch.uint8)
        genome_padding = torch.zeros((max_genome_length, D), dtype=torch.bfloat16, device=genome_embedding.device)
        genome_padding[:L] = genome_embedding
        genome_attn_mask[:L] = 1
        padded_genome_embeddings.append(genome_padding)
        genome_attn_masks.append(genome_attn_mask)

    padded_genome_embeddings = torch.stack(padded_genome_embeddings)
    genome_attn_masks = torch.stack(genome_attn_masks)

    max_text_length = 0
    for text_embedding in text_embeddings:
        if len(text_embedding) > max_text_length:
            max_text_length = len(text_embedding)

    padded_text_embeddings = []
    text_attn_masks = []
    for text_embedding in text_embeddings:
        L, D = text_embedding.shape
        text_attn_mask = torch.zeros(max_text_length, device=text_embedding.device, dtype=torch.uint8)
        text_padding = torch.zeros((max_text_length, D), dtype=torch.bfloat16, device=text_embedding.device)
        text_padding[:L] = text_embedding
        text_attn_mask[:L] = 1
        padded_text_embeddings.append(text_padding)
        text_attn_masks.append(text_attn_mask)

    padded_text_embeddings = torch.stack(padded_text_embeddings)
    text_attn_masks = torch.stack(text_attn_masks)

    # 使用 pad_sequence 填充输入
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # padded_input_ids = torch.ones([len(input_ids), 1024], dtype=input_ids.dtype) * tokenizer.pad_token_id
    # padded_input_ids[:, :input_ids.shape[-1]] = input_ids
    # input_ids = padded_input_ids
    #
    # attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    # padded_attention_mask = torch.zeros([len(input_ids), 1024], dtype=input_ids.dtype)
    # padded_attention_mask[:, :attention_mask.shape[-1]] = attention_mask
    # attention_mask = padded_attention_mask
    labels = torch.from_numpy(np.array(labels))
    # mask = labels >= -0.5  # 生成多任务回归使用的 label mask
    # labels_processed = labels.clone()  # 复制原张量以保留未满足条件的值

    # 计算实际的用来回归的值
    # labels = -torch.log10(labels / 10)

    return {
        # 'input_ids': input_ids,
        # 'attention_mask': attention_mask,
        'label': labels,
        'padded_genome_embeddings': padded_genome_embeddings,
        'genome_attn_masks': genome_attn_masks,
        'padded_text_embeddings': padded_text_embeddings,
        'text_attn_masks': text_attn_masks,
        'strain_names': strain_names,
        'mol_emb': mol_emb
    }

def collate_fn_text_only(batch):
    """
    这里把一个batch中所有的label都转换成 log 计算之后的
    """
    # input_ids = []
    # attention_mask = []
    # for item in batch:
    #     input_ids.extend([item['input_ids_1'], item['input_ids_2']])
    #     attention_mask.extend([item['attention_mask_1'], item['attention_mask_2']])
    labels = [item['label'] for item in batch]
    # genome_embeddings = [item['genome_embedding'] for item in batch]
    text_embeddings = []
    for item in batch:
        text_embeddings.extend([item['text_embedding'], item['text_embedding']])
    strain_names = [item['strain_name'] for item in batch]

    mol_emb = []
    for item in batch:
        mol_emb.extend([item['mol_emb_1'], item['mol_emb_2']])
    # mol_emb_1 = [item['mol_emb_1'] for item in batch]
    # mol_emb_2 = [item['mol_emb_2'] for item in batch]

    mol_emb = torch.stack(mol_emb)
    # mol_emb_2 = torch.stack(mol_emb_2)

    max_text_length = 0
    for text_embedding in text_embeddings:
        if len(text_embedding) > max_text_length:
            max_text_length = len(text_embedding)

    padded_text_embeddings = []
    text_attn_masks = []
    for text_embedding in text_embeddings:
        L, D = text_embedding.shape
        text_attn_mask = torch.zeros(max_text_length, device=text_embedding.device, dtype=torch.uint8)
        text_padding = torch.zeros((max_text_length, D), dtype=torch.bfloat16, device=text_embedding.device)
        text_padding[:L] = text_embedding
        text_attn_mask[:L] = 1
        padded_text_embeddings.append(text_padding)
        text_attn_masks.append(text_attn_mask)

    padded_text_embeddings = torch.stack(padded_text_embeddings)
    text_attn_masks = torch.stack(text_attn_masks)

    # 使用 pad_sequence 填充输入
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # padded_input_ids = torch.ones([len(input_ids), 1024], dtype=input_ids.dtype) * tokenizer.pad_token_id
    # padded_input_ids[:, :input_ids.shape[-1]] = input_ids
    # input_ids = padded_input_ids
    #
    # attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    # padded_attention_mask = torch.zeros([len(input_ids), 1024], dtype=input_ids.dtype)
    # padded_attention_mask[:, :attention_mask.shape[-1]] = attention_mask
    # attention_mask = padded_attention_mask
    labels = torch.from_numpy(np.array(labels))
    # mask = labels >= -0.5  # 生成多任务回归使用的 label mask
    # labels_processed = labels.clone()  # 复制原张量以保留未满足条件的值

    # 计算实际的用来回归的值
    # labels = -torch.log10(labels / 10)

    return {
        # 'input_ids': input_ids,
        # 'attention_mask': attention_mask,
        'label': labels,
        'padded_text_embeddings': padded_text_embeddings,
        'text_attn_masks': text_attn_masks,
        'strain_names': strain_names,
        'mol_emb': mol_emb
    }

class RegressionHead(nn.Module):
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
        return x

class FirstTokenAttention_genome(nn.Module):
    def __init__(self, mol_cls_embed_dim, genome_embed_dim, num_heads, dropout=0.1):
        super(FirstTokenAttention_genome, self).__init__()
        self.mol_to_genome_dim = nn.Linear(mol_cls_embed_dim, genome_embed_dim)
        # self.genome_to_mol_dim = nn.Linear(genome_embed_dim, mol_cls_embed_dim)
        # 多头注意力层
        self.key_value_projection = nn.Linear(genome_embed_dim, genome_embed_dim * 2)
        self.mha = nn.MultiheadAttention(genome_embed_dim, num_heads, dropout=dropout)
        # 残差和归一化（LayerNorm）
        self.attn_norm = nn.LayerNorm(genome_embed_dim)
        self.norm1 = nn.LayerNorm(genome_embed_dim)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(genome_embed_dim, genome_embed_dim),
            nn.GELU(),
            nn.Linear(genome_embed_dim, genome_embed_dim)
        )
        self.norm2 = nn.LayerNorm(genome_embed_dim)

    def forward(self, mol_cls_emb, genome_embs, key_padding_mask, **kwargs):
        """
        x: Tensor, shape = (batch_size, seq_len, embed_dim)
        """
        # 提取序列的第一个 token，作为 query，形状: (batch_size, 1, embed_dim)
        genome_embs_dim = genome_embs.shape[-1]
        query = self.mol_to_genome_dim(mol_cls_emb)[:, None, :]

        if torch.isnan(query).any():
            print(" query 中包含 NaN\n")

        # nn.MultiheadAttention 要求输入 shape 为 (seq_len, batch_size, embed_dim)
        query = query.transpose(0, 1)  # (1, batch_size, embed_dim)
        key_value = self.key_value_projection(genome_embs.reshape(-1, genome_embs.shape[-1])).reshape([genome_embs.shape[0], genome_embs.shape[1], -1])
        key_value = key_value.transpose(0, 1)  # (seq_len, batch_size, embed_dim)

        if torch.isnan(key_value).any():
            print(" key_value 中包含 NaN\n")

        # value = key
        query_norm = self.attn_norm(query.squeeze(0)).unsqueeze(0)
        # 计算多头注意力：只计算第一个 token 对整个序列的注意力
        attn_output, attn_weights = self.mha(query_norm, key_value[:, :, :genome_embs_dim], key_value[:, :, genome_embs_dim:], key_padding_mask = key_padding_mask.to(torch.bool))  # (1, batch_size, embed_dim)

        if torch.isnan(attn_output).any():
            print(" attn_output 中包含 NaN\n")
            print(key_padding_mask)
            print(key_padding_mask.shape)
            print(f' sum: {key_padding_mask.sum()}')
            exit(0)

        # 残差连接与归一化
        # attn_output = self.genome_to_mol_dim(attn_output.squeeze())
        query = self.norm1(query.squeeze() + attn_output.squeeze())

        # 前馈网络 + 残差连接和归一化
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)

        # 最终只输出更新后的第一个 token embedding，返回形状 (batch_size, embed_dim)
        return query

def calculate_r2(all_labels, all_preds):
    # 确保输入是 numpy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    # 计算 R^2
    ss_total = np.sum((all_labels - np.mean(all_labels)) ** 2)  # 总平方和
    ss_residual = np.sum((all_labels - all_preds) ** 2)  # 残差平方和
    r2 = 1 - (ss_residual / ss_total)

    return r2

def exclude_wrong_species_ATCC_map(Evo_MIC_data_with_genome_embedding:np.array, genome_ID_to_species_first_name_dict):
    """
    去掉那些原始 DBAASP 中连 species name 和 ATCC ID 都对不上的数据，只处理那些没有手动标注的！
    :param Evo_MIC_data_with_genome_embedding: SMIELS, strain -> MIC data
    :param genome_ID_to_species_first_name_dict: dict, {ATCC_ID: species_name }, 这个是直接从 保存的 ATCC genome embedding 文件名获得的
    :return: cleaned SMIELS, strain -> MIC data, np.array
    """
    # 记录一下清理之前有多少数据点
    original_length = len(Evo_MIC_data_with_genome_embedding)

    marked_ATCC_IDs = set()
    cleaned_data = []
    for line in Evo_MIC_data_with_genome_embedding:
        name = line[2]

        # 那些没有 ATCC ID 但是一定被手动标注了的情况
        if 'ATCC' not in name:
            cleaned_data.append(line)
            continue

        if 'ATCC' in name:
            ATCC_id = name.split('ATCC')[-1].strip()
            if 'BAA' in name:
                ATCC_id = ATCC_id.replace(" ", "-")
            if 'MY' in name:
                ATCC_id = ATCC_id.replace(" ", "")
            if 'MAY' in name:
                ATCC_id = ATCC_id.replace("MAY", "MYA")
            if 'D' in name:
                ATCC_id = ATCC_id.split("D")[0]
            if 'T' in name:
                ATCC_id = ATCC_id.split("T")[0]
            if 's' in name:
                ATCC_id = ATCC_id.split("s")[0]
            if " " in name:
                ATCC_id = ATCC_id.split(" ")[0]

        # 手动标记过 ATCC 的情况
        if genome_ID_to_species_first_name_dict.get(ATCC_id) is None:
            cleaned_data.append(line)
            marked_ATCC_IDs.add(ATCC_id)

        # 如果 species name 符合，那么是干净的数据
        elif genome_ID_to_species_first_name_dict[ATCC_id] in name:
            cleaned_data.append(line)

    cleaned_data = np.array(cleaned_data)

    wrong_ATCC_numbers = set(Evo_MIC_data_with_genome_embedding[:, 2]) - set(cleaned_data[:, 2])

    print(f'\n wrong strain names: {wrong_ATCC_numbers}')
    print(f'\n double marked_ATCC_IDs: {marked_ATCC_IDs}')

    print(f'\n original data length (no "*", no manual modification) {original_length}\n cleaned data length {len(cleaned_data)}\n')

    return cleaned_data

def get_ATCC_ID_to_species_name_map(ATCC_fasta_folder_path:Path):
    file_names = [f.name for f in ATCC_fasta_folder_path.iterdir() if f.is_file()]

    # ATCC_ID_to_species_names_map = {}

    ATCC_ID_list = []
    species_name_list = []

    for file_name in file_names:

        # 先获得这个 ATCC genome fasta 文件的 ATCC ID
        ATCC_id = file_name.split('.')[0].split('ATCC')[-1].strip()
        ATCC_id = ATCC_id.replace("_", " ").strip().replace(" ", "-")
        ATCC_ID_list.append(ATCC_id)

        # 然后获得这个 ATCC genome fasta 文件的 species name
        file_name = file_name.split('ATCC')[0]
        if 'subsp' in file_name.split('_'):
            file_name = file_name.split('subsp')[0]
        if 'pathovar' in file_name.split('_'):
            file_name = file_name.split('pathovar')[0]  # 带有 pathovar 和 var 的在 NCBI Taxonomy Browser 中都是识别不到的
        if 'var' in file_name.split('_'):
            file_name = file_name.split('var')[0]
        if 'sp' in file_name.split('_'):
            file_name = file_name.split('_sp')[0]
        species_name = file_name.replace('_', ' ').strip()
        species_name_list.append(species_name)

        # 存进这个 map 字典里
        # ATCC_ID_to_species_names_map[ATCC_id] = species_name

    ATCC_ID_to_species_names_map = dict(zip(ATCC_ID_list, species_name_list))
    species_names_to_ATCC_ID_map = {}

    ATCC_ID_list = np.array(ATCC_ID_list)
    species_name_list = np.array(species_name_list)

    for species_name in set(species_name_list):
        species_names_to_ATCC_ID_map[species_name] = ATCC_ID_list[species_name_list == species_name]

    return ATCC_ID_to_species_names_map, species_names_to_ATCC_ID_map

def get_original_strain_ID_to_species_name_map(original_text_emb_folder_path:Path):
    file_names = [f.name for f in original_text_emb_folder_path.iterdir() if f.is_file()]

    # ATCC_ID_to_species_names_map = {}

    strain_name_list = []
    species_name_list = []

    for file_name in file_names:

        # 先获得这个 ATCC genome fasta 文件的 ATCC ID
        strain_name = file_name.split('.pt')[0].replace('～', ' ').replace('^', '/')
        species_name = " ".join(strain_name.split(' ')[:2])
        strain_name_list.append(strain_name)
        species_name_list.append(species_name)

    strain_name_to_species_names_map = dict(zip(strain_name_list, species_name_list))
    species_names_to_strain_name_map = {}

    strain_name_list = np.array(strain_name_list)
    species_name_list = np.array(species_name_list)

    for species_name in set(species_name_list):
        species_names_to_strain_name_map[species_name] = strain_name_list[species_name_list == species_name]

    return strain_name_to_species_names_map, species_names_to_strain_name_map

def merge_dict(dict_1, dict_2):
    merged_dict = {}

    # 先将第一个字典中的内容全部添加到merged_dict中
    for key, value in dict_1.items():
        merged_dict[key] = list(value)  # 复制列表，防止原列表被修改

    # 遍历第二个字典
    for key, value in dict_2.items():
        if key in merged_dict:
            # 如果键已存在，则合并两个列表
            merged_dict[key].extend(value)
        else:
            # 如果键不存在，则直接添加
            merged_dict[key] = list(value)

    return  merged_dict


if __name__=='__main__':
    current_folder = Path(__file__).parent

    parser = argparse.ArgumentParser(
        description=' Cross validation',  # 在参数帮助信息之前显示的文本
    )
    parser.add_argument(
        '-p', '--parallel',  # 可选参数
        action='store_true',
        help='whether to parallel validation on multi GPUs'
    )
    parser.add_argument(
        '-t', '--test_group',  # 可选参数
        type=int,
        # choices=['Serinales', 'Betaproteobacteria', 'FCB', 'VPC', 'BFSP', 'Eurotiomycetes', 'MA', 'Bacillales', 'Enterobacterales', 'Lactobacillales', 'ALs'],  # 可选项列表
        default=None,
        help='which task to test on in this experiment'
    )
    parser.add_argument(
        '-d', '--device',  # 可选参数
        type=int,
        default=3,
        help='Which GPU to use'
    )
    parser.add_argument(
        '-e', '--epoch',  # 可选参数
        type=int,
        default=8,
        help='How many epochs to train'
    )
    parser.add_argument(
        '-w', '--weight_decay',  # 可选参数
        type=float,
        default=0,
        help='weight decay lambda'
    )
    args = parser.parse_args()

    # -------------------------
    if args.parallel:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    # -------------------------

    # if args.parallel and args.test_group is None:
    #     print('\n Please specify test group when parallel validation is on')
    #     exit(1)
    genome_embedding_scale_factor = 1e14
    text_embedding_scale_factor = 1
    # device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    lora_r_ChemBERTa = 16
    lora_config_ChemBERTa = LoraConfig(
        r=lora_r_ChemBERTa,
        lora_alpha=32,
        target_modules=["query", "key", "value", 'dense', "mol_to_genome_dim", "key_value_projection", "mha.out_proj", "ffn.0", "ffn.2", 'dense_1', 'dense_2', "out_proj"],  # 也可以包含 "dense" 等其他线性层, 看你想插在哪些层
        task_type=TaskType.FEATURE_EXTRACTION,  # 不走任何特定任务逻辑，最通用的方式
        lora_dropout=0.1,
        bias="none"
    )
    lora_r_other = 64
    lora_config_co_cross = LoraConfig(
        r=lora_r_other,
        lora_alpha=32,
        target_modules=["mol_to_genome_dim", "key_value_projection", "mha.out_proj", "ffn.0", "ffn.2"],  # 也可以包含 "dense" 等其他线性层, 看你想插在哪些层
        task_type=TaskType.FEATURE_EXTRACTION,  # 不走任何特定任务逻辑，最通用的方式
        lora_dropout=0.1,
        bias="none"
    )

    lora_r_other = 64
    lora_config_reg = LoraConfig(
        r=lora_r_other,
        lora_alpha=32,
        target_modules=['dense_1', 'dense_2', "out_proj"], # 也可以包含 "dense" 等其他线性层, 看你想插在哪些层
        task_type=TaskType.FEATURE_EXTRACTION,  # 不走任何特定任务逻辑，最通用的方式
        lora_dropout=0.1,
        bias="none"
    )

    # , 'dense_1', 'dense_2', "out_proj"
    batch_size = 320
    num_ensembles = 7  # 要集成几个 model 来做预测
    pred_strain_name = 'BAA-3170'
    weight_selection = 'DBAASP_train_best'  # 'DBAASP_train_best' / 'inhouse_best'
    random_seeds = [42, 2024, 2025, 2077, 2012, 1973, 2002, 2001, 2020, 2019, 31, 13, 55, 11, 12, 58, 72, 2010, 2008,
                    2001, 1717, 1313, 99, 83, 29, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                    1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027]
    model_save_dir = current_folder / 'Checkpoints' / 'genome_text_learnable_emb' / f'strain_wise_synergy' / f'MDLM_train_on_DBAASP_filter_on_inhouse_cls'
    if not model_save_dir.exists():
        model_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n {str(model_save_dir)} created！")
    else:
        print(f"\n {str(model_save_dir)} exist.")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 创建文件Handler
    file_handler = logging.FileHandler(model_save_dir / f'log_group_{args.test_group}.log', mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 创建控制台Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    # 添加到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # 示例输出
    logger.info("Start")


    embeddings_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Genome_embs'
    text_embeddings_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'ATCC' / 'embeddings'
    text_embeddings_wo_genome_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'wo_ATCC' / 'embeddings'

    embedded_genome_IDs, genome_ID_to_species_first_name_dict = get_embedded_genome_IDs(embeddings_folder_path)
    embedded_text_IDs, text_ID_to_species_first_name_dict = get_embedded_genome_IDs(text_embeddings_folder_path)
    Evo_MIC_count_file_path = current_folder / 'DataPrepare' / 'Data' / 'Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json'

    # Test data
    data_path = current_folder / 'DataPrepare' / 'Data' / 'inhouse_synergy' / 'processed' / 'combine_create_inhouse_synergy_Evo_smiles_seq.csv'  # 替换为你的数据路径
    synergy_data = pd.read_csv(data_path)
    synergy_data['strain_name'] = pred_strain_name
    columns_names = synergy_data.columns

    gt_test_data = synergy_data

    model_name = "ibm-research/materials.selfies-ted"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    synergy_mol_emb_dict = torch.load(current_folder / 'DataPrepare' / 'Data' / 'combine_create_synergy_inhouse_mol_emb_dict_cls_wo_pad.pt')


    embeddings_dict = load_all_genome_embeddings(embeddings_folder_path, genome_embedding_scale_factor, device,'genome')
    text_embeddings_dict = load_all_genome_embeddings(text_embeddings_folder_path, text_embedding_scale_factor,
                                                      device,'text (with corresponding genome)')

    # gt_train_dataset = SMILESDataset_with_genome_and_text(gt_train_data, tokenizer, embeddings_dict, text_embeddings_dict, 'genome-text training set', mol_emb_dict=synergy_mol_emb_dict, max_length=1024)
    gt_test_dataset = SMILESDataset_with_genome_and_text(gt_test_data, tokenizer, embeddings_dict, text_embeddings_dict, 'genome-text test set', mol_emb_dict=synergy_mol_emb_dict, max_length=1024)
    # gt_test_loader = DataLoader(gt_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    # -------------------------
    if args.parallel:
        test_sampler = DistributedSampler(gt_test_dataset, shuffle=False)
        gt_test_loader = DataLoader(
            gt_test_dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            collate_fn=collate_fn,
        )
    else:
        gt_test_loader = DataLoader(
            gt_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
    # -------------------------

    criterion = nn.MSELoss()


    test_predictions_of_ensembles = []
    for ensemble in tqdm(range(num_ensembles), desc=' Doing ensembles '):

        if weight_selection == 'inhouse_best':
            base_model_dir = current_folder / 'Checkpoints' / 'genome_text_learnable_emb' / 'strain_wise_synergy' / 'MDLM_train_on_DBAASP_test_on_inhouse_reg' / f'fold_0_ensemble_{ensemble}_best_test.ckpt'  # 这个权重在所有的 MIC data 上训练过
        elif weight_selection == 'DBAASP_train_best':
            base_model_dir = current_folder / 'Checkpoints' / 'genome_text_learnable_emb' / 'strain_wise_synergy' / 'MDLM_train_on_DBAASP_test_on_inhouse_reg' / f'fold_0_ensemble_{ensemble}_fixed_epoch.ckpt'  # 这个权重在所有的 MIC data 上训练过
        state_dict = torch.load(base_model_dir, map_location=torch.device('cpu'), weights_only=False)

        # -------------------------
        co_cross_attn_genome = FirstTokenAttention_genome(768, 8192, 4, 0.1).to(device)
        co_cross_attn_text = FirstTokenAttention_genome(768, 4096, 4, 0.1).to(device)
        co_cross_attn_genome = get_peft_model(co_cross_attn_genome, lora_config_co_cross)
        co_cross_attn_text = get_peft_model(co_cross_attn_text, lora_config_co_cross)

        reg_head = RegressionHead((8192 + 4096) * 2, (8192 + 4096) // 4, 128, 1, 0.2).to(device)

        co_cross_attn_genome.load_state_dict(state_dict['co_cross_attn_genome'])
        co_cross_attn_text.load_state_dict(state_dict['co_cross_attn_text'])
        reg_head.load_state_dict(state_dict['re_head_state_dict'])
        if args.parallel:
            co_cross_attn_genome = DDP(co_cross_attn_genome, device_ids=[local_rank], output_device=local_rank)
            co_cross_attn_text = DDP(co_cross_attn_text, device_ids=[local_rank], output_device=local_rank)
            reg_head = DDP(reg_head, device_ids=[local_rank], output_device=local_rank)
        # -------------------------

        # co_cross_attn_genome.load_state_dict(state_dict['co_cross_attn_genome'])
        co_cross_attn_genome.to(device)
        # co_cross_attn_genome.eval()
        # co_cross_attn_text.load_state_dict(state_dict['co_cross_attn_text'])
        co_cross_attn_text.to(device)
        # co_cross_attn_text.eval()
        # reg_head.load_state_dict(state_dict['re_head_state_dict'])
        reg_head.to(device)
        # reg_head.eval()

        with torch.no_grad():

            test_batch_losses = []
            test_all_labels = []
            test_all_preds = []
            gt_test_batch_losses = []
            gt_test_all_labels = []
            gt_test_all_preds = []
            t_test_batch_losses = []
            t_test_all_labels = []
            t_test_all_preds = []

            species_wise_test_labels_dict = {}
            species_wise_test_preds_dict = {}

            co_cross_attn_genome.eval()
            co_cross_attn_text.eval()
            reg_head.eval()

            ensemble_i_test_pred = []
            for gt_batch in tqdm(
                    gt_test_loader,
                    desc=f" Ensemble {ensemble + 1}/{num_ensembles} | evaluating",
                    leave=False, total=len(gt_test_loader)):
                if gt_batch is not None:
                    # input_ids = gt_batch['input_ids'].to(device)
                    # attention_mask = gt_batch['attention_mask'].to(device)
                    labels = gt_batch['label'].to(device)
                    padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
                    genome_attn_masks = gt_batch['genome_attn_masks']
                    padded_text_embeddings = gt_batch['padded_text_embeddings']  # .to(torch.float)
                    text_attn_masks = gt_batch['text_attn_masks']
                    strain_names = gt_batch['strain_names']
                    mol_cls_embedding = gt_batch['mol_emb'].to(device)

                    with torch.amp.autocast('cuda', enabled=True):
                        # outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
                        #
                        # mol_cls_embedding = outputs.last_hidden_state[:, 0, :]
                        mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_emb = mol_cls_embedding, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
                        mol_cls_embedding_text = co_cross_attn_text(mol_cls_emb = mol_cls_embedding, genome_embs = padded_text_embeddings, key_padding_mask = 1 - text_attn_masks)
                        mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
                        # logits = reg_head(features = mol_cls_embedding)
                        FICI_input_1 = torch.cat((mol_cls_embedding[::2], mol_cls_embedding[1::2]), dim=1)
                        FICI_input_2 = torch.cat((mol_cls_embedding[1::2], mol_cls_embedding[::2]), dim=1)
                        logits_1 = reg_head(FICI_input_1)
                        logits_2 = reg_head(FICI_input_2)
                        logits = (logits_1 + logits_2) / 2

                        real_FICI = 10 ** (-logits) * 10

                        ensemble_i_test_pred.extend(real_FICI.detach().cpu().tolist())

                    test_batch_labels = labels.detach().cpu().flatten().tolist()
                    test_batch_preds = torch.sigmoid(logits).detach().cpu().flatten().tolist()

                    test_all_preds.extend(test_batch_preds)
                    gt_test_all_preds.extend(test_batch_preds)

        test_predictions_of_ensembles.append(ensemble_i_test_pred)

    # ==================== 解决方案：在此处收集并统一处理 ====================

    # 1. 将每个进程的预测结果列表收集起来
    #    注意：test_predictions_of_ensembles 是一个 list of lists
    #    我们需要用 all_gather_object 来收集 python 对象
    if args.parallel:
        # 创建一个列表，用来存放从所有进程收集到的结果
        gathered_predictions = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_predictions, test_predictions_of_ensembles)

    # 2. 只在主进程 (rank 0) 上进行后续处理
    if not args.parallel or dist.get_rank() == 0:

        if args.parallel:
            # 如果是并行模式，需要将收集到的数据重新组合
            # gathered_predictions 的结构是: [proc0_preds, proc1_preds, proc2_preds, ...]
            # proc0_preds 的结构是: [[ensemble0_preds], [ensemble1_preds], ...]
            # 我们需要把它重新整理成按 ensemble 分组
            num_ensembles = len(gathered_predictions[0])
            num_samples_per_proc = len(gathered_predictions[0][0])
            world_size = dist.get_world_size()

            # 重新构建完整的 test_predictions_of_ensembles
            final_predictions = []
            for i in range(num_ensembles):
                ensemble_i_full_pred = []
                for rank_preds in gathered_predictions:
                    ensemble_i_full_pred.extend(rank_preds[i])
                final_predictions.append(ensemble_i_full_pred)

            test_predictions_of_ensembles = final_predictions

        # --- 从这里开始，是你原来的后处理逻辑 ---

        test_predictions_of_ensembles = np.mean((np.array(test_predictions_of_ensembles)), axis=0)
        indices = np.where(test_predictions_of_ensembles < 0.5)[0]

        peptide_data = pd.read_csv(current_folder / 'DataPrepare' / 'Data' / 'inhouse_synergy' / 'processed' / 'combine_create_inhouse_synergy_Evo_pep_seq.csv')
        peptide_data = peptide_data.values
        predicted_synergy_pairs = peptide_data[indices]
        predicted_synergy_pairs[:, -1] = test_predictions_of_ensembles[indices].squeeze()
        predicted_synergy_pairs = pd.DataFrame(predicted_synergy_pairs, columns=columns_names)

        predicted_synergy_pairs.to_csv(current_folder / 'DataPrepare' / 'Data' / 'inhouse_synergy' / 'processed' / f'filtered_combine_create_inhouse_synergy_Evo_pep_seq_{weight_selection}.csv', index=False)

        print(f' ✅ finished. \n length of predicted synergy_pairs: {len(predicted_synergy_pairs)}')

    # 程序结束前销毁进程组
    # -------------------------
    if args.parallel:
        dist.destroy_process_group()
    # -------------------------


