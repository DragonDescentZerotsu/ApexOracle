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
        self.remove_long_smiles()
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
    labels = -torch.log10(labels / 10)

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
    labels = -torch.log10(labels / 10)

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
    if args.parallel and args.test_group is None:
        print('\n Please specify test group when parallel validation is on')
        exit(1)
    genome_embedding_scale_factor = 1e14
    text_embedding_scale_factor = 1
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
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
    num_ensembles = 7  # 要集成几个 model 来做预测
    random_seeds = [42, 2024, 2025, 2077, 2012, 1973, 2002, 2001, 2020, 2019, 31, 13, 55, 11, 12, 58, 72, 2010, 2008,
                    2001, 1717, 1313, 99, 83, 29, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011,
                    1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027]
    model_save_dir = current_folder / 'Checkpoints' / 'genome_text_learnable_emb' / f'strain_wise_synergy' / f'MDLM_train_on_DBAASP_test_on_inhouse_reg'
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
    
    # 读取 CSV 数据
    # data_path = '/home/tianang/Projects/Synergy/DataPrepare/Data/DBAASP_id_same_as_AAseqs_SMILES_bact_MICs.csv'  # 替换为你的数据路径
    # data_path = current_folder/'DataPrepare'/'Data'/'DBAASP_id_SMILES_bact_mean_MICs.csv'  # 替换为你的数据路径
    data_path = current_folder / 'DataPrepare' / 'Data' / 'synergistic_pairs_Evo.csv'  # 替换为你的数据路径
    synergy_data = pd.read_csv(data_path)
    columns_names = synergy_data.columns
    synergy_data = synergy_data.values

    # FICI_scaler = StandardScaler()
    # synergy_data[:, -1] = FICI_scaler.fit_transform(synergy_data[:, -1].reshape(-1, 1)).squeeze()

    embeddings_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Genome_embs'
    text_embeddings_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'ATCC' / 'embeddings'
    text_embeddings_wo_genome_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'wo_ATCC' / 'embeddings'

    embedded_genome_IDs, genome_ID_to_species_first_name_dict = get_embedded_genome_IDs(embeddings_folder_path)
    embedded_text_IDs, text_ID_to_species_first_name_dict = get_embedded_genome_IDs(text_embeddings_folder_path)
    Evo_MIC_count_file_path = current_folder / 'DataPrepare' / 'Data' / 'Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json'

    original_names_with_genome_embedding_handcrafted, original_names_with_genome_embedding_DBAASP_original, origin_to_standard_name_map_dict = get_original_strain_name_with_genome_embedding(Evo_MIC_count_file_path, embedded_genome_IDs)

    # 去掉那些带 'del' 的
    del_excluded_data = []
    for synergy_data_line in tqdm(synergy_data, desc=' removing synergy data with "del" in name '):
        if 'del' not in synergy_data_line[2]:
            del_excluded_data.append(synergy_data_line)
    synergy_data = del_excluded_data

    # filter 一下留下那些有对应 strain 的 genome 的 SMILES -> MIC 对数据
    Evo_MIC_data_with_genome_embedding_handcrafted = []
    for synergy_data_line in tqdm(synergy_data, desc=' retriving synergy data with genome embeddings '):
        if synergy_data_line[2] in original_names_with_genome_embedding_handcrafted:
            Evo_MIC_data_with_genome_embedding_handcrafted.append(synergy_data_line)

    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = []
    for synergy_data_line in tqdm(synergy_data, desc=' retriving synergy data with genome embeddings '):
        if synergy_data_line[2] in original_names_with_genome_embedding_DBAASP_original:
            Evo_MIC_data_with_genome_embedding_DBAASP_origianl.append(synergy_data_line)

    text_embeddings_wo_genome_dict = load_text_wo_genome_embeddings(text_embeddings_wo_genome_folder_path, text_embedding_scale_factor, device, 'text (without corresponding genome)')
    Evo_MIC_data_wo_genome_embedding = []
    for synergy_data_line in tqdm(synergy_data, desc=' retriving FICI data with only text embeddings '):
        if len(synergy_data_line[2].split(' ')) <= 1:
            continue
        # 调试用
        # if MIC_data_line[1].split(' ')[1] in ['sp.', 'spp.', 'group']:
        #     print(1)
        if synergy_data_line[2].split(' ')[1] not in ['sp.', 'spp.', 'group'] and synergy_data_line[2] in list(text_embeddings_wo_genome_dict.keys()):
            Evo_MIC_data_wo_genome_embedding.append(synergy_data_line)

    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = np.array(Evo_MIC_data_with_genome_embedding_DBAASP_origianl)
    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = exclude_wrong_species_ATCC_map(Evo_MIC_data_with_genome_embedding_DBAASP_origianl, genome_ID_to_species_first_name_dict)

    Evo_MIC_data_with_genome_embedding = np.concatenate((np.array(Evo_MIC_data_with_genome_embedding_handcrafted), Evo_MIC_data_with_genome_embedding_DBAASP_origianl))

    Evo_MIC_data_with_genome_embedding_standard_name = []
    for line in Evo_MIC_data_with_genome_embedding:
        # 替换原始的 strain name 到 ATCC 或者是下载的genome 的 name，方便embedding载入
        line[2] = origin_to_standard_name_map_dict[line[2]]
        Evo_MIC_data_with_genome_embedding_standard_name.append(line)
    Evo_MIC_data_with_genome_embedding_standard_name = np.array(Evo_MIC_data_with_genome_embedding_standard_name)

    Evo_MIC_data_with_genome_or_text_embedding = np.concatenate((Evo_MIC_data_with_genome_embedding_standard_name, np.array(Evo_MIC_data_wo_genome_embedding)))

    embeddings_dict = load_all_genome_embeddings(embeddings_folder_path, genome_embedding_scale_factor, device, 'genome')
    text_embeddings_dict = load_all_genome_embeddings(text_embeddings_folder_path, text_embedding_scale_factor, device, 'text (with corresponding genome)')

    # TODO: 这里分成两个数据集做了，利用 set 获取全部的 MIC 数据中有哪些 strain name，并将每一种 strain name 的数据分组（dict）保存方便分割数据集
    all_name_set_train = set(Evo_MIC_data_with_genome_or_text_embedding[:, 2])  # TODO: 这里已经不仅仅是 ATCC 的 ID 和 #001 之类的了，还有只有 text 的那些完全没处理过的 strain name
    all_strain_line_group_dict_train = {}
    for standard_strain_ID in tqdm(all_name_set_train, desc=' Getting strain MIC groups '):
        indices = np.where(Evo_MIC_data_with_genome_or_text_embedding[:, 2] == standard_strain_ID)[0]
        all_strain_line_group_dict_train[standard_strain_ID] = Evo_MIC_data_with_genome_or_text_embedding[indices]

    # 利用 set 获取 MIC 数据中有哪些 ATCC ID，并将每一种 ATCC ID 的数据分组（dict）保存方便分割数据集
    all_standard_name_set_train = set(Evo_MIC_data_with_genome_embedding_standard_name[:, 2])
    standard_strain_line_group_dict_train = {}
    for standard_strain_ID in tqdm(all_standard_name_set_train, desc=' Getting strain MIC groups '):
        indices = np.where(Evo_MIC_data_with_genome_embedding_standard_name[:, 2] == standard_strain_ID)[0]
        standard_strain_line_group_dict_train[standard_strain_ID] = Evo_MIC_data_with_genome_embedding_standard_name[indices]

    ATCC_ID_to_species_name_map_dict, species_name_ATCC_IDs_map_dict = get_ATCC_ID_to_species_name_map(
        current_folder / 'DataPrepare' / 'Data' / 'Genome' / 'ATCC')
    strain_name_to_original_species_names_map, original_species_names_to_strain_name_map = get_original_strain_ID_to_species_name_map(
        current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'wo_ATCC' / 'embeddings')

    # 把两个 species 到 strain IDs 的 dict 融合一下
    merged_species_name_to_strain_name_map = merge_dict(species_name_ATCC_IDs_map_dict, original_species_names_to_strain_name_map)

    # 修正新旧命名
    with open(current_folder / 'DataPrepare' / 'Data' / 'Genome' / 'old_to_new_NCBI_taxonomy.json', 'r',
              encoding='utf-8') as f:
        old_to_new_NCBI_taxonomy_map = json.load(f)
    new_to_old_NCBI_taxonomy_map = {value: key for key, value in old_to_new_NCBI_taxonomy_map.items()}
    two_way_taxonomy_map = new_to_old_NCBI_taxonomy_map | old_to_new_NCBI_taxonomy_map


    # train_groups = [[], [], []]
    # test_groups = [[], [], []]
    train_groups = [[]]

    # for i in tqdm(range(len(train_groups)), total=len(train_groups), desc=' Generating train / test spliting for different folds ... '):

    repeated_speceis_name_NCBI = []

    for species_name, corresponding_ATCC_IDs in merged_species_name_to_strain_name_map.items():

        # 如果重复的已经处理过了就跳过
        if species_name in repeated_speceis_name_NCBI:
            continue

        mergred_corresponding_ATCC_IDs = corresponding_ATCC_IDs

        if species_name in two_way_taxonomy_map.keys():
            # 防止在此处理相同的 strains
            repeated_speceis_name_NCBI.append(two_way_taxonomy_map[species_name])
            _strains_2 = merged_species_name_to_strain_name_map.get(two_way_taxonomy_map[species_name], None)
            if _strains_2 is not None:
                mergred_corresponding_ATCC_IDs.extend(_strains_2)

        train_groups[0].extend(mergred_corresponding_ATCC_IDs)

            # mergred_corresponding_ATCC_IDs.sort()
            # if len(mergred_corresponding_ATCC_IDs) >= 6:
            #     mergred_corresponding_ATCC_IDs[1], mergred_corresponding_ATCC_IDs[2] = mergred_corresponding_ATCC_IDs[2], mergred_corresponding_ATCC_IDs[1]  # 交换第1，2个防止数据量多的都在前面
            #
            # # 只有 1 个 strain 的 species 全部放到这里的训练集里
            # if len(mergred_corresponding_ATCC_IDs) == 1:
            #     train_groups[i].extend(mergred_corresponding_ATCC_IDs)
            # elif len(mergred_corresponding_ATCC_IDs) == 2:
            #     train_groups[i].append(mergred_corresponding_ATCC_IDs[i % 2])
            #     test_groups[i].append(mergred_corresponding_ATCC_IDs[(i + 1) % 2])
            # else:
            #     # chunk_length = len(corresponding_ATCC_IDs) // (len(train_groups) - 1)  # 注意这个是真正的 fold 大小
            #     chunk_length = len(mergred_corresponding_ATCC_IDs) // len(train_groups)
            #     chunked_ATCC_IDs_for_test = mergred_corresponding_ATCC_IDs[i * chunk_length: (i + 1) * chunk_length]
            #     chunked_ATCC_IDs_for_train = list(set(mergred_corresponding_ATCC_IDs) - set(chunked_ATCC_IDs_for_test))
            #     train_groups[i].extend(chunked_ATCC_IDs_for_train)
            #     test_groups[i].extend(chunked_ATCC_IDs_for_test)





    # Test data
    data_path = current_folder / 'DataPrepare' / 'Data' / 'inhouse_synergy' / 'processed' / 'inhouse_synergy_Evo_smiles_seq.csv'  # 替换为你的数据路径
    synergy_data = pd.read_csv(data_path)
    columns_names = synergy_data.columns
    synergy_data = synergy_data.values

    # FICI_scaler = StandardScaler()
    # synergy_data[:, -1] = FICI_scaler.fit_transform(synergy_data[:, -1].reshape(-1, 1)).squeeze()

    embeddings_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Genome_embs'
    text_embeddings_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'ATCC' / 'embeddings'
    text_embeddings_wo_genome_folder_path = current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'wo_ATCC' / 'embeddings'

    embedded_genome_IDs, genome_ID_to_species_first_name_dict = get_embedded_genome_IDs(embeddings_folder_path)
    embedded_text_IDs, text_ID_to_species_first_name_dict = get_embedded_genome_IDs(text_embeddings_folder_path)
    Evo_MIC_count_file_path = current_folder / 'DataPrepare' / 'Data' / 'Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json'

    original_names_with_genome_embedding_handcrafted, original_names_with_genome_embedding_DBAASP_original, origin_to_standard_name_map_dict = get_original_strain_name_with_genome_embedding(
        Evo_MIC_count_file_path, embedded_genome_IDs)

    # 去掉那些带 'del' 的
    del_excluded_data = []
    for synergy_data_line in tqdm(synergy_data, desc=' removing synergy data with "del" in name '):
        if 'del' not in synergy_data_line[2]:
            del_excluded_data.append(synergy_data_line)
    synergy_data = del_excluded_data

    # filter 一下留下那些有对应 strain 的 genome 的 SMILES -> MIC 对数据
    Evo_MIC_data_with_genome_embedding_handcrafted = []
    for synergy_data_line in tqdm(synergy_data, desc=' retriving synergy data with genome embeddings '):
        if synergy_data_line[2] in original_names_with_genome_embedding_handcrafted:
            Evo_MIC_data_with_genome_embedding_handcrafted.append(synergy_data_line)

    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = []
    for synergy_data_line in tqdm(synergy_data, desc=' retriving synergy data with genome embeddings '):
        if synergy_data_line[2] in original_names_with_genome_embedding_DBAASP_original:
            Evo_MIC_data_with_genome_embedding_DBAASP_origianl.append(synergy_data_line)

    text_embeddings_wo_genome_dict = load_text_wo_genome_embeddings(text_embeddings_wo_genome_folder_path,
                                                                    text_embedding_scale_factor, device,
                                                                    'text (without corresponding genome)')
    Evo_MIC_data_wo_genome_embedding = []
    for synergy_data_line in tqdm(synergy_data, desc=' retriving FICI data with only text embeddings '):
        if len(synergy_data_line[2].split(' ')) <= 1:
            continue
        # 调试用
        # if MIC_data_line[1].split(' ')[1] in ['sp.', 'spp.', 'group']:
        #     print(1)
        if synergy_data_line[2].split(' ')[1] not in ['sp.', 'spp.', 'group'] and synergy_data_line[2] in list(
                text_embeddings_wo_genome_dict.keys()):
            Evo_MIC_data_wo_genome_embedding.append(synergy_data_line)

    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = np.array(Evo_MIC_data_with_genome_embedding_DBAASP_origianl)
    Evo_MIC_data_with_genome_embedding_DBAASP_origianl = exclude_wrong_species_ATCC_map(
        Evo_MIC_data_with_genome_embedding_DBAASP_origianl, genome_ID_to_species_first_name_dict)

    if len(Evo_MIC_data_with_genome_embedding_handcrafted) > 0:
        Evo_MIC_data_with_genome_embedding = np.concatenate(
            (np.array(Evo_MIC_data_with_genome_embedding_handcrafted),
             Evo_MIC_data_with_genome_embedding_DBAASP_origianl))
    else:
        Evo_MIC_data_with_genome_embedding = Evo_MIC_data_with_genome_embedding_DBAASP_origianl

    Evo_MIC_data_with_genome_embedding_standard_name = []
    for line in Evo_MIC_data_with_genome_embedding:
        # 替换原始的 strain name 到 ATCC 或者是下载的genome 的 name，方便embedding载入
        line[2] = origin_to_standard_name_map_dict[line[2]]
        Evo_MIC_data_with_genome_embedding_standard_name.append(line)
    Evo_MIC_data_with_genome_embedding_standard_name = np.array(Evo_MIC_data_with_genome_embedding_standard_name)

    if len(Evo_MIC_data_wo_genome_embedding) > 0:
        Evo_MIC_data_with_genome_or_text_embedding = np.concatenate(
            (Evo_MIC_data_with_genome_embedding_standard_name, np.array(Evo_MIC_data_wo_genome_embedding)))
    else:
        Evo_MIC_data_with_genome_or_text_embedding = Evo_MIC_data_with_genome_embedding_standard_name

    embeddings_dict = load_all_genome_embeddings(embeddings_folder_path, genome_embedding_scale_factor, device,
                                                 'genome')
    text_embeddings_dict = load_all_genome_embeddings(text_embeddings_folder_path, text_embedding_scale_factor, device,
                                                      'text (with corresponding genome)')

    # TODO: 这里分成两个数据集做了，利用 set 获取全部的 MIC 数据中有哪些 strain name，并将每一种 strain name 的数据分组（dict）保存方便分割数据集
    all_name_set_test = set(Evo_MIC_data_with_genome_or_text_embedding[:,
                       2])  # TODO: 这里已经不仅仅是 ATCC 的 ID 和 #001 之类的了，还有只有 text 的那些完全没处理过的 strain name
    all_strain_line_group_dict = {}
    for standard_strain_ID in tqdm(all_name_set_test, desc=' Getting strain MIC groups '):
        indices = np.where(Evo_MIC_data_with_genome_or_text_embedding[:, 2] == standard_strain_ID)[0]
        all_strain_line_group_dict[standard_strain_ID] = Evo_MIC_data_with_genome_or_text_embedding[indices]

    # 利用 set 获取 MIC 数据中有哪些 ATCC ID，并将每一种 ATCC ID 的数据分组（dict）保存方便分割数据集
    all_standard_name_set_test = set(Evo_MIC_data_with_genome_embedding_standard_name[:, 2])
    standard_strain_line_group_dict_test = {}
    for standard_strain_ID in tqdm(all_standard_name_set_test, desc=' Getting strain MIC groups '):
        indices = np.where(Evo_MIC_data_with_genome_embedding_standard_name[:, 2] == standard_strain_ID)[0]
        standard_strain_line_group_dict_test[standard_strain_ID] = Evo_MIC_data_with_genome_embedding_standard_name[indices]

    ATCC_ID_to_species_name_map_dict, species_name_ATCC_IDs_map_dict = get_ATCC_ID_to_species_name_map(
        current_folder / 'DataPrepare' / 'Data' / 'Genome' / 'ATCC')
    strain_name_to_original_species_names_map, original_species_names_to_strain_name_map = get_original_strain_ID_to_species_name_map(
        current_folder / 'DataPrepare' / 'Data' / 'Text_Description' / 'wo_ATCC' / 'embeddings')

    # 把两个 species 到 strain IDs 的 dict 融合一下
    merged_species_name_to_strain_name_map = merge_dict(species_name_ATCC_IDs_map_dict,
                                                        original_species_names_to_strain_name_map)

    # 修正新旧命名
    with open(current_folder / 'DataPrepare' / 'Data' / 'Genome' / 'old_to_new_NCBI_taxonomy.json', 'r',
              encoding='utf-8') as f:
        old_to_new_NCBI_taxonomy_map = json.load(f)
    new_to_old_NCBI_taxonomy_map = {value: key for key, value in old_to_new_NCBI_taxonomy_map.items()}
    two_way_taxonomy_map = new_to_old_NCBI_taxonomy_map | old_to_new_NCBI_taxonomy_map

    # train_groups = [[], [], []]
    # test_groups = [[], [], []]
    # train_groups = [[]]
    test_groups = [[]]

    # for i in tqdm(range(len(train_groups)), total=len(train_groups), desc=' Generating train / test spliting for different folds ... '):

    repeated_speceis_name_NCBI = []

    for species_name, corresponding_ATCC_IDs in merged_species_name_to_strain_name_map.items():

        # 如果重复的已经处理过了就跳过
        if species_name in repeated_speceis_name_NCBI:
            continue

        mergred_corresponding_ATCC_IDs = corresponding_ATCC_IDs

        if species_name in two_way_taxonomy_map.keys():
            # 防止在此处理相同的 strains
            repeated_speceis_name_NCBI.append(two_way_taxonomy_map[species_name])
            _strains_2 = merged_species_name_to_strain_name_map.get(two_way_taxonomy_map[species_name], None)
            if _strains_2 is not None:
                mergred_corresponding_ATCC_IDs.extend(_strains_2)

        test_groups[0].extend(mergred_corresponding_ATCC_IDs)

    group_names = ['fold 1']#, 'fold 2', 'fold 3']

    # 循环测试所有的 group
    for i, (strain_for_train, strain_for_test, test_group_name) in enumerate(zip(train_groups, test_groups, group_names)):

        # TODO: 调试用
        # if i==0:
        #     continue

        # 如果要 parallel 地 validate，那么在当前 group 不是 目标 test group 的时候直接跳过
        if args.parallel:
            # if test_group_name != args.test_group:
            if i != args.test_group:
                continue
        # print(f'\n Current test group: {test_group_name}\n')
        logger.info(f'\n Current test group: {test_group_name}\n')

        gt_strain_for_train = set(strain_for_train) & all_standard_name_set_train
        gt_strain_for_test = set(strain_for_test) & all_standard_name_set_test

        # gt_strain_for_test = set(strain_for_test) & all_standard_name_set
        # gt_strain_for_train = all_standard_name_set - gt_strain_for_test

        gt_train_data = []
        gt_test_data = []

        for strain_ID in gt_strain_for_train:
            gt_train_data.append(standard_strain_line_group_dict_train[strain_ID])
        for strain_ID in gt_strain_for_test:
            gt_test_data.append(standard_strain_line_group_dict_test[strain_ID])

        gt_train_data = pd.DataFrame(np.concatenate(gt_train_data), columns=columns_names)
        gt_test_data = pd.DataFrame(np.concatenate(gt_test_data), columns=columns_names)

        gt_train_mean_MIC = -np.log10(gt_train_data['FICI'].mean() / 10)
        # gt_train_mean_MIC = gt_train_data['FICI'].mean()

        # 新开一个只有 text embedding 的 dataset, 这里 test set 只留下那些和 genome text 都有的 test set 中不重合的的 strain，以免重复计算 test
        # t_strain_for_test = (set(strain_for_test) & all_name_set) - gt_strain_for_test
        t_strain_for_train = set(strain_for_train) & all_name_set_train  # all_name_set - t_strain_for_test - gt_strain_for_test

        t_train_data = []
        t_test_data = []

        for strain_ID in t_strain_for_train:
            t_train_data.append(all_strain_line_group_dict_train[strain_ID])
        # for strain_ID in t_strain_for_test:
        #     t_test_data.append(all_strain_line_group_dict[strain_ID])

        t_train_data = pd.DataFrame(np.concatenate(t_train_data), columns=columns_names)
        # t_test_data = pd.DataFrame(np.concatenate(t_test_data), columns=columns_names)

        t_train_mean_MIC = -np.log10(t_train_data['FICI'].mean() / 10)
        # t_train_mean_MIC = t_train_data['FICI'].mean()

        # bact_names_DBAASP = ["Escherichia coli ATCC 25922", "Pseudomonas aeruginosa ATCC 27853",
        #                      "Staphylococcus aureus ATCC 25923", "Staphylococcus aureus",
        #                      "Staphylococcus aureus ATCC 29213", "Escherichia coli", "Pseudomonas aeruginosa",
        #                      "Pseudomonas aeruginosa PAO1", "Enterococcus faecalis ATCC 29212",
        #                      "Acinetobacter baumannii ATCC 19606", "Staphylococcus epidermidis ATCC 12228",
        #                      "Candida albicans ATCC 10231", "Klebsiella pneumoniae ATCC 700603",
        #                      "Staphylococcus aureus ATCC 43300",
        #                      "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028",
        #                      "Staphylococcus aureus ATCC 6538", "Pseudomonas aeruginosa ATCC 9027", "Candida albicans",
        #                      "Klebsiella pneumoniae"]



        model_name = "ibm-research/materials.selfies-ted"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dataset = SMILESDataset(data, tokenizer)

        synergy_mol_emb_dict_1 = torch.load(current_folder / 'DataPrepare' / 'Data' / 'synergy_mol_emb_dict_cls_wo_pad.pt')
        synergy_mol_emb_dict_2 = torch.load(current_folder / 'DataPrepare' / 'Data' / 'synergy_inhouse_mol_emb_dict_cls_wo_pad.pt')
        synergy_mol_emb_dict = synergy_mol_emb_dict_1 | synergy_mol_emb_dict_2
        # SM_emb_dict = torch.load(current_directory / 'DataPrepare' / 'Data' / 'SM_emb_dict.pt')

        gt_train_dataset = SMILESDataset_with_genome_and_text(gt_train_data, tokenizer, embeddings_dict, text_embeddings_dict, 'genome-text training set', mol_emb_dict=synergy_mol_emb_dict)
        gt_test_dataset = SMILESDataset_with_genome_and_text(gt_test_data, tokenizer, embeddings_dict, text_embeddings_dict, 'genome-text test set', mol_emb_dict=synergy_mol_emb_dict)

        # 分别加载了两个 text embedding dict, 一个是只有 text embedding 的，还有一个是 genome 和 text embedding 都有的
        all_text_embedding_dict = text_embeddings_dict | text_embeddings_wo_genome_dict

        t_train_dataset = SMILESDataset_with_text_only(t_train_data, tokenizer, all_text_embedding_dict, 'text-only training set', mol_emb_dict=synergy_mol_emb_dict)
        # t_test_dataset = SMILESDataset_with_text_only(t_test_data, tokenizer, all_text_embedding_dict, 'text-only test set', mol_emb_dict=synergy_mol_emb_dict)

        # strain_wise_MIC_models_dir = current_folder / 'Checkpoints' / 'genome_text_learnable_emb' / 'strain_wise_w_SM_b_attn' / 'MDLM_MTR_fix_7_fold_ensembles'
        # strain_wise_MIC_models = [f.name for f in strain_wise_MIC_models_dir.iterdir() if 'pth' in f.name]
        base_model_dir = current_folder / 'Checkpoints' / 'genome_text_learnable_emb' / 'guidance_regressor_pad_no_mask' / 'noise_guidance_best_R2_all_peptide_epoch_100.pth'  # 这个权重在所有的 MIC data 上训练过

        test_predictions_of_ensembles = []
        for ensemble in tqdm(range(num_ensembles), desc=' Doing ensembles '):

            logging.info(f'\n Model loaded: {base_model_dir.name}')

            # 设置对应的 随机数种子
            torch.manual_seed(random_seeds[ensemble])
            torch.cuda.manual_seed(random_seeds[ensemble])

            # TODO: hyperparameters
            num_epochs = args.epoch
            min_lr = 1e-10
            batch_size = 70
            freeze_epochs = 0

            # print(f' num of frozen epochs: {freeze_epochs}\n')
            logger.info(f' num of frozen epochs: {freeze_epochs}\n')

            state_dict = torch.load(base_model_dir, map_location=torch.device('cpu'), weights_only=False)

            # ChemBERTa_model = AutoModel.from_pretrained(model_name)
            # ChemBERTa_model.load_state_dict(state_dict['ChemBERTa_state_dict'])
            #
            #
            # # 冻结预训练模型参数
            # for param in ChemBERTa_model.parameters():
            #     param.requires_grad = False

            genome_dim = gt_train_dataset[0]['genome_embedding'].shape[1]
            text_dim = gt_train_dataset[0]['text_embedding'].shape[1]

            co_cross_attn_genome = FirstTokenAttention_genome(768, gt_train_dataset[0]['genome_embedding'].shape[1], 4, 0.1)
            co_cross_attn_genome.load_state_dict(state_dict['co_cross_attn_genome'])  # TODO: 这里在正式代码中不要注释掉

            co_cross_attn_text = FirstTokenAttention_genome(768, gt_train_dataset[0]['text_embedding'].shape[1], 4, 0.1)
            co_cross_attn_text.load_state_dict(state_dict['co_cross_attn_text'])  # TODO: 这里在正式代码中不要注释掉

            reg_head = RegressionHead((genome_dim + text_dim)*2, (genome_dim + text_dim) // 4, 128, 1, 0.2)
            # reg_head.load_state_dict(state_dict['re_head_state_dict'])


            # ChemBERTa_model = get_peft_model(ChemBERTa_model, lora_config_ChemBERTa)
            # ChemBERTa_model.to(device)
            co_cross_attn_genome = get_peft_model(co_cross_attn_genome, lora_config_co_cross)
            co_cross_attn_genome.to(device)
            co_cross_attn_text = get_peft_model(co_cross_attn_text, lora_config_co_cross)
            co_cross_attn_text.to(device)
            # reg_head = get_peft_model(reg_head, lora_config_reg)
            reg_head.to(device)

            # print(f' ChemBERTa trainable parameters')
            # ChemBERTa_model.print_trainable_parameters()
            print(f' co_cross_attn_genome trainable parameters')
            co_cross_attn_genome.print_trainable_parameters()
            print(f' co_cross_attn_text trainable parameters')
            co_cross_attn_text.print_trainable_parameters()
            # print(f' reg_head trainable parameters')
            # reg_head.print_trainable_parameters()

            # FICI_head = nn.Sequential(
            #     nn.Linear(2, 4),
            #     nn.GELU(),
            #     nn.Linear(4, 1)
            # ).to(device) #nn.Linear(2, 1).to(device)

            learnable_embedding_weight = nn.Parameter(
                state_dict['learnable_embedding_weight'].to(device).detach(),
                requires_grad=False
            )

            criterion = nn.MSELoss()
            scaler = torch.cuda.amp.GradScaler()
            # scaler = torch.amp.GradScaler('cuda')
            optimizer = optim.Adam([p for p in co_cross_attn_genome.parameters() if p.requires_grad], lr=1e-5, weight_decay=args.weight_decay)  # 1e-5
            optimizer.add_param_group({'params': [p for p in co_cross_attn_text.parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': args.weight_decay})
            optimizer.add_param_group({'params': reg_head.parameters(), 'lr': 1e-5, 'weight_decay': args.weight_decay})
            # optimizer.add_param_group({'params': [learnable_embedding_weight], 'lr': 1e-5, 'weight_decay': args.weight_decay})
            # optimizer.add_param_group({'params': FICI_head.parameters(), 'lr': 1e-5, 'weight_decay': args.weight_decay})
            # optimizer.add_param_group({'params': [p for p in ChemBERTa_model.parameters() if p.requires_grad], 'lr': 1e-5, 'weight_decay': args.weight_decay * 0.1})  # TODO: ChemBERTa 和别的学习率不一样

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

            gt_train_loader = DataLoader(gt_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            gt_test_loader = DataLoader(gt_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            t_train_loader = DataLoader(t_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_text_only)
            # t_test_loader = DataLoader(t_test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_text_only)

            best_R2_test = -10
            best_spearman_test = -10
            best_pearson_test = -10
            best_test_prdictions = None
            for epoch in tqdm(range(num_epochs), desc=f' Training ensemble {ensemble + 1}/{num_ensembles} ', leave=False):

                # if epoch == freeze_epochs:
                #     # 解冻预训练模型
                #     for param in ChemBERTa_model.parameters():
                #         param.requires_grad = True
                #     # optimizer.add_param_group({'params': ChemBERTa_model.parameters(), 'lr': 1e-7})  # 在这里加的话会导致 scheduler 里面没有 ChemBERTa 的权重
                #     # print(f'\n ChemBERTa now open for training')
                #     logger.info(f'\n\n ChemBERTa now open for training')

                # 查看随机初始化状态下测试集的 R2 能到多少
                if epoch == 0:
                    with torch.no_grad():

                        test_batch_losses = []
                        test_all_labels = []
                        test_all_preds = []
                        species_wise_test_labels_dict = {}
                        species_wise_test_preds_dict = {}

                        gt_test_batch_losses = []
                        gt_test_all_labels = []
                        gt_test_all_preds = []
                        t_test_batch_losses = []
                        t_test_all_labels = []
                        t_test_all_preds = []
                        train_mean_as_test_predict = []

                        # for gt_batch, t_batch in tqdm(
                        #         itertools.zip_longest(gt_test_loader, t_test_loader, fillvalue=None),
                        #         desc=f" Epoch {epoch}/{num_epochs} | evaluating", leave=False,
                        #         total=max(len(gt_test_loader), len(t_test_loader))):
                        for gt_batch in tqdm(
                                gt_test_loader,
                                desc=f" Epoch {epoch}/{num_epochs} | evaluating", leave=False,
                                total=len(gt_test_loader)):
                            if gt_batch is not None:
                                # input_ids = gt_batch['input_ids'].to(device)
                                # attention_mask = gt_batch['attention_mask'].to(device)
                                labels = gt_batch['label'].to(device)
                                padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
                                genome_attn_masks = gt_batch['genome_attn_masks']
                                padded_text_embeddings = gt_batch['padded_text_embeddings']  # .to(torch.float)
                                text_attn_masks = gt_batch['text_attn_masks']
                                strain_names = gt_batch['strain_names']
                                mol_cls_embedding = gt_batch['mol_emb'].to(device)  # 这个是新加的

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
                                    loss = criterion(logits.squeeze(), labels.squeeze())

                                test_batch_losses.append(loss.item())
                                gt_test_batch_losses.append(loss.item())

                                test_batch_labels = labels.detach().cpu().flatten().tolist()
                                test_batch_preds = logits.detach().cpu().flatten().tolist()

                                test_all_labels.extend(test_batch_labels)
                                test_all_preds.extend(test_batch_preds)
                                gt_test_all_labels.extend(test_batch_labels)
                                gt_test_all_preds.extend(test_batch_preds)
                                train_mean_as_test_predict.extend(
                                    np.full(logits.detach().cpu().flatten().shape, gt_train_mean_MIC).tolist())

                                for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
                                    _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
                                    if _speceis_name is None:
                                        _speceis_name = strain_name_to_original_species_names_map[strain_name]
                                    if _speceis_name not in species_wise_test_preds_dict.keys():
                                        species_wise_test_preds_dict[_speceis_name] = [pred]
                                        species_wise_test_labels_dict[_speceis_name] = [label]
                                    else:
                                        species_wise_test_preds_dict[_speceis_name].append(pred)
                                        species_wise_test_labels_dict[_speceis_name].append(label)
 #
 #                             if t_batch is not None:
 #                                 # input_ids = t_batch['input_ids'].to(device)
 #                                 # attention_mask = t_batch['attention_mask'].to(device)
 #                                 labels = t_batch['label'].to(device)
 #                                 # padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
 #                                 # genome_attn_masks = gt_batch['genome_attn_masks']
 #                                 padded_text_embeddings = t_batch['padded_text_embeddings']  # .to(torch.float)
 #                                 text_attn_masks = t_batch['text_attn_masks']
 #                                 strain_names = t_batch['strain_names']
 #                                 mol_cls_embedding = t_batch['mol_emb'].to(device)
 #
 #                                 with torch.amp.autocast('cuda', enabled=True):
 #                                     # outputs = mdlm_model(input_ids=input_ids, attention_mask=attention_mask)
 #                                     #
 #                                     # mol_cls_embedding = outputs[:, 0, :]
 #                                     padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
 #                                     genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1).to(device)
 #                                     mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_emb = mol_cls_embedding, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
 #                                     # 把 learnable embedding 的 batch 纬 expand 作为 genome embedding 的替换
 #                                     mol_cls_embedding_text = co_cross_attn_text(mol_cls_emb = mol_cls_embedding, genome_embs = padded_text_embeddings, key_padding_mask = 1 - text_attn_masks)
 #                                     mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
 #                                     # logits = reg_head(features = mol_cls_embedding)
 #                                     FICI_input_1 = torch.cat((mol_cls_embedding[::2], mol_cls_embedding[1::2]), dim=1)
 #                                     FICI_input_2 = torch.cat((mol_cls_embedding[1::2], mol_cls_embedding[::2]), dim=1)
 #                                     logits_1 = reg_head(FICI_input_1)
 #                                     logits_2 = reg_head(FICI_input_2)
 #                                     logits = (logits_1 + logits_2) / 2
 #                                     loss = criterion(logits.squeeze(), labels.squeeze())
 #
 #                                 test_batch_losses.append(loss.item())
 #                                 t_test_batch_losses.append(loss.item())
 #
 #                                 test_batch_labels = labels.detach().cpu().flatten().tolist()
 #                                 test_batch_preds = logits.detach().cpu().flatten().tolist()
 #
 #                                 test_all_labels.extend(test_batch_labels)
 #                                 test_all_preds.extend(test_batch_preds)
 #                                 t_test_all_labels.extend(test_batch_labels)
 #                                 t_test_all_preds.extend(test_batch_preds)
 #                                 train_mean_as_test_predict.extend(np.full(logits.detach().cpu().flatten().shape, t_train_mean_MIC).tolist())
 #
 #                                 for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
 #                                     _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
 #                                     if _speceis_name is None:
 #                                         _speceis_name = strain_name_to_original_species_names_map[strain_name]
 #                                     if _speceis_name not in species_wise_test_preds_dict.keys():
 #                                         species_wise_test_preds_dict[_speceis_name] = [pred]
 #                                         species_wise_test_labels_dict[_speceis_name] = [label]
 #                                     else:
 #                                         species_wise_test_preds_dict[_speceis_name].append(pred)
 #                                         species_wise_test_labels_dict[_speceis_name].append(label)
 #
                        r2 = calculate_r2(test_all_labels, test_all_preds)
                        gt_r2 = calculate_r2(gt_test_all_labels, gt_test_all_preds)
                        t_r2 = calculate_r2(t_test_all_labels, t_test_all_preds)
                        r2_train_mean = calculate_r2(test_all_labels, train_mean_as_test_predict)
                        r2_MSE_spearman_pearson_species_wise = {}
                        for _speceis_name in species_wise_test_preds_dict.keys():
                            r2_species = calculate_r2(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])
                            MSE_specise = np.mean((np.array(species_wise_test_labels_dict[_speceis_name]) - np.array(species_wise_test_preds_dict[_speceis_name])) ** 2)
                            if len(species_wise_test_labels_dict[_speceis_name]) > 1:
                                spearman_species = spearmanr(species_wise_test_labels_dict[_speceis_name],
                                                             species_wise_test_preds_dict[_speceis_name])[0]
                                pearson_species = pearsonr(species_wise_test_labels_dict[_speceis_name],
                                                           species_wise_test_preds_dict[_speceis_name])[0]
                            else:
                                spearman_species = pearson_species = None
                            r2_MSE_spearman_pearson_species_wise[_speceis_name] = [r2_species, MSE_specise, spearman_species, pearson_species]
                        logger.info(f'\n Test species wise R2, MSE, Spearman, Pearson:')
                        for species_name, metrics in r2_MSE_spearman_pearson_species_wise.items():
                            formatted_metrics = ", ".join(f"{m:.4f}" if isinstance(m, float) else str(m) for m in metrics)
                            logger.info(f'    {species_name}:  {formatted_metrics}')
                        #                     print(f""" Ensemble {ensemble+1}/{num_ensembles} Epoch {epoch}/{num_epochs}
                        # Test Loss: {np.array(test_batch_losses).mean():.6f}, genome text Test Loss: {np.array(gt_test_batch_losses).mean():.6f}, text only Test Loss: {np.array(t_test_batch_losses).mean():.6f}
                        # Test R2: {r2:.6f}, genome text Test R2: {gt_r2:.6f}, text only Test R2: {t_r2:.6f}, Test train mean MIC R2: {r2_train_mean:.6f}""")
                        logger.info(f""" Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch}/{num_epochs}
 Test Loss: {np.array(test_batch_losses).mean():.6f}, genome text Test Loss: {np.array(gt_test_batch_losses).mean():.6f}, text only Test Loss: {np.array(t_test_batch_losses).mean():.6f}
 Test R2: {r2:.6f}, genome text Test R2: {gt_r2:.6f}, text only Test R2: {t_r2:.6f}, Test train mean MIC R2: {r2_train_mean:.6f}""")

                train_batch_losses = []
                train_all_labels = []
                train_all_preds = []
                gt_train_batch_losses = []
                gt_train_all_labels = []
                gt_train_all_preds = []
                t_train_batch_losses = []
                t_train_all_labels = []
                t_train_all_preds = []

                species_wise_train_labels_dict = {}
                species_wise_train_preds_dict = {}

                co_cross_attn_genome.train()
                co_cross_attn_text.train()
                reg_head.train()

                for gt_batch, t_batch in tqdm(itertools.zip_longest(gt_train_loader, t_train_loader, fillvalue=None), desc=f" Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs} | training", leave=False, total=max(len(gt_train_loader), len(t_train_loader))):
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

                        optimizer.zero_grad()

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
                            loss = criterion(logits.squeeze(), labels.squeeze())

                        # loss.backward()
                        # optimizer.step()

                        scaler.scale(loss).backward()
                        # 对模型参数的梯度进行裁剪，例如设置最大范数为 1.0
                        if epoch >= freeze_epochs:
                            # 将梯度 unscale 到正常范围
                            scaler.unscale_(optimizer)
                            # torch.nn.utils.clip_grad_norm_(mdlm_model.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(co_cross_attn_genome.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(co_cross_attn_text.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(reg_head.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        train_batch_losses.append(loss.item())
                        gt_train_batch_losses.append(loss.item())

                        train_batch_labels = labels.detach().cpu().flatten().tolist()
                        train_batch_preds = logits.detach().cpu().flatten().tolist()

                        train_all_labels.extend(train_batch_labels)
                        train_all_preds.extend(train_batch_preds)
                        gt_train_all_labels.extend(train_batch_labels)
                        gt_train_all_preds.extend(train_batch_preds)

                        for strain_name, label, pred in zip(strain_names, train_batch_labels, train_batch_preds):
                            _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
                            if _speceis_name is None:
                                _speceis_name = strain_name_to_original_species_names_map[strain_name]
                            if _speceis_name not in species_wise_train_preds_dict.keys():
                                species_wise_train_preds_dict[_speceis_name] = [pred]
                                species_wise_train_labels_dict[_speceis_name] = [label]
                            else:
                                species_wise_train_preds_dict[_speceis_name].append(pred)
                                species_wise_train_labels_dict[_speceis_name].append(label)

                    if t_batch is not None:
                        # input_ids = t_batch['input_ids'].to(device)
                        # attention_mask = t_batch['attention_mask'].to(device)
                        labels = t_batch['label'].to(device)
                        # padded_genome_embeddings = t_batch['padded_genome_embeddings']  # .to(torch.float)
                        # genome_attn_masks = t_batch['genome_attn_masks']
                        padded_text_embeddings = t_batch['padded_text_embeddings']  # .to(torch.float)
                        text_attn_masks = t_batch['text_attn_masks']
                        strain_names = t_batch['strain_names']
                        mol_cls_embedding = t_batch['mol_emb'].to(device)

                        optimizer.zero_grad()

                        with torch.amp.autocast('cuda', enabled=True):
                            # outputs = mdlm_model(input_ids=input_ids, attention_mask=attention_mask)
                            #
                            # mol_cls_embedding = outputs[:, 0, :]
                            padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                            genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1).to(device)
                            mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_emb = mol_cls_embedding, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
                            # 把 learnable embedding 的 batch 纬 expand 作为 genome embedding 的替换
                            # mol_cls_embedding_genome = learnable_embedding_weight.expand(mol_cls_embedding.shape[0], -1)
                            mol_cls_embedding_text = co_cross_attn_text(mol_cls_emb = mol_cls_embedding, genome_embs = padded_text_embeddings, key_padding_mask = 1 - text_attn_masks)
                            mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
                            # logits = reg_head(features = mol_cls_embedding)
                            FICI_input_1 = torch.cat((mol_cls_embedding[::2], mol_cls_embedding[1::2]), dim=1)
                            FICI_input_2 = torch.cat((mol_cls_embedding[1::2], mol_cls_embedding[::2]), dim=1)
                            logits_1 = reg_head(FICI_input_1)
                            logits_2 = reg_head(FICI_input_2)
                            logits = (logits_1 + logits_2) / 2
                            loss = criterion(logits.squeeze(), labels.squeeze())

                        # loss.backward()
                        # optimizer.step()

                        scaler.scale(loss).backward()
                        # 对模型参数的梯度进行裁剪，例如设置最大范数为 1.0
                        if epoch >= freeze_epochs:
                            # 将梯度 unscale 到正常范围
                            scaler.unscale_(optimizer)
                            # torch.nn.utils.clip_grad_norm_(mdlm_model.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_([learnable_embedding_weight], max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(co_cross_attn_genome.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(co_cross_attn_text.parameters(), max_norm=1.0)
                            torch.nn.utils.clip_grad_norm_(reg_head.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()

                        train_batch_losses.append(loss.item())
                        t_train_batch_losses.append(loss.item())

                        train_batch_labels = labels.detach().cpu().flatten().tolist()
                        train_batch_preds = logits.detach().cpu().flatten().tolist()

                        train_all_labels.extend(train_batch_labels)
                        train_all_preds.extend(train_batch_preds)
                        t_train_all_labels.extend(train_batch_labels)
                        t_train_all_preds.extend(train_batch_preds)

                        for strain_name, label, pred in zip(strain_names, train_batch_labels, train_batch_preds):
                            _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
                            if _speceis_name is None:
                                _speceis_name = strain_name_to_original_species_names_map[strain_name]
                            if _speceis_name not in species_wise_train_preds_dict.keys():
                                species_wise_train_preds_dict[_speceis_name] = [pred]
                                species_wise_train_labels_dict[_speceis_name] = [label]
                            else:
                                species_wise_train_preds_dict[_speceis_name].append(pred)
                                species_wise_train_labels_dict[_speceis_name].append(label)

                scheduler.step()
                # print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {np.array(batch_losses).mean()}")

                r2_train = calculate_r2(train_all_labels, train_all_preds)
                spearman_train = spearmanr(train_all_labels, train_all_preds)[0]
                pearson_train = pearsonr(train_all_labels, train_all_preds)[0]
                gt_r2_train = calculate_r2(gt_train_all_labels, gt_train_all_preds)
                gt_spearman_train = spearmanr(gt_train_all_labels, gt_train_all_preds)[0]
                gt_pearson_train = pearsonr(gt_train_all_labels, gt_train_all_preds)[0]
                t_r2_train = calculate_r2(t_train_all_labels, t_train_all_preds)
                t_spearman_train = spearmanr(t_train_all_labels, t_train_all_preds)[0]
                t_pearson_train = pearsonr(t_train_all_labels, t_train_all_preds)[0]

                r2_MSE_spearman_pearson_species_wise = {}
                for _speceis_name in species_wise_train_preds_dict.keys():
                    r2_species = calculate_r2(species_wise_train_labels_dict[_speceis_name], species_wise_train_preds_dict[_speceis_name])
                    MSE_specise = np.mean((np.array(species_wise_train_labels_dict[_speceis_name]) - np.array(species_wise_train_preds_dict[_speceis_name])) ** 2)
                    if len(species_wise_train_labels_dict[_speceis_name]) > 1:
                        spearman_species = spearmanr(species_wise_train_labels_dict[_speceis_name], species_wise_train_preds_dict[_speceis_name])[0]
                        pearson_species = pearsonr(species_wise_train_labels_dict[_speceis_name], species_wise_train_preds_dict[_speceis_name])[0]
                    else:
                        spearman_species = pearson_species = None
                    r2_MSE_spearman_pearson_species_wise[_speceis_name] = [r2_species, MSE_specise, spearman_species, pearson_species]

                logger.info(f'\n Train species wise R2, MSE, Spearman, Pearson:')
                for species_name, metrics in r2_MSE_spearman_pearson_species_wise.items():
                    formatted_metrics = ", ".join(f"{m:.4f}" if isinstance(m, float) else str(m) for m in metrics)
                    logger.info(f'    {species_name}:  {formatted_metrics}')

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

                    # for gt_batch, t_batch in tqdm(itertools.zip_longest(gt_test_loader, t_test_loader, fillvalue=None),
                    #                               desc=f" Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs} | evaluating",
                    #                               leave=False, total=max(len(gt_test_loader), len(t_test_loader))):
                    for gt_batch in tqdm(
                            gt_test_loader,
                            desc=f" Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs} | evaluating",
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
                                loss = criterion(logits.squeeze(), labels.squeeze())

                            test_batch_losses.append(loss.item())
                            gt_test_batch_losses.append(loss.item())

                            test_batch_labels = labels.detach().cpu().flatten().tolist()
                            test_batch_preds = logits.detach().cpu().flatten().tolist()

                            test_all_labels.extend(test_batch_labels)
                            test_all_preds.extend(test_batch_preds)
                            gt_test_all_labels.extend(test_batch_labels)
                            gt_test_all_preds.extend(test_batch_preds)

                            for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
                                _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
                                if _speceis_name is None:
                                    _speceis_name = strain_name_to_original_species_names_map[strain_name]
                                if _speceis_name not in species_wise_test_preds_dict.keys():
                                    species_wise_test_preds_dict[_speceis_name] = [pred]
                                    species_wise_test_labels_dict[_speceis_name] = [label]
                                else:
                                    species_wise_test_preds_dict[_speceis_name].append(pred)
                                    species_wise_test_labels_dict[_speceis_name].append(label)
                #
                #         if t_batch is not None:
                #             # input_ids = t_batch['input_ids'].to(device)
                #             # attention_mask = t_batch['attention_mask'].to(device)
                #             labels = t_batch['label'].to(device)
                #             # padded_genome_embeddings = gt_batch['padded_genome_embeddings']  # .to(torch.float)
                #             # genome_attn_masks = gt_batch['genome_attn_masks']
                #             padded_text_embeddings = t_batch['padded_text_embeddings']  # .to(torch.float)
                #             text_attn_masks = t_batch['text_attn_masks']
                #             strain_names = t_batch['strain_names']
                #             mol_cls_embedding = t_batch['mol_emb'].to(device)
                #
                #             with torch.amp.autocast('cuda', enabled=True):
                #                 # outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
                #                 #
                #                 # mol_cls_embedding = outputs.last_hidden_state[:, 0, :]
                #                 padded_genome_embeddings = learnable_embedding_weight[:, None, :].expand(mol_cls_embedding.shape[0], 1, -1)
                #                 genome_attn_masks = torch.from_numpy(np.array([1]))[None, :].expand(mol_cls_embedding.shape[0], -1).to(device)
                #                 mol_cls_embedding_genome = co_cross_attn_genome(mol_cls_emb = mol_cls_embedding, genome_embs = padded_genome_embeddings, key_padding_mask = 1 - genome_attn_masks)
                #                 # 把 learnable embedding 的 batch 纬 expand 作为 genome embedding 的替换
                #                 # mol_cls_embedding_genome = learnable_embedding_weight.expand(mol_cls_embedding.shape[0], -1)
                #                 mol_cls_embedding_text = co_cross_attn_text(mol_cls_emb = mol_cls_embedding, genome_embs = padded_text_embeddings, key_padding_mask = 1 - text_attn_masks)
                #                 mol_cls_embedding = torch.cat((mol_cls_embedding_genome.reshape(-1, 8192), mol_cls_embedding_text.reshape(-1, 4096)), dim=1)
                #                 # logits = reg_head(features = mol_cls_embedding)
                #                 FICI_input_1 = torch.cat((mol_cls_embedding[::2], mol_cls_embedding[1::2]), dim=1)
                #                 FICI_input_2 = torch.cat((mol_cls_embedding[1::2], mol_cls_embedding[::2]), dim=1)
                #                 logits_1 = reg_head(FICI_input_1)
                #                 logits_2 = reg_head(FICI_input_2)
                #                 logits = (logits_1 + logits_2) / 2
                #                 loss = criterion(logits.squeeze(), labels.squeeze())
                #
                #             test_batch_losses.append(loss.item())
                #             t_test_batch_losses.append(loss.item())
                #
                #             test_batch_labels = labels.detach().cpu().flatten().tolist()
                #             test_batch_preds = logits.detach().cpu().flatten().tolist()
                #
                #             test_all_labels.extend(test_batch_labels)
                #             test_all_preds.extend(test_batch_preds)
                #             t_test_all_labels.extend(test_batch_labels)
                #             t_test_all_preds.extend(test_batch_preds)
                #
                #             for strain_name, label, pred in zip(strain_names, test_batch_labels, test_batch_preds):
                #                 _speceis_name = ATCC_ID_to_species_name_map_dict.get(strain_name, None)
                #                 if _speceis_name is None:
                #                     _speceis_name = strain_name_to_original_species_names_map[strain_name]
                #                 if _speceis_name not in species_wise_test_preds_dict.keys():
                #                     species_wise_test_preds_dict[_speceis_name] = [pred]
                #                     species_wise_test_labels_dict[_speceis_name] = [label]
                #                 else:
                #                     species_wise_test_preds_dict[_speceis_name].append(pred)
                #                     species_wise_test_labels_dict[_speceis_name].append(label)
                #
                    print('\n Calculating metrics...')
                    r2_test = calculate_r2(test_all_labels, test_all_preds)
                    spearman_test = spearmanr(test_all_labels, test_all_preds)[0]
                    pearson_test = pearsonr(test_all_labels, test_all_preds)[0]
                    gt_r2_test = calculate_r2(gt_test_all_labels, gt_test_all_preds) if len(gt_test_all_labels) > 1 else -1000
                    gt_spearman_test = spearmanr(gt_test_all_labels, gt_test_all_preds)[0] if len(gt_test_all_labels) > 1 else -1000
                    gt_pearson_test = pearsonr(gt_test_all_labels, gt_test_all_preds)[0] if len(gt_test_all_labels) > 1 else -1000
                    t_r2_test = calculate_r2(t_test_all_labels, t_test_all_preds) if len(t_test_all_labels) > 1 else -1000
                    t_spearman_test = spearmanr(t_test_all_labels, t_test_all_preds)[0] if len(t_test_all_labels) > 1 else -1000
                    t_pearson_test = pearsonr(t_test_all_labels, t_test_all_preds)[0] if len(t_test_all_labels) > 1 else -1000

                    r2_MSE_spearman_pearson_species_wise = {}
                    for _speceis_name in species_wise_test_preds_dict.keys():
                        r2_species = calculate_r2(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])
                        MSE_specise = np.mean((np.array(species_wise_test_labels_dict[_speceis_name]) - np.array(species_wise_test_preds_dict[_speceis_name])) ** 2)
                        if len(species_wise_test_labels_dict[_speceis_name]) > 1:
                            spearman_species = spearmanr(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])[0]
                            pearson_species = pearsonr(species_wise_test_labels_dict[_speceis_name], species_wise_test_preds_dict[_speceis_name])[0]
                        else:
                            spearman_species = pearson_species = None
                        r2_MSE_spearman_pearson_species_wise[_speceis_name] = [r2_species, MSE_specise, spearman_species, pearson_species]

                    if r2_test > best_R2_test:
                        best_R2_test = r2_test
                        best_test_prdictions = test_all_preds

                        torch.save({
                            'R2': best_R2_test,
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'ChemBERTa_state_dict': mdlm_model.state_dict(),
                            're_head_state_dict': reg_head.state_dict(),
                            'co_cross_attn_genome': co_cross_attn_genome.state_dict(),
                            'co_cross_attn_text': co_cross_attn_text.state_dict(),
                            'learnable_embedding_weight': learnable_embedding_weight
                        }, model_save_dir / f'fold_{i}_ensemble_{ensemble}_best_test.ckpt')

                    if epoch==5:
                        torch.save({
                            'R2': best_R2_test,
                            'optimizer_state_dict': optimizer.state_dict(),
                            # 'ChemBERTa_state_dict': mdlm_model.state_dict(),
                            're_head_state_dict': reg_head.state_dict(),
                            'co_cross_attn_genome': co_cross_attn_genome.state_dict(),
                            'co_cross_attn_text': co_cross_attn_text.state_dict(),
                            'learnable_embedding_weight': learnable_embedding_weight
                        }, model_save_dir / f'fold_{i}_ensemble_{ensemble}_fixed_epoch.ckpt')

                    if spearman_test > best_spearman_test:
                        best_spearman_test = spearman_test

                    if pearson_test > best_pearson_test:
                        best_pearson_test = pearson_test

                    logger.info(f'\n Test species wise R2, MSE, Spearman, Pearson:')
                    for species_name, metrics in r2_MSE_spearman_pearson_species_wise.items():
                        formatted_metrics = ", ".join(f"{m:.4f}" if isinstance(m, float) else str(m) for m in metrics)
                        logger.info(f'    {species_name}:  {formatted_metrics}')

                    #                 print(f""" Ensemble {ensemble+1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs}
                    # Training Loss: {np.array(train_batch_losses).mean():.4f}, Test Loss: {np.array(test_batch_losses).mean():.4f}
                    #   Genome text Training Loss: {np.array(gt_train_batch_losses).mean():.4f}, Genome Text Test Loss: {np.array(gt_test_batch_losses).mean():.4f}
                    #   Text only Training Loss: {np.array(t_train_batch_losses).mean():.4f}, Text only Test Loss: {np.array(t_test_batch_losses).mean():.4f}
                    # Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}, Best Test R2: {best_R2_test:.4f}
                    #   Genome Text Train R2: {gt_r2_train:.4f}, Genome Text Test R2: {gt_r2_test:.4f}
                    #   Text only Train R2: {t_r2_train:.4f}, Text only Test R2: {t_r2_test:.4f}
                    # Train spearman:{spearman_train:.4f}, Test spearman:{spearman_test:.4f}, Best Test spearman:{best_spearman_test:.4f}
                    #   Genome Text Train spearman:{gt_spearman_train:.4f}, Genome Text Test spearman:{gt_spearman_test:.4f}
                    #   Text only Train spearman:{t_spearman_train:.4f}, Text only Test spearman:{t_spearman_test:.4f}
                    # Train pearson:{pearson_train:.4f}, Test pearson:{pearson_test:.4f}, Best Test pearson:{best_pearson_test:.4f}
                    #   Genome Text Train pearson:{gt_pearson_train:.4f}, Genome Text Test pearson:{gt_pearson_test:.4f}
                    #   Text onlyTrain pearson:{t_pearson_train:.4f}, Text only Test pearson:{t_pearson_test:.4f}""")
                logger.info(f"""\n Ensemble {ensemble + 1}/{num_ensembles} Epoch {epoch + 1}/{num_epochs}
    Training Loss: {np.array(train_batch_losses).mean():.4f}, Test Loss: {np.array(test_batch_losses).mean():.4f}
      Genome text Training Loss: {np.array(gt_train_batch_losses).mean():.4f}, Genome Text Test Loss: {np.array(gt_test_batch_losses).mean():.4f}
      Text only Training Loss: {np.array(t_train_batch_losses).mean():.4f}, Text only Test Loss: {np.array(t_test_batch_losses).mean():.4f}
    Train R2: {r2_train:.4f}, Test R2: {r2_test:.4f}, Best Test R2: {best_R2_test:.4f}
      Genome Text Train R2: {gt_r2_train:.4f}, Genome Text Test R2: {gt_r2_test:.4f}
      Text only Train R2: {t_r2_train:.4f}, Text only Test R2: {t_r2_test:.4f}
    Train spearman:{spearman_train:.4f}, Test spearman:{spearman_test:.4f}, Best Test spearman:{best_spearman_test:.4f}
      Genome Text Train spearman:{gt_spearman_train:.4f}, Genome Text Test spearman:{gt_spearman_test:.4f}
      Text only Train spearman:{t_spearman_train:.4f}, Text only Test spearman:{t_spearman_test:.4f}
    Train pearson:{pearson_train:.4f}, Test pearson:{pearson_test:.4f}, Best Test pearson:{best_pearson_test:.4f}
      Genome Text Train pearson:{gt_pearson_train:.4f}, Genome Text Test pearson:{gt_pearson_test:.4f}
      Text onlyTrain pearson:{t_pearson_train:.4f}, Text only Test pearson:{t_pearson_test:.4f}""")

                # if r2_test > best_R2_test:
                #     best_R2_test = r2_test
                #     torch.save({
                #         'R2': best_R2_test,
                #         'optimizer_state_dict': optimizer.state_dict(),
                #         # 'ChemBERTa_state_dict': mdlm_model.state_dict(),
                #         're_head_state_dict': reg_head.state_dict(),
                #         'co_cross_attn_genome': co_cross_attn_genome.state_dict(),
                #         'co_cross_attn_text': co_cross_attn_text.state_dict(),
                #         'learnable_embedding_weight': learnable_embedding_weight
                #     }, model_save_dir / f'fold_{i}_ensemble_{ensemble}.ckpt')

            if best_test_prdictions is not None:
                if best_R2_test > -10:
                    test_predictions_of_ensembles.append(best_test_prdictions)
        # print(f'\n len of ensembled test predictions: {len(test_predictions_of_ensembles)}')
        logger.info(f'\n len of ensembled test predictions: {len(test_predictions_of_ensembles)}')
        test_predictions_of_ensembles = np.array(test_predictions_of_ensembles)
        ensembled_predictions = np.mean(test_predictions_of_ensembles, axis=0)
        ensembled_R2 = calculate_r2(test_all_labels, ensembled_predictions)
        ensembled_spearman = spearmanr(test_all_labels, ensembled_predictions)[0]
        ensembled_pearson = pearsonr(test_all_labels, ensembled_predictions)[0]
        #
        # # print(f'\n Ensemble R2 of {args.test_group}: {ensembled_R2:.4f}')
        # # print(f' Ensemble spearman of {args.test_group}: {ensembled_spearman:.4f}')
        # # print(f' Ensemble pearson of {args.test_group}: {ensembled_pearson:.4f}')
        logger.info(f'\n Ensemble R2 of {args.test_group}: {ensembled_R2:.4f}')
        logger.info(f' Ensemble spearman of {args.test_group}: {ensembled_spearman:.4f}')
        logger.info(f' Ensemble pearson of {args.test_group}: {ensembled_pearson:.4f}')