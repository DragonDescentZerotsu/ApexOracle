import time

from google import genai
from pydantic import BaseModel
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

# import hydra
# from hydra import compose, initialize
# import models
from collections import OrderedDict
# import noise_schedule

import torch.nn.functional as F
import ast
from Bio import Phylo, SeqIO


class gene_resistance(BaseModel):
    reason: str
    conclusion: bool

if __name__ == "__main__":

    current_directory = Path(__file__).parent
    api_key_list = [
        'AIzaSyAqqOfLLgXFS3oI0rwuA8nJeuE9vdj20QM',  # lingchunsusan
                    'AIzaSyB1iHDp0zweUSPT3_x3LvqATwpHr0ZtFVo',  # yiming zhang
                    'AIzaSyDNQ2B4wTSUlJWAUTZ_Vt513iKkUTPzdiA'  # jyzhang gmail
                    ]
    api_key_index = 0
    # quota_no_count = 0
    client = genai.Client(api_key=api_key_list[api_key_index])

    strain_ID = 'BAA_3170'  # 这个名字记得注意用下划线

    genome_annotation_path = current_directory / 'Data' / 'Genome_annotation' / 'ATCC'
    Gemini_resistant_gene_folder = current_directory / 'Data' / 'Genome_annotation' / 'plausible_resistant_gene_by_Gemini'

    found_flag = False
    for f in genome_annotation_path.iterdir():
        if strain_ID in f.name:
            found_flag = True
            break

    if not found_flag:
        print("Genome annotation file not found")
        exit(1)

    gene_or_products_list = []

    contig_lengths = [0]
    current_contig_id = 0
    for seq_record in SeqIO.parse(f, "genbank"):
        contig_id = seq_record.id.split('.')[-1]
        contig_length = len(seq_record.seq)
        if contig_id != current_contig_id:
            current_contig_id = contig_id
            contig_lengths.append(contig_length)
        for feature in seq_record.features:
            if feature.type == "CDS":
                gene = feature.qualifiers.get('gene')
                products = feature.qualifiers.get('product')
                if gene is not None:
                    prompt_input = gene[0]
                else:
                    if products is not None:
                        product = products[0]
                        if 'hypothetical' in product:
                            continue
                        prompt_input = product
                    else:
                        continue

                # 注意这里 start 和 end 都已经被修改过了，是在整个 genome 上的定位，而不是某一个 contig 上的
                start = int(feature.location.start) + int(np.array(contig_lengths)[:-1].sum())
                end = int(feature.location.end) + int(np.array(contig_lengths)[:-1].sum())
                gene_or_products_list.append({'prompt_input': prompt_input,
                                              'start': start,
                                              'end': end})

    Gemini_resistant_gene_strain_file_name = f.name.replace('.gbk', '.json')
    Gemini_resistant_gene_strain_file_path = Gemini_resistant_gene_folder / Gemini_resistant_gene_strain_file_name

    # 1. 如果文件不存在，先初始化一个空的 JSON 对象
    if not Gemini_resistant_gene_strain_file_path.exists():
        with open(Gemini_resistant_gene_strain_file_path, 'w', encoding='utf-8') as f:
            json.dump({'genome_process_progress':0, 'resistant_genes':[]}, f, ensure_ascii=False, indent=4)

    # 2. 读取已有数据
    with open(Gemini_resistant_gene_strain_file_path, 'r', encoding='utf-8') as f:
        processed_resistant_genes = json.load(f)

    for gene_or_product in tqdm(gene_or_products_list, desc="finding resistant genes ..."):
        # json文件中有记录上次处理到哪里了
        if gene_or_product['start'] > processed_resistant_genes['genome_process_progress']:

            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=f"I am offering you a gene or a product that I have no idea about. Could you explain if the gene or the product is responsible for conferring colistin resistance in a bacterial strain? If it dose, answer True, if not, answer False. The gene/product offered is : {gene_or_product['prompt_input']}. please ground all your results on real publications and search results.",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": gene_resistance,
                    },
                )
                # Use the response as a JSON string.
                # print(response.text)

                # Use instantiated objects.
                gene_resistance_: gene_resistance = response.parsed
                processed_resistant_genes['genome_process_progress'] = gene_or_product['end']

                # if gene_resistance_.conclusion:
                gene_or_product['reason'] = gene_resistance_.reason
                gene_or_product['conclusion'] = gene_resistance_.conclusion
                processed_resistant_genes['resistant_genes'].append(gene_or_product)
                time.sleep(1)
                # print(gene_resistance_.conclusion)
            except Exception as e:
                # quota_no_count += 1
                print(f'old api key index:{api_key_index}')
                api_key_index += 1
                api_key_index = api_key_index % len(api_key_list)
                print(f'new api key index:{api_key_index}')
                # if quota_no_count == 10:
                #     print(e)
                #     print('no more available quota')
                #     break
                client = genai.Client(api_key=api_key_list[api_key_index])
                try:
                    response = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"I am offering you a gene or a product that I have no idea about. Could you explain if the gene or the product is responsible for conferring colistin resistance in a bacterial strain? If it dose, answer True, if not, answer False. The gene/product offered is : {gene_or_product['prompt_input']}. please ground all your results on real publications and search results.",
                        config={
                            "response_mime_type": "application/json",
                            "response_schema": gene_resistance,
                        },
                    )
                    # Use the response as a JSON string.
                    # print(response.text)

                    # Use instantiated objects.
                    gene_resistance_: gene_resistance = response.parsed
                    processed_resistant_genes['genome_process_progress'] = gene_or_product['end']

                    # if gene_resistance_.conclusion:
                    gene_or_product['reason'] = gene_resistance_.reason
                    gene_or_product['conclusion'] = gene_resistance_.conclusion
                    processed_resistant_genes['resistant_genes'].append(gene_or_product)
                    time.sleep(1)
                # 再尝试一个新的api-key，如果还是报错那就是真不行了
                except Exception as e:
                    print(e)
                    break

    # 4. 写回文件
    with open(Gemini_resistant_gene_strain_file_path, 'w', encoding='utf-8') as f:
        json.dump(processed_resistant_genes, f, ensure_ascii=False, indent=4)