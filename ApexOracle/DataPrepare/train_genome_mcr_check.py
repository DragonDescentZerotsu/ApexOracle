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

if __name__ == '__main__':
    current_directory = Path(__file__).parent

    genome_path = current_directory / 'Data' / 'Genome' / 'ATCC'
    genome_annotation_path = current_directory / 'Data' / 'Genome_annotation' / 'ATCC'

    genome_names = [f.name.split('.')[0] for f in genome_path.iterdir()]
    # genome_annotation_names = [f.name.split('.')[0] for f in genome_annotation_path.iterdir()]

    for annotation_file in tqdm(genome_annotation_path.iterdir(), total=576, desc="checking strains"):
        # 先确定确实是训练中使用过的 genome
        if annotation_file.name.split('.')[0] in genome_names:
            for seq_record in SeqIO.parse(annotation_file, "genbank"):
                for feature in seq_record.features:
                    products = feature.qualifiers.get('product')  # feature.qualifiers.get('gene / product')
                    if products is not None:
                        for product in products:
                            if 'mcr' in product or 'MCR' in product:
                                print(f'{annotation_file.name.split('.')[0]}: {product}')