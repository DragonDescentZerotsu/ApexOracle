import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from APEX_models import AMP_model
from utils import *
import pandas as pd
from fine_tune_on_DBAASP_SMILES import MultiTaskLoss, calculate_r2_per_task, R2Tracker
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import numpy as np
import wandb
import pickle
import torch.optim as optim

# ckpt_path = '/data2/tianang/projects/Synergy/compare_APEX/APEX_ckpt/APEX_2&256&2048&1e-05&0.0&0.1'
ckpt_path = '/data2/tianang/projects/Synergy/compare_APEX/APEX_ckpt/APEX_3&128&2048&1e-06&0.001&0.1'

model = torch.load(ckpt_path, weights_only=False)

wanted_weights = ['peptideEmb', 'rnn', 'layernorm', 'attn1', 'attn2', 'fc0']

partial_state = {k: v for k, v in model.state_dict().items() if k.split('.')[0] in wanted_weights}

torch.save(partial_state, '/data2/tianang/projects/Synergy/compare_APEX/APEX_ckpt/APEX_pretrained_encoder_state_dict_best.ckpt')

# print(partial_state)