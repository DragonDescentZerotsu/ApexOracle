import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path

import diffusion
import hydra
from hydra import compose, initialize
import models
from collections import OrderedDict
import noise_schedule

import torch.nn.functional as F

DIT_ckpt_path = '/data2/tianang/projects/mdlm/checkpoints/last.ckpt'
lightning_ckpt = torch.load(DIT_ckpt_path, map_location='cpu')

print(0)

