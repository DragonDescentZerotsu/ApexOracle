from datasets import Dataset, load_from_disk, concatenate_datasets
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import os

root_dir = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/selfies_hf_db_shards"

# 获取所有一级子文件夹的名字
shard_paths = [os.path.join(os.path.join(root_dir, name)) for name in os.listdir(root_dir)
              if os.path.isdir(os.path.join(root_dir, name))]
print(shard_paths)

# 分别加载已保存的 shards
# print('loading shards')
loaded_shards = []
for shard_path in tqdm(shard_paths, desc='loading shards'):
    loaded_shards.append(load_from_disk(shard_path))
# loaded_shards = [load_from_disk(path) for path in shard_paths]

# 如有需要拼接为一个完整的数据集，可以使用 concatenate_datasets
full_dataset = concatenate_datasets(loaded_shards)
print("完整数据集加载完成，共 {} 个样本".format(len(full_dataset)))

print(full_dataset[0])