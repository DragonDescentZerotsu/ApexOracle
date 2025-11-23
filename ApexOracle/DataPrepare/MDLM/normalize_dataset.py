from datasets import load_from_disk, DatasetDict
import numpy as np
from tqdm import tqdm

# 1. 加载已经保存到本地的 dataset（假设路径为 "./my_dataset"）
dataset: DatasetDict = load_from_disk("/data1/fangping2/hf_descriptor_1024_train_test_filtered")

# 2. 从 train 和 test 中提取所有 descriptors，并拼成一个大数组
train_ds = dataset["train"]  # shape: (N_train, 209)
test_ds = dataset["test"]  # shape: (N_train, 209)

# 2) 初始化统计变量
dim = 209
count = 0
mean = np.zeros(dim, dtype=np.float64)
M2 = np.zeros(dim, dtype=np.float64)

# 3) 迭代更新：对每个样本的 descriptors 执行 Welford 算法
for example in tqdm(train_ds, desc=" Computing mean and std of train"):
    desc = np.asarray(example["descriptors"], dtype=np.float64)  # shape (209,)
    if len(desc) != len(mean):
        print("Error in computing mean and std of descriptors")
        print(example['input_ids'])
        print(example['descriptors'])
        continue
    count += 1
    delta = desc - mean
    mean += delta / count
    delta2 = desc - mean
    M2 += delta * delta2

for example in tqdm(test_ds, desc=" Computing mean and std of test"):
    desc = np.asarray(example["descriptors"], dtype=np.float64)  # shape (209,)
    if len(desc) != len(mean):
        print("Error in computing mean and std of descriptors")
        print(example['input_ids'])
        print(example['descriptors'])
        continue
    count += 1
    delta = desc - mean
    mean += delta / count
    delta2 = desc - mean
    M2 += delta * delta2

# 4) 计算最终方差和标准差
if count < 2:
    var = np.zeros(dim, dtype=np.float64)
else:
    var = M2 / (count - 1)
std = np.sqrt(var)

# 5) 防止某些维度标准差为 0
std[std == 0] = 1.0

print(f'mean: {mean}')
print(f'std: {std}')
print(' saving ')
np.save('/data1/fangping2/descriptors_mean.npy', mean)
np.save('/data1/fangping2/descriptors_std.npy', std)