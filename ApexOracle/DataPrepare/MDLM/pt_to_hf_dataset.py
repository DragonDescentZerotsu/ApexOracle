import torch
from datasets import Dataset, load_from_disk, concatenate_datasets
import numpy as np
import math

# 加载 .pt 文件
print('Loading .pt file...')
data = torch.load(
    "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/tokenized_clean_all_selfies_materials.selfies-ted.pt")
input_ids = data["input_ids"]
attention_mask = data["attention_masks"]

# 将张量转换为 NumPy 数组
input_ids_np = input_ids.numpy()
attention_mask_np = attention_mask.numpy()

# 查看数据集的尺寸
num_samples, seq_length = input_ids_np.shape
print(f"数据样本数：{num_samples}，序列长度：{seq_length}")

# 计算单个 shard 内安全的最大样本数：
# 需要保证 (安全分片内样本数 * 序列长度) < 2147483647
safe_max_rows = 2147483647 // seq_length
print(f"单个 shard 安全最大样本数：{safe_max_rows}")

# 如果总样本数超过安全行数，就按 safe_max_rows 分割，否则也可以按固定 shard 数分割
if num_samples > safe_max_rows:
    num_shards = math.ceil(num_samples / safe_max_rows)
else:
    # 例如，强制拆分为 10 个 shards（你可根据需要调整）
    num_shards = 10

print(f"将数据集拆分为 {num_shards} 个 shard")

# 保存各个 shard 的路径列表
shard_paths = []
# 计算每个 shard 大致的样本数（注意最后一个 shard 可能样本数不同）
shard_size = math.ceil(num_samples / num_shards)

for i in range(num_shards):
    start_idx = i * shard_size
    end_idx = min((i + 1) * shard_size, num_samples)  # 最后一片可能不足 shard_size
    print(f"Shard {i}: 处理样本索引范围 [{start_idx}, {end_idx})")

    # 分割出对应的 numpy 数组块
    shard_input_ids = input_ids_np[start_idx:end_idx]
    shard_attention_mask = attention_mask_np[start_idx:end_idx]

    # 由于原始数据较大，为避免 PyArrow 的内部类型转换问题，
    # 这里建议先将 numpy 数组转换成 Python 列表再创建 Dataset（虽然速度会稍慢）
    shard_dataset = Dataset.from_dict({
        "input_ids": shard_input_ids,
        "attention_mask": shard_attention_mask
    })

    shard_path = f"/data1/fangping2/SELFIES_tokenized_1024_arrow_dataset/shard_{i}"
    print(f"Saving shard {i} 到 {shard_path}")
    shard_dataset.save_to_disk(shard_path)
    shard_paths.append(shard_path)

print("所有 shard 保存完成。")

# --- 后续加载各个 shard 并拼接为完整数据集的示例代码 ---

# 分别加载已保存的 shards
# loaded_shards = [load_from_disk(path) for path in shard_paths]
#
# # 如有需要拼接为一个完整的数据集，可以使用 concatenate_datasets
# full_dataset = concatenate_datasets(loaded_shards)
# print("完整数据集加载完成，共 {} 个样本".format(len(full_dataset)))