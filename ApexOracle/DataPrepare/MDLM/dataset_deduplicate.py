# from datasets import load_from_disk, Dataset, DatasetDict
# from multiprocessing import Pool, cpu_count
from tqdm import tqdm
# import os
#
# def _dedupe_shard(args):
#     """
#     每个子进程负责自己那一份 shard 的去重：
#       - dataset_path: 数据集保存目录
#       - num_shards:   一共分几份
#       - shard_idx:    本进程负责第几份
#     返回：本 shard 内去重后的样本列表
#     """
#     dataset_path, num_shards, shard_idx = args
#     ds = load_from_disk(dataset_path)
#     # 只取自己负责的那一份
#     ds_shard = ds.shard(num_shards=num_shards, index=shard_idx)
#     seen = set()
#     unique_rows = []
#     for ex in ds_shard:
#         key = tuple(ex["input_ids"])
#         if key not in seen:
#             seen.add(key)
#             unique_rows.append(ex)
#     return unique_rows
#
# def dedupe_split_parallel(dataset_path: str, num_workers: int = None) -> Dataset:
#     """
#     对单个 split 并行去重：
#       - dataset_path: 本 split 的磁盘路径（save_to_disk 的那个目录）
#       - num_workers:  并行进程数 (默认 min(cpu_count(), 8))
#     返回：去重后的 Dataset
#     """
#     # 决定并行份数
#     num_workers = num_workers or min(cpu_count(), 8)
#     # 为每个进程准备参数 (path, total_shards, shard_idx)
#     args = [(dataset_path, num_workers, i) for i in range(num_workers)]
#
#     final_seen = set()
#     final_rows = []
#
#     # 多进程遍历各个 shard
#     with Pool(processes=num_workers) as pool:
#         for shard_rows in tqdm(
#             pool.imap_unordered(_dedupe_shard, args),
#             total=num_workers,
#             desc=f"Dedupe {os.path.basename(dataset_path)}"
#         ):
#             # 收到一个 shard 的去重结果，就合并到全局，同时做一次“跨 shard 去重”
#             for ex in shard_rows:
#                 key = tuple(ex["input_ids"])
#                 if key not in final_seen:
#                     final_seen.add(key)
#                     final_rows.append(ex)
#
#     # 从列表重建 Dataset
#     return Dataset.from_list(final_rows)
#
#
# if __name__ == "__main__":
#     # 1. 指定已经保存好的路径
#     train_path = "/data1/fangping2/hf_descriptor_1024_train_test/train"
#     test_path  = "/data1/fangping2/hf_descriptor_1024_train_test/test"
#
#     # 2. 并行去重
#     dedup_train = dedupe_split_parallel(train_path, num_workers=32)
#     dedup_test  = dedupe_split_parallel(test_path,  num_workers=32)
#
#     print(f' length of deduplicated train set: {len(dedup_train)}')
#     print(f' length of deduplicated test set: {len(dedup_test)}')
#
#     # 3. 打包成 DatasetDict
#     dedup_splits = DatasetDict({
#         "train": dedup_train,
#         "test":  dedup_test,
#     })
#
#     print(' saving deduplicated dataset')
#     dedup_splits.save_to_disk('/data1/fangping2/hf_descriptor_1024_train_test_deduplicated')
#
#     # 4. （可选）把去重后的结果再保存
#     # dedup_splits["train"].save_to_disk("/data/your_dir/train_dedup")
#     # dedup_splits["test"].save_to_disk( "/data/your_dir/test_dedup")
#
#     print("去重完成：")
#     print("  train example:", dedup_splits["train"][0])
#     print("  test example: ", dedup_splits["test"][0])

from datasets import load_from_disk, DatasetDict, Dataset

# 1. 载入已经分好的 train/test DatasetDict
dataset_splits = load_from_disk("/data1/fangping2/hf_descriptor_1024_train_test")  # 目录下有 train/ 和 test/ 子文件夹

dedup_splits = {}

# 2. 针对 train 和 test 各自去重
for split in ["train", "test"]:
    ds = dataset_splits[split]
    seen = set()
    unique_rows = []
    for ex in tqdm(ds, desc=f' deduplicating {split}', total=len(ds)):
        key = tuple(ex["input_ids"])
        if key not in seen:
            seen.add(key)
            unique_rows.append(ex)
    dedup_splits[split] = Dataset.from_list(unique_rows)
    print(f"{split}: 原始 {len(ds)} → 去重后 {len(unique_rows)}")

# 3. 重建一个新的 DatasetDict
dedup_dataset = DatasetDict(dedup_splits)

print(f' length of deduplicated train set: {len(dedup_dataset['train'])}')
print(f' length of deduplicated test set: {len(dedup_dataset['test'])}')

print("  train example:", dedup_dataset["train"][0])
print("  test example: ", dedup_dataset["test"][0])

# 4. （可选）保存去重后的结果
dedup_dataset.save_to_disk("/data1/fangping2/hf_descriptor_1024_train_test_deduplicated")
# print("去重并保存完成：", dedup_dataset)