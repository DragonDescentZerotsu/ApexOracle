from datasets import load_from_disk, concatenate_datasets, DatasetDict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os


def load_single_shard(shard_path):
    """独立的shard加载函数，不传递任何不可序列化对象"""
    try:
        return load_from_disk(shard_path)
    except Exception as e:
        print(f"Error loading {shard_path}: {str(e)}")
        return None

def load_all_shards_parallel(dir_path):
    # 获取分片路径
    shard_paths = [
        os.path.join(dir_path, name)
        for name in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, name))
    ]
    print(f"Found {len(shard_paths)} shards")

    # 配置并行参数
    max_workers = max(8, len(shard_paths))

    # 并行加载（使用异步回调更新进度条）
    with tqdm(total=len(shard_paths), desc='Loading shards') as pbar:
        with Pool(processes=max_workers) as pool:
            # 使用apply_async + 回调机制
            results = []
            for path in shard_paths:
                result = pool.apply_async(
                    load_single_shard,
                    args=(path,),
                    callback=lambda _: pbar.update(1)
                )
                results.append(result)

            # 获取所有结果
            loaded_shards = [r.get() for r in results if r.get() is not None]

    # 拼接数据集
    if loaded_shards:
        full_dataset = concatenate_datasets(loaded_shards)
        print(f"完整数据集加载完成，共 {len(full_dataset)} 个样本")
        print(full_dataset[3])
    else:
        print("未成功加载任何分片")

    return full_dataset

def split_train_test(large_ds, small_ds):
    small_seqs = set()

    # print(' Getting test tokens')
    for ex in tqdm(small_ds, desc='Getting test tokens', total=len(small_ds)):
        # attention_mask==1 的位置表示有效 token
        seq = tuple(
            token for token, m in zip(ex["input_ids"], ex["attention_mask"]) if m == 1
        )
        small_seqs.add(seq)

    print(f' len small_seqs: {len(small_seqs)}')

    # 3. 定义一个过滤函数，用来判断 large_ds 中的例子是否属于 test
    def _is_in_small(example):
        # 注意：large_ds 的 input_ids 已经是 variable-length，无需再 trim
        return tuple(example["input_ids"]) in small_seqs

    # 4. 过滤出 test 集和 train 集
    print(' Filtering test dataset')
    test_ds = large_ds.filter(_is_in_small)
    print(' Filtering train dataset')
    # train 时同时排除 small 和 长序列
    train_ds = large_ds.filter(lambda ex: (not _is_in_small(ex)) and len(ex["input_ids"]) <= 1024)

    # 5. 如果 large_ds 里还有别的字段（比如 attention_mask）想删掉，可以用 remove_columns
    # train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in ("input_ids", "descriptors")])
    # test_ds = test_ds.remove_columns([c for c in test_ds.column_names if c not in ("input_ids", "descriptors")])

    print(f' Len train dataset: {len(train_ds)}')
    print(f' Len test dataset: {len(test_ds)}')

    # 6. 打包成一个 DatasetDict，方便 downstream
    dataset_splits = DatasetDict({
        "train": train_ds,
        "test": test_ds,
    })

    return dataset_splits

    # print(dataset_splits)


if __name__ == "__main__":
    large_dataset_dir = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/selfies_pep_SM_cls_hf_db_shards_v2"
    # small_dataset_dir = '/data1/fangping/SMILES_data/fullSFs/chunk0'
    save_path = '/data1/fangping2/hf_pep_SM_cls_v2'

    large_ds = load_all_shards_parallel(large_dataset_dir)
    # print(' loading small test dataset')
    # small_ds = load_single_shard(small_dataset_dir)
    #
    # dataset_splits = split_train_test(large_ds, small_ds)
    # print(f' train example:{dataset_splits["train"][0]}')
    # print(f' test example:{dataset_splits["test"][0]}')

    print(f' saving to {save_path}')
    large_ds.save_to_disk(save_path)
    print(' done')
