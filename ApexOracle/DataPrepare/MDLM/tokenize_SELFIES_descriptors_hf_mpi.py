# import os
# os.environ["OPENBLAS_MAIN_FREE"] = "1"

import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import torch
import os
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors.MoleculeDescriptors import MolecularDescriptorCalculator
import selfies
from datasets import Dataset
import math

# 配置参数
MODEL_NAME = "ibm-research/materials.selfies-ted"
CSV_PATH = "/data2/tianang/projects/Synergy/DataPrepare/MDLM/Data/all_selfies.csv"
# CSV_PATH = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_selfies.csv"
source_name = CSV_PATH.split('.csv')[0].split('/')[-1].rsplit('_', 1)[0]
CHUNKSIZE = 1000
N_WORKERS = max(1, 4)

# 初始化全局组件（主进程）
descriptor_names = [name for name, _ in Descriptors.descList if name != "Ipc"]
n_descriptors = len(descriptor_names)
global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if global_tokenizer.pad_token is None:
    global_tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# 子进程初始化函数
def init_process():
    """初始化子进程全局组件"""
    global global_calculator
    global_calculator = MolecularDescriptorCalculator(descriptor_names)


def process_chunk(chunk):
    """处理数据块并计算描述符"""
    global global_calculator

    chunk_stats = {
        'unk_smiles': 0,
        'unk_tokens': 0,
        'invalid_selfies': 0,
        'invalid_mol': 0,
        'failed_descriptor': 0,
        'lengths': [],
        'id_w_unk': [],
        'raw_tokens': [],
        'descriptors': []
    }

    unk_token_id = global_tokenizer.unk_token_id

    for _, row in chunk.iterrows():
        try:
            # Tokenization
            encoding = global_tokenizer(
                row['SELFIES'].replace("][", "] [").strip(),
                padding=False,
                truncation=False,
                return_tensors="np",
                add_special_tokens=True
            )
            input_ids = encoding['input_ids'][0].astype(np.int16)
            seq_len = input_ids.size

            # 检查UNK
            unk_count = np.count_nonzero(input_ids == unk_token_id)
            if unk_count > 0:
                chunk_stats['unk_smiles'] += 1
                chunk_stats['unk_tokens'] += unk_count
                chunk_stats['id_w_unk'].append(row['ID'])
                continue

            # SELFIES转SMILES  TODO: descriptor removed
            try:
                smiles = selfies.decoder(row['SELFIES'].strip())
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    chunk_stats['invalid_mol'] += 1
                    continue
            except:
                chunk_stats['invalid_selfies'] += 1
                continue

            # 计算描述符（使用全局计算器）  TODO: descriptor removed
            try:
                descriptors = np.array(global_calculator.CalcDescriptors(mol), dtype=np.float32)
                descriptors = np.nan_to_num(descriptors, nan=0.0, posinf=0.0, neginf=0.0)
            except:
                chunk_stats['failed_descriptor'] += 1
                continue

            # 保存有效数据
            chunk_stats['raw_tokens'].append((row['ID'], input_ids))
            # chunk_stats['descriptors'].append(descriptors)  # TODO: descriptor removed
            chunk_stats['lengths'].append(seq_len)

        except Exception as e:
            print(f"Error processing {row['ID']}: {str(e)}")

    return chunk_stats


def merge_stats(results):
    """合并统计结果"""
    final_stats = {
        'unk_smiles': 0,
        'unk_tokens': 0,
        'invalid_selfies': 0,
        'invalid_mol': 0,
        'failed_descriptor': 0,
        'lengths': [],
        'id_w_unk': [],
        'raw_tokens': [],
        'descriptors': []
    }

    for res in results:
        for k in final_stats:
            if k in res:
                if isinstance(res[k], list):
                    final_stats[k].extend(res[k])
                else:
                    final_stats[k] += res[k]

    return final_stats


if __name__ == "__main__":

    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # 当前 MPI 进程编号
    size = comm.Get_size()  # 总的 MPI 进程数

    print(f"Rank {rank} of {size} started processing.")

    # 每个 MPI 进程扫描 CSV 文件的所有块，但只处理分配给它的块 (chunk_index % size == rank)
    local_chunks = []
    chunk_indices = []  # 记录该进程处理的块号（用于调试）
    for chunk_index, chunk in tqdm(enumerate(pd.read_csv(
            CSV_PATH,
            chunksize=CHUNKSIZE,
            dtype={'ID': 'string', 'SELFIES': 'string'}
    )), desc=f'Rank {rank} loading chunks'):
        if chunk_index % size == rank:
            local_chunks.append(chunk)
            chunk_indices.append(chunk_index)

        # TODO: 调试用
        if chunk_index == 4:
            break

    print(f"Rank {rank} received {len(local_chunks)} chunks (chunk indices: {chunk_indices}).")

    # 使用 multiprocessing 在当前 MPI 进程内部并行处理各数据块
    local_results = []
    if local_chunks:
        with Pool(processes=N_WORKERS) as pool:
            # 如果需要在每个子进程中初始化其他全局组件，可以启用 initializer=init_process
            for res in tqdm(pool.imap_unordered(process_chunk, local_chunks),
                            total=len(local_chunks),
                            desc=f"Rank {rank} processing"):
                local_results.append(res)
    else:
        print(f"Rank {rank} did not receive any chunks.")

    # 合并本 MPI 进程内的结果
    local_final_stats = merge_stats(local_results)

    # 通过 MPI gather 将所有进程的统计结果汇总到 rank 0
    # all_stats = comm.gather(local_final_stats, root=0)

    # with Pool(processes=N_WORKERS, initializer=init_process) as pool:  # TODO: descriptor removed
    # with Pool(processes=N_WORKERS) as pool:
    #     reader = pd.read_csv(
    #         CSV_PATH,
    #         chunksize=CHUNKSIZE,
    #         dtype={'ID': 'string', 'SELFIES': 'string'}
    #     )
    #
    #     results = []
    #     for i, result in enumerate(pool.imap_unordered(process_chunk, reader)):
    #         results.append(result)
    #         if i % 10 == 0:
    #             print(f"Processed {i * CHUNKSIZE:,} rows...")
    #
    # final_stats = merge_stats(results)

    # 统计信息输出
    print(f"Rank {rank} finished processing.")
    total_samples = len(local_final_stats["raw_tokens"]) + sum([
        local_final_stats['unk_smiles'],
        local_final_stats['invalid_selfies'],
        local_final_stats['invalid_mol'],
        local_final_stats['failed_descriptor']
    ])
    valid_samples = len(local_final_stats["raw_tokens"])

    print(f"\n总样本量: {total_samples:,}")
    print(f"有效样本比例: {valid_samples / total_samples:.2%}")
    print("无效样本分布:")
    print(f"  - 含UNK: {local_final_stats['unk_smiles']:,}")
    print(f"  - 无效SELFIES: {local_final_stats['invalid_selfies']:,}")
    print(f"  - 无效分子: {local_final_stats['invalid_mol']:,}")
    print(f"  - 描述符错误: {local_final_stats['failed_descriptor']:,}")

    safe_max_rows = 2147483647 // 1768
    print(f"单个 shard 安全最大样本数：{safe_max_rows}")

    # 创建最终数据集
    if valid_samples > 0:
        # 计算归一化参数
        # all_descriptors = np.vstack(final_stats['descriptors']) # TODO: descriptor removed
        # mean = np.mean(all_descriptors, axis=0)
        # std = np.std(all_descriptors, axis=0)
        # std[std == 0] = 1.0  # 处理零标准差

        # 构建数据集
        print('building list dataset')
        input_ids = [t[1] for t in local_final_stats['raw_tokens']]
        num_samples, seq_length = len(input_ids)

        if num_samples > safe_max_rows:
            num_shards = math.ceil(num_samples / safe_max_rows)
        else:
            # 例如，强制拆分为 10 个 shards（你可根据需要调整）
            num_shards = 1
        print(f"将数据集拆分为 {num_shards} 个 shard")
        # normalized_descriptors = [(d - mean) / std for d in final_stats['descriptors']]  # TODO: descriptor removed

        # 转换为HuggingFace Dataset格式
        shard_size = math.ceil(num_samples / num_shards)

        for i in range(num_shards):
            start_idx = i * shard_size
            end_idx = min((i + 1) * shard_size, num_samples)  # 最后一片可能不足 shard_size
            print(f"Shard {i}: 处理样本索引范围 [{start_idx}, {end_idx})")

            # 分割出对应的 numpy 数组块
            shard_input_ids = input_ids[start_idx:end_idx]
            # shard_attention_mask = attention_mask_np[start_idx:end_idx]

            # 由于原始数据较大，为避免 PyArrow 的内部类型转换问题，
            # 这里建议先将 numpy 数组转换成 Python 列表再创建 Dataset（虽然速度会稍慢）
            shard_dataset = Dataset.from_dict({
                "input_ids": shard_input_ids,
                # "attention_mask": shard_attention_mask
            })

            shard_path = f"/data1/fangping2/SELFIES_tokenized_vary_len_dataset/shard_{i}"
            print(f"Saving shard {i} 到 {shard_path}")
            shard_dataset.save_to_disk(shard_path)

        print("所有 shard 保存完成。")
        # print('converting to huggingface dataset')
        # dataset = Dataset.from_dict({
        #     'input_ids': input_ids,
        #     # 'descriptors': normalized_descriptors # TODO: descriptor removed
        # })

        # 保存数据集
        # print('saving dataset')
        # output_filename = f"dataset_{source_name}"
        # output_path = "/data1/tianang/projects/Synergy/DataPrepare/MDLM/Data/hf_selfies_token_vary_length"
        # output_path = '/data1/fangping2/SELFIES_tokenized_vary_len'
        # dataset.save_to_disk(output_path)

        # print(f"\n数据集已保存至: {output_path}")
        # print(f"样本数量: {len(dataset)}")
        # print(f"输入ID示例: {input_ids[0][:5]}... (长度: {len(input_ids[0])})")
        # print(f"描述符示例: {normalized_descriptors[0][:5]}...") # TODO: descriptor removed

    else:
        print("\n没有有效数据需要保存")

    # 绘制长度分布图
    # plt.figure(figsize=(12, 7))
    # plt.hist(final_stats['lengths'], bins=50, color='steelblue')
    # plt.xlabel('Sequence Length', fontsize=12)
    # plt.ylabel('Count', fontsize=12)
    # plt.title(f'Sequence Length Distribution (n={valid_samples:,})', fontsize=14)
    # plt.grid(axis='y', alpha=0.5)
    # plt.show()
    # plt.close()