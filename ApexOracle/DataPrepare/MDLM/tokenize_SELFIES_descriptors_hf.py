import os
os.environ["OMP_NUM_THREADS"] = "1"

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
import argparse
from pathlib import Path

current_dir = Path(__file__).parent



parser = argparse.ArgumentParser(
    description=' which chunk to pick',  # 在参数帮助信息之前显示的文本
)
parser.add_argument(
    '-s', '--split',  # 可选参数
    type=str,
    help='Which split to compute'
)
parser.add_argument(
    '-c', '--cpu_cores',  # 可选参数
    type=int,
    default=cpu_count()-10,
    help='How many cores to use'
)
args = parser.parse_args()

# 配置参数
MODEL_NAME = "ibm-research/materials.selfies-ted"
CSV_PATH = current_dir/'Data'/'selfies_splits'/f'part_{args.split}.csv'
# CSV_PATH = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_selfies.csv"
# source_name = CSV_PATH.split('.csv')[0].split('/')[-1].rsplit('_', 1)[0]
CHUNKSIZE = 100
N_WORKERS = max(1, args.cpu_cores)  # cpu_count()

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
        'longer_than_1024':0,
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

            if len(input_ids) > 1024:
                chunk_stats['longer_than_1024'] += 1
                continue
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
            chunk_stats['descriptors'].append(descriptors)  # TODO: descriptor removed
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
        'longer_than_1024': 0,
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
    print(f' loading {CSV_PATH.name} ...')
    full_df = pd.read_csv(
        CSV_PATH,
        dtype={'ID': 'string', 'SELFIES': 'string'}
    )
    total_lines = len(full_df)

    chunks = [full_df.iloc[i: i + CHUNKSIZE] for i in range(0, total_lines, CHUNKSIZE)]

    pbar = tqdm(total=total_lines, desc=' Calculating descriptors', unit=' lines')
    with Pool(processes=N_WORKERS, initializer=init_process) as pool:  # TODO: descriptor removed
    # with Pool(processes=N_WORKERS) as pool:
    #     reader = pd.read_csv(
    #         CSV_PATH,
    #         chunksize=CHUNKSIZE,
    #         dtype={'ID': 'string', 'SELFIES': 'string'}
    #     )

        results = []
        for chunk, res in zip(chunks, pool.imap_unordered(process_chunk, chunks)):
            results.append(res)
            # 这里按实际 chunk 大小更新进度
            pbar.update(len(chunk))
            # if i % 10 == 0:
            #     print(f"Processed {i * CHUNKSIZE:,} rows...")
    pbar.close()

    final_stats = merge_stats(results)

    # 统计信息输出
    total_samples = len(final_stats["raw_tokens"]) + sum([
        final_stats['unk_smiles'],
        final_stats['invalid_selfies'],
        final_stats['invalid_mol'],
        final_stats['failed_descriptor'],
        final_stats['longer_than_1024']
    ])
    valid_samples = len(final_stats["raw_tokens"])

    print(f"\n总样本量: {total_samples:,}")
    print(f"有效样本比例: {valid_samples / total_samples:.2%}")
    print("无效样本分布:")
    print(f"  - 含UNK: {final_stats['unk_smiles']:,}")
    print(f"  - 无效SELFIES: {final_stats['invalid_selfies']:,}")
    print(f"  - 无效分子: {final_stats['invalid_mol']:,}")
    print(f"  - 描述符错误: {final_stats['failed_descriptor']:,}")
    print(f"  - 长于1024: {final_stats['longer_than_1024']:,}")

    # 创建最终数据集
    if valid_samples > 0:
        # 计算归一化参数
        all_descriptors = np.vstack(final_stats['descriptors']) # TODO: descriptor removed
        # mean = np.mean(all_descriptors, axis=0)
        # std = np.std(all_descriptors, axis=0)
        # std[std == 0] = 1.0  # 处理零标准差

        # 构建数据集
        print('building list dataset')
        input_ids = [t[1] for t in final_stats['raw_tokens']]
        normalized_descriptors = all_descriptors # [(d - mean) / std for d in final_stats['descriptors']]  # TODO: descriptor removed

        # 转换为HuggingFace Dataset格式
        print('converting to huggingface dataset')
        dataset = Dataset.from_dict({
            'input_ids': input_ids,
            'descriptors': normalized_descriptors # TODO: descriptor removed
        })

        # 保存数据集
        print('saving dataset')
        # output_filename = f"dataset_{source_name}"
        output_path = current_dir/'Data'/'selfies_hf_db_shards'/f'shard_{args.split}'
        # output_path = '/data1/fangping2/SELFIES_tokenized_vary_len'
        dataset.save_to_disk(output_path)

        print(f"\n数据集已保存至: {output_path}")
        print(f"样本数量: {len(dataset)}")
        print(f"输入ID示例: {input_ids[0][:5]}... (长度: {len(input_ids[0])})")
        print(f"描述符示例: {normalized_descriptors[0][:5]}...") # TODO: descriptor removed

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