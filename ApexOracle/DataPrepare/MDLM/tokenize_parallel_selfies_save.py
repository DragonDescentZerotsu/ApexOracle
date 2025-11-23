import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import torch
import os
import numpy as np
from tqdm import tqdm

# 配置参数
MODEL_NAME = "ibm-research/materials.selfies-ted"
CSV_PATH = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_selfies.csv"
source_name = CSV_PATH.split('.csv')[0].split('/')[-1].rsplit('_', 1)[0]
CHUNKSIZE = 100000
N_WORKERS = max(1, cpu_count() - 9)
MAX_LENGTH = 1024

# 初始化tokenizer并确保pad_token存在
global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if global_tokenizer.pad_token is None:
    global_tokenizer.add_special_tokens({'pad_token': '[PAD]'})


def process_chunk(chunk):
    """完全基于numpy的处理流程"""
    chunk_stats = {
        'unk_smiles': 0,
        'unk_tokens': 0,
        'long_sequences': 0,
        'lengths': [],
        'id_w_unk': [],
        'raw_tokens': []
    }

    # 预分配内存
    max_tokens = MAX_LENGTH + 2  # 考虑特殊token的缓冲
    temp_buffer = np.empty(max_tokens, dtype=np.int16)
    unk_token_id = global_tokenizer.unk_token_id

    for _, row in chunk.iterrows():
        try:
            # 直接编码到numpy数组
            encoding = global_tokenizer(
                row['SELFIES'].replace("][", "] ["),
                padding=False,
                truncation=False,
                return_tensors="np",  # 直接获取numpy数组
                add_special_tokens=True
            )

            # 直接处理numpy数组
            input_ids = encoding['input_ids'][0].astype(np.int16)  # 零拷贝类型转换
            seq_len = input_ids.size

            if seq_len > MAX_LENGTH:
                chunk_stats['long_sequences'] += 1
                chunk_stats['lengths'].append(seq_len)
                continue

            # 使用numpy向量化操作统计UNK
            unk_count = np.count_nonzero(input_ids == unk_token_id)
            if unk_count > 0:
                chunk_stats['unk_smiles'] += 1
                chunk_stats['unk_tokens'] += unk_count
                chunk_stats['id_w_unk'].append(row['ID'])
            else:
                # 直接存储原始数组的视图（零拷贝）
                chunk_stats['raw_tokens'].append((row['ID'], input_ids))
                chunk_stats['lengths'].append(seq_len)

        except Exception as e:
            print(f"Error processing {row['ID']}: {str(e)}")

    return chunk_stats


def merge_stats(results):
    """合并所有进程的结果"""
    final_stats = {
        'unk_smiles': 0,
        'unk_tokens': 0,
        'long_sequences': 0,
        'lengths': [],
        'id_w_unk': [],
        'raw_tokens': []
    }

    for res in results:
        final_stats['unk_smiles'] += res['unk_smiles']
        final_stats['unk_tokens'] += res['unk_tokens']
        final_stats['long_sequences'] += res['long_sequences']
        final_stats['lengths'].extend(res['lengths'])
        final_stats['id_w_unk'].extend(res['id_w_unk'])
        final_stats['raw_tokens'].extend(res['raw_tokens'])

    return final_stats


def create_padded_tensors(token_lists):
    """使用numpy预分配内存并创建tensors"""
    pad_token_id = global_tokenizer.pad_token_id

    # 预分配numpy数组
    batch_size = len(token_lists)
    input_ids_np = np.full((batch_size, MAX_LENGTH), pad_token_id, dtype=np.int16)
    attention_masks_np = np.zeros((batch_size, MAX_LENGTH), dtype=np.bool_)

    # 并行填充
    for i, (_, tokens) in tqdm(enumerate(token_lists), total=batch_size, desc='Padding'):
        seq_len = tokens.size
        input_ids_np[i, :seq_len] = tokens
        attention_masks_np[i, :seq_len] = True

    # 零拷贝转换为PyTorch tensor
    return (
        torch.from_numpy(input_ids_np),
        torch.from_numpy(attention_masks_np)
    )


if __name__ == "__main__":
    with Pool(processes=N_WORKERS) as pool:
        reader = pd.read_csv(
            CSV_PATH,
            chunksize=CHUNKSIZE,
            dtype={'ID': 'string', 'SELFIES': 'string'}
        )

        results = []
        for i, result in enumerate(pool.imap_unordered(process_chunk, reader)):
            results.append(result)
            if i % 10 == 0:
                print(f"Processed {i * CHUNKSIZE:,} rows...")

    final_stats = merge_stats(results)

    # 新增统计信息输出
    total_samples = len(final_stats["lengths"]) + final_stats['long_sequences']
    valid_samples = len(final_stats["raw_tokens"])
    print(f"\n总样本量: {total_samples:,}")
    print(f"有效样本比例: {valid_samples / total_samples:.2%}")
    print(f"无效样本分布:")
    print(f"  - 含UNK: {final_stats['unk_smiles']:,}")
    print(f"  - 过长: {final_stats['long_sequences']:,}")
    print(f"有效样本长度统计:")
    print(f"  平均: {np.mean(final_stats['lengths']):.1f}")
    print(f"  最大: {np.max(final_stats['lengths'])}")

    # 创建并保存tensors
    if valid_samples > 0:
        input_ids, attention_masks = create_padded_tensors(final_stats['raw_tokens'])
        ids = [item[0] for item in final_stats['raw_tokens']]

        output_filename = f"tokenized_clean_{source_name}_selfies_{MODEL_NAME.split('/')[-1]}.pt"
        output_path = os.path.join('/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/', output_filename)

        torch.save({
            # 'ids': ids,
            'input_ids': input_ids,
            'attention_mask': attention_masks
        }, output_path)

        # 验证保存结果
        print(f"\n保存数据验证:")
        print(f"Tensor维度: {input_ids.shape}")
        print(f"示例样本:")
        print(f"  ID: {ids[0]}")
        print(f"  Tokens: {input_ids[0][:10]}...{input_ids[0][-10:]}")
        print(f"  Mask: {attention_masks[0][:10]}...{attention_masks[0][-10:]}")
        print(f"数据已保存至: {output_path}")
    else:
        print("\n没有有效数据需要保存")

    # 绘图代码（添加长度分布分析）
    plt.figure(figsize=(12, 7))
    plt.hist(final_stats['lengths'], bins=np.arange(0, MAX_LENGTH + 50, 50), color='steelblue')
    plt.axvline(x=MAX_LENGTH, color='red', linestyle='--', label='Max Allowed Length')
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(f'Valid Sequence Length Distribution (n={valid_samples:,})', fontsize=14)
    plt.legend()
    plt.grid(axis='y', alpha=0.5)
    plt.savefig(f'length_distribution_{source_name}.png', bbox_inches='tight')
    plt.close()