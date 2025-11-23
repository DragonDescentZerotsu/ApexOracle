import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from multiprocessing import Pool, cpu_count
import numpy as np
import os

# 配置参数
# MODEL_NAME = "DeepChem/ChemBERTa-77M-MTR"
MODEL_NAME = "ibm-research/materials.selfies-ted"
CSV_PATH = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_selfies.csv"
source_name = CSV_PATH.split('.csv')[0].split('/')[-1].rsplit('_', 1)[0]
CHUNKSIZE = 100000  # 每个分块处理10万行
N_WORKERS = max(1, cpu_count() - 25)  # 保留2个核心给系统

# 初始化全局tokenizer（各进程共享）
global_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)


def process_chunk(chunk):
    """处理单个数据块的函数"""
    chunk_stats = {
        'unk_smiles': 0,
        'unk_tokens': 0,
        'lengths': [],
        'id_w_unk': []
    }

    for _, row in chunk.iterrows():
        try:
            # 更高效的tokenization调用方式
            tokens = global_tokenizer(
                row['SELFIES'].replace("][", "] ["),
                return_tensors="np",  # 使用numpy数组比PyTorch更快
                add_special_tokens=False,
                truncation=False
            )['input_ids'][0]

            # 统计逻辑
            unk_mask = (tokens == global_tokenizer.unk_token_id)
            unk_count = np.sum(unk_mask)

            if unk_count > 0:
                chunk_stats['unk_smiles'] += 1
                chunk_stats['unk_tokens'] += unk_count
                chunk_stats['id_w_unk'].append(row['ID'])

            chunk_stats['lengths'].append(len(tokens))
        except Exception as e:
            print(f"Error processing {row['ID']}: {str(e)}")

    return chunk_stats


def merge_stats(results):
    """合并所有进程的结果"""
    final_stats = {
        'unk_smiles': 0,
        'unk_tokens': 0,
        'lengths': [],
        'id_w_unk': []
    }

    for res in results:
        final_stats['unk_smiles'] += res['unk_smiles']
        final_stats['unk_tokens'] += res['unk_tokens']
        final_stats['lengths'].extend(res['lengths'])
        final_stats['id_w_unk'].extend(res['id_w_unk'])

    return final_stats


if __name__ == "__main__":
    # 并行处理流程
    with Pool(processes=N_WORKERS) as pool:
        # 分块读取器
        reader = pd.read_csv(
            CSV_PATH,
            header=None,
            names=['ID', 'SELFIES'],
            chunksize=CHUNKSIZE,
            dtype={'ID': 'string', 'SELFIES': 'string'}
        )

        # 使用imap_unordered获得最快速度
        results = []
        for i, result in enumerate(pool.imap_unordered(process_chunk, reader)):
            results.append(result)
            if i % 10 == 0:  # 每处理10个chunk报告进度
                print(f"Processed {i * CHUNKSIZE:,} rows...")

    # 合并结果
    final_stats = merge_stats(results)

    # 输出结果
    print(f"\nModel name: {MODEL_NAME}")
    print(f"包含UNK的 SELFIES 数量: {final_stats['unk_smiles']}")
    print(f"UNK token总出现次数: {final_stats['unk_tokens']}")
    print(f"平均token长度: {np.mean(final_stats['lengths']):.2f}")
    print(f"最大token长度: {np.max(final_stats['lengths'])}")
    print(f"前10个包含UNK的ID: {final_stats['id_w_unk'][:10]}")

    # 保存详细结果
    # np.save('token_lengths.npy', final_stats['lengths'])
    # with open('ids_with_unk.txt', 'w') as f:
    #     f.write('\n'.join(final_stats['id_w_unk']))

    # 绘制直方图（使用对数坐标优化显示）
    plt.figure(figsize=(12, 7))
    counts, bins, _ = plt.hist(
        final_stats['lengths'],
        bins=100,
        color='steelblue',
        edgecolor='white',
        log=True  # 对数坐标处理长尾分布
    )

    plt.xlabel('Tokenized Length', fontsize=12)
    plt.ylabel('Log-Scaled Frequency', fontsize=12)
    plt.title(f'Token Length Distribution ({MODEL_NAME})', fontsize=14)
    plt.grid(axis='y', alpha=0.5)

    # 添加统计信息标注
    textstr = '\n'.join([
        f'Total Samples: {len(final_stats["lengths"]):,}',
        f'Mean Length: {np.mean(final_stats["lengths"]):.1f}',
        f'Max Length: {np.max(final_stats["lengths"]):,}',
        f'UNK-containing: {final_stats["unk_smiles"]:,}'
    ])
    plt.gca().text(
        0.95, 0.95, textstr,
        transform=plt.gca().transAxes,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.8)
    )

    plt.savefig(
        f'/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/token_length_distribution_{source_name}_selfies_{MODEL_NAME.split('/')[-1]}.pdf',
        bbox_inches='tight',
        dpi=300,
        metadata={'CreationDate': None}  # 避免PDF时间戳变化
    )
    plt.close()