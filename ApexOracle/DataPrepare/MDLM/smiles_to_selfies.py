import pandas as pd
from multiprocessing import Pool, cpu_count
import selfies as sf

# 配置参数
INPUT_CSV = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles.csv"  # 输入文件路径
OUTPUT_CSV = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_selfies.csv"  # 输出文件路径
ERROR_LOG = "/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/conversion_errors.csv"  # 错误日志路径
CHUNKSIZE = 100000  # 每个分块处理的行数
N_WORKERS = max(1, cpu_count() - 9)  # 使用N-1个CPU核心


def process_chunk(chunk):
    """处理数据块并返回(处理后的数据, 错误列表)"""
    processed = []
    errors = []

    for _, row in chunk.iterrows():
        try:
            selfies = sf.encoder(row['SMILES'])
            processed.append((row['ID'], selfies))
        except Exception as e:
            errors.append({
                'ID': row['ID'],
                'SMILES': row['SMILES'],
                'Error': str(e)
            })

    # 创建处理后的DataFrame
    result_df = pd.DataFrame(processed, columns=['ID', 'SELFIES'])
    return result_df, errors


if __name__ == "__main__":
    # 初始化输出文件（清空已有内容）
    open(OUTPUT_CSV, 'w').close()

    with Pool(N_WORKERS) as pool:
        # 创建分块读取器（假设CSV有标题行）
        reader = pd.read_csv(
            INPUT_CSV,
            chunksize=CHUNKSIZE,
            dtype={'ID': 'string', 'SMILES': 'string'}
        )

        all_errors = []
        first_header = True  # 控制标题写入

        # 使用按顺序处理的imap
        for i, (processed_chunk, errors) in enumerate(pool.imap(process_chunk, reader)):
            # 写入处理结果
            processed_chunk.to_csv(
                OUTPUT_CSV,
                mode='a',
                header=first_header,
                index=False
            )
            if first_header:
                first_header = False

            # 收集错误
            all_errors.extend(errors)

            # 进度报告
            print(f"Processed chunk {i} | Rows: {len(processed_chunk):,} | Errors: {len(errors)}")

        # 保存错误日志
        if all_errors:
            error_df = pd.DataFrame(all_errors)
            error_df.to_csv(ERROR_LOG, index=False)
            print(f"\nSaved {len(all_errors)} errors to {ERROR_LOG}")
        else:
            print("\nAll conversions succeeded!")