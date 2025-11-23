import os
import pandas as pd
import math
from pathlib import Path
from tqdm import tqdm

current_dir = Path(__file__).parent


def split_csv_pandas(input_path, output_dir, parts=120):
    # 1. 读取全部数据
    print('loading csv file')
    df = pd.read_csv(input_path)
    total_rows = len(df)
    # 2. 计算每个文件应包含的行数（最后一个文件可能稍多）
    chunk_size = math.ceil(total_rows / parts)

    os.makedirs(output_dir, exist_ok=True)

    # 3. 按块写入
    for i in tqdm(range(parts), desc='saving splits', unit=' parts'):
        start = i * chunk_size
        end = min(start + chunk_size, total_rows)
        sub_df = df.iloc[start:end]
        if sub_df.empty:
            break
        out_path = output_dir / f'part_{i + 1:03d}.csv'
        sub_df.to_csv(out_path, index=False)
        print(f'写入 {out_path}（行 {start}–{end - 1}）')


if __name__ == '__main__':
    INPUT_CSV = current_dir / 'Data' / 'all_selfies_pep_SM_cls_v2.csv'  # 大CSV 的路径
    OUTPUT_DIR = current_dir / 'Data' / 'selfies_pep_SM_cls_splits_v2'  # 保存小文件的目录
    split_csv_pandas(INPUT_CSV, OUTPUT_DIR, parts=110)