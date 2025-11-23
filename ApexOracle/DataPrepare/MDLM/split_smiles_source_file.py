import pandas as pd
import numpy as np
from tqdm import tqdm


def split_smiles_by_source(input_file):
    # 读取数据为NumPy数组
    df = pd.read_csv(input_file)
    data = df.values  # 获取二维数组 [N rows x 2 cols]

    # 创建字典存储不同来源的数据
    source_dict = {}

    # 逐行处理数据
    for row in tqdm(data, desc="Splitting SMILES", unit=" lines"):
        original_id, smiles = row
        source_name = original_id.rsplit('_', 1)[0]
        # try:
            # 拆分ID获取来源名称
            # source_name = original_id.rsplit('_', 1)[0]
            # if len(parts) != 2:  # 跳过无效ID
            #     continue
            # source_name, _ = parts
        # except:
        #     continue

        # 存储数据（保持原始ID）
        if source_name not in source_dict:
            source_dict[source_name] = []
        source_dict[source_name].append([original_id, smiles])

    # 保存每个来源的数据
    for source_name, rows in source_dict.items():
        # 转换为DataFrame并保存
        pd.DataFrame(
            rows,
            columns=['ID', 'SMILES']
        ).to_csv(f"/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/{source_name}_smiles.csv", index=False)


if __name__ == "__main__":
    split_smiles_by_source("/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles.csv")