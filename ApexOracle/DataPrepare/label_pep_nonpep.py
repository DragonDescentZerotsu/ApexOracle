# import pandas as pd
# from smiles_to_peptide import smiles_to_pepseq
# from tqdm import tqdm
#
# print('loading data...')
# df = pd.read_csv('/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles.csv')
#
# original_columns = df.columns.tolist()
# new_columns = original_columns + ['label']
#
# data = df.values
#
# labeled_data = []
# for id, smiles in tqdm(data, desc='judging peptides'):
#     _, pep_seq = smiles_to_pepseq(smiles)
#     if pep_seq is None or 'X' in pep_seq:
#         label = 0
#     else:
#         label = 1
#
#     labeled_data.append([id, smiles, label])
#
# df_labeled = pd.DataFrame(labeled_data, columns=new_columns)
#
# print(f'saving')
# df_labeled.to_csv('/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles_pep_SM_cls_v2.csv', index=False)

import pandas as pd
from smiles_to_peptide import smiles_to_pepseq
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def process_row(row):
    """
    对单行数据进行标注：
    输入 row = [id, smiles]
    返回 [id, smiles, label]
    """
    idx, smiles = row
    _, pep_seq = smiles_to_pepseq(smiles)
    if pep_seq is None or 'X' in pep_seq:
        label = 0
    else:
        label = 1
    return [idx, smiles, label]


if __name__ == '__main__':
    print('loading data...')
    df = pd.read_csv('/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles.csv')

    # 保留原有列名，并在最后添加 'label'
    original_columns = df.columns.tolist()
    new_columns = original_columns + ['label']

    # 提取需要并行处理的数据
    # 假设原 CSV 的第一列是 id，第二列是 smiles，如有不同，请替换列名
    data = df[['ID', 'SMILES']].values.tolist()

    # 建立进程池，进程数默认使用 CPU 核心数
    with Pool(processes=cpu_count()-8) as pool:
        # imap 会保持输出顺序一致，配合 tqdm 显示进度
        results = list(tqdm(pool.imap(process_row, data),
                            total=len(data),
                            desc='judging peptides'))

    # 组装成 DataFrame 并保存
    df_labeled = pd.DataFrame(results, columns=new_columns)

    print('saving...')
    df_labeled.to_csv('/data1/tianang/Projects/Synergy/DataPrepare/MDLM/Data/all_smiles_pep_SM_cls_v2.csv',
                      index=False)
    print('done.')