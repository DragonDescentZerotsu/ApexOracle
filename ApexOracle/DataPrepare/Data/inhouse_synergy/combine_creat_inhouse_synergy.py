import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

current_dir = Path(__file__).parent

def init_worker(pd_dict):
    """
    在每个子进程中初始化全局变量 peptide_dict 和 total_peptides
    """
    global peptide_dict, total_peptides
    peptide_dict = pd_dict
    total_peptides = len(pd_dict)

def process_peptide(item):
    """
    接受 (pep_id_1, pep_seq_1) 生成所有 (pep_id_1, pep_id_2) 组合
    返回列表 of [pep_id_1, pep_id_2, 'NA', seq1, seq2, -1]
    """
    pep_id_1, pep_seq_1 = item
    results = []
    # pep_id_2 从 pep_id_1+1 一直到 total_peptides
    for pep_id_2 in range(int(pep_id_1) + 1, total_peptides + 1):
        pep_seq_2 = peptide_dict[str(pep_id_2)]
        if pep_seq_1 and pep_seq_2:
            results.append([pep_id_1, str(pep_id_2), 'NA', pep_seq_1, pep_seq_2, -1])
    return results

if __name__ == '__main__':
    # 1. 读取原始数据并构建字典
    peptide_list = pd.read_excel(
        current_dir / 'raw' / 'Master List Peptides Antimicrobial Activity (1).xlsx'
    )
    peptide_dict_local = {
        str(i + 1): seq
        for i, seq in enumerate(peptide_list['Sequence'].values)
    }

    column_names = [
        'DBAASP_id',
        'antibio_id_or_name',
        'strain_name',
        'AMP_smiles',
        'antibiotic_smiles',
        'FICI',
    ]

    # 2. 构造任务列表
    items = list(peptide_dict_local.items())

    # 3. 启动进程池：每个子进程通过 init_worker 拿到 peptide_dict_local
    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(peptide_dict_local,),
    ) as pool:
        all_results = []
        # imap 会逐个返回子任务结果，用 tqdm 包装显示总进度
        for chunk in tqdm(
            pool.imap(process_peptide, items),
            total=len(items),
            desc='Making Combinations',
        ):
            all_results.extend(chunk)

    # 4. 汇总、保存
    df = pd.DataFrame(all_results, columns=column_names)
    df.to_csv(
        current_dir / 'processed' / 'combine_create_inhouse_synergy_Evo_pep_seq.csv',
        index=False,
    )

    print('Done.')




# import pandas as pd
# from pathlib import Path
# import numpy as np
# from tqdm import tqdm
#
# current_dir = Path(__file__).parent
#
# peptide_list = pd.read_excel(current_dir / 'raw' / 'Master List Peptides Antimicrobial Activity (1).xlsx')
#
# # peptide_dict = {str(name).strip():str(seq).strip() for name, seq in zip(peptide_list['Peptide'].values, peptide_list['Sequence'].values)}
# peptide_dict = {}
# for i in range(len(peptide_list['Sequence'].values)):
#     peptide_dict[str(i+1)] = peptide_list['Sequence'].values[i]
#
# # FICI_raw_data_files = [file.name for file in (current_dir/'raw').iterdir() if file.name.startswith('ATCC')]
# #
# # strain_name_dict = {
# #     '19606': 'Acinetobacter baumannii ATCC 19606',
# #     '47085': 'Pseudomonas aeruginosa ATCC 47085'
# # }
# #
# # FICI_raw_data = []
# # strain_data = []
# # for name in FICI_raw_data_files:
# #     FICI_data = pd.read_csv(current_dir/'raw'/name).values
# #     FICI_raw_data.append(FICI_data)
# #     strain_name = name.split('_')[1]
# #     strain_data += [strain_name_dict[strain_name]] * len(FICI_data)
# #
# # FICI_raw_data = np.concatenate(FICI_raw_data, axis=0)
#
# column_names = ['DBAASP_id','antibio_id_or_name','strain_name','AMP_smiles','antibiotic_smiles','FICI']
#
# FICI_precessed_data = []
# for pep_id_1, pep_seq_1 in tqdm(peptide_dict.items(), desc='Making Combinations', total=len(peptide_dict)):
#     pep_id_2 = int(pep_id_1) + 1
#     while pep_id_2 <= len(peptide_dict):
#         pep_seq_2 = peptide_dict[str(pep_id_2)]
#
#         if pep_seq_1 is not None and pep_seq_2 is not None:
#             FICI_precessed_data.append([pep_id_1, str(pep_id_2), 'NA', pep_seq_1, pep_seq_2, -1])
#
#         pep_id_2 += 1
#
# df = pd.DataFrame(FICI_precessed_data, columns=column_names)
#
# df.to_csv(current_dir/'processed'/'combine_create_inhouse_synergy_Evo_pep_seq.csv', index=False)
#
# print(1)