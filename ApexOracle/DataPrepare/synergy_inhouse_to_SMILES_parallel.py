import pandas as pd
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from aa_seq_to_smiles import get_aa_smiles_dict, Peptide
from rdkit import Chem

# 1. 全局加载一次 aa_smiles_dict，并在子进程中共享
aa_smiles_dict_local = get_aa_smiles_dict('./Data/all_aa_smiles_new_handcrafted.csv')

def init_worker(smiles_dict):
    """
    这个初始化函数会在每个子进程启动时被调用，
    将 aa_smiles_dict_local 绑定到子进程的全局变量 aa_smiles_dict 上，
    避免在每次调用时重复传输大字典。
    """
    global aa_smiles_dict
    aa_smiles_dict = smiles_dict

def process_row(row):
    """
    对单行记录进行转换：
    - 跳过包含特殊字符的序列
    - 用 Peptide 类生成 RDKit Mol
    - 转 SMILES 并替换原来的两列
    返回处理后的行列表；若跳过则返回 None。
    """
    seq1, seq2 = row[3], row[4]
    # 跳过带括号等特殊符号的
    for ch in ('(', ')'):
        if ch in str(seq1) or ch in str(seq2):
            return None

    # 生成 Peptide 对象并转 SMILES
    pep1 = Peptide(seq1, aa_smiles_dict=aa_smiles_dict)
    pep2 = Peptide(seq2, aa_smiles_dict=aa_smiles_dict)

    smi1 = Chem.MolToSmiles(pep1.ncTerminus_modified_mols[0])
    smi2 = Chem.MolToSmiles(pep2.ncTerminus_modified_mols[0])

    # 就地修改
    row[3], row[4] = smi1, smi2
    return row

if __name__ == '__main__':
    # 2. 读入 CSV 并转为 Python list of lists
    df = pd.read_csv('./Data/inhouse_synergy/processed/combine_create_inhouse_synergy_Evo_pep_seq.csv')
    data = df.values.tolist()

    # 3. 并行处理并显示进度条
    with Pool(
        processes=cpu_count(),
        initializer=init_worker,
        initargs=(aa_smiles_dict_local,)
    ) as pool:
        processed = []
        for out in tqdm(
            pool.imap(process_row, data),
            total=len(data),
            desc='Seq → SMILES'
        ):
            if out is not None:
                processed.append(out)

    # 4. 汇总结果并写回 CSV
    df_new = pd.DataFrame(processed, columns=df.columns)

    print(' Saving file.')
    df_new.to_csv(
        './Data/inhouse_synergy/processed/combine_create_inhouse_synergy_Evo_smiles_seq.csv',
        index=False
    )

    print('Done.')