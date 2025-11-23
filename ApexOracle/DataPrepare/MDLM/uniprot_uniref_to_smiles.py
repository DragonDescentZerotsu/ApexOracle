import sys
from pathlib import Path

# 获取当前脚本的绝对路径
current_file = Path(__file__).resolve()

# 计算项目根目录（Synergy目录）
# 路径结构：/.../Synergy/DataPrepare/MDLM/uniprot_uniref_to_smiles.py
project_root = current_file.parent.parent.parent  # 向上三级到Synergy目录

# 添加项目根目录到Python路径
sys.path.insert(0, str(project_root))
from tqdm import tqdm
from DataPrepare.aa_seq_to_smiles import *
from rdkit import Chem
import json
import multiprocessing
from functools import partial
import os

current_dir = Path(__file__).parent

# --- 文件配置 ---
input_peptide_file = current_dir/'Data'/'unique_peptide_sequences.txt'
output_smiles_file = current_dir/'Data'/"unique_peptides_as_smiles.txt"
aa_smiles_dict_path = current_dir.parent/'Data'/'all_aa_smiles_new_handcrafted.csv'

# 标准氨基酸集合
canonical_aas = {'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                'a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l',
                'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y'}

def init_pool(aa_smiles_path):
    """初始化每个进程的aa_smiles_dict"""
    global global_aa_smiles_dict
    global_aa_smiles_dict = get_aa_smiles_dict(aa_smiles_path)

def process_peptide(raw_peptide):
    """处理单个肽序列"""
    try:
        pep_obj = Peptide(raw_peptide, aa_smiles_dict=global_aa_smiles_dict)
        return Chem.MolToSmiles(pep_obj.ncTerminus_modified_mols[0])
    except Exception as e:
        return None

def contains_non_canonical(peptide):
    """检测非标准氨基酸"""
    return any(aa not in canonical_aas for aa in peptide)

def main():
    print(f"正在从 '{input_peptide_file}' 读取肽序列...")

    # 预读取并过滤无效序列
    valid_peptides = []
    invalid_count = 0
    with open(input_peptide_file, 'r') as f:
        for line_num, line in enumerate(tqdm(f, desc="预读取和过滤"), 1):
            raw_peptide = line.strip()
            if not raw_peptide:
                continue
            if contains_non_canonical(raw_peptide):
                print(f"行号 {line_num}: 发现非标准氨基酸 - {raw_peptide}")
                invalid_count += 1
                continue
            valid_peptides.append(raw_peptide)

    total = len(valid_peptides)
    print(f"总有效肽序列: {total}, 无效序列: {invalid_count}")

    # 设置并行参数
    num_workers = os.cpu_count() - 9  # 留9个核心给系统
    chunk_size = max(100, len(valid_peptides) // (num_workers * 10))  # 动态分块大小

    print(f"\n启动 {num_workers} 个进程进行SMILES转换...")
    with multiprocessing.Pool(
        processes=num_workers,
        initializer=init_pool,
        initargs=(aa_smiles_dict_path,)
    ) as pool:
        results = []
        with tqdm(total=total, desc="SMILES转换进度") as pbar:
            for result in pool.imap(process_peptide, valid_peptides, chunksize=chunk_size):
                results.append(result)
                pbar.update()

    # 写入结果
    success_count = 0
    with open(output_smiles_file, 'w') as outfile:
        for smiles in tqdm(results, desc="写入结果"):
            if smiles:
                outfile.write(smiles + '\n')
                success_count += 1

    print(f"\n处理完成。成功转换: {success_count}/{total}")
    print(f"转换成功率: {success_count/total:.2%}")

if __name__ == "__main__":
    main()