"""
最后保存一个两两比较的文件，格式为：
DBAASP_id1, DBAASP_id2, smiles1, smiles2, different_places_1, different_places_2, similarity_to_MCS_1, similarity_to_MCS_2
"""

import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from rdkit.Chem import rdDepictor
import re
from pathlib import Path
from visualize_mol_diff import find_nonH_atoms_indices, locate_difference_index
from multiprocessing import Pool, cpu_count

def comp_similarity(mol_A, mol_to_compare, max_time_for_matching):
    """
    计算两个分子之间的相似性并返回两个分子分别和 MCS (Max Common Structure?) 的相似度
    :param mol_A:
    :param smi:
    :return:
    """
    # params = rdFMCS.MCSParameters()
    # params.BondCompare = rdFMCS.BondCompare.CompareOrder
    # params.RingMatchesRingOnly = True
    # params.MatchValences = True
    # params.timeout = max_time_for_matching
    # print('matching two mols...')
    mcs_result = rdFMCS.FindMCS([mol_A, mol_to_compare], timeout=max_time_for_matching, bondCompare=rdFMCS.BondCompare.CompareOrder, ringMatchesRingOnly=True)# params)
    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    # 获取A、B中匹配公共子结构的原子索引
    matchA = mol_A.GetSubstructMatch(mcs_mol)
    matchB = mol_to_compare.GetSubstructMatch(mcs_mol)

    allAtomsA = set(range(mol_A.GetNumAtoms()))
    allAtomsB = set(range(mol_to_compare.GetNumAtoms()))

    # molA_copy = Chem.Mol(mol_A)
    # molB_copy = Chem.Mol(mol_to_compare)

    diffAtomsA = allAtomsA - set(matchA)
    diffAtomsB = allAtomsB - set(matchB)

    molA_copy = Chem.Mol(mol_A)
    molB_copy = Chem.Mol(mol_to_compare)

    for idx in diffAtomsA:
        molA_copy.GetAtomWithIdx(idx).SetAtomMapNum(999)

    for idx in diffAtomsB:
        molB_copy.GetAtomWithIdx(idx).SetAtomMapNum(999)

    A_similarity = len(matchA) / len(allAtomsA)
    B_similarity = len(matchB) / len(allAtomsB)

    smiA_marked = Chem.MolToSmiles(molA_copy, canonical=False)
    smiB_marked = Chem.MolToSmiles(molB_copy, canonical=False)

    highlight_atoms_A = [i for i in range(mol_A.GetNumAtoms()) if i not in matchA]
    highlight_atoms_B = [i for i in range(molB_copy.GetNumAtoms()) if i not in matchB]

    return A_similarity, B_similarity, smiA_marked, smiB_marked, highlight_atoms_A, highlight_atoms_B

# def locate_difference_index(smiles):
#     """
#     locate 到不一样的 [CH:999] 之类的地方之后，从第一个不一样的地方开始标记，
#     :param smiles:
#     :return:
#     """
#     pattern = r"\[.*?:999\]"
#     matches = re.findall(pattern, smiles)


current_folder = Path(__file__).parent

df = pd.read_csv(current_folder / 'Data' / 'DBAASP_id_SMILES_merged.csv')
id_smiles_list = list(zip(df['DBAASP_id'], df['SMILES']))

similarity_threshold = 0.65
max_time_for_matching = 5  # seconds
window_length = 30

similar_pairs = []

for i, (id, smi) in tqdm(enumerate(id_smiles_list), desc="Outer Loop"):
    # 比较 i 的前后20行来看有没有很像的 smiles
    if i == 5:
        break
    elif i==0:
        continue

    lower_limit = i+1
    if lower_limit == len(id_smiles_list):
        break
    upper_limit = i+window_length if i+window_length <len(id_smiles_list) else len(id_smiles_list)-1
    # 生成要比较的行号
    line_idxs_to_compare = list(range(lower_limit, upper_limit+1))
    # 去掉其中的 i
    # line_idxs_to_compare.remove(i)
    # 把第 i 个的 smiles 转换成 mol 对象方便比较
    mol_A = Chem.MolFromSmiles(smi)
    # 逐个比较前后20个 smiles
    for line_idx in tqdm(line_idxs_to_compare, desc="Inner Loop", leave=False, total=len(line_idxs_to_compare)-window_length):
        smi_to_compare = id_smiles_list[line_idx][1]
        mol_to_compare = Chem.MolFromSmiles(smi_to_compare)
        A_similarity, B_similarity, smiA_marked, smiB_marked, highlight_atoms_A, highlight_atoms_B = comp_similarity(mol_A, mol_to_compare, max_time_for_matching)

        # 相似性高于一定范围才继续处理
        if min(A_similarity, B_similarity) > similarity_threshold:
            smiA_original = Chem.MolToSmiles(mol_A, canonical=False)
            smiB_original = Chem.MolToSmiles(mol_to_compare, canonical=False)
            A_original_tokens = find_nonH_atoms_indices(smiA_original)
            B_original_tokens = find_nonH_atoms_indices(smiB_original)
            A_marked_tokens = find_nonH_atoms_indices(smiA_marked)
            B_marked_tokens = find_nonH_atoms_indices(smiB_marked)
            A_original_diff_indices = locate_difference_index(A_original_tokens, smiA_marked, A_marked_tokens)
            B_original_diff_indices = locate_difference_index(B_original_tokens, smiB_marked, B_marked_tokens)

            similar_pairs.append([id, id_smiles_list[line_idx][0], smiA_original, smiB_original, A_original_diff_indices, B_original_diff_indices, A_similarity, B_similarity])

            img = Draw.MolsToGridImage([mol_A, mol_to_compare], molsPerRow=2, subImgSize=(2000, 2000),
                                       highlightAtomLists=[highlight_atoms_A, highlight_atoms_B])
            # 使用 matplotlib 展示图片
            plt.figure(figsize=(20, 10))
            plt.imshow(img)
            plt.axis("off")  # 去掉坐标轴
            plt.show()

# 保存数据
df_save = pd.DataFrame(similar_pairs)
df_save.to_csv(current_folder / 'Data' /'DBAASP_id_SMILES_compare.csv', index=False)