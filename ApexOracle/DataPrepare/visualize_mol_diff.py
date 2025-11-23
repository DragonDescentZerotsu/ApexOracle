from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
from rdkit.Chem import rdDepictor
import re

def find_nonH_atoms_indices(smiles: str):
    """
    用最简单的思路：
      - 用正则 r'[A-Z][a-z]?|[a-z]' 匹配所有“元素-like”子串
      - 跳过匹配到的 'H' 或 'h'
      - 其余全部计为“非氢原子”
    返回一个列表，每个元素是 (start_index, ordinal)，其中：
      - start_index: 该“原子”在 SMILES 中出现的起始下标（0-based）
      - ordinal: 这是扫描到的第几个“非氢”原子（1-based）
    """

    pattern = re.compile(r'(Cl|Br|Si|Al|Mg|Na|Zn|(?<!H)H[ea-fl-rt-z])|[A-GI-Z]|[a-gi-z]')  # 大写字母 + 可选小写字母，或单个小写字母

    results = []
    nonH_count = 0

    # finditer 可以同时获得匹配内容(原子符号)和它在原串中的位置
    for match in pattern.finditer(smiles):
        atom_str = match.group(0)  # 匹配到的原子字符串
        start_pos = match.start()  # 在 SMILES 中的起始下标

        if atom_str.upper() == 'H':
            # 如果匹配到氢原子 'H' 或 'h'，跳过
            continue
        else:
            # 其余都视为非氢原子
            nonH_count += 1
            results.append((start_pos, nonH_count, match.group()))

    return results

def locate_difference_index(atom_tokenize_result_origin, smiles_marked, atom_tokenize_result_masked):
    """
    处理一条 smiles，获得其上不一样的地方
    :param atom_tokenize_result_origin: 原始未被 mark 的 smiles 的 tokenize 结果，格式为 (atom在string中的index, 这是第几个重原子, 原子符号)
    :param smiles_marked:
    :param atom_tokenize_result_masked:
    :return:
    """
    # 获得 marked 的 smiles 中哪些部分是不一样的
    pattern = r"\[[A-Za-z0-9@+\-\(\)=#]+:999\]"
    matches = re.finditer(pattern, smiles_marked)

    different_atom_counts = []
    different_atom_indices = []
    for match in matches:
        for atom_pos, atom_count, atom_str in atom_tokenize_result_masked:
            if match.start() <= atom_pos < match.end():
                different_atom_counts.append(atom_count)
                break

    for atom_pos, atom_count, atom_str in atom_tokenize_result_origin:
        if atom_count in different_atom_counts:
            different_atom_indices.append(atom_pos)

    return different_atom_indices

if __name__ == "__main__":
    molA = Chem.MolFromSmiles("CC(C)C[C@H](NC(=O)CCCCCCN)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O)C(C)C)[C@@H](C)O")
    molB = Chem.MolFromSmiles("CCCCCCC(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](CCCCN)C(=O)N[C@H](C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](CCCCN)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(N)=O)C(C)C)[C@@H](C)O")

    # params = rdFMCS.MCSParameters()
    # params.BondCompare = rdFMCS.BondCompare.CompareOrder
    # params.RingMatchesRingOnly = True
    # params.MatchValences = True
    print('matching two mols...')
    mcs_result = rdFMCS.FindMCS([molA, molB], atomCompare=rdFMCS.AtomCompare.CompareAnyHeavyAtom, bondCompare=rdFMCS.BondCompare.CompareAny, ringMatchesRingOnly=False, timeout=5)

    # 首先为两个分子生成2D坐标
    # AllChem.Compute2DCoords(molA)
    # AllChem.Compute2DCoords(molB)

    # 将B对齐到A上，利用最大公共子结构信息
    # 1. 从 MCS 得到 SMARTS，并转换成分子对象

    mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
    # 获取A、B中匹配公共子结构的原子索引
    matchA = molA.GetSubstructMatch(mcs_mol)
    matchB = molB.GetSubstructMatch(mcs_mol)
    print(f'matchA: {matchA}')
    print(f'matchB: {matchB}')
    highlight_atoms_A = [i for i in range(molA.GetNumAtoms()) if i not in matchA]
    highlight_atoms_B = [i for i in range(molB.GetNumAtoms()) if i not in matchB]

    # 找出molA、molB中所有原子索引
    allAtomsA = set(range(molA.GetNumAtoms()))
    allAtomsB = set(range(molB.GetNumAtoms()))

    molA_copy = Chem.Mol(molA)
    molB_copy = Chem.Mol(molB)

    diffAtomsA = set(range(molA_copy.GetNumAtoms())) - set(matchA)
    diffAtomsB = set(range(molB_copy.GetNumAtoms())) - set(matchB)
    # 公共部分原子
    # 3. 使用2D对齐函数，让B的子结构对齐到A
    # atomMap = [[b, a] for b, a in zip(matchB, matchA) if b < molB.GetNumAtoms() and a < molA.GetNumAtoms()]
    # rdDepictor.GenerateDepictionMatching2DStructure(molA, molB, atomMap=atomMap)

    # 用于在 SMILES 中标记不一样的
    for idx in diffAtomsA:
        molA_copy.GetAtomWithIdx(idx).SetAtomMapNum(999)

    for idx in diffAtomsB:
        molB_copy.GetAtomWithIdx(idx).SetAtomMapNum(999)

    print(f'A_similarity: {len(matchA) / len(allAtomsA)*100}%')
    print(f'B_similarity: {len(matchB) / len(allAtomsB)*100}%')

    # 设置成 canonical=False 可以防止因为 SetAtomMapNum=999 导致原子在SMILES中的顺序被改变
    smiA_marked = Chem.MolToSmiles(molA_copy, canonical=False)
    smiB_marked = Chem.MolToSmiles(molB_copy, canonical=False)

    print(f'smiA_marked: {smiA_marked}\nsmiA_original: {Chem.MolToSmiles(molA, canonical=False)}')
    print(f'smiB_marked: {smiB_marked}\nsmiB_original: {Chem.MolToSmiles(molB, canonical=False)}')

    pattern = r"\[[A-Za-z0-9@+\-\(\)=#]+:999\]"
    matches = re.finditer(pattern, smiB_marked)
    print([(match.start(), match.group(), match.end()) for match in matches])

    result_original = find_nonH_atoms_indices(Chem.MolToSmiles(molB, canonical=False))
    result_marked = find_nonH_atoms_indices(smiB_marked)

    print(f'original atoms: {result_original}')
    print(f'masked atoms: {result_marked}')

    B_different_indices = locate_difference_index(result_original, smiB_marked, result_marked)

    # 绘制并展示对齐后的结果
    img = Draw.MolsToGridImage([molA, molB], molsPerRow=2, subImgSize=(2000,2000), highlightAtomLists=[highlight_atoms_A, highlight_atoms_B])
    # 使用 matplotlib 展示图片
    plt.figure(figsize=(20, 10))
    plt.imshow(img)
    plt.axis("off")  # 去掉坐标轴
    plt.show()