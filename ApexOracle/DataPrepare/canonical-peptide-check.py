from rdkit import Chem
from rdkit.Chem import AllChem

# 20种canonical氨基酸的简化SMILES（N端和C端截断版，方便子结构匹配）
canonical_aa_smiles = {
    'A': 'N[C@@H](C)C(=O)O',  # Alanine
    'R': 'N[C@@H](CCCNC(N)=N)C(=O)O',  # Arginine
    'N': 'N[C@@H](CC(=O)N)C(=O)O',  # Asparagine
    'D': 'N[C@@H](CC(=O))C(=O)O',  # Aspartic acid
    'C': 'N[C@@H](CS)C(=O)O',  # Cysteine
    'E': 'N[C@@H](CCC(=O))C(=O)O',  # Glutamic acid
    'Q': 'N[C@@H](CCC(=O)N)C(=O)O',  # Glutamine
    'G': 'NCC(=O)O',  # Glycine
    'H': 'N[C@@H](CC1=CN=CN1)C(=O)O',  # Histidine
    'I': 'N[C@@H](C(C)CC)C(=O)O',  # Isoleucine
    'L': 'N[C@@H](CC(C)C)C(=O)O',  # Leucine
    'K': 'N[C@@H](CCCCN)C(=O)O',  # Lysine
    'M': 'N[C@@H](CCSC)C(=O)O',  # Methionine
    'F': 'N[C@@H](CC1=CC=CC=C1)C(=O)O',  # Phenylalanine
    'P': 'N1CCC[C@H]1C(=O)O',  # Proline
    'S': 'N[C@@H](CO)C(=O)O',  # Serine
    'T': 'N[C@@H](C(O)C)C(=O)O',  # Threonine
    'W': 'N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)O',  # Tryptophan
    'Y': 'N[C@@H](CC1=CC=C(O)C=C1)C(=O)O',  # Tyrosine
    'V': 'N[C@@H](C(C)C)C(=O)O',  # Valine
    'A_': 'N[C@@H](C)C(=O)',  # Alanine
    'R_': 'N[C@@H](CCCNC(N)=N)C(=O)',  # Arginine
    'N_': 'N[C@@H](CC(=O)N)C(=O)',  # Asparagine
    'D_': 'N[C@@H](CC(=O))C(=O)',  # Aspartic acid
    'C_': 'N[C@@H](CS)C(=O)',  # Cysteine
    'E_': 'N[C@@H](CCC(=O))C(=O)',  # Glutamic acid
    'Q_': 'N[C@@H](CCC(=O)N)C(=O)',  # Glutamine
    'G_': 'NCC(=O)',  # Glycine
    'H_': 'N[C@@H](CC1=CN=CN1)C(=O)',  # Histidine
    'I_': 'N[C@@H](C(C)CC)C(=O)',  # Isoleucine
    'L_': 'N[C@@H](CC(C)C)C(=O)',  # Leucine
    'K_': 'N[C@@H](CCCCN)C(=O)',  # Lysine
    'M_': 'N[C@@H](CCSC)C(=O)',  # Methionine
    'F_': 'N[C@@H](CC1=CC=CC=C1)C(=O)',  # Phenylalanine
    'P_': 'N1CCC[C@H]1C(=O)',  # Proline
    'S_': 'N[C@@H](CO)C(=O)',  # Serine
    'T_': 'N[C@@H](C(O)C)C(=O)',  # Threonine
    'W_': 'N[C@@H](CC1=CNC2=CC=CC=C12)C(=O)',  # Tryptophan
    'Y_': 'N[C@@H](CC1=CC=C(O)C=C1)C(=O)',  # Tyrosine
    'V_': 'N[C@@H](C(C)C)C(=O)'  # Valine
}

canonical_aa_mols = {k: Chem.MolFromSmiles(v) for k, v in canonical_aa_smiles.items()}


def split_peptide_into_residues(mol):
    """
    利用肽键断开肽链，返回单个氨基酸片段的Mol对象列表
    """
    # 找肽键的定义：C(=O)-N，断开C-N键
    peptide_bond_idxs = []
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom()
        end_atom = bond.GetEndAtom()
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE:
            # 判断是否是肽键：C连接羰基氧，N连接氨基
            if begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'N':
                # 进一步判定C原子有一个双键氧
                oxy_found = False
                for nbr in begin_atom.GetNeighbors():
                    if nbr.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(begin_atom.GetIdx(),
                                                                          nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        oxy_found = True
                        break
                if oxy_found:
                    peptide_bond_idxs.append(bond.GetIdx())
            elif begin_atom.GetSymbol() == 'N' and end_atom.GetSymbol() == 'C':
                oxy_found = False
                for nbr in end_atom.GetNeighbors():
                    if nbr.GetSymbol() == 'O' and mol.GetBondBetweenAtoms(end_atom.GetIdx(),
                                                                          nbr.GetIdx()).GetBondType() == Chem.rdchem.BondType.DOUBLE:
                        oxy_found = True
                        break
                if oxy_found:
                    peptide_bond_idxs.append(bond.GetIdx())

    # 断开肽键
    if not peptide_bond_idxs:
        # 说明没断开，可能是单个残基
        return [mol]
    fragmented_mol = Chem.FragmentOnBonds(mol, peptide_bond_idxs, addDummies=False)
    frags = Chem.GetMolFrags(fragmented_mol, asMols=True, sanitizeFrags=True)
    return frags


def is_canonical_residue(mol):
    """
    判断一个残基Mol是否属于canonical氨基酸
    """
    for aa_mol in canonical_aa_mols.values():
        print(f'aa: {Chem.MolToSmiles(aa_mol)}')
        print(f'mol: {Chem.MolToSmiles(mol)}')
        # 这里用同构匹配判断结构相似
        if mol.HasSubstructMatch(aa_mol) and aa_mol.HasSubstructMatch(mol):
            return True
    return False


def check_peptide_all_canonical(peptide_smiles):
    mol = Chem.MolFromSmiles(peptide_smiles)
    if mol is None:
        raise ValueError("输入的SMILES无效")
    residues = split_peptide_into_residues(mol)
    for res in residues:
        if not is_canonical_residue(res):
            return False
    return True


# 示例用法：
peptide = 'N[C@@H](C)C(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O'  # Ala-Phe
print(check_peptide_all_canonical(peptide))  # True

peptide_noncanonical = 'N[C@@H](C)C(=O)N[C@@H](CCN=O)C(=O)O'  # 第二个残基不是canonical
print(check_peptide_all_canonical(peptide_noncanonical))  # False