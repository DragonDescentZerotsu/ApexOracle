"""
把 APEX 训练使用的 inhouse data 转换成 SMILES，顺便处理 inf 到 512，-1000 到 -1
"""

from aa_seq_to_smiles import *
from rdkit import Chem
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
# 使用从 OPSIN 获得的 SMILES
smiles = "N[C@H](C(=O)OCC=C)CCC(=O)O"
aa_seq = 'VTCDILSVEAKGVKLNDAACAAHCLFRGRSGGYCNGKRVCVCR'
# 创建分子对象
aa = AAs(smiles)
print(aa.N_terminal_atom.GetIdx(), aa.C_terminal_atom.GetIdx())

# for atom in aa.mol.GetAtoms():
#     if atom.HasProp('N_terminal'):
#         print(atom.GetProp('N_terminal'))
#
# aa.add_dummy_atoms()
# print(Chem.MolToSmiles(aa.rwmol))

# 获得aa_smiles_dict
aa_smiles_dict = get_aa_smiles_dict('./Data/all_aa_smiles_new_handcrafted.csv')

df = pd.read_csv('./Data/APEX 1.1 Data.csv')
colume_names = list(df.columns)[:-1]
colume_names[0] = 'SMILES'

inhouse_data = df.values

inhouse_data[:, 1:-1][inhouse_data[:, 1:-1]<-500] = -1
inhouse_data[:, 1:-1][inhouse_data[:, 1:-1]==np.inf] = 512

inhouse_train = []
inhouse_test = []
for line in tqdm(inhouse_data):
    processed_line = []
    pep_obj = Peptide(line[0], aa_smiles_dict=aa_smiles_dict)
    pep_smiles = Chem.MolToSmiles(pep_obj.ncTerminus_modified_mols[0])
    processed_line.append(pep_smiles)
    processed_line.extend(line[1:-1])
    if line[-1] == 'CV':
        inhouse_train.append(processed_line)
    else:
        inhouse_test.append(processed_line)

print(f' train set length: {len(inhouse_train)}')
print(f' test set length: {len(inhouse_test)}')

df_train = pd.DataFrame(inhouse_train, columns=colume_names)
df_test = pd.DataFrame(inhouse_test, columns=colume_names)

df_train.to_csv('./Data/APEX_train_SMILES.csv', index=False)
df_test.to_csv('./Data/APEX_test_SMILES.csv', index=False)


# peptide = Peptide(aa_seq, aa_smiles_dict=aa_smiles_dict)
# print(Chem.MolToSmiles(peptide.main_chain_linked_mol[0]))

# 读取 DBAASP 可处理数据的代码，这部分应该变成一个函数
# with open('./Data/processable_data_wo_cid_SMILES_new.json', 'r', encoding='utf-8') as file:
#     processable_data_wo_cid_SMILES = json.load(file)

# df = pd.read_csv("./Data/terminal_modifications/c_terminal_smiles_from_PubChem_handcrafted.csv")
# c_terminus_name_smiles = dict(zip(df['name'], df['SMILES']))
#
# df = pd.read_csv("./Data/terminal_modifications/n_terminal_smiles_from_PubChem_handcrafted.csv")
# n_terminus_name_smiles = dict(zip(df['name'], df['SMILES']))

# 测试单独 peptide 的代码
# for AMP in processable_data_wo_cid_SMILES:
#     if AMP['id'] in [5602]:
#         cTerminus = AMP.get('cTerminus')
#         cTerminus_name = None
#         if cTerminus and cTerminus.get('name') is not None:
#             cTerminus_name = cTerminus.get('name')
#         nTerminus = AMP.get('nTerminus')
#         nTerminus_name = None
#         if nTerminus and nTerminus.get('name') is not None:
#             nTerminus_name = nTerminus.get('name')
#         AMP_peptide = Peptide(AMP['sequence'], aa_smiles_dict, AMP['id'], AMP['intrachainBonds'], AMP['interchainBonds'], AMP['unusualAminoAcids'], cTerminus_name, nTerminus_name, c_terminus_name_smiles)

# 测试单独 peptide 的代码
# for AMP in processable_data_wo_cid_SMILES:
#     if AMP['id'] in [15374]:
#         cTerminus = AMP.get('cTerminus')
#         cTerminus_name = None
#         if cTerminus and cTerminus.get('name') is not None:
#             cTerminus_name = cTerminus.get('name')
#         nTerminus = AMP.get('nTerminus')
#         nTerminus_name = None
#         if nTerminus and nTerminus.get('name') is not None:
#             nTerminus_name = nTerminus.get('name')
#             if nTerminus_name == '3MeACl':
#                 print(AMP['id'])
#         AMP_peptide = Peptide(AMP['sequence'], aa_smiles_dict, AMP['id'], AMP['intrachainBonds'], AMP['interchainBonds'], AMP['unusualAminoAcids'], cTerminus_name, nTerminus_name, c_terminus_name_smiles, n_terminus_name_smiles)
#
# intra_bond_linked_id_smiles = []
#
# for AMP in tqdm(processable_data_wo_cid_SMILES, desc='Processing  AMPs'):
#     if AMP['complexity']['name'] == 'Monomer':
#         cTerminus = AMP.get('cTerminus')
#         cTerminus_name = None
#         if cTerminus and cTerminus.get('name') is not None:
#             cTerminus_name = cTerminus.get('name')
#             # if cTerminus_name == 'DiMIQ':
#             #     print(AMP['id'])
#         nTerminus = AMP.get('nTerminus')
#         nTerminus_name = None
#         if nTerminus and nTerminus.get('name') is not None:
#             nTerminus_name = nTerminus.get('name')
#             # if nTerminus_name == 'Chol':
#             #     print(AMP['id'])
#         AMP_peptide = Peptide(AMP['sequence'], aa_smiles_dict, AMP['id'], AMP['intrachainBonds'], AMP['interchainBonds'], AMP['unusualAminoAcids'], cTerminus_name, nTerminus_name, c_terminus_name_smiles, n_terminus_name_smiles)
#         if not AMP_peptide.noise_data_flag:
#             intra_bond_linked_id_smiles.append([AMP['id'], Chem.MolToSmiles(AMP_peptide.ncTerminus_modified_mols[0])])
    # elif AMP['complexity']['name'] == 'Multimer':
    #     seqs = []
    #     multimer_id = AMP['id']
    #     intrachain_bonds = []
    #     interchain_bonds = []
    #     unusual_aas = []
    #     for monomer in AMP['monomers']:
    #         seqs.append(monomer['sequence'])
    #         intrachain_bonds.append(monomer['intrachainBonds'])
    #         interchain_bonds.append(monomer['interchainBonds'])
    #         unusual_aas.append(monomer['unusualAminoAcids'])
    #     AMP_peptide = Peptide(seqs, aa_smiles_dict, multimer_id, intrachain_bonds, interchain_bonds, unusual_aas)

# output_file_path = './Data/DBAASP_id_wo_existing_smiles_intra_linked_smiles.csv'
# # 列名
# print(f'len: {len(intra_bond_linked_id_smiles)}')
# df = pd.DataFrame(intra_bond_linked_id_smiles, columns=["DBAASP_id", "SMILES"])
# df.to_csv(output_file_path, index=False)


