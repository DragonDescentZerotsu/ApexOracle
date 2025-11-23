"""
DBAASP 提供的 SMILES 不考虑氨基酸的手性，所以这里要重新拼接这些 peptide 的 SMILES
"""

import json
from aa_seq_to_smiles import *
from pathlib import Path
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
import numpy as np

current_directory = Path(__file__).parent

json_path = current_directory/'Data'/'all_peptides_data.json'

# 打开并读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 将JSON内容加载为Python列表

# 打印读取的数据
print("number of AMPs: ", len(data))
id_dict={}

# 转换成字典 AMP id 映射到 某一个行 index
for i, AMP in enumerate(data):
    id_dict[AMP['id']] = i

# 到 DBAASP 的 SMILES 的路径
SMILES_from_DBAASP_path = current_directory/'Data'/'DBAASP_id_wo_PubChem_SMILES_w_DBAASP_smiles.csv'

DBAASP_ids_offering_SMILES = pd.read_csv(SMILES_from_DBAASP_path)[['DBAASP_id', 'SMILES']].values

aa_smiles_dict = get_aa_smiles_dict('./Data/all_aa_smiles_new_handcrafted.csv')

df = pd.read_csv("./Data/terminal_modifications/c_terminal_smiles_from_PubChem_handcrafted.csv")
c_terminus_name_smiles = dict(zip(df['name'], df['SMILES']))

df = pd.read_csv("./Data/terminal_modifications/n_terminal_smiles_from_PubChem_handcrafted.csv")
n_terminus_name_smiles = dict(zip(df['name'], df['SMILES']))

intra_bond_linked_id_smiles = []

for DBAASP_id, DBAASP_offered_smiles in tqdm(DBAASP_ids_offering_SMILES, ' Correcting SMILES '):

    # print(DBAASP_id)
    AMP = data[id_dict[DBAASP_id]]
    if AMP['complexity']['name'] == 'Monomer':

        # 这些部分单独提出来是因为如果进行正常的拼接的话会报错，其中有错误的 intrachain bond 的 data
        if DBAASP_id in [20478, 20637, 21675, 21676, 21677, 21678, 21751, 21770]:
            print(DBAASP_id)
            intra_bond_linked_id_smiles.append([DBAASP_id, DBAASP_offered_smiles])
            continue

        cTerminus = AMP.get('cTerminus')
        cTerminus_name = None
        if cTerminus and cTerminus.get('name') is not None:
            cTerminus_name = cTerminus.get('name')
            # if cTerminus_name == 'DiMIQ':
            #     print(AMP['id'])
        nTerminus = AMP.get('nTerminus')
        nTerminus_name = None
        if nTerminus and nTerminus.get('name') is not None:
            nTerminus_name = nTerminus.get('name')
            # if nTerminus_name == 'Chol':
            #     print(AMP['id'])
        AMP_peptide = Peptide(AMP['sequence'], aa_smiles_dict, AMP['id'], AMP['intrachainBonds'], AMP['interchainBonds'],
                              AMP['unusualAminoAcids'], cTerminus_name, nTerminus_name, c_terminus_name_smiles,
                              n_terminus_name_smiles)
        if not AMP_peptide.noise_data_flag:
            intra_bond_linked_id_smiles.append([AMP['id'], Chem.MolToSmiles(AMP_peptide.ncTerminus_modified_mols[0])])
    else:
        print(DBAASP_id)
        intra_bond_linked_id_smiles.append([DBAASP_id, DBAASP_offered_smiles])

output_file_path = './Data/DBAASP_id_wo_PubChem_SMILES_w_DBAASP_smiles_remaked.csv'

print(f'len: {len(intra_bond_linked_id_smiles)}')
df = pd.DataFrame(intra_bond_linked_id_smiles, columns=["DBAASP_id", "SMILES"])
df.to_csv(output_file_path, index=False)

# 把这部分修改过的 SMILES 替换回一直使用的 DBAASP_id_SMILES_merged 还有 DBAASP_id_bact_name_SMILES_MIC_Evo

merged_smiles_data_path = current_directory / 'Data' / 'DBAASP_id_SMILES_merged.csv'
merged_smiles = pd.read_csv(merged_smiles_data_path)  # DBAASP_id, SMILES
merged_smiles_columns = merged_smiles.columns
merged_smiles = merged_smiles.values

Evo_strain_MIC_data_path = current_directory / 'Data' / 'DBAASP_id_bact_name_SMILES_MIC_Evo.csv'
all_Evo_MIC_data = pd.read_csv(Evo_strain_MIC_data_path)  # DBAASP_id, strain_name, SMILES, MIC
all_Evo_MIC_data_columns = all_Evo_MIC_data.columns
all_Evo_MIC_data = all_Evo_MIC_data.values

intra_bond_linked_id_smiles = dict(intra_bond_linked_id_smiles)

replacing_record = []
for i, (DBAASP_id, SMILES) in tqdm(enumerate(merged_smiles), desc = ' fixing merged '):
    if DBAASP_id in intra_bond_linked_id_smiles.keys():
        replacing_record.append([i, intra_bond_linked_id_smiles[DBAASP_id]])
for replace in replacing_record:
    merged_smiles[replace[0]][1] = replace[1]

df = pd.DataFrame(merged_smiles, columns = merged_smiles_columns)
df.to_csv(merged_smiles_data_path, index=False)

replacing_record = []
for i, (DBAASP_id, strain_name, SMILES, MIC) in tqdm(enumerate(all_Evo_MIC_data), desc=' fixing merged '):
    if DBAASP_id in intra_bond_linked_id_smiles.keys():
        replacing_record.append([i, intra_bond_linked_id_smiles[DBAASP_id]])
for replace in replacing_record:
    all_Evo_MIC_data[replace[0]][2] = replace[1]

df = pd.DataFrame(all_Evo_MIC_data, columns = all_Evo_MIC_data_columns)
df.to_csv(Evo_strain_MIC_data_path, index=False)