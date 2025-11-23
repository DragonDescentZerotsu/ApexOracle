"""
把 APEX 训练使用的 inhouse data 转换成 SMILES，顺便处理 inf 到 512，-1000 到 -1
"""

from aa_seq_to_smiles import *
from rdkit import Chem
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path

current_dir = Path(__file__).parent

# 使用从 OPSIN 获得的 SMILES
smiles = "N[C@H](C(=O)OCC=C)CCC(=O)O"
aa_seq = 'VTCDILSVEAKGVKLNDAACAAHCLFRGRSGGYCNGKRVCVCR'
# 创建分子对象
aa = AAs(smiles)
print(aa.N_terminal_atom.GetIdx(), aa.C_terminal_atom.GetIdx())

# 获得aa_smiles_dict
aa_smiles_dict = get_aa_smiles_dict('./Data/all_aa_smiles_new_handcrafted.csv')

df = pd.read_csv('./Data/APEX 1.1 Data.csv')
strain_names = list(df.columns)[1:-1]
# colume_names[0] = 'SMILES'
Evo_style_colume_names = ['DBAASP_id', 'strain_name', 'SMILES', 'MIC']

inhouse_data = df.values

inhouse_data[:, 1:-1][inhouse_data[:, 1:-1]<-500] = -1
inhouse_data[:, 1:-1][inhouse_data[:, 1:-1]==np.inf] = 512

print('\n counting data points')
inhouse_data_num = 0
for i in range(len(strain_names)):
    num_data_points = np.sum(inhouse_data[:, i+1] > -1)
    print(f' {strain_names[i]}: {num_data_points}')
    inhouse_data_num += num_data_points
print(f' total: {inhouse_data_num}')

Evo_style_MIC_data = []
for i, line in tqdm(enumerate(inhouse_data), desc=' Formatting inhouse MIC data', total=len(inhouse_data)):
    pep_obj = Peptide(line[0], aa_smiles_dict=aa_smiles_dict)
    pep_smiles = Chem.MolToSmiles(pep_obj.ncTerminus_modified_mols[0])
    for j, MIC_value in enumerate(line[1:-1]):
        if MIC_value > -1:
            Evo_style_MIC_data.append([f'#{i}', strain_names[j], pep_smiles, MIC_value])

df_inhouse_Evo_style = pd.DataFrame(Evo_style_MIC_data, columns=Evo_style_colume_names)

df_inhouse_Evo_style.to_csv(current_dir / 'Data' / 'inhouse_Evo_style_SMILES_MIC.csv', index=False)

DBAASP_Evo_strain_MIC_data_path = current_dir / 'Data' / 'DBAASP_id_bact_name_SMILES_MIC_Evo.csv'
DBAASP_Evo_strain_MIC_data = pd.read_csv(DBAASP_Evo_strain_MIC_data_path)

DBAASP_inhouse_AMP_SMILES_MIC_Evo = pd.concat([DBAASP_Evo_strain_MIC_data, df_inhouse_Evo_style], ignore_index=True)

DBAASP_inhouse_AMP_SMILES_MIC_Evo.to_csv(current_dir / 'Data' / 'DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv', index=False)