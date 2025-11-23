"""
提取感兴趣的 19 种 strain 的 MIC value
"""
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
import re
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import copy

def is_number(s):
    return bool(re.fullmatch(r'^(>|>=|<|<=)?\d+(\.\d+)?(±\d+(\.\d+)?)?(\-\d+(\.\d+)?)?$', s))

json_path = './Data/all_peptides_data.json'

# 打开并读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 将JSON内容加载为Python列表

# 打印读取的数据
print("number of AMPs: ", len(data))
id_dict = {}
for i, AMP in enumerate(data):
    id_dict[AMP['id']] = i

huge_bact_names = ["Escherichia coli ATCC 25922", "Pseudomonas aeruginosa ATCC 27853", "Staphylococcus aureus ATCC 25923", "Staphylococcus aureus", "Staphylococcus aureus ATCC 29213", "Escherichia coli", "Pseudomonas aeruginosa", "Pseudomonas aeruginosa PAO1", "Enterococcus faecalis ATCC 29212", "Acinetobacter baumannii ATCC 19606", "Staphylococcus epidermidis ATCC 12228", "Candida albicans ATCC 10231", "Klebsiella pneumoniae ATCC 700603", "Staphylococcus aureus ATCC 43300", "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028", "Staphylococcus aureus ATCC 6538", "Pseudomonas aeruginosa ATCC 9027", "Candida albicans", "Klebsiella pneumoniae"]

df = pd.read_csv('./Data/DBAASP_id_SMILES_merged.csv')

DBAASPid_SMILES = np.array(df[['DBAASP_id', 'SMILES']].values.tolist())
concentration_all = []
DBAASPid_SMILES_bact_MICs = []
for AMP in tqdm(data):
    if str(AMP['id']) not in DBAASPid_SMILES[:, 0]:
        continue

    bact_MICs = np.full(len(huge_bact_names), -1.0)

    line_index = np.where(DBAASPid_SMILES[:, 0] == str(AMP['id']))[0][0]
    # 记录一个 AMP 对每一种细菌的杀菌 entry 的单位和浓度
    bact_measure_value_unit = {}
    if AMP['targetActivities'] is not None:
        # 统计这个 AMP 中所有的原始格式的 针对不同菌株的 MIC
        for bact in AMP['targetActivities']:
            # if bact['unit'] is not None:
            #     if (bact['activityMeasureValue'], bact['unit']['name']) not in list(measure_unit.keys()):
            #         measure_unit[(bact['activityMeasureValue'], bact['unit']['name'])] = 1
            #     else:
            #         measure_unit[(bact['activityMeasureValue'], bact['unit']['name'])] += 1
            if bact['targetSpecies'] is None:
                continue
            bact_name = bact['targetSpecies']['name']
            # 确定是不是我们要的菌株
            if bact_name in huge_bact_names:
                if bact_name not in list(bact_measure_value_unit.keys()):
                    bact_measure_value_unit[bact_name] = {}
                if bact['unit'] is not None:
                    if (bact['activityMeasureValue'], bact['unit']['name']) not in list(bact_measure_value_unit[bact_name].keys()):
                        if bact['activityMeasureValue'] == 'MIC' or 'inhibition' in bact['activityMeasureValue'] or 'inhibiton' in bact['activityMeasureValue'] or 'Inhibition' in bact['activityMeasureValue']:
                            bact_measure_value_unit[bact_name][(bact['activityMeasureValue'], bact['unit']['name'])] = bact['concentration']
        # print(bact_measure_value_unit)
        for bact_name, value in list(bact_measure_value_unit.items()):
            # value 就是 (bact['activityMeasureValue'], bact['unit']['name'])：bact['concentration'] 对
            if len(value) == 0:
                # 如果没有测量 MIC 的那就不要这个细菌的数据了
                del bact_measure_value_unit[bact_name]
                continue

            # 如果是 µg/ml 则需要单位转换
            unit_transfer = False
            if ('MIC', 'µM') in value.keys():
                concentration = value[('MIC', 'µM')].strip()
            elif ('MIC', 'µg/ml') in value.keys():
                unit_transfer = True
                concentration = value[('MIC', 'µg/ml')].strip()

            # 除了上面这两种 MIC 就是各种 inhibition, Inhibition, inhibiton
            else:
                inhibition_percent_max = 0
                inhibition_percent_max_key = None
                MIC_final = 0
                for (value_name, unit_name), concentration in list(value.items()):
                    if '%' not in value_name:
                        print(f'wrong inhibition percent: {value_name}')
                        exit(1)
                    inhibition_percent = value_name.split('%')[0]
                    if '(' in inhibition_percent:
                        inhibition_percent = value_name.split(')')[0].split('(')[-1]
                    if '>=' in inhibition_percent:
                        inhibition_percent = inhibition_percent.split('>=')[-1]
                    if '<' in inhibition_percent:
                        if inhibition_percent[-1] == '<':
                            inhibition_percent = inhibition_percent.split('<')[0]
                        else:
                            inhibition_percent = inhibition_percent.split('<')[-1]
                    if '±' in inhibition_percent:
                        inhibition_percent = float(inhibition_percent.split('±')[0])
                    if '-' in inhibition_percent:
                        inhibition_percent = sum(float(comp) for comp in inhibition_percent.split('-')) / len(inhibition_percent.split('-'))
                    else:
                        inhibition_percent = float(inhibition_percent)

                    # 同一个 AMP 可能会有不同 百分比 抑制的数据记录，要最高的那个
                    if inhibition_percent > inhibition_percent_max:
                        inhibition_percent_max = inhibition_percent
                        inhibition_percent_max_key = (value_name, unit_name)

                # 抑制率一定要 95% 以上才要
                if inhibition_percent_max < 95:
                    del bact_measure_value_unit[bact_name]
                    continue

                else:
                    concentration = value[inhibition_percent_max_key].strip()
                    if unit_name == 'µg/ml':
                        unit_transfer = True
                    if unit_name not in ['µg/ml', 'µM']:
                        print(f'wrong unit name: {unit_name}')
                        exit(1)

            if not is_number(concentration):
                concentration_all.append(concentration)
            concentration_copy = concentration
            if ' - =>' in concentration:
                sum(float(comp) for comp in concentration.split(' - =>')) / len(concentration.split(' - =>'))
            if ' ' in concentration.strip():
                concentration = concentration.replace(' ', '')
            if '->' in concentration:
                sum(float(comp) for comp in concentration.split('->')) / len(concentration.split('->'))
            if '≥' in concentration:
                concentration = concentration.split('≥')[-1]
            if '>=' in concentration:
                concentration = concentration.split('>=')[-1]
            if '<=' in concentration:
                concentration = concentration.split('<=')[-1]
            if '>>' in concentration:
                concentration = concentration.split('>>')[-1]
            if '>' in concentration:
                concentration = concentration.split('>')[-1]
            if '<' in concentration:
                if concentration[-1] == '<':
                    concentration = concentration.split('<')[0]
                else:
                    concentration = concentration.split('<')[-1]
            if '±' in concentration:
                concentration = concentration.split('±')[0]
            if ',' in concentration:
                concentration = concentration.replace(',', '.')
            if '-' in concentration:
                concentration = sum(float(comp) for comp in concentration.split('-')) / len(concentration.split('-'))
            concentration = float(concentration)

            if unit_transfer:
                mol = Chem.MolFromSmiles(DBAASPid_SMILES[line_index, 1])
                if mol is None:
                    print(f'AMP: {AMP['id']}, bact: {bact_name}, value: {value}')
                mol_wt = rdMolDescriptors.CalcExactMolWt(mol)
                concentration = concentration / mol_wt * 1000

            if '>' in concentration_copy or '>=' in concentration_copy:
                if '->' not in concentration_copy and ' - =>' not in concentration_copy and '>>' not in concentration_copy:
                    concentration *= 2
            if '>>' in concentration_copy:
                concentration *= 3

            bact_MICs[huge_bact_names.index(bact_name)] = concentration

    id_smiles_MICs = np.concatenate((DBAASPid_SMILES[line_index], bact_MICs))
    if max(bact_MICs) > -1:
        DBAASPid_SMILES_bact_MICs.append(id_smiles_MICs)

print(concentration_all)
column_names = ['DBAASP_id', 'SMILES']
column_names.extend(huge_bact_names)
df = pd.DataFrame(DBAASPid_SMILES_bact_MICs, columns=column_names)
df.to_csv('./Data/DBAASP_id_SMILES_bact_mean_MICs.csv', index=False)
print(f"number of peptides: {len(df)}")