"""
提取所有有 MIC value 的样本，存下来对应的 DBAASP_id; Strain Name; SMILES; MIC value
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
print( 'Reading data from ' + json_path)
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 将JSON内容加载为Python列表

# 打印读取的数据
print("number of AMPs: ", len(data))
id_dict = {}
for i, AMP in enumerate(data):
    id_dict[AMP['id']] = i

# 注意这里最后面加了 mean ，是所有菌株的均值
# huge_bact_names = ["Escherichia coli ATCC 25922", "Pseudomonas aeruginosa ATCC 27853", "Staphylococcus aureus ATCC 25923", "Staphylococcus aureus", "Staphylococcus aureus ATCC 29213", "Escherichia coli", "Pseudomonas aeruginosa", "Pseudomonas aeruginosa PAO1", "Enterococcus faecalis ATCC 29212", "Acinetobacter baumannii ATCC 19606", "Staphylococcus epidermidis ATCC 12228", "Candida albicans ATCC 10231", "Klebsiella pneumoniae ATCC 700603", "Staphylococcus aureus ATCC 43300", "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028", "Staphylococcus aureus ATCC 6538", "Pseudomonas aeruginosa ATCC 9027", "Candida albicans", "Klebsiella pneumoniae", 'mean']

bact_count_dict = {}

df = pd.read_csv('./Data/DBAASP_id_SMILES_merged.csv')

DBAASPid_SMILES = np.array(df[['DBAASP_id', 'SMILES']].values.tolist())
concentration_all = []
DBAASPid_SMILES_bact_MICs = []
for AMP in tqdm(data):

    # 原始数据不在有 SMILES 的数据里面
    if str(AMP['id']) not in DBAASPid_SMILES[:, 0]:
        continue

    # TODO: 调试用
    # if AMP['id'] == 105:
    #     print(AMP['id'])
    # bact_MICs = np.full(len(huge_bact_names), -1.0)

    # 找出 DBAASPid_SMILES 第一列中第一个与 str(AMP['id']) 相匹配的元素所在的行索引，方便下面访问这个 DBAASP id 对应的 SMILES
    line_index = np.where(DBAASPid_SMILES[:, 0] == str(AMP['id']))[0][0]

    # 记录一个 AMP 对每一种 细菌 的杀菌 entry 的单位和浓度
    bact_measure_value_unit = {}
    if AMP['targetActivities'] is not None:
        # 先统计这个 AMP 中所有的原始格式的 针对不同菌株的 MIC
        for bact in AMP['targetActivities']:
            # if bact['unit'] is not None:
            #     if (bact['activityMeasureValue'], bact['unit']['name']) not in list(measure_unit.keys()):
            #         measure_unit[(bact['activityMeasureValue'], bact['unit']['name'])] = 1
            #     else:
            #         measure_unit[(bact['activityMeasureValue'], bact['unit']['name'])] += 1
            if bact['targetSpecies'] is None:
                continue
            bact_name = bact['targetSpecies']['name']

            # 用来控制最后得到的到底是什么编号的 strain
            if 'ATCC' in bact_name:
                continue

            # TODO: 调试用
            # if bact_name not in huge_bact_names:
            #     print(bact_name, '\n not in huge_bact_names')

            # 不管是不是我们感兴趣的菌株都要
            if bact_name not in list(bact_measure_value_unit.keys()):
                bact_measure_value_unit[bact_name] = {}
            if bact['unit'] is not None:
                if (bact['activityMeasureValue'], bact['unit']['name']) not in list(bact_measure_value_unit[bact_name].keys()):
                    if bact['activityMeasureValue'] == 'MIC' or 'inhibition' in bact['activityMeasureValue'] or 'inhibiton' in bact['activityMeasureValue'] or 'Inhibition' in bact['activityMeasureValue']:
                        bact_measure_value_unit[bact_name][(bact['activityMeasureValue'], bact['unit']['name'])] = bact['concentration']
        # print(bact_measure_value_unit)

        # 上面统计处理完所有的可能的格式之后其实是存储了该 AMP 下 每一种有记录的 bact 的 各种 MIC 和 单位数据，下面开始处理这些存储的数据
        # num_bact_count = 0
        # mean_mic = 0
        for bact_name, value in list(bact_measure_value_unit.items()):

            # TODO: 调试用
            # if bact_name not in huge_bact_names:
            #     print(bact_name, ' not in huge_bact_names')

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
                    value_key_copy = (value_name, unit_name)
                    if '%' not in value_name:
                        print(f'wrong inhibition percent: {value_name}')
                        # exit(1)
                        value_name = value_name.split(' ')[0]+'%'
                    inhibition_percent = value_name.split('%')[0]
                    if '(' in inhibition_percent:
                        inhibition_percent = value_name.split(')')[0].split('(')[-1]
                    if '±' in inhibition_percent:
                        inhibition_percent = inhibition_percent.split('±')[0]
                    if '>=' in inhibition_percent:
                        inhibition_percent = inhibition_percent.split('>=')[-1]
                    if '<' in inhibition_percent:
                        if inhibition_percent[-1] == '<':
                            inhibition_percent = inhibition_percent.split('<')[0]
                        else:
                            inhibition_percent = inhibition_percent.split('<')[-1]
                    if '>' in inhibition_percent:
                        if inhibition_percent[-1] == '>':
                            inhibition_percent = inhibition_percent.split('>')[0]
                        else:
                            inhibition_percent = inhibition_percent.split('>')[-1]
                    if 'e' in inhibition_percent:
                        inhibition_percent = str(float(inhibition_percent.split('±')[0]))
                    if '-' in inhibition_percent:
                        inhibition_percent = sum(float(comp) for comp in inhibition_percent.split('-')) / len(inhibition_percent.split('-'))
                    else:
                        inhibition_percent = float(inhibition_percent)

                    # 同一个 AMP 可能会有不同 百分比 抑制的数据记录，要最高的那个
                    if inhibition_percent > inhibition_percent_max:
                        inhibition_percent_max = inhibition_percent
                        inhibition_percent_max_key = value_key_copy

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
            # TODO: 调试用
            # print(f'concentration: {concentration}')
            # 如果 concentration 是空字符串就直接删掉原始细菌 entry
            if not concentration:
                del bact_measure_value_unit[bact_name]
                continue
            if ' - =>' in concentration:
                concentration = str(sum(float(comp) for comp in concentration.split(' - =>')) / len(concentration.split(' - =>')))
            if '->=' in concentration:
                concentration = str(sum(float(comp) for comp in concentration.split('->=')) / len(concentration.split('->=')))
            if ' - >=' in concentration:
                concentration = str(sum(float(comp) for comp in concentration.split(' - >=')) / len(concentration.split(' - >=')))
            if ' ' in concentration.strip():
                concentration = concentration.replace(' ', '')
            if '->' in concentration:
                concentration = str(sum(float(comp) for comp in concentration.split('->')) / len(concentration.split('->')))
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
                concentration = str(sum(float(comp) for comp in concentration.split('-')) / len(concentration.split('-')))
            if '–' in concentration:
                concentration = str(sum(float(comp) for comp in concentration.split('–')) / len(concentration.split('–')))
            if len(concentration.split('.')) > 2:
                concentration = '.'.join(concentration.split('.')[:2])
            # print(concentration_copy)
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

            if bact_name not in bact_count_dict.keys():
                bact_count_dict[bact_name] = 1
            else:
                bact_count_dict[bact_name] += 1

            id_bact_name_smiles_MIC = np.array((DBAASPid_SMILES[line_index][0], bact_name, DBAASPid_SMILES[line_index][1], concentration))
            DBAASPid_SMILES_bact_MICs.append(id_bact_name_smiles_MIC)

            # if bact_name in huge_bact_names:
            #     bact_MICs[huge_bact_names.index(bact_name)] = concentration

            # mean_mic = (mean_mic * num_bact_count + concentration) / (num_bact_count+1)
            # num_bact_count += 1

        # if 语句防止有完全没有 MIC 记录的数据混在里面
        # bact_MICs[-1] = mean_mic if mean_mic > 0 else -1


    # id_smiles_MICs = np.concatenate((DBAASPid_SMILES[line_index], bact_MICs))
    # if max(bact_MICs) > -1:
    #     DBAASPid_SMILES_bact_MICs.append(id_smiles_MICs)

print(concentration_all)
print(f' num of datapoints for each strain:\n{json.dumps(dict(sorted(bact_count_dict.items(), key=lambda item: item[1], reverse=True)), indent=4)}')

# 保存上面打印出来的数据分布
with open(f'./Data/Evo_no_ATCC_only_edition_1_MIC_data_count_{len(DBAASPid_SMILES_bact_MICs)}.json', 'w', encoding='utf-8') as f:
    json.dump(dict(sorted(bact_count_dict.items(), key=lambda item: item[1], reverse=True)), f, ensure_ascii=False, indent=4)

column_names = ['DBAASP_id', 'strain_name', 'SMILES', 'MIC']
# column_names.extend(huge_bact_names)
df = pd.DataFrame(DBAASPid_SMILES_bact_MICs, columns=column_names)
df.to_csv('./Data/DBAASP_id_bact_name_SMILES_MIC_Evo_no_ATCC_only.csv', index=False)
print(f"number of datapoint: {len(df)}")