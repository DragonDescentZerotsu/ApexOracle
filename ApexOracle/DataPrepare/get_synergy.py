import json
from pathlib import Path
from typing import List, Any
import requests
import time
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_antimicrobial_SMILES(url: str, name, retries=3, delay=1) -> tuple[str, Any | None]:
    """
    获得并返回 antimicrobial 的SMILES
    :param url: url模版
    :param name: 名称用于PubChem下载
    :return: antimicrobial name, smiles
    """
    smiles = None
    for retry in range(retries):
        # 如果第一次没有成功，retry>0
        if retry>0:
            print(f'retry: {retry}')
        response = requests.get(url.format(name))
        if response.status_code == 200:
            json_info = json.loads(response.text.strip())
            smiles = json_info['PropertyTable']['Properties'][0]['IsomericSMILES']
            break
        elif response.status_code == 503:
            print(f"Service unavailable (503) for name: {name}, retrying in", delay, "seconds...")
            time.sleep(delay)
        else:
            print(f"Failed to retrieve data of name: {name}: response.status_code: {response.status_code}")
            break
    if retry==retries-1:
        print(f"Failed to retrieve data of name: {name}: response.status_code: {response.status_code}")
        # if smiles is not None:
        #     break
    return smiles

current_path = Path(__file__).parent

json_path = current_path/'Data'/'all_peptides_data.json'

# 打开并读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 将JSON内容加载为Python列表

# 打印读取的数据
print("number of AMPs: ", len(data))
id_dict = {}
for i, AMP in enumerate(data):
    id_dict[AMP['id']] = i

print("id_dict:", id_dict)

synergy_id_name_data = {}

for AMP in tqdm(data, desc=' get raw synergy data'):
    # 如果有 synergy 数据
    if len(AMP['synergies']) > 0:
        for syn in AMP['synergies']:
            # 获得 FICI 值
            FICI = syn['fici'].strip()
            # print(FICI)
            if '<=' in FICI:
                FICI = FICI.split('<=')[-1]
            if '>=' in FICI:
                FICI = str(float(FICI.split('>=')[-1])*1.5)
            if '>' in FICI:
                FICI = str(float(FICI.split('>')[-1])*2)
            if '<' in FICI:
                FICI = FICI.split('<')[-1]
            if '-' in FICI:
                if FICI.strip() == '-' :
                    continue
                FICI = str(np.array([float(seg) for seg in FICI.split('-')]).mean())
            if '–' in FICI:
                FICI = str(np.array([float(seg) for seg in FICI.split('–')]).mean())
            if '±' in FICI:
                FICI = str(float(FICI.split('±')[0]))

            if len(FICI.strip()) == 0:
                continue
            FICI = float(FICI)

            # 如果是和 peptide 有 synergy 作用
            if syn['antibioticId'] is not None:
                # 单调获取数据防止重复
                if syn['antibioticId'] > syn['peptideId']:
                    if (syn['peptideId'], syn['antibioticId']) not in synergy_id_name_data.keys():
                        synergy_id_name_data[(syn['peptideId'], syn['antibioticId'])] = [FICI]
                    else:
                        synergy_id_name_data[(syn['peptideId'], syn['antibioticId'])].append(FICI)

            # 如果是和 其他 antimicrobial 有 synergy 作用
            if syn['antibioticName'] is not None:
                # "" 和 None 没区别，也是要去掉的
                if syn['antibioticName'].strip() == "":
                    continue
                if (syn['peptideId'], syn['antibioticName'].strip()) not in synergy_id_name_data.keys():
                    synergy_id_name_data[(syn['peptideId'], syn['antibioticName'].strip())] = [FICI]
                else:
                    synergy_id_name_data[(syn['peptideId'], syn['antibioticName'].strip())].append(FICI)

# 刚刚收集的数据都是列表，其中每一个数值代表对应于某一种 strain 的 FICI，这里要求平均并且变成最后 bin 分类的数据
bined_synergy_id_name_data = {}
for key, value in synergy_id_name_data.items():
    # 平均
    mean_FICI = np.array(value).mean()

    # 分 bin
    if 0 <= mean_FICI < 0.5:
        mean_FICI = 0
    elif 0.5 <= mean_FICI < 4:
        mean_FICI = 1
    elif mean_FICI >= 4:
        mean_FICI = 2
    else:
        print(f' Wrong FICI number: {FICI}')

    # 存入数据
    bined_synergy_id_name_data[key] = mean_FICI

antimicrobial_smiles_path = current_path/'Data'/'antimicrobial_SMILES_handcrafted.json'

# TODO: 一定记得删
# 这里的代码时用啦统计
# antimicrobial_smiles_path.unlink(missing_ok=True)
# TODO: 一定记得删


if not antimicrobial_smiles_path.exists():
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/IsomericSMILES/JSON"
    name_smiles_dict = {}
    no_smiles_names = set()  # 记录哪些 antimicrobio 的 smiles 没被找到，下次就不要找了
    all_names = set()
    count_dict = {}
    for AMP_id, antibio_id_or_name in tqdm(bined_synergy_id_name_data.keys(), desc=' Get antibiotic smiles from PubChem'):

        if antibio_id_or_name in all_names:
            count_dict[str(antibio_id_or_name)] += 1
            continue
        all_names.add(antibio_id_or_name)
        count_dict[str(antibio_id_or_name)] = 1
        # 防止第二是 AMP_id 也送进去 PubMed 找smiels
        if isinstance(antibio_id_or_name, str):
            # 如果这个name 以前没有被标记为找不到smiles就找一下
            if antibio_id_or_name not in no_smiles_names:
                smiles = get_antimicrobial_SMILES(url, antibio_id_or_name)
            else:
                smiles = None
        else:
            smiles = None
        if smiles is not None:
            name_smiles_dict[antibio_id_or_name] = smiles
        else:
            if isinstance(antibio_id_or_name, str):
                print(f' {antibio_id_or_name} not found.')
                no_smiles_names.add(antibio_id_or_name)
    print(f' Saving {antimicrobial_smiles_path}...')
    with open(antimicrobial_smiles_path, 'w', encoding="utf-8") as f:
        json.dump(name_smiles_dict, f, ensure_ascii=False, indent=4)
    print(f' Saved')

else:
    with open(antimicrobial_smiles_path, 'r', encoding="utf-8") as f:
        name_smiles_dict = json.load(f)

# print(f' all antibiotic names:\n{all_names}\n\n not found names:\n{no_smiles_names}')
# print(f' antibiotic smiles:\n {json.dumps(name_smiles_dict, indent=4)}')
# no_smiles_count_dict = {name: count for name, count in count_dict.items() if name in no_smiles_names}
# with_smiles_count_dict = {name: count for name, count in count_dict.items() if name in all_names-no_smiles_names}
# print(f' no smiles count:\n{json.dumps(no_smiles_count_dict, indent=4)}')
# print(f' with smiles count:\n{json.dumps(with_smiles_count_dict, indent=4)}')
# print(f' data point count:{np.array([count for count in with_smiles_count_dict.values()]).sum()+102}')

# TODO: 获取数据中所有的 name antimicrobial，并且从 PubChem 获取对应的 SMILES
df = pd.read_csv(current_path / 'Data' / 'DBAASP_id_SMILES_merged.csv')

AMP_smiles_dict = dict(zip(df['DBAASP_id'], df['SMILES']))

synergistic_datas = []
c_count = [0,0,0]
for (AMP_id, antibio_id_or_name), synergy_class in tqdm(bined_synergy_id_name_data.items(), desc=' Creating synergistic pairs'):
    AMP_smiles = AMP_smiles_dict.get(AMP_id, None)
    if AMP_smiles is not None:
        if isinstance(antibio_id_or_name, str):
            antibiotic_smiles = name_smiles_dict.get(antibio_id_or_name, None)
        elif isinstance(antibio_id_or_name, int):
            antibiotic_smiles = AMP_smiles_dict.get(antibio_id_or_name, None)
        else:
            print(f' {antibio_id_or_name} error data type.')
            antibiotic_smiles = None

        if antibiotic_smiles is None:
            continue

        # 在这里说明两个都不是 None，是有数据的
        synergistic_datas.append([AMP_id, antibio_id_or_name, AMP_smiles, antibiotic_smiles, synergy_class])
        c_count[synergy_class] += 1

print(f' class count: {c_count}')
df = pd.DataFrame(synergistic_datas, columns=['DBAASP_id', 'antibio_id_or_name', 'AMP_smiles', 'antibiotic_smiles', 'synergy_class'])
df.to_csv(current_path / 'Data' / 'synergistic_pairs.csv', index=False)







