'''
提取出所有有synvergy作用记录的peptide
'''

import json
from tqdm import tqdm

json_path = '/home/tianang/Projects/Synergy/DataPrepare/Data/all_peptides_data.json'

# 打开并读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 将JSON内容加载为Python字典

synergies = []
c=0
for peptide in tqdm(data, desc="Processing Peptides"):
    # TODO: 这里还没改完呢
    if len(peptide['synergies']) != 0:
        print(peptide['complexity']['name'])
    # if len(peptide['intrachainBonds']) != 0 and peptide['complexity']['name'] == 'Monomer':
    #     c+=1
    #     print(json.dumps(peptide, indent=4, ensure_ascii=False))
    #     break

print(c)
