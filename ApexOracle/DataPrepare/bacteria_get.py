from tqdm import tqdm
import json

json_path = './Data/all_peptides_data.json'
save_path = './Data/strain_number_count.json'

# 打开并读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # 将JSON内容加载为Python列表

# 打印读取的数据
print("number of AMPs: ", len(data))
id_dict = {}
for i, AMP in enumerate(data):
    id_dict[AMP['id']] = i

# print("id_dict:", id_dict)

bact_list = []
bact_list_2 = []
bact_count = {}
bact_count_2 = {}
measure_unit = []
for AMP in tqdm(data):
    bact_names = []
    bact_names_2 = []

    if AMP['targetActivities'] is not None:
        for bact in AMP['targetActivities']:
            if bact['unit'] is not None:
                if (bact['activityMeasureValue'], bact['unit']['name']) not in measure_unit:
                    measure_unit.append((bact['activityMeasureValue'], bact['unit']['name']))
            if bact['targetSpecies'] is not None:
                # if bact['targetSpecies']['name'] not in list(bact_count.keys()):
                #     bact_count[bact['targetSpecies']['name']] = 1
                # else:
                #     bact_count[bact['targetSpecies']['name']] += 1
                bact_names.append(bact['targetSpecies']['name'])
                # if bact['targetSpecies']['name'] == 'Pseudomonas aeruginosa PAO1':
                #     print(AMP['id'], 0)
                # bact name 2 是没有编号的 细菌名字
                bact_names_2.append(' '.join(bact['targetSpecies']['name'].split()[:2]))
            else:
                print(f'{AMP['id']}')
                continue

        # if 'Pseudomonas aeruginosa PAO1' in bact_names:
        #     print(AMP['id'])

        bact_list.append(set(bact_names))
        bact_list_2.append(set(bact_names_2))
        for name in set(bact_names):
            # if name == 'Pseudomonas aeruginosa PAO1':
            #     print(AMP['id'], 1)
            if name not in list(bact_count.keys()):
                bact_count[name] = 1
            else:
                bact_count[name] += 1
        for name in set(bact_names_2):
            if name not in list(bact_count_2.keys()):
                bact_count_2[name] = 1
            else:
                bact_count_2[name] += 1
    else:
        print(AMP['id'])

from functools import reduce
result = reduce(lambda x, y: x & y, bact_list)

print(result)
result = reduce(lambda x, y: x & y, bact_list_2)
print(result)
print(json.dumps(dict(sorted(bact_count.items(), key=lambda item: item[1], reverse=True)), indent=4, ensure_ascii=False))
with open(save_path, 'w', encoding='utf-8') as f:
    json.dump(dict(sorted(bact_count.items(), key=lambda item: item[1], reverse=True)), f, ensure_ascii=False, indent=4)
print(json.dumps(dict(sorted(bact_count_2.items(), key=lambda item: item[1], reverse=True)), indent=4, ensure_ascii=False))
print(measure_unit)
