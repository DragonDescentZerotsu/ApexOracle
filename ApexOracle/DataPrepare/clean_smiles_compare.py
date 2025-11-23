import pandas as pd
import re
from tqdm import tqdm
import ast


def find_stereo_centers(smiles):
    # 正则表达式匹配 C@@H, C@H, C@, C@@
    pattern = r'C@{1,2}H?|C@{1,2}'

    # 使用 re.finditer() 找到所有匹配的位置
    matches = [(match.start(), match.group()) for match in re.finditer(pattern, smiles)]

    return matches

if __name__ == "__main__":

    df = pd.read_csv('./Data/DBAASP_id_SMILES_compare.csv')
    origi_columns = df.columns
    data = df.values

    # for row in data:
    #     if row[6]>0.98 and row[7]>0.98:
    #         print(row)
    filtered_data = []
    for row in tqdm(data, desc=' Filtering SMILES pairs'):
        # 如果这两个 smiles 在 rdkit 看来完全相等
        if row[6] == row[7] == 1:
            # 如果其实不是相等的，说明有异构的差异

            if row[2] != row[3]:
                different_places_1 = []
                different_places_2 = []
                # 有异构差异就进行 C@@H 和 C@H 的比对
                match_1 = find_stereo_centers(row[2])
                match_2 = find_stereo_centers(row[3])
                for (index_1, match_group_1), (index_2, match_group_2) in zip(match_1, match_2):
                    if match_group_1 != match_group_2:
                        different_places_1.append(index_1)
                        different_places_2.append(index_2)
                row[4] = str(different_places_1)
                row[5] = str(different_places_2)
                filtered_data.append(row)

        # 如果不完全相等直接放过
        else:
            filtered_data.append(row)

    print(f' length of filtered data: {len(filtered_data)}\n length of original data: {len(data)}')

    df = pd.DataFrame(filtered_data, columns=origi_columns)
    df.to_csv('./Data/DBAASP_id_SMILES_compare_cleaned.csv', index=False)