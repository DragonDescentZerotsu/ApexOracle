import pandas as pd
from tqdm import tqdm

mean_MIC_df = pd.read_csv('./Data/DBAASP_id_SMILES_bact_mean_MICs.csv')

mean_MIC_data = mean_MIC_df.values

# 创建一个从 DBAASP_id 到 mean_MIC 映射的 dict
mean_MIC_dict = {DBAASP_id:mean_MIC for DBAASP_id, mean_MIC in zip(mean_MIC_data[:, 0], mean_MIC_data[:, -1])}

# 读取 SMILES compare记结果，并为其配上对应的 mean MIC
df = pd.read_csv('./Data/DBAASP_id_SMILES_compare_cleaned.csv')

origi_columns = list(df.columns)

smiles_comp_data = df.values
smiles_comp_data_w_mean_MIC = []
for line in tqdm(smiles_comp_data, desc=' completing MIC'):
    # print(line)
    # exit(0)
    mic_1 = mean_MIC_dict.get(line[0], None)
    mic_2 = mean_MIC_dict.get(line[1], None)
    if mic_1 and mic_2 is not None:
        # 还需要两个数值有一定的差距
        if 0.8*max(mic_1, mic_2) > min(mic_1, mic_2):
            line = line.tolist()
            line.extend([mic_1, mic_2])
            smiles_comp_data_w_mean_MIC.append(line)
        # print(line)
        # break

print(f' length of filtered data: {len(smiles_comp_data_w_mean_MIC)}\n length of original data: {len(smiles_comp_data)}')
origi_columns.extend(['mean_MIC_1', 'mean_MIC_2'])
df = pd.DataFrame(smiles_comp_data_w_mean_MIC, columns=origi_columns)
# df
df.to_csv('./Data/DBAASP_id_SMILES_compare_cleaned_w_mean_MIC.csv')