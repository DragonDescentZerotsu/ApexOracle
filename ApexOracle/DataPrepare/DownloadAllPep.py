import requests
import json
from tqdm import tqdm

# 保存所有肽数据的列表
all_peptides_data = []

# 定义下载范围
max_peptide_id = 22621
peptide_id = 1

# 定义URL的基本格式
base_url = "https://dbaasp.org/peptides/"

# 使用tqdm显示进度条，设置总任务数量为 max_peptide_id
with tqdm(total=max_peptide_id, desc="下载进度", unit="peptide") as pbar:
    # 循环下载数据直到编号达到最大值
    while peptide_id <= max_peptide_id:
        # 拼接URL
        url = base_url + str(peptide_id)

        # 发送GET请求
        response = requests.get(url)

        # 检查请求是否成功
        if response.status_code == 200:
            try:
                # 获取数据并解析为JSON
                peptide_data = response.json()

                # 如果数据为空，说明这个编号没有数据，跳过该编号
                if not peptide_data:
                    print(f"编号为 {peptide_id} 的数据为空，跳过。")
                else:
                    # 将数据添加到总数据列表中
                    all_peptides_data.append(peptide_data)
                    print(f"成功下载编号为 {peptide_id} 的数据。")

            except requests.exceptions.JSONDecodeError:
                print(f"无法解析编号为 {peptide_id} 的JSON响应，跳过。")

        else:
            print(f"编号为 {peptide_id} 的数据无法下载（错误码: {response.status_code}），跳过。")

        # 增加编号，继续下载下一个肽
        peptide_id += 1
        pbar.update(1)  # 每次循环更新进度条

# 将所有肽数据保存到本地JSON文件
with open("/home/tianang/Projects/Synergy/DataPrepare/Data/all_peptides_data.json", "w", encoding="utf-8") as json_file:
    json.dump(all_peptides_data, json_file, ensure_ascii=False, indent=4)

print("所有数据已成功保存到 'all_peptides_data.json'")