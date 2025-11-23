import requests
import json

# 获取肽列表的URL
pep_id= 1
url = f"https://dbaasp.org/peptides/{pep_id}"

# 发送GET请求
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    try:
        # 获取数据并解析为JSON
        peptides_data = response.json()

        # 将JSON数据保存到本地文件
        with open(f"/home/tianang/Projects/Synergy/DataPrepare/Data/peptide_{pep_id}.json", "w", encoding="utf-8") as json_file:
            json.dump(peptides_data, json_file, ensure_ascii=False, indent=4)

        print(f"数据已成功保存到 'peptide_{pep_id}.json'")

    except requests.exceptions.JSONDecodeError:
        print("无法解析JSON响应")
else:
    print(f"错误: {response.status_code}")