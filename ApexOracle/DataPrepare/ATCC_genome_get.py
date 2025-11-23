import os
import requests
from pathlib import Path
import json
from tqdm import tqdm
import time

current_script_path = Path(__file__).resolve()  # 获取当前脚本的绝对路径
current_directory = current_script_path.parent  # 获取当前脚本所在的目录

def get_ATCC_number(file_path):
    """
    获取 DBAASP 中所有strain所对应的 ATCC id
    :param file_path: 处理好的替代文件路径
    :return: ATCC id number set
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # 解析 JSON 文件

    ATCC_id_set = set()

    for name in data.keys():
        if '*' in name:
            name = name.split('*')[-1]
        if 'del' in name:
            continue
        if '(' in name:
            name = name.split('(')[0]
        if 'ATCC' in name:
            ATCC_id = name.split('ATCC')[-1].strip()
            if 'BAA' in name:
                ATCC_id = ATCC_id.replace(" ", "-")
            if 'MY' in name:
                ATCC_id = ATCC_id.replace(" ", "")
            if 'MAY' in name:
                ATCC_id = ATCC_id.replace("MAY", "MYA")
            if 'D' in name:
                ATCC_id = ATCC_id.split("D")[0]
            if 'T' in name:
                ATCC_id = ATCC_id.split("T")[0]
            if 's' in name:
                ATCC_id = ATCC_id.split("s")[0]
            if " " in name:
                ATCC_id = ATCC_id.split(" ")[0]
            ATCC_id_set.add(ATCC_id)

    return ATCC_id_set

def get_stored_ATCC_IDs(folder_path):
    """
    检查哪些 ATCC ID 已经被下载了
    :param folder_path:
    :return:
    """
    stored_ATCC_IDs = []
    files = [f.name for f in folder_path.iterdir() if f.is_file()]
    for file_name in files:
        file_name = file_name.split('.')[0]
        file_name = file_name.split('ATCC')[-1]
        components = file_name.split('_')[1:]
        if len(components) == 2:
            stored_ATCC_IDs.append('-'.join(components))
        else:
            stored_ATCC_IDs.append(components[0])

    return stored_ATCC_IDs

DBAASP_strain_file_path = current_directory / 'Data' / 'Evo_edition_3_MIC_data_handcrafted_no_ATCC_to_custom_ATCC.json'
# ATCC_id_set = get_ATCC_number(DBAASP_strain_file_path)  # TODO: 这里手动修改了需要下载的ATCC
ATCC_id_set = {'BAA-3170', 'BAA-3197'}
print(f'\n Number of different ATCC IDs: {len(ATCC_id_set)}')

# 获取已经下载好的 genome 的 ATCC number
stored_ATCC_IDs = set(get_stored_ATCC_IDs(current_directory / 'Data' / 'Genome' / 'ATCC'))

delay = 1
retries = 3

# 设置您的 API 密钥
API_KEY = "00b8dfee2aa1d1d043454b547f0204b8"

# 要搜索的 product_id
# product_id = "25922"  # 示例 product_id，请根据实际情况替换

# 搜索基因组的 API URL
search_api_url = "https://genomes.atcc.org/api/genomes/search"

# 设置请求头，添加 API 密钥
headers = {"X-API-Key": API_KEY,}

found_ATCC_ID_set = set()
# 发起 POST 请求，搜索基因组
for product_id in tqdm(ATCC_id_set, desc=' Getting ATCC ID Genome... '):  # 要搜索的 product_id

    if product_id in stored_ATCC_IDs:
        continue
    # 设置搜索参数
    search_params = {"product_id": product_id}

    search_response = requests.post(search_api_url, headers=headers, json=search_params)
    if search_response.status_code == 200:
        search_results = search_response.json()
        if search_results:
            # 假设第一个结果是我们需要的
            genome = search_results[0]
            genome_id = genome.get("id")
            genome_name = genome.get("name", "unknown_genome")
            if genome_id:
                # 下载组装文件的 API URL
                download_api_url = f"https://genomes.atcc.org/api/genomes/{genome_id}/download_assembly"
                # 发起 GET 请求，下载组装文件
                download_response = requests.get(download_api_url, headers=headers)
                # 防止频繁访问导致的出错
                if download_response.status_code == 503:
                    print(f"  Service unavailable (503) for ATCC ID {product_id} during getting Genome ID, retrying in", delay, "seconds...")
                    time.sleep(delay)
                if download_response.status_code == 200:
                    download_data = download_response.json()
                    file_url = download_data.get("url")
                    filename = download_data.get("save_as_filename", f"{genome_name}.fasta")
                    if file_url:
                        # print(f"开始下载文件：{filename}")
                        # 使用流式请求下载文件内容
                        # with requests.get(file_url, stream=True) as r:
                        #     r.raise_for_status()
                        #     with open(filename, "wb") as f:
                        #         for chunk in r.iter_content(chunk_size=8192):
                        #             if chunk:
                        #                 f.write(chunk)
                        file_response = requests.get(file_url)
                        # 防止频繁访问导致的出错
                        if file_response.status_code == 503:
                            print(f"  Service unavailable (503) for ATCC ID {product_id} during downloading Genome, retrying in", delay, "seconds...")
                            time.sleep(delay)
                        if file_response.status_code == 404:
                            print(f"  Service unavailable (404) for ATCC ID {product_id} during downloading Genome, retrying in", delay, "seconds...")
                            time.sleep(delay)
                            for retry in range(retries):
                                print(f" retry #{retry+1}")
                                file_response = requests.get(file_url)
                                if file_response.status_code == 200:
                                    break
                        if file_response.status_code == 200:
                            with open(current_directory / 'Data' / 'Genome' / 'ATCC' / filename, "wb") as f:
                                f.write(file_response.content)  # 直接写入整个文件
                            found_ATCC_ID_set.add(product_id)
                            # print(f"文件已保存到：{filename}")
                        else:
                            print(f" ATCC {product_id} 下载失败，状态码：{file_response.status_code}")
                    else:
                        print(f" ATCC {product_id} 未找到下载链接。")
                else:
                    print(f" ATCC {product_id} 下载请求失败，状态码：{download_response.status_code}")
            else:
                print(f" ATCC {product_id} 未找到基因组 ID: {product_id}。")
        else:
            print(f" ATCC {product_id} 未找到匹配的基因组。")
    else:
        print(f" ATCC {product_id} 搜索请求失败，状态码：{search_response.status_code}")

print(f'\n Found ATCC IDs length: {len(found_ATCC_ID_set)}\n{found_ATCC_ID_set}')
print(f'\n Not found ATCC IDs length: {len(ATCC_id_set - found_ATCC_ID_set -stored_ATCC_IDs)}\n{ATCC_id_set - found_ATCC_ID_set -stored_ATCC_IDs}')