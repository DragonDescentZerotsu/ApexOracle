from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence
from sklearn.cluster import AgglomerativeClustering
from Bio import Phylo
from triton.language import bfloat16
from scipy.stats import pearsonr, spearmanr
import json
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from discription_generation_w_ATCC import description_generation

current_directory = Path(__file__).parent

def get_embedded_genome_IDs(folder_path):
    """
    检查哪些 genome ID 的genome已经被转成 Evo2 的 embedding 了
    :param folder_path: 保存 Genome 的 Evo2 embed 的文件夹路径
    :return: 不带 ATCC 的纯 ID list  | e.g. ['25332', '11060', ‘BAA-252', ...]
    """
    stored_genome_IDs = []
    genome_ID_to_species_first_name_dict = {}
    files = [f.name for f in folder_path.iterdir() if f.is_file()]
    for file_name in files:
        file_name = file_name.split('.')[0]
        file_name_temp = file_name.split('ATCC')[-1]
        components = file_name_temp.split('_')[1:]
        if len(components) == 2:
            ATCC_ID = '-'.join(components)
            stored_genome_IDs.append(ATCC_ID)  # 组装成形如 ‘BAA-252' 或者 'MYA-730'
        else:
            ATCC_ID = components[0]
            stored_genome_IDs.append(ATCC_ID)  # 就是普通的 '25922'

        genome_ID_to_species_first_name_dict[ATCC_ID] = file_name.split('_')[0]

    return stored_genome_IDs, genome_ID_to_species_first_name_dict

def get_original_strain_name_with_genome_embedding(Evo_MIC_count_file_path, embedded_genome_IDs):
    with open(Evo_MIC_count_file_path, 'r', encoding='utf-8') as f:
        strain_count_data = json.load(f)  # 解析 JSON 文件

    origin_to_standard_name_map_list_handcrafted = []  # [(original_name, standard_name (ATCC ID)), (Staphylococcus aureus ATCC 25923, 25923)...]
    origin_to_standard_name_map_list_DBAASP_original = []
    for name, count in strain_count_data.items():

        # 先处理手动标记的 strain
        if '*' in name:
            original_name, standard_name = name.split('*')
            if 'ATCC' in standard_name:
                standard_name = standard_name.split('ATCC')[-1].strip()
            else:
                # 包含那些没有 ATCC 但是单独下载了 Genome 数据的
                standard_name = standard_name.strip()
            origin_to_standard_name_map_list_handcrafted.append((original_name.strip(), standard_name))

        # 如果没有手动标记，那就只处理原始 strain 中就有 ATCC ID 的那些
        else:
            if 'ATCC' in name:
                original_name = name
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

                origin_to_standard_name_map_list_DBAASP_original.append((original_name.strip(), ATCC_id))

    origin_to_standard_name_map_list = np.array(origin_to_standard_name_map_list_handcrafted + origin_to_standard_name_map_list_DBAASP_original)

    original_names_with_genome_embedding_handcrafted = []  # 提取出那些有对应 Evo2 embedding 的 DBAASP 中的完整 strain name
    for line_idx, (original_name, standard_name) in enumerate(origin_to_standard_name_map_list_handcrafted):
        # 检查这些 ATCC ID 是不是已经在有 Evo2 embedding 的 strain 里
        if standard_name in embedded_genome_IDs:
            original_names_with_genome_embedding_handcrafted.append(original_name)

    original_names_with_genome_embedding_DBAASP_original = []  # 提取出那些有对应 Evo2 embedding 的 DBAASP 中的完整 strain name
    for line_idx, (original_name, standard_name) in enumerate(origin_to_standard_name_map_list_DBAASP_original):
        # 检查这些 ATCC ID 是不是已经在有 Evo2 embedding 的 strain 里
        if standard_name in embedded_genome_IDs:
            original_names_with_genome_embedding_DBAASP_original.append(original_name)

    return original_names_with_genome_embedding_handcrafted, original_names_with_genome_embedding_DBAASP_original, dict(origin_to_standard_name_map_list)

def load_all_genome_embeddings(embeddings_folder_path, scale, device):
    """
    返回一个 genome ID 到 Evo2 embedding 字典
    :param embeddings_folder_path: 保存 Genome 的 Evo2 embed 的文件夹路径
    :param scale: Evo2 的 embedding 量级大概在 1e-15 左右，和模型参数 1e-2 左右的量级差太多了，所以需要缩放匹配
    :param device: 提前将所有的 Evo2 embedding 载入到显存之中，减少加载时间
    :return: dict  e.g. {'25922': torch.tensor([...], dtype=torch.bfloat16), ...}
    """
    file_paths = [embeddings_folder_path / f.name for f in embeddings_folder_path.iterdir() if f.is_file()]
    embeddings_dict = {}
    for file_path in tqdm(file_paths, desc=' loading embeddings ... '):
        embedding = torch.load(file_path).to(device)
        file_name = file_path.name.split('.')[0]
        if 'ATCC' in file_name:
            file_name = file_name.split('ATCC')[-1]
            components = file_name.split('_')[1:]
            if len(components) == 2:
                ID = '-'.join(components)
            else:
                ID = components[0]
        else:
            # 自己下载的情况
            ID = file_name
        embeddings_dict[ID] = embedding * scale

    return embeddings_dict

def exclude_wrong_species_ATCC_map(Evo_MIC_data_with_genome_embedding:np.array, genome_ID_to_species_first_name_dict):
    """
    去掉那些原始 DBAASP 中连 species name 和 ATCC ID 都对不上的数据，只处理那些没有手动标注的！
    :param Evo_MIC_data_with_genome_embedding: SMIELS, strain -> MIC data
    :param genome_ID_to_species_first_name_dict: dict, {ATCC_ID: species_name }, 这个是直接从 保存的 ATCC genome embedding 文件名获得的
    :return: cleaned SMIELS, strain -> MIC data, np.array
    """
    # 记录一下清理之前有多少数据点
    original_length = len(Evo_MIC_data_with_genome_embedding)

    marked_ATCC_IDs = set()
    cleaned_data = []
    for line in Evo_MIC_data_with_genome_embedding:
        name = line[1]

        # 那些没有 ATCC ID 但是一定被手动标注了的情况
        if 'ATCC' not in name:
            cleaned_data.append(line)
            continue

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

        # 手动标记过 ATCC 的情况
        if genome_ID_to_species_first_name_dict.get(ATCC_id) is None:
            cleaned_data.append(line)
            marked_ATCC_IDs.add(ATCC_id)

        # 如果 species name 符合，那么是干净的数据
        elif genome_ID_to_species_first_name_dict[ATCC_id] in name:
            cleaned_data.append(line)

    cleaned_data = np.array(cleaned_data)

    wrong_ATCC_numbers = set(Evo_MIC_data_with_genome_embedding[:, 1]) - set(cleaned_data[:, 1])

    print(f'\n wrong strain names: {wrong_ATCC_numbers}')
    print(f'\n double marked_ATCC_IDs: {marked_ATCC_IDs}')

    print(f'\n original data length (no "*", no manual modification) {original_length}\n cleaned data length {len(cleaned_data)}\n')

    return cleaned_data

def get_ATCC_ID_to_species_name_map(ATCC_fasta_folder_path:Path):
    file_names = [f.name for f in ATCC_fasta_folder_path.iterdir() if f.is_file()]

    # ATCC_ID_to_species_names_map = {}

    ATCC_ID_list = []
    species_name_list = []

    for file_name in file_names:

        # 先获得这个 ATCC genome fasta 文件的 ATCC ID
        ATCC_id = file_name.split('.')[0].split('ATCC')[-1].strip()
        ATCC_id = ATCC_id.replace("_", " ").strip().replace(" ", "-")
        ATCC_ID_list.append(ATCC_id)

        # 然后获得这个 ATCC genome fasta 文件的 species name
        file_name = file_name.split('ATCC')[0]
        if 'subsp' in file_name.split('_'):
            file_name = file_name.split('subsp')[0]
        if 'pathovar' in file_name.split('_'):
            file_name = file_name.split('pathovar')[0]  # 带有 pathovar 和 var 的在 NCBI Taxonomy Browser 中都是识别不到的
        if 'var' in file_name.split('_'):
            file_name = file_name.split('var')[0]
        if 'sp' in file_name.split('_'):
            file_name = file_name.split('_sp')[0]
        species_name = file_name.replace('_', ' ').strip()
        species_name_list.append(species_name)

        # 存进这个 map 字典里
        # ATCC_ID_to_species_names_map[ATCC_id] = species_name

    ATCC_ID_to_species_names_map = dict(zip(ATCC_ID_list, species_name_list))
    species_names_to_ATCC_ID_map = {}

    ATCC_ID_list = np.array(ATCC_ID_list)
    species_name_list = np.array(species_name_list)

    for species_name in set(species_name_list):
        species_names_to_ATCC_ID_map[species_name] = ATCC_ID_list[species_name_list == species_name]

    return ATCC_ID_to_species_names_map, species_names_to_ATCC_ID_map

def judge_text_gen(client, strain_name):

    completion = client.chat.completions.create(
        model="qwen-max-0125",
        # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a scientist working on Antibiotic resistance.'},
            {'role': 'user',
             'content': f"""Your task is to verify the existence of the bacterial strain provided below.
             
             Strain Name: {strain_name}

        Important Instructions:**

1. Base your conclusion strictly on current scientific knowledge and credible sources. Provide relevant references clearly supporting your decision.
2. If the strain doesn’t include an ATCC identifier, carefully confirm whether there are detailed scientific evidence (e.g., peer-reviewed papers or reputable resources) is found describing specific characteristics of this strain—such as:
    
    1. Unique Mutations: Any distinctive genetic mutations identified in this strain compared to the wild-type strain of the same species, particularly those affecting virulence factors, metabolic pathways, or the plasma membrane.
    2. Antibiotics and antimicrobial peptides Resistance: Any known antibiotics and antimicrobial peptides resistance mechanisms associated with this strain, including specific resistance genes or mutations.
    3. Antibiotics and antimicrobial peptides Sensitivity: Identify antibiotics and antimicrobial peptides to which this strain is known to be sensitive.
    
3. If the strain includes an ATCC identifier, carefully confirm whether this specific ATCC ID exists in the official ATCC database.
    •	If the ATCC ID cannot be confirmed, classify the strain as “Not Exist”
4. Exception to Rule 3:
    If scientific evidence as described in Instruction 2 is found—even if the ATCC ID cannot be confirmed—you may still conclude the strain as “Exist.”
5. If there isn't an identifier in the strain name, classify the strain as “Not Exist”

Response Format:
Respond strictly following this format:
**Explanation:** (A very concise but clear justification supporting your conclusion regarding the existence of the strain.)
**Conclusion:** (Choose only one: **“Exist”** or **“Not Exist”**)"""}],
        extra_body={
            "enable_search": True
        }
    )

    description = completion.choices[0].message.content.strip()

    return description

def get_strains_w_judge_text(judge_text_dir:Path):
    file_names = [f.name for f in judge_text_dir.iterdir() if f.is_file()]
    file_names = [name.split('.txt')[0].replace('～', ' ').replace('^', '/') for name in file_names]
    return file_names

def load_all_judge_text(judge_text_dir:Path):
    """
    找到那些被 judge 判断为存在的 strain
    :param judge_text_dir:
    :return:
    """
    file_names = [f.name for f in judge_text_dir.iterdir() if f.is_file()]
    # file_names = [name.split('.txt')[0].replace('～', ' ').replace('^', '/') for name in file_names]
    strain_name_exist = []
    for file_name in tqdm(file_names, desc=' Loading Judge Text'):
        strain_name = file_name.split('.txt')[0].replace('～', ' ').replace('^', '/')
        with open(judge_text_dir / file_name, 'r') as f:
            judge_text = f.read()
            if '**Conclusion:**' in judge_text:
                conclusion_txt = judge_text.split('**Conclusion:**')[1]
                if 'Not' not in conclusion_txt:
                    strain_name_exist.append(strain_name)
            else:
                print(f' {file_name} wrong content: {judge_text}')
            # file_name_judge_text_pair.append((file_name, judge_text))

    print(f' Judge text loaded.\n Num of strains with scientific literature: {len(strain_name_exist)}')
    return strain_name_exist

def get_wo_ATCC_strain_names(wo_ATCC_text_path:Path):

    text_files = [f.name for f in wo_ATCC_text_path.iterdir() if f.is_file()]
    text_strain_names = [file.split('.')[0].replace('～', ' ').replace('^', '/') for file in text_files]

    # ATCC_strain_names 用于 生成 prompt
    return text_strain_names


embeddings_folder_path = current_directory / 'Data' / 'Genome_embs'

embedded_genome_IDs, genome_ID_to_species_first_name_dict = get_embedded_genome_IDs(embeddings_folder_path)
Evo_MIC_count_file_path = current_directory / 'Data' / 'Evo_edition_4_MIC_data_handcrafted_no_ATCC_to_custom_ATCC_and_inhouse.json'

# original_names_with_genome_embedding: 那些有对应 Evo2 embedding 的 DNAASP 中的完整 strain name
original_names_with_genome_embedding_handcrafted, original_names_with_genome_embedding_DBAASP_original, origin_to_standard_name_map_dict = get_original_strain_name_with_genome_embedding(Evo_MIC_count_file_path, embedded_genome_IDs)

# 读取原始 DBAASP 中那些有 MIC 的数据
Evo_strain_MIC_data_path = current_directory / 'Data' / 'DBAASP_inhouse_AMP_SMILES_MIC_Evo.csv'
all_Evo_MIC_data = pd.read_csv(Evo_strain_MIC_data_path)
columns_names = all_Evo_MIC_data.columns
all_Evo_MIC_data = all_Evo_MIC_data.values

# 去掉那些带 'del' 的
# del_excluded_data = []
# for MIC_data_line in tqdm(all_Evo_MIC_data, desc=' removing MIC data with "del" in name '):
#     if 'del' not in MIC_data_line[1]:
#         del_excluded_data.append(MIC_data_line)
# all_Evo_MIC_data = del_excluded_data

# filter 一下留下那些没有对应 strain 的 genome 的 SMILES -> MIC 对数据
Evo_MIC_data_wo_genome_embedding = []
for MIC_data_line in tqdm(all_Evo_MIC_data, desc=' retriving MIC data with out genome embeddings '):
    if MIC_data_line[1] not in original_names_with_genome_embedding_handcrafted and MIC_data_line[1] not in original_names_with_genome_embedding_DBAASP_original:
        Evo_MIC_data_wo_genome_embedding.append(MIC_data_line)

unique_strains, count = np.unique(np.array(Evo_MIC_data_wo_genome_embedding)[:, 1], return_counts=True)
sorted_indices = np.argsort(count)[::-1]

# Use the sorted indices to sort unique_elements and counts
unique_elements_sorted = unique_strains[sorted_indices]
counts_sorted = count[sorted_indices]

#TODO: 1. 写一个函数来 judge 某个 strain 是否是真实存在的，可以复用很多之前的 description 生成代码
#      2. 生成的 judge 的文本都存下来，然后每一个读取判断是否存在

# np.savetxt(current_directory / 'Data' / "original_names_in_DBAASP_wo_ATCC.txt", unique_elements_sorted, fmt='%s')

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key='sk-325b8ad3d9624c778c31a34245ea202b',  # TODO: 记得清除 API key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
)

judge_text_dir = current_directory / 'Data' / 'Text_Description' / 'wo_ATCC' / 'judge_exist_text'
names_w_judge_txt = get_strains_w_judge_text(judge_text_dir)
names_wo_judge_txt = set(unique_elements_sorted) - set(names_w_judge_txt) #-set(['Bacteroides sp.', 'Psychrobacter sp.', 'Flavobacterium sp.', 'Streptococcus sp.', 'Bacillus sp.', 'Pseudomonas sp.'])

for strain_name in tqdm(names_wo_judge_txt, desc=' Generating judgement text'):
    judge_text = judge_text_gen(client, strain_name)
    text_file_save_name = strain_name.replace(' ', '～').replace('/', '^') + '.txt'
    save_path = judge_text_dir / text_file_save_name
    with open(save_path, mode='w', encoding='utf-8') as text_file:
        text_file.write(judge_text)

# if len(names_wo_judge_txt) == 0:
strain_name_exist = load_all_judge_text(judge_text_dir)
indices = [np.where(unique_elements_sorted == val)[0][0] for val in strain_name_exist]
all_count = counts_sorted[indices].sum()
print(f' Number of data ponints with text description but no genome data: {all_count}')

wo_genome_w_text_strains = get_wo_ATCC_strain_names(current_directory / 'Data' / 'Text_Description' / 'wo_ATCC' / 'Text')

strain_name_exist_description_not_generated = set(strain_name_exist) - set(wo_genome_w_text_strains)

for prompt_strain_name in tqdm(strain_name_exist_description_not_generated, desc=' Generating description '):
    description = description_generation(client, prompt_strain_name)
    if description is None:
        print(f'\n {prompt_strain_name} failed to generate description.')
    else:
        text_file_save_name = prompt_strain_name.replace(' ', '～').replace('/', '^') + '.txt'
        save_path = current_directory / 'Data' / 'Text_Description' / 'wo_ATCC' / 'Text' / text_file_save_name
        with open(save_path, mode='w', encoding='utf-8') as text_file:
            text_file.write(description)


# def process_strain(strain_name, client, judge_text_dir):
#     # Generate judgement text and write it to a file
#     judge_text = judge_text_gen(client, strain_name)
#     text_file_save_name = strain_name.replace(' ', '～').replace('/', '^') + '.txt'
#     save_path = judge_text_dir / text_file_save_name
#     with open(save_path, mode='w', encoding='utf-8') as text_file:
#         text_file.write(judge_text)
#
# # Adjust the number of workers based on your workload and system capabilities.
# num_workers = 10
#
# # Using partial to include constant parameters (client and judge_text_dir)
# process_fn = partial(process_strain, client=client, judge_text_dir=judge_text_dir)
#
# # Create a thread pool and submit the tasks
# with ThreadPoolExecutor(max_workers=num_workers) as executor:
#     futures = [executor.submit(process_fn, strain_name) for strain_name in names_wo_judge_txt]
#
#     # Use tqdm to display a progress bar as futures complete
#     for future in tqdm(as_completed(futures), total=len(futures), desc=' Generating judgement text'):
#         # Optionally, handle exceptions here if needed
#         future.result()

# print(1)
# Evo_MIC_data_with_genome_embedding_DBAASP_origianl = []
# for MIC_data_line in tqdm(all_Evo_MIC_data, desc=' retriving MIC data with genome embeddings '):
#     if MIC_data_line[1] in original_names_with_genome_embedding_DBAASP_original:
#         Evo_MIC_data_with_genome_embedding_DBAASP_origianl.append(MIC_data_line)