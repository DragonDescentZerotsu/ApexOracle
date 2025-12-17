import os
from openai import OpenAI
from pathlib import Path
import numpy as np
import copy
from tqdm import tqdm

current_directory = Path(__file__).parent
dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")

if not dashscope_api_key:
    raise EnvironmentError("DASHSCOPE_API_KEY environment variable is required.")

client = OpenAI(
    api_key=dashscope_api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
)

def description_generation(client, strain_name):

    completion = client.chat.completions.create(
        model="qwen-max-0125",  # 此处以qwen-plus为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a scientist working on Antibiotic resistance.'},
            {'role': 'user', 'content': f"""Your task is to provide a concise and informative description of the bacterial strain {strain_name}, including:
    
    1. Species Information: Identify the species to which this strain belongs, specifying whether it is Gram-positive, Gram-negative, Fungi, Archaea, or Protozoa. Describe its notable physiological traits.
    2. Unique Mutations: Describe any distinctive genetic mutations identified in this strain compared to the wild-type strain of the same species, particularly those affecting virulence factors, metabolic pathways, or the plasma membrane. Explain how these mutations modify the bacterium’s behavior, physiology, or pathogenicity, and how they may contribute to the development of antimicrobial resistance.
    3. Antibiotics and antimicrobial peptides Resistance: Outline any known antibiotics and antimicrobial peptides resistance mechanisms associated with this strain, including specific resistance genes or mutations. Describe the molecular mechanisms by which these confer resistance to particular antibiotics. Make sure to include the corresponding MIC value if you can find it. If you can't find corresponding MIC and molecular mechanisms, don't mention anything about that and keep your response as concise as you can.
    4. Antibiotics and antimicrobial peptides Sensitivity: Identify antibiotics and antimicrobial peptides to which this strain is known to be sensitive. Explain the mechanisms by which these antibiotics exert their effects on the strain. Make sure to include the corresponding MIC value if you can find it. If you can't find corresponding MIC and molecular mechanisms, don't mention anything about that and keep your response as concise as you can.
    In your response, do not provide a summary; instead, list the information as separate points as follows:
    Species Information: ...
    Unique Mutations: ...
    Antibiotic and antimicrobial peptides Resistance: ...
    Antibiotic and antimicrobial peptides Sensitivity: ...
    References: ... (This is just a place holder section, you MUST Ensure that the description is based on current scientific knowledge and has a reference but don't include this section in your response)
    
    Important Instructions:
    1. Ensure that the description is based on current scientific knowledge and includes relevant references where applicable, do not insert references before the Reference section.
    2. Although the strain ID I provided is from ATCC, the same strain may also be cataloged under different identifiers in other databases, such as DSM, KCTC, NCTC, JCM, or other unique numbering systems beyond these examples. Please make sure to cross-reference these alternative ID systems when searching for relevant information.
    3. Do not include any bulletin points in your response.
    4. If information about the strain is unavailable or cannot be found, JUST respond with ‘None’ in the corresponding section! Do not respond with any further explanation!"""}],
        extra_body={
            "enable_search": True
        }
        )

    description = completion.choices[0].message.content.strip()

    return description

def get_ATCC_strain_names(ATCC_genome_path:Path, ATCC_text_path:Path):
    files = [f.name for f in ATCC_genome_path.iterdir() if f.is_file()]
    ATCC_strain_names = [file.split('.')[0].replace('_', ' ') for file in files]

    text_files = [f.name for f in ATCC_text_path.iterdir() if f.is_file()]
    text_ATCC_strain_names = [file.split('.')[0].replace('_', ' ') for file in text_files]

    ATCC_strain_names = list(set(ATCC_strain_names) - set(text_ATCC_strain_names))

    # 处理那些从 NCBI 下载但是被处理成自定义 ATCC ID 的 strain name
    custom_ATCC_to_original_NCBI_map = {
        '#001': 'Escherichia coli K88',
        '#002': 'Pseudomonas aeruginosa PA14',
        '#003': 'Escherichia coli DH5alpha',
        '#004': 'Escherichia coli BW25113',
        '#005': 'Escherichia coli BL21AI'
    }

    # 记录那些需要被上面 custom_ATCC_to_original_NCBI_map 替换的行
    replace_list = []
    for line, ATCC_strain_name in enumerate(ATCC_strain_names):
        if ATCC_strain_name.split(' ')[-1] in custom_ATCC_to_original_NCBI_map.keys():
            replace_list.append((line, custom_ATCC_to_original_NCBI_map[ATCC_strain_name.split(' ')[-1]]))



    strain_names_to_save = copy.deepcopy(ATCC_strain_names)

    # 替换掉 ATCC_strain_names 中
    for replace_line in replace_list:
        ATCC_strain_names[replace_line[0]] = replace_line[1]

    # ATCC_strain_names 用于 生成 prompt
    return ATCC_strain_names, strain_names_to_save



ATCC_genome_path = current_directory/'Data'/'Genome'/'ATCC'
ATCC_text_path = current_directory/'Data'/'Text_Description'/'ATCC'/'Text'
strain_names_for_prompt, strain_names_to_save = get_ATCC_strain_names(ATCC_genome_path, ATCC_text_path)

for prompt_strain_name, save_strain_name in tqdm(zip(strain_names_for_prompt, strain_names_to_save), desc=' Generating Description Text... ', total=len(strain_names_for_prompt)):
    description = description_generation(client, prompt_strain_name)
    if description is None:
        print(f'\n {prompt_strain_name} failed to generate description.')
    else:
        text_file_save_name = save_strain_name.replace(' ', '_') + '.txt'
        save_path = ATCC_text_path / text_file_save_name
        with open(save_path, mode='w', encoding='utf-8') as text_file:
            text_file.write(description)

# description_generation(client, 'Mycobacterium smegmatis ATCC 700084')