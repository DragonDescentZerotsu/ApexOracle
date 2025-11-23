# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

current_dir = Path(__file__).parent

def get_all_text(text_folder_path:Path, text_embed_fold_path:Path):
    text_file_names = set([file.name.split('.txt')[0] for file in text_folder_path.iterdir() if file.is_file()])
    embeded_files = set([f.name.split('.txt')[0] for f in text_embed_fold_path.iterdir() if f.is_file()])

    # 只处理那些没有被 embedding 过的文件
    non_embeded_files = list(text_file_names - embeded_files)

    file_name_text_content_pair = []

    for file_name in non_embeded_files:
        text_file_path = text_folder_path / (file_name + '.txt')
        with open(text_file_path, 'r', encoding='utf-8') as text_file:
            text = text_file.read()

            text = text.replace(file_name.replace('～', ' ').replace('^', '/').replace('subsp', 'subsp.'), 'This strain')
            text = text.replace(file_name.replace('～', ' ').replace('^', '/'), 'This strain')
            # ATCC_ID = file_name.replace('_', ' ').split('ATCC')[-1].strip()
            # if ATCC_ID in text:
            #     print(f'\n {file_name}: replacement could be wrong, please check:\n {text}')

            # 检查可能出错的没有内容的 QWen API 返回的文本
            if len(text) <= 4:
                print(f' {file_name} is too short, could be wrong:\n {text}')
            file_name_text_content_pair.append([file_name, text])

    return np.array(file_name_text_content_pair)

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

ATCC_text_dir_path = current_dir / 'Data' / 'Text_Description' / 'wo_ATCC' / 'Text' # current_dir / 'Data' / 'Text_Description' / 'ATCC' / 'Text'
ATCC_text_embed_dir_path = current_dir / 'Data' / 'Text_Description' / 'wo_ATCC' / 'embeddings' # current_dir / 'Data' / 'Text_Description' / 'ATCC' / 'embeddings'
file_name_text_content_pair = get_all_text(ATCC_text_dir_path, ATCC_text_embed_dir_path)

tokenizer = AutoTokenizer.from_pretrained("YBXL/Med-LLaMA3-8B")
model = AutoModelForCausalLM.from_pretrained("YBXL/Med-LLaMA3-8B").to(device)

for file_name, text in tqdm(file_name_text_content_pair, desc=' Embedding text descriptions ... '):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    hidden_states = model(input_ids, output_hidden_states=True).hidden_states[-2].squeeze()
    torch.save(hidden_states.cpu().detach(), ATCC_text_embed_dir_path / (file_name + '.pt'))
