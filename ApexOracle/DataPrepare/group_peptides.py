import numpy as np
from huggingface_hub.utils import tqdm
import multiprocessing
import json
import pandas as pd
from skbio.alignment import StripedSmithWaterman
from skbio.sequence import SubstitutionMatrix

blosum50 = SubstitutionMatrix.by_name("BLOSUM50")

json_path = './Data/all_peptides_data.json'

# 打开并读取JSON文件
with open(json_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

AMP_id_AAseqs = []
for AMP in tqdm(data, desc='Processing DBAASP AMPs'):
    if AMP['complexity']['name'] == 'Monomer':
        AMP_id_AAseqs.append([AMP['id'], AMP['sequence']])

inhouse_data = pd.read_csv('./Data/APEX 1.1 Data.csv').values
for i, line in tqdm(enumerate(inhouse_data), desc='Processing inhouse AMPs', total=len(inhouse_data)):
    AMP_id_AAseqs.append([f'#{i}', line[0]])

AMP_id_AAseqs = np.array(AMP_id_AAseqs)
all_list = AMP_id_AAseqs[:, 1].tolist()

# 创建一个 len(all_list) x len(all_list) 的相似度矩阵，初始值为 0
simi_matrix = np.zeros((len(all_list), len(all_list)))

# 定义函数：计算每个序列和自身的比对得分
def self_align(all_list):
    all_score_dict = {}  # 用于存储每个序列与自己的比对得分
    for i in all_list:
        print(i)  # 打印当前处理的序列（用于调试）
        query = StripedSmithWaterman(i, protein=True, substitution_matrix=blosum50, score_only=True)  # 创建比对器
        alignment = query(i)  # 将序列与自身比对
        all_score_dict[i] = alignment['optimal_alignment_score']  # 记录自比对得分
    return all_score_dict  # 返回字典

# 获取所有序列的自比对得分
all_score_dict = self_align(all_list)

# 定义函数：计算第 i 个序列与所有其他序列的归一化相似度
def seq_alignment(i):
    seq = all_list[i]  # 获取第 i 个序列
    simi_vec = np.zeros(len(all_list))  # 初始化相似度向量

    query = StripedSmithWaterman(seq, protein=True, substitution_matrix=blosum50, score_only=True)  # 创建比对器
    value1 = all_score_dict[seq]  # 当前序列的自比对得分

    counter = 0  # 用于追踪 simi_vec 的索引
    for seq2 in all_list:
        value2 = all_score_dict[seq2]  # 目标序列的自比对得分
        alignment = query(seq2)  # 执行序列比对
        score = alignment['optimal_alignment_score']  # 获取原始得分
        # 归一化得分，避免被比对分数本身的大小影响，同时加上小常数防止除以 0
        normalized_score = score / (float(np.sqrt(value1) * np.sqrt(value2)) + 1e-12)
        simi_vec[counter] = normalized_score  # 将得分填入相似度向量
        counter += 1  # 移动到下一个位置

    simi_vec[i] = 1.0  # 自己与自己相似度设为 1（防止精度误差）
    return [simi_vec, i]  # 返回相似度向量和索引

# 创建索引列表 [0, 1, ..., len(all_list)-1]
index_list = np.arange(len(all_list)).tolist()

# 创建多进程池，使用 230 个进程并行执行 seq_alignment
pool = multiprocessing.Pool(processes=230)
results = pool.map(seq_alignment, index_list)  # 并行计算每个序列与所有序列的相似度
pool.close()  # 关闭进程池

# 将每个结果写入 simi_matrix 对应的行
for i in results:
    simi_vec, ind = i
    simi_matrix[ind] = simi_vec  # 将结果写入矩阵第 ind 行

