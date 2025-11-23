import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from APEX_models import AMP_model
from utils import *
import pandas as pd
from fine_tune_on_DBAASP_SMILES import MultiTaskLoss, calculate_r2_per_task, R2Tracker
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import numpy as np
import wandb
import pickle
import torch.optim as optim


def load_APEX_inhouse(path):

    f = open(path, 'rb')
    inhouse_pkl = pickle.load(f)
    f.close()

    Seq_val, _, MIC_val = inhouse_pkl['val_set']
    Seq_tune, _, MIC_tune = inhouse_pkl['tune_set']

    print ('Tune data size', len(Seq_tune), np.shape(MIC_tune))
    print ('Val data size', len(Seq_val), np.shape(MIC_val))


    MIC_tune[np.where(np.isinf(MIC_tune))] = 512
    MIC_val[np.where(np.isinf(MIC_val))] = 512


    Y_mask_tune = np.ones((np.shape(MIC_tune)[0], np.shape(MIC_tune)[1]))
    Y_mask_tune[np.where(MIC_tune == -1000.0)] = 0
    MIC_tune[np.where(MIC_tune==-1000.0)] = 1000
    MIC_tune =  -np.log10(MIC_tune/float(10))


    Y_mask_val = np.ones((np.shape(MIC_val)[0], np.shape(MIC_val)[1]))
    Y_mask_val[np.where(MIC_val == -1000.0)] = 0
    MIC_val[np.where(MIC_val==-1000.0)] = 1000
    MIC_val =  -np.log10(MIC_val/float(10))

    inhouse_seq = np.concatenate((Seq_tune, Seq_val), axis=0)
    inhouse_MIC = np.concatenate((MIC_tune, MIC_val), axis=0)
    inhouse_mask = np.concatenate((Y_mask_tune, Y_mask_val), axis=0)

    return inhouse_seq, inhouse_MIC, inhouse_mask

class inhouse_APEX_Dataset(Dataset):
    def __init__(self, inhouse_seq, inhouse_MIC, inhouse_mask, max_length, word2vec):
        self.inhouse_seq = onehot_encoding(inhouse_seq, max_length, word2vec)
        self.inhouse_MIC = inhouse_MIC
        self.inhouse_mask = inhouse_mask

    def __len__(self):
        return len(self.inhouse_seq)

    def __getitem__(self, idx):
        seq = self.inhouse_seq[idx]
        MIC = self.inhouse_MIC[idx]
        mask = self.inhouse_mask[idx]
        return {
            'input_ids': seq,
            'label': MIC,
            'label_mask': mask
        }

def collate_fn_inhouse(batch):
    input_ids = [torch.from_numpy(item['input_ids']) for item in batch]
    input_ids = torch.stack(input_ids, dim=0)
    labels = [torch.from_numpy(item['label']) for item in batch]
    labels = torch.stack(labels, dim=0)
    masks = [torch.from_numpy(item['label_mask']) for item in batch]
    masks = torch.stack(masks, dim=0)

    return {
        'input_ids': input_ids,
        'label': labels,
        'label_mask': masks
    }


class AAseqsDataset(Dataset):
    def __init__(self, dataframe, max_length, word2vec):
        self.dataframe = dataframe
        self.original_length = len(self.dataframe)
        self.max_length = max_length
        self.target_columns = self.dataframe.columns.tolist()[2:]
        self.remove_long_smiles()
        # 转换成输入的格式
        self.seqs = onehot_encoding(self.dataframe['AAseqs'].tolist(), max_length, word2vec)

    def remove_long_smiles(self):
        self.dataframe = self.dataframe[self.dataframe['AAseqs'].apply(lambda x: len(x) <= self.max_length)]
        self.dataframe = self.dataframe.reset_index(drop=True)  # 重置索引
        return self.dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        AAseq = self.seqs[idx]
        DBAASP_id = self.dataframe.iloc[idx]['DBAASP_id']
        # target_columns = self.dataframe.columns.tolist()[2:]
        target = self.dataframe.loc[idx, self.target_columns].values.tolist()
        return {
            'input_ids': AAseq,  # 这个在过 onehot_encoding 的时候其实就已经 pandding 过了
            'label': torch.tensor(target, dtype=torch.float)
        }

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    # attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    # 使用 pad_sequence 填充输入
    # input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    # attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels, dim=0)
    mask = labels >= -0.5  # 生成多任务回归使用的 label mask
    labels_processed = labels.clone()  # 复制原张量以保留未满足条件的值

    # 计算实际的用来回归的值
    labels_processed[mask] = -torch.log10(labels[mask] / 10)
    mask = mask.int()

    return {
        'input_ids': torch.from_numpy(np.array(input_ids)),
        'label': labels_processed,
        'label_mask': mask
    }

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)  # Xavier初始化权重
        if module.bias is not None:
            nn.init.zeros_(module.bias)  # 偏置初始化为0
    elif isinstance(module, nn.GRU):
        for name, param in module.named_parameters():
            if 'weight_ih' in name:  # 输入到隐藏层的权重
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:  # 隐藏层到隐藏层的权重
                nn.init.orthogonal_(param)  # 正交初始化
            elif 'bias' in name:  # 偏置
                nn.init.zeros_(param)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)  # 初始化为1
        nn.init.zeros_(module.bias)  # 初始化为0

bact_names_DBAASP = ["Escherichia coli ATCC 25922", "Pseudomonas aeruginosa ATCC 27853", "Staphylococcus aureus ATCC 25923", "Staphylococcus aureus", "Staphylococcus aureus ATCC 29213", "Escherichia coli", "Pseudomonas aeruginosa", "Pseudomonas aeruginosa PAO1", "Enterococcus faecalis ATCC 29212", "Acinetobacter baumannii ATCC 19606", "Staphylococcus epidermidis ATCC 12228", "Candida albicans ATCC 10231", "Klebsiella pneumoniae ATCC 700603", "Staphylococcus aureus ATCC 43300", "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028", "Staphylococcus aureus ATCC 6538", "Pseudomonas aeruginosa ATCC 9027", "Candida albicans", "Klebsiella pneumoniae"]

max_len = 52 # maximun peptide length

word2idx, idx2word = make_vocab()
#emb = PC6('./fpScales.csv', word2idx)
#emb = FP_scale('./fpScales.csv', word2idx)
emb, AAindex_dict = AAindex('aaindex1.csv', word2idx)  # emb 从这来的，应该有这个就够了
vocab_size = len(word2idx)
emb_size = np.shape(emb)[1]

# data_path = '/home/tianang/Projects/Synergy/DataPrepare/Data/DBAASP_id_same_as_SMILES_AAseqs_bact_MICs.csv'  # 替换为你的数据路
# data_path = './DBAASP_id_same_as_SMILES_AAseqs_bact_MICs_512_limit.csv'  # 替换为你的数据路
data_path = '/data2/tianang/projects/Synergy/DataPrepare/Data/DBAASP_id_same_as_SMILES_AAseqs_bact_MICs_512_limit.csv'  # H100
data = pd.read_csv(data_path)

dataset = AAseqsDataset(data, max_len, word2idx)

# inhouse_seq, inhouse_MIC, inhouse_mask = load_APEX_inhouse('/home/tianang/Projects/APEX_train/zoonomia_APEX/inhouse_pathogen_pkl')
# inhouse_dataset = inhouse_APEX_Dataset(inhouse_seq, inhouse_MIC, inhouse_mask, max_len, word2idx)

print(dataset[0])
print(f'Current Dataset length: {len(dataset)}, Original Dataset length: {dataset.original_length}, cutting off length: {dataset.max_length}')

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 设置训练参数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 200
batch_size = 200
freeze_epochs = 5

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Synergy",
    name=f'APEX_DBAASP_AASeq_data_{num_epochs}_epochs_{batch_size}_batch_size_5_fold_mean',

    # track hyperparameters and run metadata
    config={
        "learning_rate": 1e-4,
        "architecture": "APEX",
        "dataset": data_path,
        "epochs": num_epochs,
    }
)
# best_mean_R2s = [0.45065, 0.455, 0.4183, 0.4696]
best_mean_R2s = []
for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)): # TODO
    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="Synergy",
    #     name=f'APEX_standard_data_{batch_size}batch_size',
    #
    #     # track hyperparameters and run metadata
    #     config={
    #         'fold': fold + 1,
    #         "learning_rate": 1e-4,
    #         "architecture": "APEX",
    #         "dataset": data_path,
    #         "epochs": num_epochs,
    #     }
    # )
    # if fold != 4:
    #     continue

    print(f"Fold {fold + 1}")
    train_subset = torch.utils.data.Subset(dataset, train_idx)  # TODO
    test_subset = torch.utils.data.Subset(dataset, test_idx)  # TODO

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)  # TODO
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  # TODO

    # 加载模型
    model = AMP_model(emb, emb_size, num_rnn_layers = 3, dim_h = 256, dim_latent = 256, num_fc_layers=3, num_task = 19)  # TODO
    # model.apply(init_weights)
    model.to(device)

    # 定义损失函数和优化器
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    r2_tracker = R2Tracker(num_tasks=19)  # TODO
    best_R2_score = 0.0
    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | training"):
            input_ids = batch['input_ids'].to(device)
            # attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            label_masks = batch['label_mask'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels, label_masks)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

        # 模型评估
        model.eval()
        all_labels = []
        all_preds = []
        all_label_masks = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | evaluating"):
                input_ids = batch['input_ids'].to(device)
                # attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                label_masks = batch['label_mask'].to(device)

                logits = model(input_ids)
                # probs = torch.softmax(logits, dim=1)[:, 1]  # 取正类的概率

                all_labels.extend(labels.cpu().numpy())
                # all_preds.extend(probs.cpu().numpy())
                all_preds.extend(logits.cpu().numpy())
                all_label_masks.extend(label_masks.cpu().numpy())

        # 计算 average_precision_score
        # ap_score = average_precision_score(all_labels, all_preds)
        R2_per_task = calculate_r2_per_task(all_labels, all_preds, all_label_masks)
        # 记录每个 fold 的最高 AP 分数
        # if ap_score > best_ap_score:
        #     best_ap_score = ap_score
        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, val AP Score: {R2_per_task}, best AP Score: {best_R2_score}")
        r2_tracker.update_best_r2(R2_per_task)
        R2_mean = np.array(R2_per_task).mean()
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}\nmean R2 Score: {R2_mean}\nR2 Score per task: {R2_per_task}\nbest R2 Score: {r2_tracker.get_best_r2()}\n")
        if R2_mean > best_R2_score:
            best_R2_score = R2_mean
            # torch.save(model.state_dict(), f'./compare_APEX/checkpoint/APEX_model_fold_{fold + 1}.pt')
        wandb.log({"loss": loss.item(), "R2_mean": R2_mean, "fold": fold + 1})
    best_mean_R2s.append(best_R2_score)
wandb.log({"best_mean_R2_across_folds": np.array(best_mean_R2s).mean()})
        # R2_best = r2_tracker.get_best_r2()
        # wandb.log({"epoch": epoch + 1}, commit=False)
        # wandb.log({f"{bact_names_DBAASP[i]}": R2_per_task[i] for i in range(len(R2_per_task))}, commit=False)
        # wandb.log({f"best_{bact_names_DBAASP[i]}": R2_best[i] for i in range(len(R2_best))}, commit=False)
        # wandb.log({"loss": loss.item(), "R2_mean": R2_mean, "fold": fold + 1})
        # print(
        #     f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}\nmean R2 Score: {R2_mean}\nR2 Score per task: {R2_per_task}\nbest R2 Score: {r2_tracker.get_best_r2()}\n")
    # wandb.finish()
    # break
