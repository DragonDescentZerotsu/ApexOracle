import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
import wandb
from tqdm import tqdm
from pathlib import Path


# 自定义 PyTorch Dataset
class SMILESDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.dataframe = dataframe
        self.original_length = len(self.dataframe)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_columns = self.dataframe.columns.tolist()[2:]
        self.remove_long_smiles()

    def remove_long_smiles(self):
        self.dataframe = self.dataframe[self.dataframe['SMILES'].apply(lambda x: len(self.tokenizer(x, return_tensors='pt', padding=False, truncation=False)['input_ids'].squeeze(0)) <= self.max_length)]
        self.dataframe = self.dataframe.reset_index(drop=True)  # 重置索引
        # self.dataframe.to_csv('/home/tianang/Projects/Synergy/DataPrepare/Data/DBAASP_id_SMILES_bact_MICs_512_limit.csv', index=False)
        # print(f'new data file saved to /home/tianang/Projects/Synergy/DataPrepare/Data/DBAASP_id_SMILES_bact_MICs_512_limit.csv')
        return self.dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        smiles = self.dataframe.iloc[idx]['SMILES']
        DBAASP_id = self.dataframe.iloc[idx]['DBAASP_id']
        # target_columns = self.dataframe.columns.tolist()[2:]
        target = self.dataframe.loc[idx, self.target_columns].values.tolist()
        inputs = self.tokenizer(smiles, return_tensors='pt', padding=False, truncation=False)  #, max_length=self.max_length)
        inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # 去掉 batch 维度
        return {
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'label': torch.tensor(target, dtype=torch.float)
        }

    # @staticmethod

class synthetic_dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.origin_data = dataframe.values
        self.original_length = len(self.origin_data)
        self.reordered_synergy_smiles = []
        self.reordered_antibiotic_id_or_name = []
        self.tokenizer = tokenizer
        self.reordered_synergy_labels = []
        self.max_length = max_length
        self.input_ids = []
        self.attention_mask = []

        self.reorder_synergistic_data()
        self._tokennize()

    def reorder_synergistic_data(self):
        """
        把 smiles 比较数据中 peptide lose 和 peptide win 排序，peptide lose 永远在 peptide win 之前
        :return:
        """
        for line in tqdm(self.origin_data, desc=' reordering synergy dataset to pairs'):
            mic_1 = line[2]
            mic_2 = line[3]
            id_or_name_1 = line[0]
            id_or_name_2 = line[1]
            label = line[-1]

            self.reordered_antibiotic_id_or_name.extend([id_or_name_1, id_or_name_2])
            self.reordered_synergy_smiles.extend([mic_1, mic_2])
            self.reordered_synergy_labels.extend([label, label])

    def _tokennize(self):
        smiles_indices_to_remove = set()
        for idx, smiles in tqdm(enumerate(self.reordered_synergy_smiles), desc=' Tokenizing SMILES...', total=len(self.reordered_synergy_smiles)):
            inputs = self.tokenizer(smiles, return_tensors='pt', padding=False, truncation=False, return_offsets_mapping=True)

            # 如果输入的长度太长了，就直接跳过，并且记录哪些smiles应该从数据集中删除
            if len(inputs['input_ids'][0]) > self.max_length:
                smiles_indices_to_remove.add(idx)
                if idx % 2 == 0:
                    # 如果
                    smiles_indices_to_remove.add(idx + 1)
                else:
                    smiles_indices_to_remove.add(idx - 1)

            inputs = {key: val.squeeze(0) for key, val in inputs.items()}

            self.input_ids.append(inputs['input_ids'])
            self.attention_mask.append(inputs['attention_mask'])

        self.reordered_synergy_smiles = np.delete(np.stack(self.reordered_synergy_smiles), list(smiles_indices_to_remove))
        self.reordered_synergy_labels = np.delete(np.stack(self.reordered_synergy_labels), list(smiles_indices_to_remove))
        self.attention_mask = [mask for idx, mask in enumerate(self.attention_mask) if idx not in smiles_indices_to_remove]
        self.input_ids = [input_id for idx, input_id in enumerate(self.input_ids) if idx not in smiles_indices_to_remove]
        self.reordered_antibiotic_id_or_name = [DBAASP_id for idx, DBAASP_id in enumerate(self.reordered_antibiotic_id_or_name) if idx not in smiles_indices_to_remove]

    def __len__(self):
        # 实际只有 input_ids 长度一半那么多的 pair
        return len(self.input_ids) // 2

    def __getitem__(self, idx):
        return {
            'input_ids_1': self.input_ids[idx * 2],
            'input_ids_2': self.input_ids[idx * 2 + 1],
            'attention_mask_1': self.attention_mask[idx * 2],
            'attention_mask_2': self.attention_mask[idx * 2 + 1],
            'label': self.reordered_synergy_labels[idx * 2],
        }

def collate_fn_synergy(batch):
    """
    成对地整理 batch 中的 synergy data
    """
    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        input_ids.extend([item['input_ids_1'], item['input_ids_2']])
        attention_mask.extend([item['attention_mask_1'], item['attention_mask_2']])
        labels.append(item['label'])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': torch.from_numpy(np.array(labels))
    }

# 定义分类模型
class ClassificationModel(nn.Module):
    def __init__(self, model_name, num_labels):
        super(ClassificationModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)  # hidden_size = 768
        print(self.bert.config.max_position_embeddings)
        self.classifier = RegressionHead(self.bert.config.hidden_size, num_targets=num_labels)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # 提取 <cls> token 嵌入
        logits = self.classifier(cls_embedding)
        return logits

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes,
        pooler_dropout: float=0.2,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class RegressionHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        hidden_dim_1 = 384,
        hidden_dim_2 = 128,
        num_targets = 19,
        pooler_dropout: float=0.2,
    ):
        """
        Initialize the classification head.

        :param input_dim: Dimension of input features.
        :param inner_dim: Dimension of the inner layer.
        :param num_classes: Number of classes for classification.
        :param activation_fn: Activation function name.
        :param pooler_dropout: Dropout rate for the pooling layer.
        """
        super().__init__()
        self.dense_1 = nn.Linear(input_dim, hidden_dim_1)
        self.dense_2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.activation_fn = nn.GELU()
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(hidden_dim_2, num_targets)

    def forward(self, features, **kwargs):
        """
        Forward pass for the classification head.

        :param features: Input features for classification.

        :return: Output from the classification head.
        """
        x = self.dense_1(features)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.dense_2(x)
        x = self.activation_fn(x)
        x = self.dropout(x)

        x = self.out_proj(x)
        return x


class FirstTokenAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(FirstTokenAttention, self).__init__()
        # 多头注意力层
        self.key_projection = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        # 残差和归一化（LayerNorm）
        self.norm1 = nn.LayerNorm(embed_dim)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, key_padding_mask):
        """
        x: Tensor, shape = (batch_size, seq_len, embed_dim)
        """
        # 提取序列的第一个 token，作为 query，形状: (batch_size, 1, embed_dim)
        query = x[:, 0:1, :]
        # nn.MultiheadAttention 要求输入 shape 为 (seq_len, batch_size, embed_dim)
        query = query.transpose(0, 1)  # (1, batch_size, embed_dim)
        key = self.key_projection(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        key = key.transpose(0, 1)  # (seq_len, batch_size, embed_dim)
        value = key

        # 计算多头注意力：只计算第一个 token 对整个序列的注意力
        attn_output, attn_weights = self.mha(query, key, value, key_padding_mask = key_padding_mask.to(torch.bool))  # (1, batch_size, embed_dim)
        # 残差连接与归一化
        query = self.norm1(query + attn_output)

        # 前馈网络 + 残差连接和归一化
        ffn_output = self.ffn(query)
        query = self.norm2(query + ffn_output)

        # 最终只输出更新后的第一个 token embedding，返回形状 (batch_size, embed_dim)
        return query.squeeze(0)

class MolMergeFFN(nn.Module):
    def __init__(self, embed_dim):
        super(MolMergeFFN, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(approximate='tanh')
        )

    def forward(self, x):
        return self.ffn(x)

def collate_fn(batch):
    """
    这里把一个batch中所有的label都转换成 log 计算之后的
    """
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['label'] for item in batch]

    # 使用 pad_sequence 填充输入
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.stack(labels, dim=0)
    mask = labels >= -0.5  # 生成多任务回归使用的 label mask
    labels_processed = labels.clone()  # 复制原张量以保留未满足条件的值

    # 计算实际的用来回归的值
    labels_processed[mask] = -torch.log10(labels[mask] / 10)
    mask = mask.int()

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels_processed,
        'label_mask': mask
    }

class MultiTaskLoss(nn.Module):
    def __init__(self, reduction='mean', device=torch.device('cpu')):
        super(MultiTaskLoss, self).__init__()
        self.reduction = reduction
        self.device = device

    def forward(self, y_pred, y_true, mask, mean_weight = 1):
        """
        Args:
            y_pred: [batch_size, num_tasks] 模型预测值
            y_true: [batch_size, num_tasks] 真实值
            mask:   [batch_size, num_tasks] 掩码 (1 表示计算损失，0 表示忽略)
            mean_weight: 对最后一维 mean 的 loss 施加的权重
        Returns:
            loss:   单一标量表示的损失
        """
        # 计算 MSE 损失（或其他损失）
        loss = (y_pred - y_true) ** 2
        # weight_mask = torch.ones_like(loss, device=self.device)
        # weight_mask[:, -1] = mean_weight
        # loss = loss * weight_mask

        # 应用掩码
        masked_loss = loss * mask

        # 根据 reduction 方式计算最终损失
        if self.reduction == 'mean':
            # 避免掩码全为 0 的情况，计算有效元素的均值
            return masked_loss.sum() / (mask.sum() + 1e-8)
        elif self.reduction == 'sum':
            return masked_loss.sum()
        else:  # 'none'
            return masked_loss

def calculate_r2_per_task(all_labels, all_preds, all_label_masks):
    """
    计算每个任务的 R^2 值

    Args:
        all_labels (np.array): 实际值数组，形状为 [batch_size, num_tasks]
        all_preds (np.array): 预测值数组，形状为 [batch_size, num_tasks]
        all_label_masks (np.array): 掩码矩阵，形状为 [batch_size, num_tasks]

    Returns:
        list: 每个任务的 R^2 值，任务无效时为 None
    """
    # 确保输入是 numpy 数组
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_label_masks = np.array(all_label_masks)

    num_tasks = all_labels.shape[1]  # 任务数量
    r2_per_task = []

    for task_idx in range(num_tasks):
        # 获取当前任务的有效掩码
        mask = all_label_masks[:, task_idx].astype(bool)

        # 筛选有效的标签和预测值
        y_true = all_labels[mask, task_idx]
        y_pred = all_preds[mask, task_idx]

        # 如果有效样本数不足，则返回 None
        if len(y_true) == 0:
            r2_per_task.append(None)
            continue

        # 计算 R^2
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)  # 总平方和
        ss_residual = np.sum((y_true - y_pred) ** 2)  # 残差平方和
        r2 = 1 - (ss_residual / ss_total)

        r2_per_task.append(r2)

    return r2_per_task

class R2Tracker:
    def __init__(self, num_tasks):
        self.best_r2_per_task = [None] * num_tasks  # 初始化最佳 R² 为 None

    def update_best_r2(self, current_r2_per_task):
        """
        更新最佳 R² 值
        Args:
            current_r2_per_task (list): 当前批次的每个任务 R² 值
        """
        for task_idx, current_r2 in enumerate(current_r2_per_task):
            if current_r2 is not None:  # 仅更新有效任务
                if self.best_r2_per_task[task_idx] is None or current_r2 > self.best_r2_per_task[task_idx]:
                    self.best_r2_per_task[task_idx] = current_r2

    def get_best_r2(self):
        """
        获取每个任务的最佳 R²
        Returns:
            list: 每个任务的最佳 R² 值
        """
        return self.best_r2_per_task


if __name__=='__main__':
    current_folder = Path(__file__).parent
    # 读取 CSV 数据
    # data_path = '/home/tianang/Projects/Synergy/DataPrepare/Data/DBAASP_id_same_as_AAseqs_SMILES_bact_MICs.csv'  # 替换为你的数据路径
    # data_path = current_folder/'DataPrepare'/'Data'/'DBAASP_id_SMILES_bact_mean_MICs.csv'  # 替换为你的数据路径
    data_path = current_folder / 'DataPrepare' / 'Data' / 'synergistic_pairs.csv'  # 替换为你的数据路径
    data = pd.read_csv(data_path)


    # 加载分词器和定义数据集
    # model_name = "seyonec/ChemBERTa_zinc250k_v2_40k"
    bact_names_DBAASP = ["Escherichia coli ATCC 25922", "Pseudomonas aeruginosa ATCC 27853",
                         "Staphylococcus aureus ATCC 25923", "Staphylococcus aureus",
                         "Staphylococcus aureus ATCC 29213", "Escherichia coli", "Pseudomonas aeruginosa",
                         "Pseudomonas aeruginosa PAO1", "Enterococcus faecalis ATCC 29212",
                         "Acinetobacter baumannii ATCC 19606", "Staphylococcus epidermidis ATCC 12228",
                         "Candida albicans ATCC 10231", "Klebsiella pneumoniae ATCC 700603",
                         "Staphylococcus aureus ATCC 43300",
                         "Salmonella enterica subsp. enterica serovar Typhimurium ATCC 14028",
                         "Staphylococcus aureus ATCC 6538", "Pseudomonas aeruginosa ATCC 9027", "Candida albicans",
                         "Klebsiella pneumoniae"]
    model_name = "DeepChem/ChemBERTa-77M-MTR"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dataset = SMILESDataset(data, tokenizer)

    synergy_dataset = synthetic_dataset(data, tokenizer)

    # print(synergy_dataset[0])
    print(f'\n Current Dataset length: {len(synergy_dataset)}\n Original Dataset length: {synergy_dataset.original_length}\n cutting off length: {synergy_dataset.max_length}\n')

    # 定义 KFold 交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 设置训练参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 20
    min_lr = 1e-6
    batch_size = 100
    freeze_epochs = 5
    mean_weight = 1
    lora_r = 28
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=32,
        target_modules=["query", "key", "value", 'dense'],  # 也可以包含 "dense" 等其他线性层, 看你想插在哪些层
        task_type=TaskType.FEATURE_EXTRACTION,  # 不走任何特定任务逻辑，最通用的方式
        lora_dropout=0.1,
        bias="none"
    )

    # 5-fold 交叉验证训练和评估
    all_ap_scores = []

    wandb.init(
        # set the wandb project where this run will be logged
        project="Synergy",
        name=f'Synergy, LoRA, LoRA_r = {lora_r}, L40S, DBAASP only, no old <cls> embedding',

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
            "architecture": "ChemBERTa-77M-MTR",
            "dataset": data_path,
            "epochs": num_epochs,
        }
    )
    # wandb.login(key="5518c5d9d8cbd8bc6edc7fbe88f6d5ab4e2832f9")
    # best_mean_R2s = []
    best_ap_scores = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(synergy_dataset)):
        # wandb.init(
        #     # set the wandb project where this run will be logged
        #     project="Synergy",
        #     name=f'ChemBERTa_APEX_data_{num_epochs}epoch_{batch_size}batch size',
        #
        #     # track hyperparameters and run metadata
        #     config={
        #         'fold': fold + 1,
        #         "learning_rate": 1e-4,
        #         "architecture": "ChemBERTa-77M-MTR",
        #         "dataset": data_path,
        #         "epochs": num_epochs,
        #     }
        # )
        print(f"Fold {fold + 1}")

        ChemBERTa_model = AutoModel.from_pretrained(model_name)
        pretrained_dict = torch.load(current_folder / 'Checkpoints' / 'best_all_data_for_synergy.pth')
        ChemBERTa_model.load_state_dict(pretrained_dict['ChemBERTa_state_dict'])
        perf_ChemBERTa_model = get_peft_model(ChemBERTa_model, lora_config)
        perf_ChemBERTa_model.to(device)
        perf_ChemBERTa_model.print_trainable_parameters()
        peft_model_trainable_parameters = [p for p in perf_ChemBERTa_model.parameters() if p.requires_grad]
        lora_keys = [k for k in perf_ChemBERTa_model.state_dict().keys() if "lora" in k]

        # 生成训练和验证集
        train_subset = torch.utils.data.Subset(synergy_dataset, train_idx)
        test_subset = torch.utils.data.Subset(synergy_dataset, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_synergy)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_synergy)

        # 这一个模块用来融合两个分子的 <cls> embedding
        merge_layer = MolMergeFFN(ChemBERTa_model.config.hidden_size)
        merge_layer.to(device)

        co_cross_attn = FirstTokenAttention(ChemBERTa_model.config.hidden_size, 2, 0.1)
        co_cross_attn.to(device)

        cls_head = RegressionHead(ChemBERTa_model.config.hidden_size, num_targets=3)
        cls_head.to(device)

        # 冻结预训练模型参数
        # for param in model.bert.parameters():
        #     param.requires_grad = False

        # 定义损失函数和优化器
        # criterion = MultiTaskLoss(device=device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(merge_layer.parameters(), lr=1e-4)
        optimizer.add_param_group({'params': peft_model_trainable_parameters, 'lr': 1e-4})
        optimizer.add_param_group({'params': cls_head.parameters(), 'lr': 1e-4})
        optimizer.add_param_group({'params': co_cross_attn.parameters(), 'lr': 1e-4})

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=min_lr)

        best_ap_score = 0  # 初始化每个 fold 的最佳 AP 分数
        # r2_tracker = R2Tracker(num_tasks=19)
        # best_R2_score = 0.0

        # 训练模型
        for epoch in range(num_epochs):
            # if epoch + 1 == freeze_epochs:
            #     # 解冻预训练模型
            #     for param in model.bert.parameters():
            #         param.requires_grad = True
            #     optimizer.add_param_group({'params': model.bert.parameters(), 'lr': 1e-4})

            perf_ChemBERTa_model.train()
            merge_layer.train()
            co_cross_attn.train()
            cls_head.train()
            train_loss_list = []
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | training"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                # label_masks = batch['label_mask'].to(device)

                optimizer.zero_grad()
                outputs = perf_ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)

                cls_embedding_1 = outputs.last_hidden_state[::2, 0, :]  # [B/2, D]
                cls_embedding_2 = outputs.last_hidden_state[1::2, 0, :]  # [B/2, D]
                two_mol_concat_emb = torch.cat((cls_embedding_1, cls_embedding_2), dim=1)  # [B/2, 2D]
                merged_emd = merge_layer(two_mol_concat_emb)[:, None, :]  # [B/2, D]

                seq_token_emd_1 = outputs.last_hidden_state[::2, 1:, :]  # [B/2, T-1, D]
                seq_token_emd_2 = outputs.last_hidden_state[1::2, 1:, :]  # [B/2, T-1, D]
                merged_seq_emd = torch.cat((merged_emd, seq_token_emd_1, seq_token_emd_2), dim=1)  # [B/2, 2T-1, D]

                padding_mask_1 = 1 - attention_mask[::2]  # [B/2, T]
                padding_mask_2 = 1 - attention_mask[1::2, 1:]  # [B/2, T-1]
                key_padding_mask = torch.cat((padding_mask_1, padding_mask_2), dim=1)  # [B/2, 2T-1]

                merged_cls_emd = co_cross_attn(merged_seq_emd, key_padding_mask)

                # concat_cls_emd = torch.cat((cls_embedding_1, merged_cls_emd, cls_embedding_2), dim=1)  # 与最开始不同的是完全不要未更新过的 <cls> embedding

                # logits = cls_head(concat_cls_emd)  # 与最开始不同的是完全不要未更新过的 <cls> embedding
                logits = cls_head(merged_cls_emd)  # 与最开始不同的是完全不要未更新过的 <cls> embedding

                # loss = criterion(logits, labels, label_masks, mean_weight)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss.item())

            scheduler.step()
            wandb.log({'train loss': np.array(train_loss_list).mean()}, commit=False)

            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

            # 模型评估
            # model.eval()
            perf_ChemBERTa_model.eval()
            merge_layer.eval()
            co_cross_attn.eval()
            cls_head.eval()
            all_labels = []
            all_preds = []
            # all_label_masks = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | evaluating"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label']
                    # label_masks = batch['label_mask'].to(device)

                    optimizer.zero_grad()
                    outputs = perf_ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)

                    cls_embedding_1 = outputs.last_hidden_state[::2, 0, :]  # [B/2, D]
                    cls_embedding_2 = outputs.last_hidden_state[1::2, 0, :]  # [B/2, D]
                    two_mol_concat_emb = torch.cat((cls_embedding_1, cls_embedding_2), dim=1)  # [B/2, 2D]
                    merged_emd = merge_layer(two_mol_concat_emb)[:, None, :]  # [B/2, D]

                    seq_token_emd_1 = outputs.last_hidden_state[::2, 1:, :]  # [B/2, T-1, D]
                    seq_token_emd_2 = outputs.last_hidden_state[1::2, 1:, :]  # [B/2, T-1, D]
                    merged_seq_emd = torch.cat((merged_emd, seq_token_emd_1, seq_token_emd_2), dim=1)  # [B/2, 2T-1, D]

                    padding_mask_1 = 1 - attention_mask[::2]  # [B/2, T]
                    padding_mask_2 = 1 - attention_mask[1::2, 1:]  # [B/2, T-1]
                    key_padding_mask = torch.cat((padding_mask_1, padding_mask_2), dim=1)  # [B/2, 2T-1]

                    merged_cls_emd = co_cross_attn(merged_seq_emd, key_padding_mask)

                    # concat_cls_emd = torch.cat((cls_embedding_1, merged_cls_emd, cls_embedding_2), dim=1)  # 与最开始不同的是完全不要未更新过的 <cls> embedding

                    # logits = cls_head(concat_cls_emd)  # 与最开始不同的是完全不要未更新过的 <cls> embedding
                    logits = cls_head(merged_cls_emd)  # 与最开始不同的是完全不要未更新过的 <cls> embedding

                    all_labels.extend(labels.numpy())
                    # all_preds.extend(probs.cpu().numpy())
                    # all_preds.extend(logits[:,:-1].cpu().numpy())
                    all_preds.extend(logits.cpu().numpy())
                    # all_label_masks.extend(label_masks.cpu().numpy())

            # 计算 average_precision_score
            # ap_score = average_precision_score(all_labels, all_preds)
            all_labels.append(np.int64(2))
            all_preds.append(np.array([0, 0, 1], dtype=np.float32))
            roc_auc = roc_auc_score(all_labels, F.softmax(torch.from_numpy(np.array(all_preds))), multi_class="ovr", average="macro")
            # R2_per_task = calculate_r2_per_task(all_labels, all_preds, all_label_masks)
            # 记录每个 fold 的最高 AP 分数
            # if ap_score > best_ap_score:
            #     best_ap_score = ap_score
            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, val AP Score: {R2_per_task}, best AP Score: {best_R2_score}")
            # r2_tracker.update_best_r2(R2_per_task)
            # R2_mean = np.array(R2_per_task).mean()
            loss = criterion(torch.tensor(all_preds), torch.tensor(all_labels))

            print(f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {loss.item()}")  # \nR2 Score per task: {R2_per_task}\nbest R2 Score: {r2_tracker.get_best_r2()}")
            if roc_auc > best_ap_score:
                best_ap_score = roc_auc
                lora_state_dict = {k: perf_ChemBERTa_model.state_dict()[k] for k in lora_keys}
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lora_state_dict': lora_state_dict,
                    'merge_layer_state_dict': merge_layer.state_dict(),
                    'co_cross_attn_state_dict': co_cross_attn.state_dict(),
                    'cls_head_state_dict': cls_head.state_dict(),
                }, current_folder / 'Checkpoints' / f'synergy_best_AP_fold{fold}.pth')
                # torch.save(model.state_dict(), f'./compare_APEX/checkpoint/APEX_model_fold_{fold + 1}.pt')
            wandb.log({"synergy test loss": loss.item(), "test roc_auc": roc_auc, "fold": fold + 1, "epoch": epoch + fold * num_epochs})
        # best_mean_R2s.append(best_R2_score)
        best_ap_scores.append(best_ap_score)
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'lora_state_dict': lora_state_dict,
            'merge_layer_state_dict': merge_layer.state_dict(),
            'co_cross_attn_state_dict': co_cross_attn.state_dict(),
            'cls_head_state_dict': cls_head.state_dict(),
        }, current_folder / 'Checkpoints' / f'synergy_final_AP_fold{fold}.pth')
    wandb.log({"best_mean_ap_scores_folds": np.array(best_ap_scores).mean()})