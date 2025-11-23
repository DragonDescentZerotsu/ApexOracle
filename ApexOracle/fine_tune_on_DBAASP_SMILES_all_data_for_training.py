import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from torch.nn.utils.rnn import pad_sequence
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
        # DBAASP_id = self.dataframe.iloc[idx]['DBAASP_id']
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
    data_path = current_folder / 'DataPrepare' / 'Data' / 'DBAASP_id_SMILES_bact_MICs.csv'  # 替换为你的数据路径
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
    dataset = SMILESDataset(data, tokenizer)

    print(dataset[0])
    print(f'Current Dataset length: {len(dataset)}, Original Dataset length: {dataset.original_length}, cutting off length: {dataset.max_length}')


    # max_length = 0
    # max_length_list = []
    # longer_512_count = 0
    # for i in range(len(dataset)):
    #     current_length = len(dataset[i]['input_ids'])
    #     if current_length == max_length:
    #         max_length_list.append(max_length)
    #     if current_length > max_length:
    #         max_length = len(dataset[i]['input_ids'])
    #         max_length_list.append(max_length)
    #     if current_length > 512:
    #         longer_512_count += 1

    # 定义 KFold 交叉验证
    # kf = KFold(n_splits=1, shuffle=True, random_state=42)

    # 设置训练参数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_epochs = 1100
    batch_size = 100
    freeze_epochs = 5
    mean_weight = 1

    # 5-fold 交叉验证训练和评估
    all_ap_scores = []

    wandb.init(
        # set the wandb project where this run will be logged
        project="Synergy",
        name=f'ChemBERTa_all_data_{num_epochs}epoch_{batch_size}batch size, L40S, all data used training',

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-4,
            "architecture": "ChemBERTa-77M-MTR",
            "dataset": data_path,
            "epochs": num_epochs,
        }
    )
    # wandb.login(key="5518c5d9d8cbd8bc6edc7fbe88f6d5ab4e2832f9")
    best_mean_R2s = []
    # for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
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
        # print(f"Fold {fold + 1}")

        # 生成训练和验证集
    # train_subset = torch.utils.data.Subset(dataset, train_idx)
    # test_subset = torch.utils.data.Subset(dataset, test_idx)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # 加载模型
    model = ClassificationModel(model_name, num_labels=19)
    model.to(device)

    # 冻结预训练模型参数
    for param in model.bert.parameters():
        param.requires_grad = False

    # 定义损失函数和优化器
    criterion = MultiTaskLoss(device = device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4)

    best_ap_score = 0  # 初始化每个 fold 的最佳 AP 分数
    r2_tracker = R2Tracker(num_tasks=19)
    best_R2_score = 0.0

    # 训练模型
    for epoch in range(num_epochs):
        if epoch+1 == freeze_epochs:
            # 解冻预训练模型
            for param in model.bert.parameters():
                param.requires_grad = True
            optimizer.add_param_group({'params': model.bert.parameters(), 'lr': 1e-4})

        model.train()
        all_labels = []
        all_preds = []
        all_label_masks = []
        for batch in tqdm(data_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            label_masks = batch['label_mask'].to(device)

            optimizer.zero_grad()
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels, label_masks, mean_weight)
            loss.backward()
            optimizer.step()

            all_labels.extend(labels.detach().cpu().numpy())
            all_preds.extend(logits.detach().cpu().numpy())
            all_label_masks.extend(label_masks.detach().cpu().numpy())

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

        # 模型评估
        # model.eval()
        # all_labels = []
        # all_preds = []
        # all_label_masks = []
        # with torch.no_grad():
        #     for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | evaluating"):
        #         input_ids = batch['input_ids'].to(device)
        #         attention_mask = batch['attention_mask'].to(device)
        #         labels = batch['label'].to(device)#[:,:-1]
        #         label_masks = batch['label_mask'].to(device)#[:,:-1]
        #
        #         logits = model(input_ids=input_ids, attention_mask=attention_mask)
        #         # probs = torch.softmax(logits, dim=1)[:, 1]  # 取正类的概率
        #
        #
        #         all_labels.extend(labels.cpu().numpy())
        #         # all_preds.extend(probs.cpu().numpy())
        #         # all_preds.extend(logits[:,:-1].cpu().numpy())
        #         all_preds.extend(logits.cpu().numpy())
        #         all_label_masks.extend(label_masks.cpu().numpy())

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
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}\nR2 Score per task: {R2_per_task}\nbest R2 Score: {r2_tracker.get_best_r2()}")
        if R2_mean > best_R2_score:
            best_R2_score = R2_mean
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'ChemBERTa_state_dict': model.bert.state_dict(),
                're_head_state_dict': model.classifier.state_dict()
            }, current_folder / 'Checkpoints' / f'best_all_data_for_synergy.pth')
            # torch.save(model.state_dict(), f'./compare_APEX/checkpoint/APEX_model_fold_{fold + 1}.pt')
        wandb.log({"loss": loss.item(), "R2_mean": R2_mean, "epoch": epoch})
    best_mean_R2s.append(best_R2_score)
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'ChemBERTa_state_dict': model.bert.state_dict(),
        're_head_state_dict': model.classifier.state_dict()
    }, current_folder / 'Checkpoints' / f'final_all_data_for_synergy.pth')
wandb.log({"best_mean_R2_across_folds": np.array(best_mean_R2s).mean()})
            # R2_best = r2_tracker.get_best_r2()
            # wandb.log({"epoch": epoch + 1}, commit=False)
            # wandb.log({f"{bact_names_DBAASP[i]}": R2_per_task[i] for i in range(len(R2_per_task))}, commit=False)
            # wandb.log({f"best_{bact_names_DBAASP[i]}": R2_best[i] for i in range(len(R2_best))}, commit=False)
            # wandb.log({"loss": loss.item(), "R2_mean": R2_mean, "fold": fold + 1})
            # print(
            #     f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}\nR2 Score per task: {R2_per_task}\nbest R2 Score: {r2_tracker.get_best_r2()}")

        # all_ap_scores.append(best_ap_score)
        # wandb.finish()
        # break

    # 输出平均 AP 分数
    # mean_ap_score = sum(all_ap_scores) / len(all_ap_scores)
    # print(f"Mean AP Score: {mean_ap_score}")