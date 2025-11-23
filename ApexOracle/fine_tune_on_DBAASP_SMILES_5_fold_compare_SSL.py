import pandas as pd
import torch
torch.cuda.empty_cache()
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
import ast


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
            'DBAASP_id': DBAASP_id,
            'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'label': torch.tensor(target, dtype=torch.float)
        }

    # @staticmethod
class CompareDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.origin_data = dataframe.values
        self.reordered_DBAASP_id = []
        self.reordered_smiles = []
        self.reordered_diff_places = []
        self.reorder_lose_win()
        self.input_ids = []
        self.attention_mask = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.more_attention_masks = []  # 这个 more_attention_mask 的意思是 需要 pay more attention 的 token 是哪些
        self.generate_more_attention_mask()

    def reorder_lose_win(self):
        """
        把 smiles 比较数据中 peptide lose 和 peptide win 排序，peptide lose 永远在 peptide win 之前
        :return:
        """
        for line in tqdm(self.origin_data, desc=' reordering compare dataset to lose, win'):
            mic_1 = line[-2]
            mic_2 = line[-1]
            if mic_1 > mic_2:
                self.reordered_DBAASP_id.append(line[0])
                self.reordered_DBAASP_id.append(line[1])
                self.reordered_smiles.append(line[2])
                self.reordered_smiles.append(line[3])
                self.reordered_diff_places.append(ast.literal_eval(line[4]))
                self.reordered_diff_places.append(ast.literal_eval(line[5]))
            else:
                self.reordered_DBAASP_id.append(line[1])
                self.reordered_DBAASP_id.append(line[0])
                self.reordered_smiles.append(line[3])
                self.reordered_smiles.append(line[2])
                self.reordered_diff_places.append(ast.literal_eval(line[5]))
                self.reordered_diff_places.append(ast.literal_eval(line[4]))

    def generate_more_attention_mask(self):
        """
        利用 different places 标记和 tokenizer 确定每一个 smiles 应该更注意的 token 是哪些，并顺便过滤掉太长的 smiles，同时
        生成所有smiles的input_token，这样载入数据就不用再算了
        :return:
        """
        smiles_indices_to_remove = set()
        for idx, smiles in tqdm(enumerate(self.reordered_smiles),desc=' generating more attention mask', total=len(self.reordered_smiles)):
            inputs = self.tokenizer(smiles, return_tensors='pt', padding=False, truncation=False, return_offsets_mapping=True)

            # 如果输入的长度太长了，就直接跳过，并且记录哪些smiles应该从数据集中删除
            if len(inputs['input_ids'][0]) > self.max_length:
                smiles_indices_to_remove.add(idx)
                if idx % 2 == 0:
                    # 如果
                    smiles_indices_to_remove.add(idx+1)
                else:
                    smiles_indices_to_remove.add(idx-1)
                # continue

            # if idx in smiles_indices_to_remove:
            #     continue

            inputs = {key: val.squeeze(0) for key, val in inputs.items()}  # 去掉 batch 维度
            # more_attention_mask = np.zeros(len(inputs['input_ids']))
            # for diff_place in self.reordered_diff_places[idx]:
            #     for token_id, token_range in enumerate(inputs['offset_mapping']):
            #         if token_range[0] <= diff_place < token_range[1]:
            #             more_attention_mask[token_id] = 1
            #             break

            # 这段代码是为了寻找匹配的 different places 和 token offset
            more_attention_mask = np.zeros(len(inputs['input_ids']))
            offsets = inputs['offset_mapping'].numpy()[:-1]  # 去掉最后一个 [0, 0] 的特殊offset防止出错
            token_starts = offsets[:, 0]
            token_ends = offsets[:, 1]
            diff_places = np.array(self.reordered_diff_places[idx])
            token_ids = np.searchsorted(token_starts, diff_places, side='right') - 1
            valid = (token_ids >= 0) & (diff_places < token_ends[token_ids])
            more_attention_mask[token_ids[valid]] = 1

            # same = False in more_attention_mask == more_attention_mask_new
            # print(same)

            self.input_ids.append(inputs['input_ids'])
            self.attention_mask.append(inputs['attention_mask'])
            self.more_attention_masks.append(more_attention_mask)

        self.reordered_smiles = np.delete(np.stack(self.reordered_smiles), list(smiles_indices_to_remove))
        self.more_attention_masks = [mask for idx, mask in enumerate(self.more_attention_masks) if idx not in smiles_indices_to_remove]
        self.attention_mask = [mask for idx, mask in enumerate(self.attention_mask) if idx not in smiles_indices_to_remove]
        self.input_ids = [input_id for idx, input_id in enumerate(self.input_ids) if idx not in smiles_indices_to_remove]
        self.reordered_DBAASP_id = [DBAASP_id for idx, DBAASP_id in enumerate(self.reordered_DBAASP_id) if idx not in smiles_indices_to_remove]
        # print(0)

    def exclude_test_data(self, test_DBAASP_ids, outpu_test_compare_pairs = False):
        """
        测试集中的数据最好是也不能参与 RL ，这里都剔除掉
        :param outpu_test_compare_pairs: 是否要输出被去掉的与test有关的数据是哪些
        :param test_DBAASP_ids: 所有在测试集中的 DBAASP id
        :return: 可选，可选是否输出与test有关的compare pairs
        """
        print(' exclude test data from compare dataset')

        self.reordered_DBAASP_id = np.array(self.reordered_DBAASP_id)
        test_DBAASP_ids = np.array(test_DBAASP_ids)

        # 找到 test set 中那些出现过的 DBAASP_id 在 compare 中原始的 line index 在哪
        initial_line_indexs_to_remove = np.where(np.isin(self.reordered_DBAASP_id, test_DBAASP_ids))[0]
        pair_line_indexs_to_remove = set()
        for initial_line_id in initial_line_indexs_to_remove:
            if initial_line_id % 2 == 0:
                pair_line_indexs_to_remove.update([initial_line_id, initial_line_id + 1])
            else:
                pair_line_indexs_to_remove.update([initial_line_id, initial_line_id - 1])

        compare_test_common_DBAASP_ids = set(self.reordered_DBAASP_id[initial_line_indexs_to_remove])
        print(f'\n num of compare test overlap DBAASP ids: {len(compare_test_common_DBAASP_ids)}')
        print(f' num of test DBAASP ids: {len(test_DBAASP_ids)}')
        print(f' num of comapre DBAASP ids: {len(set(self.reordered_DBAASP_id))}')
        print(f' overlap rate to test: {len(compare_test_common_DBAASP_ids)/len(test_DBAASP_ids)*100}%')
        print(f' overlap rate to compare: {len(compare_test_common_DBAASP_ids) / len(set(self.reordered_DBAASP_id)) * 100}%\n')

        # 这个是用于返回测试 test 上分类正确率的数据, 已经tokenize好的
        # self.input_ids = np.array(self.input_ids, dtype=object)
        test_compare_pairs_input_ids = np.array(self.input_ids, dtype=object)[np.sort(np.array(list(pair_line_indexs_to_remove)))]
        # test_compare_pairs_input_ids = self.input_ids[np.sort(np.array(pair_line_indexs_to_remove))]
        # self.attention_mask = np.array(self.attention_mask, dtype=object)
        test_compare_pairs_attn_masks = np.array(self.attention_mask, dtype=object)[np.sort(np.array(list(pair_line_indexs_to_remove)))]
        test_compare_pairs_more_attn_masks = np.array(self.more_attention_masks, dtype=object)[np.sort(np.array(list(pair_line_indexs_to_remove)))]
        test_compare_pairs_DBAASP_ids = np.array(self.reordered_DBAASP_id)[np.sort(np.array(list(pair_line_indexs_to_remove)))]
        # test_compare_pairs_attn_masks = self.attention_mask[np.sort(np.array(pair_line_indexs_to_remove))]

        print(f' removing overlaping data, which is {len(pair_line_indexs_to_remove) / len(self.reordered_DBAASP_id) * 100}% of the original compare data')
        # self.reordered_smiles = np.delete(self.reordered_smiles, list(pair_line_indexs_to_remove))
        self.more_attention_masks = [mask for idx, mask in enumerate(self.more_attention_masks) if idx not in pair_line_indexs_to_remove]
        self.attention_mask = [mask for idx, mask in enumerate(self.attention_mask) if idx not in pair_line_indexs_to_remove]
        self.input_ids = [input_id for idx, input_id in enumerate(self.input_ids) if idx not in pair_line_indexs_to_remove]
        self.reordered_DBAASP_id = [DBAASP_id for idx, DBAASP_id in enumerate(self.reordered_DBAASP_id) if idx not in pair_line_indexs_to_remove]
        print(f' removed')

        if outpu_test_compare_pairs:
            return test_compare_pairs_input_ids, test_compare_pairs_attn_masks, test_compare_pairs_more_attn_masks, test_compare_pairs_DBAASP_ids
        else:
            return None, None, None, None

    def update_test_compare_data(self, ChemBERTa_model, re_head, batch_size, test_DBAASP_ids):
        """
        做 SSL，把去掉的和 test 数据有关的 data 预测之后加回原来的 data 之中，形状应该是没变的
        :param ChemBERTa_model:
        :param re_head:
        :param batch_size:
        :return:
        """
        batch = 0
        mean_weight = torch.softmax(torch.tensor(
            [0.5330289900302887, 0.5252614915370941, 0.4671872854232788, 0.31565332412719727, 0.5086256861686707,
             0.4470508098602295, 0.3673310875892639, 0.5014685392379761, 0.5517476797103882, 0.6112076640129089,
             0.5257684588432312, 0.48106539249420166, 0.6668699979782104, 0.5823871493339539, 0.5998848974704742,
             0.544399231672287, 0.5091746151447296, 0.20608174800872803, 0.4069221019744873]) * 10, dim=0).to(
            device=ChemBERTa_model.device)
        all_pred_mean_MICs = []
        paired_input_ids, paired_input_attn_masks, test_compare_pairs_more_attn_masks, test_compare_pairs_DBAASP_ids = self.exclude_test_data(test_DBAASP_ids, outpu_test_compare_pairs = True)
        pbar = tqdm(total=len(paired_input_ids), desc=' predicting test mean MIC')
        while batch * batch_size < len(paired_input_ids):
            input_ids = paired_input_ids[batch * batch_size:(batch + 1) * batch_size]
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            attention_mask = paired_input_attn_masks[batch * batch_size:(batch + 1) * batch_size]
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
            input_ids = input_ids.to(device=ChemBERTa_model.device)
            attention_mask = attention_mask.to(device=ChemBERTa_model.device)

            with torch.no_grad():
                outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                logits = re_head(cls_embedding)
                mean_MIC = torch.sum(logits * mean_weight, dim=1)
                all_pred_mean_MICs.extend(mean_MIC.detach().cpu().numpy().tolist())

            batch += 1
            pbar.update(len(input_ids))

        pbar.close()
        reordered_input_ids = []
        reordered_input_attn_masks = []
        reordered_more_attn_masks = []
        reordered_DBAASP_ids = []

        all_pred_mean_MICs = -np.array(all_pred_mean_MICs)  # 注意这里得到的是 log(MIC/10)

        for input_ids_1, input_ids_2, attn_mask_1, attn_mask_2, more_attn_mask_1, more_attn_mask_2, mean_MIC_1, mean_MIC_2, DBAASP_id1, DBAASP_id2 in zip(paired_input_ids[::2], paired_input_ids[1::2], paired_input_attn_masks[::2], paired_input_attn_masks[1::2], test_compare_pairs_more_attn_masks[::2], test_compare_pairs_more_attn_masks[1::2], all_pred_mean_MICs[::2], all_pred_mean_MICs[1::2], test_compare_pairs_DBAASP_ids[::2], test_compare_pairs_DBAASP_ids[1::2]):
            if mean_MIC_1 > mean_MIC_2:
                reordered_input_ids.extend([input_ids_1, input_ids_2])
                reordered_input_attn_masks.extend([attn_mask_1, attn_mask_2])
                reordered_more_attn_masks.extend([more_attn_mask_1, more_attn_mask_2])
                reordered_DBAASP_ids.extend([DBAASP_id1, DBAASP_id2])
            else:
                reordered_input_ids.extend([input_ids_2, input_ids_1])
                reordered_input_attn_masks.extend([attn_mask_2, attn_mask_1])
                reordered_more_attn_masks.extend([more_attn_mask_2, more_attn_mask_1])
                reordered_DBAASP_ids.extend([DBAASP_id2, DBAASP_id1])

        self.input_ids.extend(reordered_input_ids)
        self.attention_mask.extend(reordered_input_attn_masks)
        self.more_attention_masks.extend(reordered_more_attn_masks)
        self.reordered_DBAASP_id.extend(reordered_DBAASP_ids)


    def __len__(self):
        # 实际只有 input_ids 长度一半那么多的 pair
        return len(self.input_ids) // 2

    def __getitem__(self, idx):
        return {
            'input_ids_1': self.input_ids[idx * 2],
            'input_ids_2': self.input_ids[idx * 2 + 1],
            'attention_mask_1': self.attention_mask[idx * 2],
            'attention_mask_2': self.attention_mask[idx * 2 + 1],
            'more_attention_mask_1': self.more_attention_masks[idx * 2],
            'more_attention_mask_2': self.more_attention_masks[idx * 2 + 1],
        }

def assess_test_classification_acc(ChemBERTa_model, re_head, paired_input_ids, paired_input_attn_masks, batch_size, pad_token_id):
    """
    计算一个已经在训练集上训练的不错的 model 在 test 数据的分类情况下正确率有多少
    :param ChemBERTa_model:
    :param re_head:
    :param paired_input_ids:
    :param paired_input_attn_masks:
    :param batch_size:
    :param pad_token_id:
    :return:
    """
    mean_weight = torch.softmax(torch.tensor([0.5330289900302887, 0.5252614915370941, 0.4671872854232788, 0.31565332412719727, 0.5086256861686707, 0.4470508098602295, 0.3673310875892639, 0.5014685392379761, 0.5517476797103882, 0.6112076640129089, 0.5257684588432312, 0.48106539249420166, 0.6668699979782104, 0.5823871493339539, 0.5998848974704742, 0.544399231672287, 0.5091746151447296, 0.20608174800872803, 0.4069221019744873])*10, dim=0).to(device=ChemBERTa_model.device)
    batch = 0
    all_pred_labels = []
    all_label_masks = []
    pbar = tqdm(total=len(paired_input_ids), desc=' predicting test mean MIC')
    while batch*batch_size < len(paired_input_ids):
        input_ids = paired_input_ids[batch*batch_size:(batch+1)*batch_size]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = paired_input_attn_masks[batch*batch_size:(batch+1)*batch_size]
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        input_ids = input_ids.to(device=ChemBERTa_model.device)
        attention_mask = attention_mask.to(device=ChemBERTa_model.device)

        with torch.no_grad():
            outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            logits = re_head(cls_embedding)

            mean_MIC = torch.sum(logits * mean_weight, dim=1)
            pred_labels = ((mean_MIC[::2]-mean_MIC[1::2]) < 0).to(dtype=torch.int)
            real_mean_MIC = torch.sum(torch.exp(-logits)*10 * mean_weight, dim=1)
            paired_mean_MIC = torch.stack([real_mean_MIC[::2], real_mean_MIC[1::2]], dim=1)
            label_mask = (torch.max(paired_mean_MIC, dim=1)[0] / torch.min(paired_mean_MIC, dim=1)[0] > 1.1).to(dtype=torch.int)
            all_label_masks.append(label_mask.detach().cpu().numpy())
            # pred_labels = (((((logits[::2] - logits[1::2]) < 0).to(dtype=torch.int)).sum(dim=1)) >= 10).to(dtype=torch.int)
            all_pred_labels.append(pred_labels.detach().cpu().numpy())

        batch += 1
        pbar.update(len(input_ids))

    pbar.close()
    all_pred_labels = np.concatenate(all_pred_labels).reshape(-1)
    all_label_masks = np.concatenate(all_label_masks).reshape(-1)
    acc = np.sum(all_pred_labels * all_label_masks) / np.sum(all_label_masks)
    print(f' test classification accuracy: {acc * 100}%')


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

def collate_fn_compare(batch):
    input_ids = []
    attention_mask = []
    more_attention_mask = []
    for item in batch:
        input_ids.extend([item['input_ids_1'], item['input_ids_2']])
        attention_mask.extend([item['attention_mask_1'], item['attention_mask_2']])
        more_attention_mask.extend([torch.from_numpy(item['more_attention_mask_1']), torch.from_numpy(item['more_attention_mask_2'])])

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    more_attention_mask = pad_sequence(more_attention_mask, batch_first=True, padding_value=0)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'more_attention_mask': more_attention_mask
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
    # a = dataset[0]

    # compare_dataset = CompareDataset(pd.read_csv(current_folder / 'DataPrepare' / 'Data' / 'DBAASP_id_SMILES_compare_cleaned_w_mean_MIC.csv'), tokenizer)

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
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 设置训练参数
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    num_epochs = 400
    RL_start_epoch = 200
    RL_initialize_epoch = 10
    batch_size = 100
    compare_batch_size = 50
    freeze_epochs = 1
    mean_weight = 1
    attn_loss_weight = 0.2

    # 5-fold 交叉验证训练和评估
    all_ap_scores = []

    wandb.init(
        # set the wandb project where this run will be logged
        project="Synergy",
        name=f'ChemBERTa_all_data_{num_epochs}epoch_{batch_size}batch size, L40S, debug',

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
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
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
        # compare_loader = DataLoader(compare_dataset, batch_size=compare_batch_size, shuffle=True, collate_fn=collate_fn_compare)
        ChemBERTa_model = AutoModel.from_pretrained(model_name)
        ChemBERTa_model.to(device)
        # 生成训练和验证集
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        test_subset = torch.utils.data.Subset(dataset, test_idx)
        # a = test_subset[0]
        # 去除compare data中重合的test data
        test_subset_DBAASP_id = [data['DBAASP_id'] for data in test_subset]
        compare_dataset = CompareDataset(
            pd.read_csv(current_folder / 'DataPrepare' / 'Data' / 'DBAASP_id_SMILES_compare_cleaned_w_mean_MIC.csv'),
            tokenizer)

        test_compare_pairs_input_ids, test_compare_pairs_attn_masks, _, _ = compare_dataset.exclude_test_data(test_subset_DBAASP_id, outpu_test_compare_pairs=True)

        # 加载模型
        re_head = RegressionHead(ChemBERTa_model.config.hidden_size, num_targets=19)
        re_head.to(device)
        # TODO: 以下是调试代码
        checkpoint = torch.load(current_folder / 'Checkpoints' / 'no_RL_best_R2_fold0.pth', map_location=device)
        ChemBERTa_model.load_state_dict(checkpoint['ChemBERTa_state_dict'])
        re_head.load_state_dict(checkpoint['re_head_state_dict'])
        assess_test_classification_acc(ChemBERTa_model, re_head, test_compare_pairs_input_ids, test_compare_pairs_attn_masks, 200, tokenizer.pad_token_id)
        # TODO: 以上是调试代码

        compare_loader = DataLoader(compare_dataset, batch_size=compare_batch_size, shuffle=True, collate_fn=collate_fn_compare)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # model = ClassificationModel(model_name, num_labels=19)
        # model.to(device)
        compare_head = RegressionHead(ChemBERTa_model.config.hidden_size, num_targets=1)
        compare_head.to(device)

        # 冻结预训练模型参数
        for param in ChemBERTa_model.parameters():
            param.requires_grad = False

        # 定义损失函数和优化器
        criterion = MultiTaskLoss(device = device)

        # 添加 回归头 和 比较头 的 parameter 更新
        optimizer = optim.Adam(re_head.parameters(), lr=1e-4)
        optimizer.add_param_group({'params': compare_head.parameters(), 'lr': 1e-4})

        best_ap_score = 0  # 初始化每个 fold 的最佳 AP 分数
        r2_tracker = R2Tracker(num_tasks=19)
        best_R2_score = 0.0

        # 训练模型
        for epoch in range(num_epochs):
            if epoch+1 == freeze_epochs:
                # 解冻预训练模型
                for param in ChemBERTa_model.parameters():
                    param.requires_grad = True
                optimizer.add_param_group({'params': ChemBERTa_model.parameters(), 'lr': 1e-4})

            ChemBERTa_model.train()
            re_head.train()
            compare_head.train()

            # RL_start_epoch 个 epoch 之后才开始训练 RL
            # if epoch+1 == RL_start_epoch:
            #     checkpoint = torch.load(current_folder / 'Checkpoints' / f'RL_best_R2_fold{fold}.pth', map_location=device)
            #     ChemBERTa_model.load_state_dict(checkpoint['ChemBERTa_state_dict'])
            #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            #     re_head.load_state_dict(checkpoint['re_head_state_dict'])

            if epoch+1>=RL_start_epoch:

                # 刚开始的几个 epoch 内不更新 ChemBERTa 基座权重
                if epoch+1 < RL_start_epoch+RL_initialize_epoch:
                    for param in ChemBERTa_model.parameters():
                        param.requires_grad = False
                else:
                    for param in ChemBERTa_model.parameters():
                        param.requires_grad = True

                compare_dataset.update_test_compare_data(ChemBERTa_model, re_head, compare_batch_size, test_subset_DBAASP_id)
                for batch in tqdm(compare_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | training comparison"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    more_attention_mask = batch['more_attention_mask'].to(device)

                    outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]  # 提取 <cls> token 嵌入
                    psudo_MIC = compare_head(cls_embedding).squeeze()
                    psudo_MIC_loss = torch.mean(-torch.log(torch.sigmoid(-psudo_MIC[::2]+psudo_MIC[1::2])))

                    attention_score = torch.mean(outputs.attentions[-1], dim=1)[:, 0]  # (num_layers, batch_size, num_heads, seq_len, seq_len) -> (batch_size, seq_len, seq_len)

                    atten_loss_mask = (torch.sum(more_attention_mask, dim=1)>0).to(torch.int)

                    sample_level_attn_loss = -torch.log(torch.sigmoid(torch.sum(attention_score * more_attention_mask, dim=1) / (torch.sum(more_attention_mask * attention_mask, dim=1) + 1e-4)
                                                                      - torch.sum(attention_score * (1-more_attention_mask) * attention_mask, dim=1) / (torch.sum((1-more_attention_mask) * attention_mask, dim=1) + 1e-4)))
                    batch_attn_loss = torch.sum(sample_level_attn_loss * atten_loss_mask) / torch.sum(atten_loss_mask)

                    final_loss = attn_loss_weight * batch_attn_loss + psudo_MIC_loss

                    optimizer.zero_grad()
                    final_loss.backward()
                    optimizer.step()

                    wandb.log({'train compare loss': final_loss.item()})

                # a = 1

            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | training regression"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                label_masks = batch['label_mask'].to(device)

                outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                logits = re_head(cls_embedding)
                # logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels, label_masks, mean_weight)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

            # 模型评估
            ChemBERTa_model.eval()
            re_head.eval()
            all_labels = []
            all_preds = []
            all_label_masks = []
            with torch.no_grad():
                for batch in tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs} | evaluating"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)#[:,:-1]
                    label_masks = batch['label_mask'].to(device)#[:,:-1]

                    outputs = ChemBERTa_model(input_ids=input_ids, attention_mask=attention_mask)
                    cls_embedding = outputs.last_hidden_state[:, 0, :]
                    logits = re_head(cls_embedding)

                    # logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    # probs = torch.softmax(logits, dim=1)[:, 1]  # 取正类的概率


                    all_labels.extend(labels.cpu().numpy())
                    # all_preds.extend(probs.cpu().numpy())
                    # all_preds.extend(logits[:,:-1].cpu().numpy())
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
                f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}\nR2 Score per task: {R2_per_task}\nbest R2 Score: {r2_tracker.get_best_r2()}")
            if R2_mean > best_R2_score:
                best_R2_score = R2_mean
                torch.save({
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'ChemBERTa_state_dict': ChemBERTa_model.state_dict(),
                    're_head_state_dict': re_head.state_dict(),
                    'compare_head_state_dict': compare_head.state_dict()
                }, current_folder / 'Checkpoints' / f'RL_best_R2_fold{fold}.pth')
                # torch.save(model.state_dict(), f'./compare_APEX/checkpoint/APEX_model_fold_{fold + 1}.pt')
            wandb.log({"loss": loss.item(), "R2_mean": R2_mean, "fold": fold + 1, "R2_mean_compare":R2_mean, "epoch": epoch + fold * num_epochs})
        best_mean_R2s.append(best_R2_score)
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