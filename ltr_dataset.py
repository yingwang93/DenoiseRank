import torch
from torch.utils.data import Dataset
from sklearn.exceptions import NotFittedError
import numpy as np
import random


class LearningToRankDataset(Dataset):
    def __init__(self, df, label_column, list_id_column, transform=None, user_model=None, seed=None):

        df.sort_values(by=list_id_column, inplace=True)

        feat_columns = df.columns.difference([label_column, list_id_column])
        self.feat = df[feat_columns].values
        if transform is not None:
            try:
                self.feat = transform.transform(self.feat)
            except NotFittedError:
                self.feat = transform.fit_transform(self.feat)

        self.qid = torch.from_numpy(df[list_id_column].values).int()
        self.feat = torch.from_numpy(self.feat).float()
        self.target = torch.from_numpy(df[label_column].values).float()
        self.length = torch.from_numpy(df[list_id_column].value_counts(sort=False).values)
        self.cum_length = torch.cumsum(self.length, dim=0)

    def __getitem__(self, item):

        if item == 0:
            start_idx = 0
        else:
            start_idx = self.cum_length[item-1]
        end_idx = self.cum_length[item].item()

        item_dict = {
            'feat': self.feat[start_idx:end_idx],
            'target': self.target[start_idx:end_idx],
            'length': self.length[item].reshape(1),
            'qid': self.qid[start_idx].reshape(1)
        }
        return item_dict

    def __len__(self):
        return self.length.shape[0]

    # For diversity test
    def getRandomQuery(self, num_list=20, require_list_len=10):
        # 过滤出所有大于require_list_len的元素
        filtered_indices = [i for i, x in enumerate(self.length) if x >= require_list_len]

        # # 检查是否足够选择
        if len(filtered_indices) > num_list:
        #     # 随机选择指定数量的元素
            filtered_indices = random.sample(filtered_indices, num_list)
        
        print('filtered_indices ids: ', filtered_indices)
        queries = []
        for item in filtered_indices:
            if item == 0:
                start_idx = 0
            else:
                start_idx = self.cum_length[item-1]
            end_idx = self.cum_length[item].item()

            item_dict = {
                'feat': self.feat[start_idx:end_idx],
                'target': self.target[start_idx:end_idx],
                'length': self.length[item].reshape(1),
                'qid': self.qid[start_idx].reshape(1)
            }
                
            queries.append(item_dict)
        return queries

    @staticmethod
    def collate_fn(batches):
        batch_example = batches[0]
        batch = {key: torch.cat([batch_vals[key] for batch_vals in batches]) for key in batch_example.keys()}
        return batch

    @property
    def input_dim(self):
        return self.feat.shape[1]

    @property
    def max_target(self):
        return self.target.max().cpu().int().item()

