import torch

import numpy as np
from sklearn.metrics import ndcg_score


class Average:
    def __init__(self):
        self.sum = 0.
        self.total = 0.

    def update(self, val, weight=1):
        self.sum += val
        self.total += weight

    def compute(self):
        self.agg = self.sum / self.total
        self.sum = 0.
        self.total = 0.
        return self.agg


class NDCG:
    def __init__(self, kind="exponential", k=10):
        # k<=0时表示不截断
        self.kind = kind
        self.k = k

        self.ndcg_sum = 0.
        self.total = 0.

    def update(self, score, target, length):
        cum_length = np.cumsum(length.cpu().numpy())
        score_per_list = np.split(score.cpu().numpy(), cum_length[:-1])
        target_per_list = np.split(target.cpu().numpy(), cum_length[:-1])

        for score_of_list, target_of_list in zip(score_per_list, target_per_list):
            if self._should_count_list(target_of_list):
                gain_of_list = self._compute_gain(target_of_list)
                k = self.k if self.k> 0 else len(score_of_list)
                self.ndcg_sum += ndcg_score([gain_of_list], [score_of_list], k=k)
            else:
                self.ndcg_sum += 1
                
            self.total += 1

    def _compute_gain(self, target):
        if self.kind == "exponential":
            return np.power(2, target) - 1
        elif self.kind == "linear":
            return target
        else:
            raise ValueError(f"kind={self.kind} is not supported")

    def _should_count_list(self, target):
        if np.all(target == target[0]):
            return False
        return True

    def compute(self):
        if self.total == 0:
            return torch.nan

        self.agg = self.ndcg_sum / self.total
        self.ndcg_sum = 0.
        self.total = 0.
        return self.agg


class TopNDCG(NDCG):
    def __init__(self, max_target=None, **kwargs):
        super().__init__(**kwargs)
        self.max_target = max_target

    def _should_count_list(self, target):
        if target.max() < self.max_target:
            return False
        return super()._should_count_list(target)

class ERR:
    def __init__(self, k=10, max_rel=4):
        self.k = k
        self.max_rel = max_rel

        self.err_sum = 0.
        self.total = 0.

    def update(self, score, target, length):
        cum_length = np.cumsum(length.cpu().numpy())
        score_per_list = np.split(score.cpu().numpy(), cum_length[:-1])
        target_per_list = np.split(target.cpu().numpy(), cum_length[:-1])

        for score_of_list, target_of_list in zip(score_per_list, target_per_list):
            self.err_sum += compute_err_at_k(score_of_list, target_of_list, self.k, self.max_rel) 
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.nan

        self.agg = self.err_sum / self.total
        self.err_sum = 0.
        self.total = 0.
        return self.agg

def compute_err_at_k(preds, labels, k: int, max_rel: int = 4):
    """
    计算PyTorch下的ERR@K指标
    :param preds: 预测相关度分数，形状为[n]
    :param labels: 真实相关度标签（多级，如0-4），形状为[n]
    :param k: 仅计算前K个位置的贡献, k<=0时表示不截断
    :param max_rel: 最大相关度等级（默认Web30K为4）
    :return: ERR@K标量值
    """
    # 1. 输入验证
    preds = np.asarray(preds, dtype=np.float32)  # 强制转为浮点数组
    labels = np.asarray(labels, dtype=np.int32)  # 标签为整数类型
    n = len(preds)
    if k<=0: k=n 
    k = min(k, n)  # 处理n < K的情况
    
    sorted_indices = np.argsort(-preds, axis=0).flatten()  # 
    sorted_labels = labels[sorted_indices][:k]  # 仅保留前K个
    
    sorted_labels_float = sorted_labels.astype(np.float32)  
    R = (np.power(2.0, sorted_labels_float) - 1.0) / np.power(2.0, max_rel)
    
    ones = np.ones(1, dtype=np.float32)  
    one_minus_R = np.concatenate([ones, 1 - R])  # 添加初始值1以处理累积乘积
    cum_prod = np.cumprod(one_minus_R)[:-1]  # 移除最后一个元素
    
    p = R * cum_prod
    ranks = np.arange(1, k + 1, dtype=np.float32)  
    err = np.sum(p / ranks) 
    
    return float(err)


class MRR:
    def __init__(self, k=10, min_rel=1):
        self.k = k
        self.min_rel = min_rel

        self.err_sum = 0.
        self.total = 0.

    def update(self, score, target, length):
        cum_length = np.cumsum(length.cpu().numpy())
        score_per_list = np.split(score.cpu().numpy(), cum_length[:-1])
        target_per_list = np.split(target.cpu().numpy(), cum_length[:-1])

        for score_of_list, target_of_list in zip(score_per_list, target_per_list):
            self.err_sum += compute_mrr_at_k(score_of_list, target_of_list, self.k, self.min_rel) 
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.nan

        self.agg = self.err_sum / self.total
        self.err_sum = 0.
        self.total = 0.
        return self.agg

def compute_mrr_at_k(preds, labels, k: int, min_rel: int = 1):
    """
    计算PyTorch下的MRR@K指标
    :param preds: 预测相关度分数，形状为[n]
    :param labels: 真实相关度标签（多级，如0-4），形状为[n]
    :param k: 仅计算前K个位置的贡献, k<=0时表示不截断
    :param min_rel: 最小相关度阈值（默认≥2视为相关）
    :return: MRR@K标量值
    """
    # 1. 输入验证
    preds = np.asarray(preds, dtype=np.float32)  # 强制转为浮点数组
    labels = np.asarray(labels, dtype=np.int32)  # 标签为整数类型
    n = len(preds)
    if k<=0: k=n 
    k = min(k, n)  # 处理n < K的情况
    
    # 2. 根据预测分数对真实标签排序（降序）
    sorted_indices = np.argsort(-preds, axis=0).flatten()  # 
    sorted_labels = labels[sorted_indices][:k]  # 仅保留前K个
    
    # 3. 定位首个相关文档（≥min_rel为相关）
    relevant = sorted_labels >= min_rel
    hit_indices = np.where(relevant)[0]
    if hit_indices.size > 0:
        first_hit_rank = hit_indices[0] + 1  # 排名从1开始
        mrr = 1.0 / first_hit_rank
    else:
        mrr = 0.0
    
    return float(mrr)

class Precision:
    def __init__(self, k=10, min_rel=1):
        self.k = k
        self.min_rel = min_rel

        self.err_sum = 0.
        self.total = 0.

    def update(self, score, target, length):
        cum_length = np.cumsum(length.cpu().numpy())
        score_per_list = np.split(score.cpu().numpy(), cum_length[:-1])
        target_per_list = np.split(target.cpu().numpy(), cum_length[:-1])

        for score_of_list, target_of_list in zip(score_per_list, target_per_list):
            self.err_sum += compute_pre_at_k(score_of_list, target_of_list, self.k, self.min_rel) 
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.nan

        self.agg = self.err_sum / self.total
        self.err_sum = 0.
        self.total = 0.
        return self.agg

def compute_pre_at_k(preds, labels, k: int, min_rel: int = 1):
    """
    计算PyTorch下的Precision@K指标
    :param preds: 预测相关度分数，形状为[n]
    :param labels: 真实相关度标签（多级，如0-4），形状为[n]
    :param k: 仅计算前K个位置的贡献, k<=0时表示不截断
    :param min_rel: 最小相关度阈值（默认≥2视为相关）
    :return: Precision@K标量值
    """
    # 1. 输入验证
    preds = np.asarray(preds, dtype=np.float32)  # 强制转为浮点数组
    labels = np.asarray(labels, dtype=np.int32)  # 标签为整数类型
    n = len(preds)
    if k<=0: k=n 
    k = min(k, n)  # 处理n < K的情况
    
    # 2. 根据预测分数对真实标签排序（降序）
    sorted_indices = np.argsort(-preds, axis=0).flatten()  # 
    sorted_labels = labels[sorted_indices][:k]  # 仅保留前K个
    
    # 3. 统计前K个结果中的相关文档数（标签≥min_rel）
    relevant_count = np.sum(sorted_labels >= min_rel)
    
    # 4. 计算Precision@K = 相关数 / K
    precision = relevant_count / k
    
    return float(precision)


class MAP:
    def __init__(self, k=10, min_rel=1):
        self.k = k
        self.min_rel = min_rel

        self.err_sum = 0.
        self.total = 0.

    def update(self, score, target, length):
        cum_length = np.cumsum(length.cpu().numpy())
        score_per_list = np.split(score.cpu().numpy(), cum_length[:-1])
        target_per_list = np.split(target.cpu().numpy(), cum_length[:-1])

        for score_of_list, target_of_list in zip(score_per_list, target_per_list):
            self.err_sum += compute_map_at_k(score_of_list, target_of_list, self.k, self.min_rel) 
            self.total += 1

    def compute(self):
        if self.total == 0:
            return torch.nan

        self.agg = self.err_sum / self.total
        self.err_sum = 0.
        self.total = 0.
        return self.agg

def compute_map_at_k(preds, labels, k: int, min_rel: int = 1):
    """
    计算单查询的AP@k指标（平均精度）
    :param preds: 预测分数，形状为[n]
    :param labels: 真实标签，形状为[n]
    :param k: 计算前K个位置的贡献, k<=0时表示不截断
    :param min_rel: 最小相关度阈值（默认≥2视为相关）
    :return: AP@k标量值（实际为单个查询的平均精度）
    """
    # 1. 输入验证
    preds = np.asarray(preds, dtype=np.float32)  # 强制转为浮点数组
    labels = np.asarray(labels, dtype=np.int32)  # 标签为整数类型
    n = len(preds)
    if k<=0: k=n 
    k = min(k, n)  # 处理n < K的情况
    
    # 2. 根据预测分数对真实标签排序（降序）
    sorted_indices = np.argsort(-preds, axis=0).flatten()  # 
    sorted_labels = labels[sorted_indices][:k]  # 仅保留前K个
    
    # 统计相关位置
    relevant_positions = np.where(sorted_labels >= min_rel)[0]
    num_relevant = len(relevant_positions)
    
    if num_relevant == 0:
        return 0.0
    else:
        hit_count = 0
        precisions = []
        # 遍历前k个预测结果
        for r in range(k):
            if sorted_labels[r] >= min_rel:
                hit_count += 1
                precisions.append(hit_count / (r + 1))  # 计算当前精度，r从0开始，所以要+1
        # 分母取实际相关数与k的较小值
        return sum(precisions) / min(num_relevant, k)
