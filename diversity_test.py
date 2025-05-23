import os
import sys
import copy
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, distributed

from ltr_dataset import LearningToRankDataset
from metrics import NDCG, TopNDCG, Average, ERR, MRR, MAP, Precision
from model import DenoiseRank, MLP
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('--epocs', type=int, default=300, help='train epocs')  
parser.add_argument('--datasets', default='web30k', help='Selection: web30k istella yahoo')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')  
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument("--hidden_size", default=136, type=int, help="hidden size of model")
parser.add_argument("--mlp_hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.3, help='Dropout of representation')
parser.add_argument('--tf_input_dim', type=int, default=8, help='tf_input_dim')
parser.add_argument('--num_MLP_layer', type=int, default=4, help='Number of denoise network layers')
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--num_heads', type=int, default=4, help='Number of Transformer heads')
parser.add_argument('--head_hidden_layers', nargs='+', type=int, default=[128], help='head_hidden_layers')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Selection: cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt')  ## 
parser.add_argument('--diffusion_steps', type=int, default=1000, help='Maximum diffusion timestep')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--rescale_timesteps', default=False, help='rescal timesteps')
parser.add_argument('--loss_type', default='ListNet', help='Selection: softmax approxNDCG MSE Oridinal Pointwise ListNet RankNet NDCG2PP LambdaLoss')
parser.add_argument('--input_SA', default=True, help='should input doc feat caculate through transformer.')

args = parser.parse_args()
print(args)

SEED = 0
ROOT_DIR = os.path.join(os.path.dirname(__file__))
DEBUG = False  # Set to True to run on a small subset of the data
DEVICE = 'cpu'
_CUTOFF = 30.
NUM_WORKERS = 0
class log1pTransform:
    def __init__(self):
        self.minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

    def transform(self, X):
        log1p_x = np.log1p(np.abs(X)) 
        sign_x = np.sign(X)  # 计算符号
        X_transformed = log1p_x * sign_x
        X_scaled = np.clip(
            X_transformed, 
            min=-_CUTOFF, 
            max=_CUTOFF
        )
        return X_scaled


def main():
    """ DDP Training """
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=7200))
        print(datetime.datetime.now(), "ddp init success ...", rank)
    except KeyError:
        print(datetime.datetime.now(), "ddp init fail ...")
        world_size = 1
        rank = 0
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )

    torch.cuda.set_device(rank)
    DEVICE = torch.device('cuda', rank)
    print(datetime.datetime.now(), "Parameter local rank ", dist.get_rank())
    random.seed(SEED) # 用种子值 SEED 初始化随机数生成器；设置相同的种子值，每次生成的随机数序列都是一样的
    np.random.seed(SEED) # 同上
    torch.manual_seed(SEED) # 同上
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.reset_peak_memory_stats()   # 监测显存峰值
    
    transform = None
    hidden_size = 136
    test_data = None
    if args.datasets=='web30k':
        data_dir = os.path.join(ROOT_DIR, 'MSLR-WEB30K', f"Fold{SEED % 5 + 1}")
        transform = log1pTransform()
        hidden_size = 136
        test_data = load_data(data_dir, transform, 'test', feat_dim=hidden_size)
    elif args.datasets=='istella':
        data_dir = os.path.join(ROOT_DIR, 'ISTELLA', f"Fold1")
        transform = log1pTransform()
        hidden_size = 220
        test_data = load_data(data_dir, transform, 'test', feat_dim=hidden_size)
    elif args.datasets=='yahoo':
        data_dir = os.path.join(ROOT_DIR, 'YAHOO', f"Fold1")
        transform = QuantileTransformer(output_distribution='normal')
        hidden_size = 700
        test_data = load_data_featFill(data_dir, transform, 'test', feat_dim=hidden_size)
    
    args.hidden_size = hidden_size
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=LearningToRankDataset.collate_fn,
                             num_workers=NUM_WORKERS)

    
    model = torch.load(os.path.join(ROOT_DIR, 'model_save', 'bestModel_web30k_0519.pth'), weights_only=False)
    model = model.to(DEVICE)
    model.eval()
    
    # random
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('----------------------随机20个----------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    process_diversity_test(model, test_data, test_loader, DEVICE, args, eva_times=10, num_list=20, require_list_len=20)
    # # # 全部
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    print('----------------------全部--------------------------------')
    print('---------------------------------------------------------')
    print('---------------------------------------------------------')
    process_diversity_test(model, test_data, test_loader, DEVICE, args, random=False, eva_times=10, require_list_len=30)



def process_diversity_test(model, test_data, test_loader, DEVICE, args, random=True, eva_times=10, num_list=20, require_list_len=50):
    cur_model = copy.deepcopy(model)
    print("Testing...")
    cur_model.eval()

    queries = None
    if random == True:
        queries = test_data.getRandomQuery(num_list, require_list_len)
        print('eva_times: ', eva_times, 'num_list: ', num_list, 'require_list_len: ', require_list_len, 'querylen: ', len(queries))
        
    else:
        queries = test_loader
        
    cur_model.diffu.num_timesteps = args.diffusion_steps
    test_metrics = {
        'ndcg1': NDCG(k=1),
        'ndcg5': NDCG(k=5),
        'ndcg10': NDCG(k=10),
        'ndcg20': NDCG(k=20),
        'ndcg30': NDCG(k=30),
        'loss': Average()
    }
    score_res_set = []
    target_res_set = []
    with torch.no_grad():
        for batch in queries:
            feat, length, target = batch['feat'].to(DEVICE), batch['length'].to(DEVICE), batch['target'].to(DEVICE)
            score_res = []
            for i in range(eva_times):
                    
                score = cur_model(feat, target, length, False, True)
                loss = cur_model.compute_loss(score, target, length, args)
                if isinstance(score, tuple):
                    score = score[0]
                update_metrics(batch, score, loss, test_metrics)
                score_res.append(score)
                
            score_res_set.append(score_res)
            target_res_set.append(target)

    diversoty_list_k1 = []
    diversoty_list_k5 = []
    diversoty_list_k10 = []
    diversoty_list_k20 = []
    diversoty_list_k30 = []
    for i in range(len(score_res_set)):
        score_list = torch.stack(score_res_set[i]) 
        t_sorted_indices = torch.argsort(score_list, descending=True, dim=1) # 排序多样性
        unique_sequences = torch.unique(t_sorted_indices[:, :1], dim=0) # 唯一排序序列的数量，也就是排序多样性
        diversoty_list_k1.append(unique_sequences.size(0))
        unique_sequences = torch.unique(t_sorted_indices[:, :5], dim=0) 
        diversoty_list_k5.append(unique_sequences.size(0))
        unique_sequences = torch.unique(t_sorted_indices[:, :10], dim=0) 
        diversoty_list_k10.append(unique_sequences.size(0))
        unique_sequences = torch.unique(t_sorted_indices[:, :20], dim=0) 
        diversoty_list_k20.append(unique_sequences.size(0))
        unique_sequences = torch.unique(t_sorted_indices[:, :30], dim=0) 
        diversoty_list_k30.append(unique_sequences.size(0))

    print('diversity@1: ', np.mean(diversoty_list_k1),
    'diversity@5: ', np.mean(diversoty_list_k5),
    'diversity@10: ', np.mean(diversoty_list_k10), 
    'diversity@20: ', np.mean(diversoty_list_k20),
     'diversity@30: ', np.mean(diversoty_list_k30))

    agg = compute_metrics(test_metrics)
    print('Test----------------------step--------------------------------')
    print({f"{key}": val for key, val in agg.items()})
    print('Test_End----------------------step--------------------------------')
    sys.stdout.flush()

def load_data(data_dir, transform, stage, feat_dim):
    path = os.path.join(data_dir, f"{stage}.txt")
    nrows = 1000 if DEBUG else None
    df = pd.read_csv(path, sep=' ', header=None, nrows=nrows).dropna(axis=1)
    df.loc[:, 1:] = df.loc[:, 1:].apply(lambda row: [el.split(':')[1] for el in row])
    df.columns = ['target'] + ['qid'] + [f'feat_{i}' for i in range(1, feat_dim+1)]
    df = df.astype(float)
    df[['target', 'qid']] = df[['target', 'qid']].astype(int)

    user_model = {
        'seen_max': 16,
        'seen_bootstrap': 10,
        'click_noise': .1,
        'purchase_intent_kappa': .1,
        'purchase_noise': 0.
    }
    user_model = None
    data = LearningToRankDataset(df, label_column='target', list_id_column='qid', transform=transform,
                                 user_model=user_model, seed=SEED)
    return data

def load_data_featFill(data_dir, transform, stage, feat_dim):
    path = os.path.join(data_dir, f"{stage}.txt")
    data = []
    with open(path, 'r') as file:
        for line in file:
            parts = line.split(' ')
            target = parts[0]
            query_id = parts[1].split(':')[1]
            features = {f"feat_{i}": float(0.0) for i in range(feat_dim)}
            for part in parts[2:] :
                features[f"feat_{part.split(':')[0]}"] = float(part.split(':')[1])
            features['qid'] = query_id
            features['target'] = target
            data.append(features)
    df = pd.DataFrame(data)  
    df = df.astype(float)
    df[['target', 'qid']] = df[['target', 'qid']].astype(int)

    data = LearningToRankDataset(df, label_column='target', list_id_column='qid', transform=transform,
                                 user_model=None, seed=SEED)
    return data

def update_metrics(batch, score, loss, metrics):
    target = batch['target']
    length = batch['length']

    metrics['loss'].update(loss.item() * length.shape[0], weight=length.shape[0])
    if 'ndcg' in metrics: metrics['ndcg'].update(score, target, length)
    if 'ndcg10' in metrics: metrics['ndcg10'].update(score, target, length)
    if 'ndcg1' in metrics: metrics['ndcg1'].update(score, target, length)
    if 'ndcg3' in metrics: metrics['ndcg3'].update(score, target, length)
    if 'ndcg5' in metrics: metrics['ndcg5'].update(score, target, length)
    if 'ndcg15' in metrics: metrics['ndcg15'].update(score, target, length)
    if 'ndcg20' in metrics: metrics['ndcg20'].update(score, target, length)
    if 'err1' in metrics: metrics['err1'].update(score, target, length)
    if 'err5' in metrics: metrics['err5'].update(score, target, length)
    if 'err10' in metrics: metrics['err10'].update(score, target, length)
    if 'err' in metrics: metrics['err'].update(score, target, length)
    if 'err20' in metrics: metrics['err20'].update(score, target, length)
    if 'MRR' in metrics: metrics['MRR'].update(score, target, length)
    if 'MRR1' in metrics: metrics['MRR1'].update(score, target, length)
    if 'MRR5' in metrics: metrics['MRR5'].update(score, target, length)
    if 'MRR10' in metrics: metrics['MRR10'].update(score, target, length)
    if 'MRR20' in metrics: metrics['MRR20'].update(score, target, length)
    if 'MAP' in metrics: metrics['MAP'].update(score, target, length)
    if 'MAP1' in metrics: metrics['MAP1'].update(score, target, length)
    if 'MAP5' in metrics: metrics['MAP5'].update(score, target, length)
    if 'MAP10' in metrics: metrics['MAP10'].update(score, target, length)
    if 'MAP20' in metrics: metrics['MAP20'].update(score, target, length)
    if 'Precision' in metrics: metrics['Precision'].update(score, target, length)
    if 'Precision1' in metrics: metrics['Precision1'].update(score, target, length)
    if 'Precision5' in metrics: metrics['Precision5'].update(score, target, length)
    if 'Precision10' in metrics: metrics['Precision10'].update(score, target, length)
    if 'Precision20' in metrics: metrics['Precision20'].update(score, target, length)
    if 'explicit_target' in batch:
        explicit_target = batch['explicit_target'].to(DEVICE)
        metrics['ndcg_explicit'].update(score, explicit_target, length)


def compute_metrics(metrics):
    return {key: metric.compute() for key, metric in metrics.items()}


if __name__ == '__main__':
    main()
