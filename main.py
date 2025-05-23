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
from metrics import NDCG, Average, ERR, MRR, MAP, Precision
from model import DenoiseRank
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler

parser = argparse.ArgumentParser()
parser.add_argument('--epocs', type=int, default=300, help='train epocs')  
parser.add_argument('--datasets', default='istella', help='Selection: web30k istella yahoo')
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
DEBUG = False  
DEVICE = 'cpu'
_CUTOFF = 30.
NUM_WORKERS = 0


def main():
    """ DDP Training """
    try:
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group("nccl")
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
    random.seed(SEED) 
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    
    transform = None
    hidden_size = 136
    train_data = None
    test_data = None
    if args.datasets=='web30k':
        data_dir = os.path.join(ROOT_DIR, 'MSLR-WEB30K', f"Fold{SEED % 5 + 1}")#{SEED % 5 + 1}
        transform = log1pTransform()
        hidden_size = 136
        train_data = load_data(data_dir, transform, 'train', feat_dim=hidden_size)
        test_data = load_data(data_dir, transform, 'test', feat_dim=hidden_size)
    elif args.datasets=='istella':
        data_dir = os.path.join(ROOT_DIR, 'ISTELLA', f"Fold1")
        transform = log1pTransform()
        hidden_size = 220
        train_data = load_data(data_dir, transform, 'train', feat_dim=hidden_size)
        test_data = load_data(data_dir, transform, 'test', feat_dim=hidden_size)
    elif args.datasets=='yahoo':
        data_dir = os.path.join(ROOT_DIR, 'YAHOO', f"Fold1")
        transform = QuantileTransformer(output_distribution='normal')
        hidden_size = 700
        train_data = load_data_featFill(data_dir, transform, 'train', feat_dim=hidden_size)
        test_data = load_data_featFill(data_dir, transform, 'test', feat_dim=hidden_size)
    
    args.hidden_size = hidden_size
    train_sampler = distributed.DistributedSampler(train_data, shuffle=True)
    train_loader = DataLoader(train_data, batch_size=args.batch_size,  collate_fn=LearningToRankDataset.collate_fn,
                              num_workers=NUM_WORKERS, sampler=train_sampler)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=LearningToRankDataset.collate_fn,
                             num_workers=NUM_WORKERS)
    model = DenoiseRank(args=args).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.1)#weight_decay=.1
    learning_decay_start = 20
    learning_decay = -.5
    decay_fn = lambda t: (t - learning_decay_start + 1) ** learning_decay if t >= learning_decay_start else 1.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, decay_fn)
    

    best_model = None
    best_metrics_dict = {}

    ##################################
    ############ Training ############
    ##################################
    model.train()
    train_metrics = {
        'ndcg10': NDCG(k=10),
        'ndcg1': NDCG(k=1),
        'ndcg5': NDCG(k=5),
        'ndcg': NDCG(k=0),
        'loss': Average()
    }
    ep_num = args.epocs
    loop = tqdm(range(ep_num))
    for _epoch in loop:
        model.train()
        torch.distributed.barrier()
        for batch in train_loader:
            feat, length, target = batch['feat'].to(DEVICE), batch['length'].to(DEVICE), batch['target'].to(DEVICE)
            
            score = model(feat, target, length, True)
            loss = model.compute_loss(score, target, length, args)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if isinstance(score, tuple):
                score = score[0]
            update_metrics(batch, score.detach(), loss, train_metrics)
        scheduler.step()

        agg = compute_metrics(train_metrics)
        loop.set_postfix("")
        if dist.get_rank()==0:
            print('----------------------',_epoch,'--------------------------------')
            print({f"train_{key}": val for key, val in agg.items()})     
            
        sys.stdout.flush()
        
        ##################################
        ############ Evaluate ############
        ##################################
        if _epoch>0 and (_epoch%10==0 or _epoch==ep_num-1) and dist.get_rank()==0:
            print('当前模型学习率：', optimizer.param_groups[0]['lr'])
            
            test_metrics_list = process_test(model, test_loader, [args.diffusion_steps], DEVICE, args)
            test_metrics = test_metrics_list[-1]
            # 记录最佳指标和模型
            need_record = False
            if best_model == None:
                need_record = True
            else:
                need_record = (best_metrics_dict['Best_ndcg10'] < test_metrics['ndcg10'].agg)
                
            if need_record:
                best_model = copy.deepcopy(model)
                for key_temp, values_temp in test_metrics.items():
                    best_metrics_dict['Best_' + key_temp] = values_temp.agg


    # torch.save(best_model, os.path.join(ROOT_DIR, 'model_save', 'bestModel_web30k_0519.pth'))
    # # best_model_test(best_model, test_loader, pre_train_model, DEVICE, args)

                    
    print('Best----------------------Best--------------------------------')
    print({f"{key}": val for key, val in best_metrics_dict.items()})
    sys.stdout.flush()
    
    
def best_model_test(model, test_loader, DEVICE, args):
    steps = [args.diffusion_steps]
    process_test(model, test_loader, steps, DEVICE, args)

def process_test(model, test_loader, steps, DEVICE, args):
    cur_model = copy.deepcopy(model)
    
    cur_model.eval()
    metrics_list = []
    for s in steps:
        cur_model.diffu.num_timesteps = s
        test_metrics = {
            'ndcg10': NDCG(k=10),
            'ndcg1': NDCG(k=1),
            'ndcg5': NDCG(k=5),
            'loss': Average(),
            'ndcg': NDCG(k=0),
            'ndcg3': NDCG(k=3),
            'ndcg15': NDCG(k=15),
            'ndcg20': NDCG(k=20),
            'err': ERR(k=0),
            'err1': ERR(k=1),
            'err5': ERR(k=5),
            'err10': ERR(k=10),
            'err20': ERR(k=20),
            'MRR': MRR(k=0),
            'MRR1': MRR(k=1),
            'MRR5': MRR(k=5),
            'MRR10': MRR(k=10),
            'MRR20': MRR(k=20),
            'MAP': MAP(k=0),
            'MAP1': MAP(k=1),
            'MAP5': MAP(k=5),
            'MAP10': MAP(k=10),
            'MAP20': MAP(k=20),
            'Precision': Precision(k=0),
            'Precision1': Precision(k=1),
            'Precision5': Precision(k=5),
            'Precision10': Precision(k=10),
            'Precision20': Precision(k=20),
        }
        with torch.no_grad():
            for batch in test_loader:
                feat, length, target = batch['feat'].to(DEVICE), batch['length'].to(DEVICE), batch['target'].to(DEVICE)
                    
                score = cur_model(feat, target, length, False)
                loss = cur_model.compute_loss(score, target, length, args)

                if isinstance(score, tuple):
                    score = score[0]
                update_metrics(batch, score, loss, test_metrics)
                
        agg = compute_metrics(test_metrics)
        print('Test----------------------timestep:', s,'--------------------------------')
        print({f"{key}": val for key, val in agg.items()})
        print('Test_End----------------------timestep:', s,'--------------------------------')
        sys.stdout.flush()
        metrics_list.append(test_metrics)
    return metrics_list

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

# Some dims of the feature are none and need to fill
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

class log1pTransform:
    def __init__(self):
        self.minmax_scaler = MinMaxScaler(feature_range=(-1, 1))

    def transform(self, X):
        log1p_x = np.log1p(np.abs(X))  
        sign_x = np.sign(X)  
        X_transformed = log1p_x * sign_x
        X_scaled = np.clip(
            X_transformed, 
            min=-_CUTOFF, 
            max=_CUTOFF
        )
        return X_scaled




if __name__ == '__main__':
    main()
