import torch
from torch.nn.utils.rnn import pad_sequence

from loss import OrdinalLoss, SoftmaxLoss, ApproxNDCGLoss, MSELoss, OrdinalLoss, PointwiseLoss, ListNetLoss, RankNetLoss, NDCGLoss2PPLoss, LambdaLoss
from diffusion import DenoiseRankDiffusion

class DenoiseRank(torch.nn.Module):
    def __init__(self, args
                 ):

        super().__init__()
        
        # 设置嵌入维度（隐藏大小）
        self.emb_dim = args.hidden_size
        # 初始化核心predict模型
        self.diffu = DenoiseRankDiffusion(args)

        self.rank_loss_fn = MSELoss()

    # 6. concat rank loc noise to input doc and denoise
    def forward(self, feat, labels, length, isTraining = True):

        feat_per_list = feat.split(length.tolist())
        label_per_list = labels.split(length.tolist())
        
        pad_feat = pad_sequence(feat_per_list, batch_first=True, padding_value=0)
        pad_label = pad_sequence(label_per_list, batch_first=True, padding_value=0)
        
        pad_label = pad_label.float() / 4.0   # 2.直接除以最大值，归一化
        
        # 掩膜
        padding_mask = torch.ones((pad_feat.shape[0], pad_feat.shape[1]), dtype=torch.bool).to(pad_feat.device)
        for i, list_len in enumerate(length):
            padding_mask[i, :list_len] = False

        rank_score = None
        if isTraining:
            ###
            rank_score, weights, t = self.diffu(pad_feat, pad_label, padding_mask)
            rank_score = rank_score[~padding_mask]
        else:
            noise_x_t = torch.rand_like(pad_label)
            rank_score = self.reverse(pad_feat, noise_x_t, padding_mask)
            rank_score = rank_score[~padding_mask]
        return rank_score
    
    def continuous_mapping(self, logits):
        """
        输入: logits.shape = [n, m, 5]
        输出: [n, m] (0~4之间的连续值)
        """
        probs = torch.softmax(logits, dim=-1)            # 生成概率分布 [n, m, 5]
        indices = torch.arange(5, device=logits.device) # 创建索引张量 [0,1,2,3,4]
        weighted_sum = torch.sum(probs * indices, dim=-1) # 加权求和 [n, m]
        return weighted_sum / 4.0
    
    def reverse(self, item_rep, noise_x_t, mask_seq):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, noise_x_t, mask_seq)
        # 返回逆向传播的结果
        return reverse_pre
    
    def compute_loss(self, score, target, length, args):
        rank_loss = None
        target = target/4.0
        if args.loss_type == 'softmax':
            self.rank_loss_fn = SoftmaxLoss()
        elif args.loss_type == 'approxNDCG':
            self.rank_loss_fn = ApproxNDCGLoss()
        elif args.loss_type == 'MSE':
            self.rank_loss_fn = MSELoss()
        elif args.loss_type == 'Oridinal':
            self.rank_loss_fn = OrdinalLoss()
        elif args.loss_type == 'Pointwise':
            self.rank_loss_fn = PointwiseLoss()
        elif args.loss_type == 'ListNet':
            self.rank_loss_fn = ListNetLoss()
        elif args.loss_type == 'RankNet':
            self.rank_loss_fn = RankNetLoss()
        elif args.loss_type == 'NDCG2PP':
            self.rank_loss_fn = NDCGLoss2PPLoss()
        elif args.loss_type == 'LambdaLoss':
            self.rank_loss_fn = LambdaLoss()
        rank_loss = self.rank_loss_fn.forward_per_list(score, target, length)
            
        return rank_loss


class MLP(torch.nn.Module):
    def __init__(self, input_dim,
                 hidden_layers=None,
                 output_dim=1,
                 dropout=0.):
        super().__init__()

        net = []
        for h_dim in hidden_layers:
            net.append(torch.nn.Linear(input_dim, h_dim))
            net.append(torch.nn.ReLU())
            if dropout > 0.:
                net.append(torch.nn.Dropout(dropout))
            input_dim = h_dim
        net.append(torch.nn.Linear(input_dim, output_dim))

        self.net = torch.nn.Sequential(*net)
        self.rank_loss_fn = SoftmaxLoss()

    def forward(self, feat, *_args):
        score = self.net(feat).squeeze(dim=-1)
        return score

    def compute_loss(self, score, target, length):
        loss = self.rank_loss_fn.forward_per_list(score, target, length)
        return loss
