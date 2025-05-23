import torch.nn as nn
import torch as th
from step_sample import create_named_schedule_sampler
import numpy as np
import math
import torch
import torch.nn.functional as F
from loss import OrdinalLoss, SoftmaxLoss, ApproxNDCGLoss

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar in the limit of num_diffusion_timesteps. Beta schedules may be added, but should not be removed or changed once they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(num_diffusion_timesteps, lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,)
    elif schedule_name == 'sqrt':
        return betas_for_alpha_bar(num_diffusion_timesteps,lambda t: 1-np.sqrt(t + 0.0001),  )
    elif schedule_name == "trunc_cos":
        return betas_for_alpha_bar_left(num_diffusion_timesteps, lambda t: np.cos((t + 0.1) / 1.1 * np.pi / 2) ** 2,)
    elif schedule_name == 'trunc_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_end = scale * 0.02 + 0.01
        if beta_end > 1:
            beta_end = scale * 0.001 + 0.01
        return np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif schedule_name == 'pw_lin':
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001 + 0.01
        beta_mid = scale * 0.0001  #scale * 0.02
        beta_end = scale * 0.02
        first_part = np.linspace(beta_start, beta_mid, 10, dtype=np.float64)
        second_part = np.linspace(beta_mid, beta_end, num_diffusion_timesteps - 10 , dtype=np.float64)
        return np.concatenate([first_part, second_part])
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and produces the cumulative product of (1-beta) up to that part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):  ## 2000
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def betas_for_alpha_bar_left(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, but shifts towards left interval starting from 0
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    betas.append(min(1-alpha_bar(0), max_beta))
    for i in range(num_diffusion_timesteps-1):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SiLU(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, hidden_size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(hidden_size, hidden_size*4)
        self.w_2 = nn.Linear(hidden_size*4, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_1.weight)
        nn.init.xavier_normal_(self.w_2.weight)

    def forward(self, hidden):
        hidden = self.w_1(hidden)
        activation = 0.5 * hidden * (1 + torch.tanh(math.sqrt(2 / math.pi) * (hidden + 0.044715 * torch.pow(hidden, 3))))
        return self.w_2(self.dropout(activation))


class MultiHeadedAttention(nn.Module):
    def __init__(self, heads, hidden_size, dropout):
        super().__init__()
        assert hidden_size % heads == 0
        self.size_head = hidden_size // heads
        self.num_heads = heads
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
        self.w_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_normal_(self.w_layer.weight)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q, k, v = [l(x).view(batch_size, -1, self.num_heads, self.size_head).transpose(1, 2) for l, x in zip(self.linear_layers, (q, k, v))]
        corr = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
        
        if mask is not None:
            mask = mask.unsqueeze(1).repeat([1, corr.shape[1], 1]).unsqueeze(-1).repeat([1,1,1,corr.shape[-1]])
            corr = corr.masked_fill(mask == 0, -1e9)
        prob_attn = F.softmax(corr, dim=-1)
        if self.dropout is not None:
            prob_attn = self.dropout(prob_attn)
        hidden = torch.matmul(prob_attn, v)
        hidden = self.w_layer(hidden.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.size_head))
        return hidden


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size, attn_heads, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(heads=attn_heads, hidden_size=hidden_size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, dropout=dropout)
        self.input_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.output_sublayer = SublayerConnection(hidden_size=hidden_size, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, hidden, mask):
        hidden = self.input_sublayer(hidden, lambda _hidden: self.attention.forward(_hidden, _hidden, _hidden, mask=mask))
        hidden = self.output_sublayer(hidden, self.feed_forward)
        return self.dropout(hidden)


class Transformer_rep(nn.Module):
    def __init__(self, args):
        super(Transformer_rep, self).__init__()
        self.hidden_size = args.hidden_size
        self.heads = 4
        self.dropout = args.dropout
        self.n_blocks = args.num_blocks
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(self.hidden_size, self.heads, self.dropout) for _ in range(self.n_blocks)])

    def forward(self, hidden, mask):
        for transformer in self.transformer_blocks:
            hidden = transformer.forward(hidden, mask)
        return hidden

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
    
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, dropout=0.):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        net = []
        net.append(nn.Linear(num_in, num_out))
        net.append(torch.nn.Softplus())
        if dropout > 0.:
            net.append(nn.Dropout(dropout))
        self.net = torch.nn.Sequential(*net)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.net(x)
        gamma = self.embed(t)
        g_v = gamma.unsqueeze(1)
        rout = g_v * out
            
        return rout

class ConditionalFeedwordNetwork(nn.Module):
    def __init__(self, x_dim, y_dim, timesteps, out_dim, dropout=0., num_MLP_layers=4, hidden_size=128):
        super(ConditionalFeedwordNetwork, self).__init__()
        self.num_MLP_layers = num_MLP_layers
        n_steps = timesteps + 1
        self.cat_x = True
        self.cat_y = True
        data_dim = 0
        if self.cat_y:
            data_dim += y_dim
        if self.cat_x:
            data_dim += x_dim
        self.lin1 = ConditionalLinear(data_dim, hidden_size, n_steps, dropout)
        if num_MLP_layers>2: self.lin2 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        if num_MLP_layers>3: self.lin3 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        if num_MLP_layers>4: self.lin5 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        if num_MLP_layers>5: self.lin6 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        if num_MLP_layers>6: self.lin7 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        if num_MLP_layers>7: self.lin8 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        if num_MLP_layers>8: self.lin9 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        if num_MLP_layers>9: self.lin10 = ConditionalLinear(hidden_size, hidden_size, n_steps, dropout)
        self.lin4 = nn.Linear(hidden_size, out_dim)

    def forward(self, x, y_t, t):
        
        eps_pred = torch.cat((y_t, x), dim=-1)
        eps_pred = self.lin1(eps_pred, t)
        if self.num_MLP_layers>2: eps_pred = self.lin2(eps_pred, t)
        if self.num_MLP_layers>3: eps_pred = self.lin3(eps_pred, t)
        if self.num_MLP_layers>4: eps_pred = self.lin5(eps_pred, t)
        if self.num_MLP_layers>5: eps_pred = self.lin6(eps_pred, t)
        if self.num_MLP_layers>6: eps_pred = self.lin7(eps_pred, t)
        if self.num_MLP_layers>7: eps_pred = self.lin8(eps_pred, t)
        if self.num_MLP_layers>8: eps_pred = self.lin9(eps_pred, t)
        if self.num_MLP_layers>9: eps_pred = self.lin10(eps_pred, t)
        return self.lin4(eps_pred)

    
    
class DenoiseNeuralNetwork(nn.Module):
    def __init__(self, hidden_size, args):
        """
        初始化函数

        参数:
        - hidden_size: 隐藏层大小
        - args: 其他参数
        """
        # 调用父类的初始化方法
        super(DenoiseNeuralNetwork, self).__init__()
        self.input_SA = args.input_SA
        # 时间步
        self.diffusion_steps = args.diffusion_steps
        self.hidden_size = hidden_size  # 初始化隐藏层大小属性
        self.linear_item = nn.Linear(self.hidden_size, self.hidden_size)  # 线性层，输入和输出维度均为隐藏层大小
        self.linear_xt = nn.Linear(self.hidden_size, self.hidden_size)  # 另一个线性层
        self.linear_t = nn.Linear(self.hidden_size, self.hidden_size)  # 还有一个线性层
        
        self.time_dim = args.tf_input_dim
        time_embed_dim = self.time_dim * 4  # 计算时间嵌入维度
        # 构建时间嵌入的序列模块
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, time_embed_dim),  # 线性层
            SiLU(),  # 激活函数
            nn.Linear(time_embed_dim, self.time_dim)  # 线性层
        )
        
        # 构建标签嵌入的序列模块
        self.label_dim = args.tf_input_dim
        label_embed_dim = args.tf_input_dim * 4  # 计算时间嵌入维度
        self.label_embed = nn.Sequential(
            nn.Linear(1, label_embed_dim),  # 线性层
            SiLU(),  # 激活函数
            nn.Linear(label_embed_dim, self.label_dim)  # 线性层
        )
        self.tf_emb_dim = args.hidden_size
        encoder_layer = torch.nn.TransformerEncoderLayer(self.tf_emb_dim, nhead=args.num_heads,
                                                         dim_feedforward=512, dropout=args.dropout,
                                                         activation='gelu', batch_first=True,
                                                         norm_first=True)
        # Note: the 'norm' parameter is set to 'None' here, because the TransformerEncoderLayer already computes it
        self.att = torch.nn.TransformerEncoder(encoder_layer, num_layers=args.num_blocks, norm=None)
        
        # 1.原始 rank score MLP
        self.rank_score_input_dim = self.tf_emb_dim
        
        # 2.conditional MLP
        self.rank_score_net = ConditionalFeedwordNetwork(x_dim = self.rank_score_input_dim,
                                                     y_dim = 1,
                                                     timesteps = self.diffusion_steps,
                                                     out_dim = 5,
                                                     dropout=args.dropout,
                                                     num_MLP_layers=args.num_MLP_layer,
                                                     hidden_size=args.mlp_hidden_size)
        
        self.dropout = nn.Dropout(args.dropout) 
        self.norm_diffu_rep = LayerNorm(self.hidden_size)  

    def timestep_embedding(self, timesteps, dim, max_period=10000):
        """
        创建正弦时间步嵌入

        参数:
        - timesteps: 一维张量，每个元素对应一个批次元素，可能是分数
        - dim: 输出的维度
        - max_period: 控制嵌入的最小频率

        返回:
        - 一个 [N x dim] 的位置嵌入张量
        """
        half = dim // 2  
        freqs = th.exp(-math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half).to(device=timesteps.device)  
        args = timesteps[:, None].float() * freqs[None]  # 计算参数
        embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
        if dim % 2:  
            embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
        return embedding  

    def forward(self, rep_item, x_t, t, mask_seq):
        """
        前向传播函数

        参数:
        - rep_item: 文档列表 [N,M,D]
        - x_t: 输入，这里以labels取加噪声，[N, M]
        - t: 时间步
        - mask_seq: 掩码序列

        返回:
        - rank_score: [N，M] 即预测出来的标签，去噪后的结果，连续值
        """ 
        rep_diffu = None
        if self.input_SA:
            rep_diffu = self.att(rep_item, src_key_padding_mask=mask_seq)  
        else:
            rep_diffu = rep_item
        rank_score = self.rank_score_net(x=rep_diffu,
                                         y_t=x_t.unsqueeze(-1),
                                         t=t)
        # onehot_ranks = torch.softmax(rank_score, dim=-1) # 归一化
        rank_score = self.continuous_mapping(rank_score) 

        return rank_score # [N，M] 即预测出来的标签，去噪后的结果，连续值 | [N，M, 5]one-hot归一化的输出

    def continuous_mapping(self, logits):
        """
        输入: logits.shape = [n, m, 5]
        输出: [n, m] (0~4之间的连续值)
        """
        probs = torch.softmax(logits, dim=-1)            # 生成概率分布 [n, m, 5]
        indices = torch.arange(5, device=logits.device) # 创建索引张量 [0,1,2,3,4]
        weighted_sum = torch.sum(probs * indices, dim=-1) # 加权求和 [n, m]
        return weighted_sum / 4.0

# 定义DiffuRec类，继承自nn.Module
class DenoiseRankDiffusion(nn.Module):
    # 初始化函数，接收参数args
    def __init__(self, args):
        # 调用父类初始化函数
        super(DenoiseRankDiffusion, self).__init__()
        
        self.hidden_size = args.hidden_size
        self.schedule_sampler_name = args.schedule_sampler_name
        self.diffusion_steps = args.diffusion_steps
        
        self.use_timesteps = space_timesteps(self.diffusion_steps, [self.diffusion_steps])
        
        self.noise_schedule = args.noise_schedule
        betas = self.get_betas(self.noise_schedule, self.diffusion_steps)
        self.betas = np.array(betas, dtype=np.float64)
        assert len(self.betas.shape) == 1, "betas must be 1-D"
        assert (self.betas > 0).all() and (self.betas <= 1).all()
        alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)

        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef3 = (1.0 + ((self.sqrt_alphas_cumprod-1.0)*(np.sqrt(alphas) + np.sqrt(self.alphas_cumprod_prev)))/(1.0 - self.alphas_cumprod))

        self.posterior_variance = (self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        self.num_timesteps = int(self.betas.shape[0])
        
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.num_timesteps)  ## lossaware (schedule_sample)
        self.timestep_map = self.time_map()
        self.rescale_timesteps = args.rescale_timesteps
        self.original_num_steps = len(self.betas)

        self.xstart_model = DenoiseNeuralNetwork(self.hidden_size, args)

    def get_betas(self, noise_schedule, diffusion_steps):
        betas = get_named_beta_schedule(noise_schedule, diffusion_steps)
        return betas

    def q_sample(self, x_start, t, noise=None, mask=None):
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        x_t = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise 
        )  
            
        if mask == None:
            return x_t
        else:
            mask = th.broadcast_to(mask.unsqueeze(dim=-1), x_start.shape)  
            return th.where(mask==0, x_start, x_t)

    def time_map(self):
        timestep_map = []
        for i in range(len(self.alphas_cumprod)):
            if i in self.use_timesteps:
                timestep_map.append(i)
        return timestep_map

    def _scale_timesteps(self, t):
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        assert x_start.shape == x_t.shape
        
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        ) 
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, rep_item, x_t, t, mask_seq):
        model_output = self.xstart_model(rep_item, x_t, self._scale_timesteps(t), mask_seq)
        x_0 = model_output 
        
        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = _extract_into_tensor(model_log_variance, t, x_t.shape)
        
        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t, t=t)  
        return model_mean, model_log_variance

    def p_sample(self, item_rep, noise_x_t, t, mask_seq):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t, mask_seq)
        noise = th.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))
        sample_xt = model_mean + nonzero_mask * th.exp(0.5 * model_log_variance) * noise
        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t, mask_seq):
        device = next(self.xstart_model.parameters()).device
        indices = list(range(self.num_timesteps))[::-1]
        for i in indices:
            t = th.tensor([i] * item_rep.shape[0], device=device)
            with th.no_grad():
                noise_x_t = self.p_sample(item_rep, noise_x_t, t, mask_seq)
        return noise_x_t 

    def forward(self, item_rep, item_tag, mask_seq):     
        """
        输入: 
            item_rep：[N, M, D]
            item_tag：[N, M]
            mask_seq: [N, M]
            
        输出: [n, m] (0~4之间的连续值)
        """   
        noise = th.randn_like(item_tag)
        t, weights = self.schedule_sampler.sample(item_rep.shape[0], item_tag.device) 
        
        x_t = self.q_sample(item_tag, t, noise=noise)
        
        x_0 = self.xstart_model(item_rep, x_t, self._scale_timesteps(t), mask_seq) 
        return x_0, weights, t
