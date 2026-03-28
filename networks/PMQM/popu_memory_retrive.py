"""
PMQM - retrieval version
(major differences are in MemoryCacheEncoding-forward )
the implementation of 'sinkhorn_knopp' and 'MemoryCacheEncoding' are inspired by https://github.com/facebookresearch/swav/blob/main/main_swav.py

(Thanks!)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import math


@torch.no_grad()
def sinkhorn_knopp(out):
    """
    Non-distributed version of Sinkhorn-Knopp algorithm
    """
    # parameters for SK
    epsilon = 0.03  # for numerical stability of Sinkhorn-Knopp
    sinkhorn_iterations = 3  # iters for Sinkhorn-Knopp

    # compute the Q matrix
    Q = torch.exp(out / epsilon).t()  # Q: K×B
    B = Q.shape[1]  # num of samples (batch size)
    K = Q.shape[0]  # memory length (a.k.a prototype length)

    # Normalize the matrix so that the sum equals 1
    Q /= torch.sum(Q)

    # Sinkhorn-Knopp iteration
    for _ in range(sinkhorn_iterations):
        # Normalize each row: the total weight of each prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # Normalize each column: the total weight of each sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  #  Column sums must equal 1 to ensure Q is an assignment matrix
    return Q.t()


class QKVAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        """
        QKV Attention implementation
        Args:
            d_model: Input dimension
            dropout: Dropout rate
        """
        super().__init__()

        self.d_model = d_model

        # Linear transformations for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output linear transformation
        self.W_o = nn.Linear(d_model, d_model)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, query, key, value, mask=None):
        """
        Scaled Dot-Product Attention computation
        Args:
            query: Query matrix [batch_size, d_k]
            key: Key matrix [N, d_k]
            value: Value matrix [N, d_k]
            mask: Mask matrix [batch_size, d_k]
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.T) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    def forward(self, query, key, value, mask=None):
        """
        Forward pass for QKV Attention
        Args:
            query: Query tensor [batch_size, d_model]
            key: Key tensor [N, d_model]
            value: Value tensor [N, d_model]
            mask: Mask tensor [batch_size, d_model] (optional)
        """
        # Apply linear transformations to Q, K, V
        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        # Compute scaled dot-product attention
        attn_output, attention_weights = self.scaled_dot_product_attention(query, key, value, mask)

        # Apply the output linear transformation
        output = self.W_o(attn_output)  # [batch_size, d_model]

        return output, attention_weights

class TimeAwaredWeightedSum(nn.Module):
    def __init__(self, t_len):
        super().__init__()
        self.time_weight = nn.Sequential(nn.Linear(t_len, t_len, bias=False),
                                         nn.Sigmoid())  # calculate learnable weight according to time
        nn.init.kaiming_normal_(self.time_weight[0].weight, a=0, mode='fan_out')

    def forward(self, x, time):
        # x: [N,t_len,dm], time: [N,t_len,1]
        time = time.float()
        w_t = self.time_weight(time.squeeze(-1)).unsqueeze(-1) # [N,t_len,1]
        weighted_x = (x * w_t).sum(dim=1, keepdim=False)  # [N,dm]
        return weighted_x


class DownSample(nn.Module):
    def __init__(self, N, L, d_model, dropout=0.1):
        super().__init__()
        self.spatial_conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=N, out_channels=N, kernel_size=3,  padding='same', padding_mode='reflect'),
            nn.GELU(),
            nn.LayerNorm([N, d_model]),
        )
        self.spatial_conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=N, out_channels=N, kernel_size=5, padding='same', padding_mode='reflect'),
            nn.GELU(),
            nn.LayerNorm([N, d_model]),
        )
        self.dropout_sc1 = nn.Dropout(p=dropout)

        self.spatial_conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=N, out_channels=1, kernel_size=1),
            nn.GELU(),
            nn.LayerNorm([1, d_model]))
        self.dropout_sc2 = nn.Dropout(p=dropout)
        self.temporal_weight_sum = TimeAwaredWeightedSum(L)
        self.final_linear = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(p=dropout)
        )

    def spatial_conv(self, x):
        # x: [B*L, N, d_model]
        x = self.dropout_sc1(self.spatial_conv_1(x) + self.spatial_conv_2(x) + x) #  [B*L, N, d_model] combine 2 scales feature map and residual
        x = self.spatial_conv_3(x)  #  [B*L, 1, d_model]
        return x

    def forward(self, x, t_seq):  # x: [B, L, N, d_model]
        """
        Scaled Dot-Product Attention computation
        Args:
            query: Query matrix [batch_size, d_k]
            key: Key matrix [N, d_k]
            value: Value matrix [N, d_k]
            mask: Mask matrix [batch_size, d_k]
        """
        B, L, N, D = x.shape
        x = x.reshape(B * L, N, D)  # [B*L, N, d_model]
        x = self.spatial_conv(x)  #  [B*L, 1, d_model]
        x = x.reshape(B, L, D)  # [B, L, d_model]
        x = self.temporal_weight_sum(x, t_seq)  # [B, d_model]
        x = self.final_linear(x)  # [B, d_model]
        return x

class MemoryCacheEncoding(nn.Module):
    def __init__(self, d_model=768, memory_len: int=100, patch_num: int=256, num_crop: int=2, seq_length=6, device="cuda",dropout=0.1):
        """
        Memory Cache Encoding
        Args:
            d_model (int): Dimension of the model features. Default: 768
            memory_len (int): Number of samples to store in memory cache. Default: 100
            seq_length (int): Length of input sequences. Default: 6
            patch_num (int): number of patches for image patch embedding
            num_crop(int): number of crop for SwAV
            device (str): Device to store the memory bank. Default: "cuda"
        """
        super(MemoryCacheEncoding, self).__init__()

        self.d_model = d_model # d_model wrt image encoder
        self.device = device
        self.seq_length = seq_length
        self.memory_len = memory_len
        assert memory_len >= 1, "memory_len must be an integer greater than or equal to 1"
        # init banks
        self.register_buffer('feature_bank', torch.zeros(memory_len, self.seq_length, self.d_model))
        self.register_buffer('time_bank', torch.zeros(memory_len, self.seq_length, 1))
        self.register_buffer('label_bank', torch.zeros(memory_len, self.seq_length, 1))


        self.num_crop = num_crop
        # init prototype (define directly that I don't need to call nn.Linear.weight if I use nn.Linear implementation as SwAV)
        # (similar to the nn.Linear implementation as SwAV github. For nn.Linear, its weight is prototype here)
        self.prototype = nn.Parameter(torch.randn(self.d_model,  self.memory_len))
        self.xt_embd_proj = DownSample(N=int(patch_num/num_crop), L=seq_length, d_model=d_model, dropout=dropout)
        self.swav_loss_value = None


    def norm_prototype(self):
        with torch.no_grad():
            self.prototype.data = torch.nn.functional.normalize(self.prototype.data, dim=0, p=2)

    def encoding_crops(self, crops, t_seq):
        _out = []
        for idx in range(len(crops)):
            _out.append(self.xt_embd_proj(crops[idx], t_seq))
        out = torch.cat(_out, dim=0)
        return out

    def get_assignment(self, pred):
        # take one crop for prediction
        bs= int(pred.size(0) // 2)
        pred = pred[bs:]
        pred_prob = F.softmax(pred, dim=1)
        return pred_prob

    def get_prototype_scores(self, embd):
        """
        calculate codes

        as the section in SwAV - A.1 Implementation details of SwAV training

        """

        # step 1: l2 norm projection (project feature into the unit sphere)
        embd = nn.functional.normalize(embd, dim=1, p=2)

        # step 2: compute prototype scores by matmul: (2B, dm) @ (dm, memory_len) = (2B, memory_len) (same as nn.linear implementation in github, for the nn.linear, the linear.weight is prototype here)
        scores = torch.matmul(embd, self.prototype)
        return scores

    def swav_loss(self, scores):
        """
        calculate swav loss

        as the section in SwAV - 3.1 Online clustering - Swapped prediction problem & Computing codes online.

        """
        bs = int(scores.size(0) // 2)

        crops_for_assign = [i for i in range(self.num_crop)]  # list of crops id used for computing assignments
        nmb_crops = [self.num_crop]  # list of number of crops (example: [2, 6])
        temperature = 0.1  # temperature parameter in training loss

        loss = 0
        for i, crop_id in enumerate(crops_for_assign):
            with torch.no_grad():
                out = scores[bs * crop_id: bs * (crop_id + 1)].detach()
                q = sinkhorn_knopp(out)[-bs:]

            # cluster assignment prediction
            subloss = 0
            for v in np.delete(np.arange(np.sum(nmb_crops)), crop_id):
                x = scores[bs * v: bs * (v + 1)] / temperature
                subloss -= torch.mean(torch.sum(q * F.log_softmax(x, dim=1), dim=1))
            loss += subloss / (np.sum(nmb_crops) - 1)
        loss /= len(crops_for_assign)
        return loss


    def select_crops(self, xt_embd):
        B, L, N, dm = xt_embd.shape
        assert ((N % self.num_crop) == 0), f"the shape N of xt_embd (B,L,N,dm) must be divisible by num_crop {self.num_crop}"

        crop_size = N // self.num_crop

        indices = torch.randperm(N)

        crops = []
        for i in range(self.num_crop):
            start_idx = i * crop_size
            end_idx = start_idx + crop_size
            crop_indices = indices[start_idx:end_idx]
            crop = xt_embd[:, :, crop_indices, :]
            crops.append(crop)

        return crops

    def forward(self, xt_seq_embd, t_seq, call_assign=False):
        if self.training:
            """ clustering """

            # step 1: normalize the prototype
            self.norm_prototype()
            # step 2: get multi-res crops for contrastive learning (here we select random 0.5N patchs from same patient)
            crops = self.select_crops(xt_seq_embd)
            # step 3: encoding crops to get features (B, L, 0.5N, dm) -> (B,dm), repeat twice, then get (2B,dm)
            embd = self.encoding_crops(crops, t_seq)
            # step 4: get scores for the prototype
            scores = self.get_prototype_scores(embd)
            # step 5: swav forward and calculate swav loss
            self.swav_loss_value = self.swav_loss(scores)

            # step 6: calculate assignment using score if the model is well-trained
            # if I want the score for assignment of each sample:
            # bs = xt_seq_embd.size(0)
            # score = scores[:bs]
            return self.prototype
        else:
            ''' the implementation of computing assigment for retrieve'''
            if call_assign:
                with torch.no_grad():
                    # step 1: normalize the prototype
                    self.norm_prototype()
                    # step 2: get multi-res crops for contrastive learning (here we select random 0.5N patchs from same patient)
                    crops = self.select_crops(xt_seq_embd)
                    # step 3: encoding crops to get features (B, L, 0.5N, dm) -> (B,dm), repeat twice, then get (2B,dm)
                    embd = self.encoding_crops(crops, t_seq)
                    # step 4: get scores for the prototype
                    scores = self.get_prototype_scores(embd)
                    pred = self.get_assignment(scores)
                    prototype = self.prototype
            else:
                pred = None
                prototype = self.prototype
            return prototype, pred



class PopuMemoryNet(nn.Module):
    """ the implementation of PMQM """
    def __init__(self, d_model=128, seq_len=6, memory_len=100, dropout=0.1):
        super(PopuMemoryNet,self).__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.dropout = dropout
        self.memory_len = memory_len  # the length of the memory bank (num of selected cases)

        self.mce = MemoryCacheEncoding(d_model=d_model, memory_len=memory_len, patch_num=256, num_crop=2, dropout=self.dropout)
        self.attention = QKVAttention(d_model=d_model,dropout=self.dropout)

        # Layer normalization
        self.norm = nn.LayerNorm(d_model)

    def get_kv(self, xt_seq_embd, t_seq, call_assign):
        """ build up k and v from memory bank """
        prototype, assign_prob = self.mce(xt_seq_embd, t_seq, call_assign=call_assign)  # (d_model, N)
        prototype = prototype.permute(1,0)  # (N, d_model)
        return prototype, prototype, assign_prob # k = prototype, v = prototype, assign_prob: assignment prob for retrival

    def attn(self, q, k, v):
        attn_output, attn_output_weights = self.attention(q, k, v)
        return attn_output, attn_output_weights

    def forward(self,ch, xt_seq_embd, t_seq, call_assign=False):
        '''
        return ch and assign_prob
        '''
        q = ch
        k,v, assign_prob = self.get_kv(xt_seq_embd, t_seq, call_assign=call_assign)
        attn_out, _ = self.attn(q,k,v)
        return self.norm(attn_out + ch), assign_prob # add & norm


def debug_propnet():
    import torch
    device = "cpu"
    B, L, N, dm = 20, 6, 256, 128
    xt_seq_embd = torch.randn((B, L, N, dm)).to(device)
    t_seq = generate_nonuniform_time_seq(B, L).long().to(device)

    net_config = {
        'd_model': dm,
        'seq_len': 6,
        'memory_len': 70,
        'dropout': 0.1,
    }


    pmn = PopuMemoryNet(**net_config).to(device)
    pmn.train()
    ch = torch.randn((B,128)).to(device)
    out, assign_prob = pmn(ch, xt_seq_embd, t_seq, call_assign=True)
    print("ch", out.shape)
    print("assign", assign_prob.shape)
    return out, assign_prob

def generate_nonuniform_time_seq(b, L, min_delta=1, max_delta=4):
    deltas = torch.randint(min_delta, max_delta + 1, (b, L))
    deltas[:, 0] = 0
    t_seq = torch.cumsum(deltas, dim=1)  # shape: [b, L]
    return t_seq.float()


if __name__ == '__main__':
    # example of usage
    debug_propnet()
