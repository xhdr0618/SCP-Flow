import torch
import torch.nn as nn
from networks.tMSHF.tMSHF_imgfeature import MultiscaleSpatialTemporalTransformer
from networks.tMSHF.seq_embedding import ImgTimeLabelEmbeddingModule
from networks.PMQM.popu_memory import PopuMemoryNet


class ConditionGenMSTFCMPopuMemory(nn.Module):
    """ condition generator """
    def __init__(self,d_model, start_memory_step=1500, dropout=0.1):
        super(ConditionGenMSTFCMPopuMemory, self).__init__()

        self.d_model = d_model
        self.seq_len = 6
        self.img_in_shape = (3,256,256)
        self.memory_len = 70
        # updating schedule
        self.start_memory_step = start_memory_step


        """ config for seq_embds """
        embd_config = {
            "in_channels": self.img_in_shape[0],  # channel of the input img
            "seq_length": self.seq_len,  # Length of the longitudinal data
            "patch_embd_size": 16,  # Patch size for image embedding
            "d_model": self.d_model,  # Embedding dimension
        }

        """ config for t-MSHF """
        mstformer_config = {
            # Input settings
            "in_shape": self.img_in_shape,  # Input shape for Patch Embedding
            "seq_length": self.seq_len,  # Length of the longitudinal data

            # Architecture settings
            "num_scales": 3,  # Number of scales in the network
            "num_encoder_layers": 1,  # Number of encoder layers per scale
            "num_decoder_layers": 1,  # Number of decoder layers per scale
            "patch_embd_size": 16,  # Patch size for image embedding

            # Model dimensions
            "d_model": self.d_model,  # Embedding dimension
            "num_heads": 8,  # Number of attention heads

            # Regularization
            "dropout": dropout,  # Dropout rate

            # Output settings
            "num_classes": None,  # Number of output classes (if none just output features)
            "OMEGA": True,  # Adapting Time aware scaling
        }
        """ config for popu memory net """
        pmn_config = {
            'd_model': self.d_model,
            'seq_len': self.seq_len,
            'memory_len': self.memory_len,
            'dropout': dropout,
        }

        self.seq_emb = ImgTimeLabelEmbeddingModule(**embd_config).to("cuda")
        self.mstformer = MultiscaleSpatialTemporalTransformer(**mstformer_config).to("cuda")
        self.pmn = PopuMemoryNet(**pmn_config).to("cuda")

        self.img_token = nn.Parameter(torch.zeros(1, 1, 1, self.d_model))
        self.label_token = nn.Parameter(torch.zeros(1, 1, self.d_model))


    def get_ch(self, xt_seq_embd, t_seq, l_seq_embd):
        """
        processing the t-MSHF to get the individual historical condition (ch)
        """
        # get first 5 embd
        xt_seq_embd, l_seq_embd = xt_seq_embd[:, :5, ...], l_seq_embd[:, :5, ...]

        assert xt_seq_embd.shape[1] == self.seq_len - 1, f" xt_seq_embd - the t-MSHF model can only see first {self.seq_len - 1} images feature to predict {self.seq_len}th, but received {xt_seq_embd.shape}"
        assert t_seq.shape[1] == self.seq_len, f" t_seq - the t-MSHF model need all {self.seq_len} visit times to predict {self.seq_len}th, but received {t_seq.shape}"
        assert l_seq_embd.shape[1] == self.seq_len - 1, f" l_seq_embd - the t-MSHF model can only see first {self.seq_len - 1} label feature to predict {self.seq_len}th, but received{l_seq_embd.shape}"

        # processing token expand
        b, l, n, dm = xt_seq_embd.size()
        device = xt_seq_embd.device
        img_token = self.img_token.expand(b, 1, n, dm).to(device)
        label_token = self.label_token.expand(b, 1, dm).to(device)

        xt_seq_embd = torch.cat([xt_seq_embd, img_token], dim=1)  # cat in L dimension
        l_seq_embd = torch.cat([l_seq_embd, label_token.long()], dim=1)  # cat in L dimension

        x_fea = self.mstformer(xt_seq_embd=xt_seq_embd, t_seq=t_seq, l_seq_embd=l_seq_embd)

        # proj last token
        ch = x_fea[:, -1, :]
        return ch

    def get_cp(self,ch, xt_seq_embd, t_seq, x_id=None):
        """
        processing the PMQM to get the populational condition (cp) according to ch
        args:
            xt_seq_embd: (B,L,N(num of patches),dm)
            t_seq: (B,L)
            (optional) x_id: (B,L) the actural id for each image (good for retrival)
        return：
            cp: (B, dm)
        """
        assert xt_seq_embd.shape[1] == self.seq_len, f" xt_seq_embd - the t-PMQM model need all {self.seq_len} images feature to predict {self.seq_len}th, but received {xt_seq_embd.shape}"
        assert t_seq.shape[1] == self.seq_len, f" t_seq - the PMQM model need all {self.seq_len} visit times to predict {self.seq_len}th, but received {t_seq.shape}"
        cp = self.pmn(ch=ch, xt_seq_embd=xt_seq_embd, t_seq=t_seq)
        return cp


    def forward(self, x_seq, t_seq, l_seq, current_step=None):
        assert x_seq.shape[1] == self.seq_len, f"the embd need all {self.seq_len} imgs, but received {x_seq.shape}"
        assert t_seq.shape[1] == self.seq_len, f"the embd need all {self.seq_len} visit times, but received {t_seq.shape}"
        assert l_seq.shape[1] == self.seq_len, f"the embd need all {self.seq_len} imgs, but received {l_seq.shape}"
        assert (self.training and current_step is not None) or (not self.training and current_step is None), f'Training mode and current_step must be aligned: training={self.training}, step={current_step}'

        # take all 6 seq_embd
        t_seq = t_seq.float()
        xt_seq_embd, l_seq_embd = self.seq_emb(x_seq=x_seq, t_seq=t_seq, l_seq=l_seq)

        if current_step is None:  # evaluation
            ch = self.get_ch(xt_seq_embd=xt_seq_embd, t_seq=t_seq, l_seq_embd=l_seq_embd)
            cp = self.get_cp(ch=ch, xt_seq_embd=xt_seq_embd, t_seq=t_seq)
            return ch, cp


        else: # training
            # get historical feature
            ch = self.get_ch(xt_seq_embd=xt_seq_embd, t_seq=t_seq, l_seq_embd=l_seq_embd)

            # using PMN according to current steps
            if current_step <= self.start_memory_step: # train ch first to get stable query, don't use PMN
                cp = None
                return ch, cp
            else: # update PMN param
                cp = self.get_cp(ch=ch, xt_seq_embd=xt_seq_embd, t_seq=t_seq)
                return ch, cp


def generate_nonuniform_time_seq(b, L, min_delta=1, max_delta=4):
    """
    generate irregular increasing time seq (int) for debugging
    Args:
        b: batch size
        L: sequence length
        min_delta: min time interval (int)
        max_delta: max time interval (int)
    """
    # generate random time interval
    deltas = torch.randint(min_delta, max_delta + 1, (b, L))

    # set the first time (t0) be 0
    deltas[:, 0] = 0

    # cumsum the interval to get t_seq for debug
    t_seq = torch.cumsum(deltas, dim=1)  # shape: [b, L]

    return t_seq.float()

if __name__ == '__main__':
    # Example of usage
    b = 2
    L = 6  # seq_length
    N = 16  # patch size
    c,h,w = (3,256,256)
    device = "cuda"

    x_seq = torch.randn((b, L, c,h,w),device=device).float()
    t_seq = generate_nonuniform_time_seq(b, L).long().to(device)
    l_seq = torch.randint(low=0, high=1, size=(b, L)).long().to(device)
    x_id = [f"id_{i}" for i in range(1, b+1)]

    model = ConditionGenMSTFCMPopuMemory(d_model=128).to(device)
    model.train()
    ch,cp = model(x_seq, t_seq, l_seq, current_step=1000)
    print(ch.shape)