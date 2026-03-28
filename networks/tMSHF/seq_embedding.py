import torch.nn as nn
import torch
import math
class PatchEmbedding(nn.Module):
    """Image Patch Embedding Layer

    Input: (B, T, C, H, W)
        B: batch size
        T: sequence length
        C: input channels (default 3)
        H, W: input image height and width

    Output: (B, T, N, D)
        N: number of patches = (H/patch_size) * (W/patch_size)
        D: embedding dimension (default 256)

    Parameters:
        in_channels: number of input channels (default 3)
        embed_dim: embedding dimension (default 256)
        patch_size: size of each patch (default 16)
    """
    def __init__(self, in_channels=3, embed_dim=256, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.proj(x)  # (B*T, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B*T, H'*W', embed_dim)
        x = self.norm(x)
        return x.view(B, T, -1, x.size(-1))  # (B, T, H'*W', embed_dim)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        div_term_single = torch.exp(torch.arange(0, d_model - 1, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term_single)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class OutputEmbedding(nn.Module):
    def __init__(self, vocab_size=6, d_model=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class SpaceTimePositionalEncoding(nn.Module):
    """
    Space-Time Positional Encoding (STP) for transformer-based models.

    Input dimensions:
        - x: [B, T, N, D] - Input token embeddings where:
            B: batch size
            T: sequence length
            N: number of patches
            D: embedding dimension
        - time_intervals (Δt_{l,1}): [B, T] - Time intervals between current visit and first examination

    Output dimension:
        - Enhanced embeddings: [B, T, N, D] - Original embeddings with positional information added
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        # Create indices for dimensions
        self.register_buffer('i', torch.arange(d_model))

        # Pre-compute scaling term 10000^(2i/d_m)
        self.register_buffer('div_term', torch.pow(10000, ( self.i / d_model)).view(1, 1, 1, -1))

    def forward(self, x, time_intervals):
        """
        Compute and add the space-time positional encoding according to Eq. (2).

        Args:
            x: Tensor [B, T, N, D] containing input embeddings
            time_intervals: Tensor [B, T] containing Δt_{l,1} values

        Returns:
            Tensor [B, T, N, D] containing embeddings with positional encoding added
        """
        B, T, N, D = x.shape

        # Expand time_intervals from [B, T] to [B, T, N, 1]
        time_intervals = time_intervals.view(B, T, 1, 1).expand(-1, -1, N, 1)

        # Generate patch indices [B, T, N, 1]
        patch_indices = torch.arange(N, device=x.device).view(1, 1, N, 1).expand(B, T, -1, -1)

        # Compute STP according to Eq. (2)
        pe = torch.zeros_like(x)

        # For even indices [0, 2, 4, ...]
        pe[..., 0::2] = torch.sin(time_intervals / self.div_term[..., 0::2]) + torch.sin(patch_indices / self.div_term[..., 0::2])

        # For odd indices [1, 3, 5, ...]
        pe[..., 1::2] = torch.cos(time_intervals / self.div_term[..., 1::2]) + torch.cos(patch_indices / self.div_term[..., 1::2])

        # Add positional encoding to input embeddings
        return self.dropout(x + pe)

class ImgTimeLabelEmbeddingModule(nn.Module):
    def __init__(
            self,
            in_channels=3,
            seq_length=6,
            patch_embd_size=16,
            d_model=256,
    ):
        super().__init__()

        self.patch_embed = PatchEmbedding(
            in_channels=in_channels,
            embed_dim=d_model,
            patch_size=patch_embd_size
        )
        self.output_embed = OutputEmbedding(
            vocab_size=seq_length,
            d_model=d_model
        )

        self.spacetime_pos_encoder = SpaceTimePositionalEncoding(d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, x_seq, t_seq, l_seq):
        """
        Args:
            x_seq: Input image sequence tensor [B, L, C, H, W]
            t_seq: Time points tensor [B, L]
            l_seq: Disease label tensor [B, L]

        Returns:
            xt_seq_embd: Patch embedded and time encoded sequence [B, L, N, D]
            l_seq_embd: Label embedded and encoded sequence [B, L, D]
        """
        # processing img and time seq
        x_seq_embd = self.patch_embed(x_seq)  # patch embd, [B, L, N, D]
        xt_seq_embd = x_seq_embd + self.spacetime_pos_encoder(x_seq_embd, t_seq)  # add S-T pos enc, [B, L, N, D]


        # processing label seq
        l_seq_embd = self.output_embed(l_seq)  # label embedding [B, L, D]
        l_seq_embd = l_seq_embd + self.pos_encoder(l_seq_embd)  # positional encoding [B, L, D]

        return xt_seq_embd, l_seq_embd
