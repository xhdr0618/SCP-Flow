"""
tMSHF - implementation

Continuous-MHA Adapted from github: https://github.com/microsoft/PhysioPro/blob/main/physiopro/network/contiformer.py
(Thanks!)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.tMSHF.module.linear import ODELinear, InterpLinear

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q, k, v, mask=None):
        """
        Args:
            q: Query tensor of shape [B, L, N, D]
            k: Key tensor of shape [B, L, N, D]
            v: Value tensor of shape [B, L, N, D]
            mask: Optional attention mask
            where:
                B = batch size
                L = sequence length
                N = number of spatial tokens
                D = embedding dimension
        Returns:
            Output tensor of shape [B, L, N, D]
        """
        B, L, N, D = q.shape

        # Save original query for residual connection
        residual = q

        # Linear projections and reshape to combine B and L
        q = self.q_proj(q).reshape(B * L, N, D)  # [B*L, N, D]
        k = self.k_proj(k).reshape(B * L, N, D)  # [B*L, N, D]
        v = self.v_proj(v).reshape(B * L, N, D)  # [B*L, N, D]

        # Split heads: [B*L, N, D] -> [B*L, H, N, D/H]
        q = q.reshape(B * L, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B * L, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B * L, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*L, H, N, N]

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # [B, N, N]
                mask = mask.unsqueeze(1)  # [B, 1, N, N]
                mask = mask.repeat_interleave(L, dim=0)  # [B*L, 1, N, N]
            elif mask.dim() == 4:  # [B, L, N, N]
                mask = mask.reshape(B * L, 1, N, N)
            attn = attn.masked_fill(mask == 0, float('-inf'))

        # Apply softmax and dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # [B*L, H, N, D/H]

        # Combine heads and reshape back to original dimensions
        out = out.transpose(1, 2)  # [B*L, N, H, D/H]
        out = out.reshape(B * L, N, D)  # [B*L, N, D]
        out = self.out_proj(out)  # [B*L, N, D]
        out = self.proj_dropout(out)

        # Reshape back to 4D
        out = out.reshape(B, L, N, D)

        # Add residual connection and normalize
        return self.norm(residual + out)


class ScaledDotProductAttention(nn.Module):
    """Continuous Time MHA - Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, omega=None, mask=None):
        # if q is ODELinear, attn = (q.transpose(2, 3).flip(dims=[-2]) / self.temperature * k).sum(dim=-1).sum(dim=-1)
        if omega is not None: # Apply Time-aware scaling
            omega = omega[:, None, :, :, None, None]
            attn = (q / self.temperature * k * omega).sum(dim=-1).sum(dim=-1)
        else:
            attn = (q / self.temperature * k).sum(dim=-1).sum(dim=-1)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        output = (attn.unsqueeze(-1) * v.sum(dim=-2)).sum(dim=-2)
        return output, attn

    def interpolate(self, q, k, v, omega=None, mask=None):
        if omega is not None:
            omega = omega[:, None, :, :, None, None]
            attn = (q / self.temperature * k * omega).sum(dim=-1).sum(dim=-1)
        else:
            attn = (q / self.temperature * k).sum(dim=-1).sum(dim=-1)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)

        output = (attn.unsqueeze(-1) * v.sum(dim=-2)).sum(dim=-2)
        return output, attn

class ContinuousTimeMultiHeadAttention(nn.Module):
    """ Continuous Time Multi-Head Attention module (Continuous Time MHA)"""

    def __init__(self, n_head, d_model, d_k=None, d_v=None, dropout=0.1, normalize_before=True, args_ode=None):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head

        if d_k is None:
            assert d_model % n_head == 0, f"using d_model // n_head as d_k, but d_model % n_head != 0 "
            self.d_k = d_model // n_head
        else:
            self.d_k = d_k
        if d_v is None:
            assert d_model % n_head == 0, f"using d_model // n_head as d_v, but d_model % n_head != 0 "
            self.d_v = d_model // n_head
        else:
            self.d_v = d_v


        self.w_qs = InterpLinear(d_model, n_head * self.d_k, args_ode)
        self.w_ks = ODELinear(d_model, n_head * self.d_k, args_ode)
        self.w_vs = ODELinear(d_model, n_head * self.d_v, args_ode)

        self.fc = nn.Linear(self.d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=self.d_k**0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, t, omega=None, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q, t).view(sz_b, len_q, len_q, -1, n_head, d_k)
        k = self.w_ks(k, t).view(sz_b, len_k, len_k, -1, n_head, d_k)
        v = self.w_vs(v, t).view(sz_b, len_v, len_v, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 4, 1, 2, 3, 5), k.permute(0, 4, 1, 2, 3, 5), v.permute(0, 4, 1, 2, 3, 5)

        if mask is not None:
            mask = mask[None,None,:,:] # For batch and head axes broadcasting.

        output, attn = self.attention(q, k, v, omega=omega, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output += residual

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn

    def interpolate(self, q, k, v, t, qt, omega=None, mask=None):
        d_k, n_head = self.d_k, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        len_qt = qt.size(1)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs.interpolate(q, t, qt, mask=mask).view(sz_b, len_qt, len_q, -1, n_head, d_k)
        k = self.w_ks.interpolate(k, t, qt).view(sz_b, len_qt, len_k, -1, n_head, d_k)
        v = self.w_vs.interpolate(v, t, qt).view(sz_b, len_qt, len_v, -1, n_head, d_k)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.permute(0, 4, 1, 2, 3, 5), k.permute(0, 4, 1, 2, 3, 5), v.permute(0, 4, 1, 2, 3, 5)

        if mask is not None:
            mask = mask = mask[None,None,:,:]  # For batch and head axes broadcasting.

        output, _ = self.attention.interpolate(q, k, v, omega=omega, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = output.transpose(1, 2).contiguous().view(sz_b, len_qt, -1)
        output = self.fc(output)

        return output


class TimeAwareAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, OMEGA=True, dropout=0.1):
        super().__init__()
        self.dim = d_model
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5

        # Layer norms
        self.norm = nn.LayerNorm(d_model)

        # Continuous time MHA args
        args_ode = {
            "actfn": "sigmoid",
            "layer_type": "concat",
            "zero_init": True,
            "atol": 1e-1,
            "rtol": 1e-1,
            "method": "rk4",
            "regularize": False,
            "approximate_method": "bilinear",
            "nlinspace": 3,
            "linear_type": "before",
            "interpolate": "linear",
            "itol": 1e-2,
        }
        args_ode = AttrDict(args_ode)
        self.ct_mha = ContinuousTimeMultiHeadAttention(
            n_head=self.num_heads,
            d_model=d_model,
            dropout=dropout,
            normalize_before=True,
            args_ode=args_ode,
        )
        self.OMEGA = OMEGA
        self.proj_dropout = nn.Dropout(dropout)


    def temporal_attention(self, x, t=None, mask=None):
        # x shape: [B, L, dm]

        if self.OMEGA:  # Apply time-aware scaling
            omega = self.time_attention_scaling(t)
            out, attn = self.ct_mha(x, x, x, t, omega=omega, mask=mask) # out [B,L,dm]
        else:
            out, attn = self.ct_mha(x, x, x, t, mask=mask)  # out [B,L,dm]
        return out

    def time_attention_scaling(self, delta_t, N=None, alpha=-0.5, beta=0.5):
        """
        Calculate time-aware attention scaling values based on sequence of time points.

        Args:
            delta_t (torch.Tensor): Sequence of time points, shape (B, L)
            N (int): Number of image patches
            alpha (float): Hyperparameter controlling time scaling (default: -0.5)
            beta (float): Hyperparameter controlling time scaling (default: 0.5)
            (original setting in paper is alpha=0.5 and beta=0.5, which is wrong)
        Returns:
            torch.Tensor: Attention scaling matrix Ω with shape (B, L, L)
        """
        B, L = delta_t.shape

        # Expand delta_t to calculate time matrix
        t_i = delta_t.unsqueeze(2)  # [B, L, 1]
        t_j = delta_t.unsqueeze(1)  # [B, 1, L]

        # Calculate time differences matrix (Δt_i,j)
        # Result shape: [B, L, L]
        delta_t_matrix = torch.matmul(t_i, t_j)


        if N is not None:
            # repeat to B*N shape
            # Result shape: [B*N, L, L]
            delta_t_matrix = delta_t_matrix.repeat(N, 1, 1)

        # Create a mask for valid positions (upper triangular excluding diagonal)
        mask = torch.tril(torch.ones(L, L), diagonal=-1).bool()
        mask = mask.to(delta_t.device)

        # Calculate scaling values for all positions: Ω_i,j = 1 / (1 + e^(α*Δt_(i,j) - β))
        omega = 1.0 / (1.0 + torch.exp(alpha * delta_t_matrix - beta))

        # Zero out invalid positions (upper triangular part)
        omega = omega.masked_fill(mask[None, :, :], 0.0)

        return omega

    def forward(self, x, t=None, mask=None):
        """
        Args:
            x: Input tensor of shape [B, L, dm]
            t: input tensor of shape [B, L]
        Returns:
            Output tensor of shape [B, L, dm]
        """
        # Temporal attention
        residual = x
        x = self.norm(x)
        x = self.temporal_attention(x, t, mask)
        x = self.proj_dropout(x + residual)

        return x

class SpatialTemporalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, OMEGA=True, dropout=0.1, Npatch=None):
        super().__init__()
        self.dim = d_model
        self.num_heads = num_heads
        head_dim = d_model // num_heads
        self.scale = head_dim ** -0.5

        # Dropout layers
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        """ spatial attention settings """
        # Linear projections for spatial attention
        self.q_spatial = nn.Linear(d_model, d_model)
        self.k_spatial = nn.Linear(d_model, d_model)
        self.v_spatial = nn.Linear(d_model, d_model)

        # MLP after spatial attention
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model)
        )
        """ temporal attention settings """
        # Continuous time MHA args
        args_ode = {
            "actfn": "sigmoid",
            "layer_type": "concat",
            "zero_init": True,
            "atol": 1e-1,
            "rtol": 1e-1,
            "method": "rk4",
            "regularize": False,
            "approximate_method": "bilinear",
            "nlinspace": 3,
            "linear_type": "before",
            "interpolate": "linear",
            "itol": 1e-2,
        }
        args_ode = AttrDict(args_ode)
        self.ct_mha = ContinuousTimeMultiHeadAttention(
            n_head=self.num_heads,
            d_model=d_model,
            dropout=dropout,
            normalize_before = True,
            args_ode=args_ode,
        )
        self.OMEGA = OMEGA  # Apply time-aware scaling

        # init conv_in and conv_out for temporal attn later
        if Npatch is not None:
            self.conv_in = nn.Sequential(
                nn.Conv1d(Npatch * d_model, d_model, kernel_size=1),
                nn.Dropout(dropout)
            )
            self.conv_out = nn.Sequential(
                nn.Conv1d(d_model, Npatch * d_model, kernel_size=1),
                nn.Dropout(dropout)
            )


    def spatial_attention(self, x):
        # x shape: [B, L, N, dm]
        B, L, N, dm = x.shape

        # Merge batch and time dimensions for spatial attention
        x = x.reshape(B * L, N, dm)

        # Linear projections
        q = self.q_spatial(x)  # [B*L, N, dm]
        k = self.k_spatial(x)  # [B*L, N, dm]
        v = self.v_spatial(x)  # [B*L, N, dm]

        # Calculate attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*L, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B*L, N, dm]
        out = self.proj_dropout(out)

        # Reshape back to original dimensions
        out = out.reshape(B, L, N, dm)

        return out

    def temporal_attention(self, x, t, mask=None):
        # x shape: [B, L, N, dm]
        B, L, N, dm = x.shape

        """  downsample w.r.t image related dims (N, dm) """
        """  [B, L, N, dm] -reshape->  [B, L, N*dm] -conv_in-> [B, L, dm] -CTMHA-> [B, L, dm] -conv_out-> [B, L, N*dm] -reshape-> [B, L, N, dm]"""
        # Reshape for temporal attention (B, L, N*dm)
        x = x.reshape(B, L, N*dm)

        # conv_in down sample
        x = x.permute(0, 2, 1)  # [B, L, N*dm] -> [B, N*dm, L]
        x = self.conv_in(x)
        x = x.permute(0, 2, 1)  # [B, dm, L] -> [B, L, dm]

        if self.OMEGA:  # Apply time-aware scaling
            omega = self.time_attention_scaling(t)
            out, attn = self.ct_mha(x, x, x, t, omega=omega, mask=mask)
        else:
            out, attn = self.ct_mha(x, x, x, t, mask=mask)  # out [B,L,dm]

        # conv_out up sample
        out = out.permute(0, 2, 1)  # [B, L, dm] -> [B, dm, L]
        out = self.conv_out(out)  # [B, dm, L] -> [B, N*dm, L]
        out = out.permute(0, 2, 1)  # [B, N*dm, L] -> [B, L, N*dm]

        # Reshape back to original dimensions
        out = out.reshape(B, L, N, dm) # [B, L, N, dm]

        return out

    def time_attention_scaling(self, delta_t, N=None, alpha=-0.5, beta=0.5):
        """
        Calculate time-aware attention scaling values based on sequence of time points.

        Args:
            delta_t (torch.Tensor): Sequence of time points, shape (B, L)
            N (int): Number of image patches
            alpha (float): Hyperparameter controlling time scaling (default: -0.5)
            beta (float): Hyperparameter controlling time scaling (default: 0.5)
            (original setting in paper is alpha=0.5 and beta=0.5, which is wrong)
        Returns:
            torch.Tensor: Attention scaling matrix Ω with shape (B, L, L)
        """
        B, L = delta_t.shape

        # Expand delta_t to calculate time matrix
        t_i = delta_t.unsqueeze(2)  # [B, L, 1]
        t_j = delta_t.unsqueeze(1)  # [B, 1, L]

        # Calculate time differences matrix (Δt_i,j)
        # Result shape: [B, L, L]
        delta_t_matrix = torch.matmul(t_i, t_j)


        if N is not None:
            # repeat to B*N shape
            # Result shape: [B*N, L, L]
            delta_t_matrix = delta_t_matrix.repeat(N, 1, 1)

        # Create a mask for valid positions (upper triangular excluding diagonal)
        mask = torch.tril(torch.ones(L, L), diagonal=-1).bool()
        mask = mask.to(delta_t.device)

        # Calculate scaling values for all positions: Ω_i,j = 1 / (1 + e^(α*Δt_(i,j) - β))
        omega = 1.0 / (1.0 + torch.exp(alpha * delta_t_matrix - beta))

        # Zero out invalid positions (upper triangular part)
        omega = omega.masked_fill(mask[None, :, :], 0.0)

        return omega

    def forward(self, x, t=None, mask=None):
        """
        Args:
            x: Input tensor of shape [B, L, N, dm]
            t: input tensor of shape [B, L]
        Returns:
            Output tensor of shape [B, L, N, dm]
        """
        # Spatial attention
        residual = x
        x = self.norm1(x)
        x = self.spatial_attention(x)
        x = self.proj_dropout(x + residual)

        # MLP
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.proj_dropout(x + residual)

        # Temporal attention
        residual = x
        x = self.norm3(x)
        x = self.temporal_attention(x, t, mask)
        x = self.proj_dropout(x + residual)

        return x


class ScaleTransition(nn.Module):
    def __init__(self, patch_size: int = 2, stride: int = 2):
        """
        Initialize the Scale Transition module using unfold operation.
        Args:
            patch_size (int): Size of the patch for merging
            stride (int): Stride for patch sliding. If None, uses patch_size//2 (default: 2, no overlapping)
        """
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform scale transition using unfold operation for efficiency.
        """
        B, L, N, D = x.shape
        H = W = int(N ** 0.5)
        assert H * W == N, "Input number of patches must be a perfect square"

        # Reshape to [B*L, D, H, W] for unfold operation
        x = x.view(B, L, H, W, D).permute(0, 1, 4, 2, 3).reshape(B * L, D, H, W)

        # Use unfold to create patches
        patches = F.unfold(x,
                           kernel_size=(self.patch_size, self.patch_size),
                           stride=self.stride)

        # Reshape patches and compute mean
        P = self.patch_size * self.patch_size
        new_H = ((H - self.patch_size) // self.stride) + 1
        new_W = ((W - self.patch_size) // self.stride) + 1
        patches = patches.view(B, L, D, P, new_H * new_W)

        # Average pooling over patch dimension
        output = patches.mean(dim=3)  # [B, L, D, new_H*new_W]

        # Reshape to final format
        output = output.permute(0, 1, 3, 2)  # [B, L, new_H*new_W, D]

        return output


class MST_ScaleTransition(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = None, patch_size: int = 2):
        """
        Complete Scale Transition module with optional dimension adjustment.
        Args:
            input_dim (int): Input embedding dimension
            hidden_dim (int, optional): Output embedding dimension after transition
            patch_size (int): Size of patches to merge
        """
        super().__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim or input_dim

        self.scale_transition = ScaleTransition(patch_size)

        # Optional dimension adjustment
        self.dim_adjust = (nn.Linear(input_dim, self.hidden_dim) if input_dim != hidden_dim else nn.Identity())

        # Layer normalization for stability
        self.norm = nn.LayerNorm(self.hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass including scale transition and dimension adjustment.
        Args:
            x: Input tensor of shape (B, L, H*W, d_m)
        Returns:
            Processed tensor of shape (B, L, (H/patch_size)*(W/patch_size), hidden_dim)
        """
        # Perform scale transition
        x = self.scale_transition(x)

        # Adjust dimensions if needed
        x = self.dim_adjust(x)

        # Apply normalization
        x = self.norm(x)

        return x

class AddNorm(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_output):
        return self.layer_norm(x + self.dropout(sublayer_output))

class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, embed_dim * mlp_ratio)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(embed_dim * mlp_ratio, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        out = self.dropout1(self.act(self.fc1(x)))
        out = self.dropout2(self.fc2(out))
        return self.norm(x + self.dropout3(out))

class ImgTimeDecodingLayer(nn.Module):
    def __init__(self, d_model, num_heads, OMEGA=True, dropout=0.1, Npatch=None):
        super().__init__()
        # Conti-time S-T Attention
        self.spatial_temporal_attention = SpatialTemporalSelfAttention(d_model, num_heads, OMEGA=OMEGA, dropout=dropout, Npatch=Npatch)
        self.add_norm1 = AddNorm(d_model, dropout)

        self.mlp = MLP(d_model,dropout=dropout)
        self.add_norm3 = AddNorm(d_model, dropout)

        # Multi-Head Attention for encoder outputs
        self.multiheadattn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x, encoder_output, t):
        # Mask for Decoder's Masked Continuous-time Spatial-temporal Attention
        L = x.shape[1]
        mask = torch.tril(torch.ones(L, L)).to(x.device).bool()

        # Multi-Head Spatial-Temporal Attention
        attn_output = self.spatial_temporal_attention(x, t, mask=mask)
        x = self.add_norm1(x, attn_output)
        B,L,N,D = x.shape

        # multi head attention
        encoder_output = encoder_output.unsqueeze(dim=2).repeat(1,1,N,1)
        cross_attn_output = self.multiheadattn(q=x, k=encoder_output, v=encoder_output)
        x = self.add_norm2(x, cross_attn_output)

        # MLP
        mlp_output = self.mlp(x)
        x = self.add_norm2(x, mlp_output)

        return x


class LabelEncodingLayer(nn.Module):
    def __init__(self, d_model, num_heads, OMEGA=True, dropout=0.1):
        super().__init__()
        # Multi-Head Time-aware Attention
        self.self_attention = TimeAwareAttention(d_model, num_heads, OMEGA=OMEGA, dropout=dropout)
        self.add_norm1 = AddNorm(d_model, dropout)

        # Final MLP
        self.mlp = MLP(d_model)
        self.add_norm2 = AddNorm(d_model, dropout)

    def forward(self, x,t):
        self_attn_output = self.self_attention(x, t)
        x = self.add_norm1(x, self_attn_output)

        # MLP
        mlp_output = self.mlp(x)
        x = self.add_norm2(x, mlp_output)

        return x


class SpatialTemporalEncoderDecoder(nn.Module):
    def __init__(self, d_model, num_heads, OMEGA, Npatch=None, num_encoder_layers=2, num_decoder_layers=2, dropout=0.1):
        super().__init__()
        # Encoder
        self.label_encoder_layers = nn.ModuleList([
            LabelEncodingLayer(d_model, num_heads, OMEGA=OMEGA, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])

        # Decoder
        self.imgtime_decoder_layers = nn.ModuleList([
            ImgTimeDecodingLayer(d_model, num_heads,OMEGA=OMEGA, dropout=dropout, Npatch=Npatch)
            for _ in range(num_encoder_layers)
        ])



    def forward(self, label_encoder_input, imgtime_encoder_input, t_seq):
        # Label Encoder
        label_encoder_output = label_encoder_input
        for label_encoder_layer in self.label_encoder_layers:
            label_encoder_output = label_encoder_layer(label_encoder_output, t_seq)

        # ImgTime Decoder
        imgtime_decoder_output = imgtime_encoder_input
        for imgtime_decoder_layer in self.imgtime_decoder_layers:
            imgtime_decoder_output = imgtime_decoder_layer(imgtime_decoder_output, label_encoder_output, t_seq)

        return label_encoder_output, imgtime_decoder_output

class ScaleFeatureReducing(nn.Module):
    """Scale feature reducing module that supports multiple reduction methods and final projection.

    Args:
        num_scales (int): Number of scales to process
        scales_size (list): List of input sizes for each scale
        d_model (int): d_model size for the final linear transformation
        method (str): Reduction method ('max_pool', 'linear', or 'avg_pool')
        scale_out_size (int): Target output size for each scale before concatenation
    """

    def __init__(self, num_scales, scales_size, d_model, method='avg', scale_out_size=1):
        super().__init__()
        self.num_scales = num_scales
        self.scales_size = scales_size
        self.d_model = d_model
        self.method = method
        self.scale_out_size = scale_out_size

        # Validate inputs
        assert len(scales_size) == num_scales, "scales_size length must match num_scales"
        assert method in ['max', 'linear', 'avg'], "method must be one of ['max', 'linear', 'avg']"

        # Initialize reduction layers for each scale
        self.reduction_layers = nn.ModuleList()

        for scale_size in scales_size:
            if method == 'linear':
                # Linear projection layer
                self.reduction_layers.append(
                    nn.Linear(scale_size*d_model, scale_out_size)
                )
            elif method in ['max', 'avg']:
                # For pooling methods, we don't need trainable parameters
                self.reduction_layers.append(None)

        # Initialize output projection for linear method only
        if method == 'linear':
            self.out_projection = nn.Linear(num_scales * d_model, d_model)
        else:
            self.out_projection = None

    def _apply_pooling(self, x, input_size, pool_type='avg'):
        """Apply pooling reduction to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [B, L, N, D]
            input_size (int): Input size (N dimension)
            pool_type (str): Type of reduction ('max' or 'avg')

        Returns:
            torch.Tensor: Reduced tensor of shape [B, L, scale_out_size, D]
        """
        B, L, N, D = x.shape

        # Reshape for pooling operation
        x = x.reshape(B * L, 1, N, D)

        # Calculate kernel size and stride for the desired output size
        kernel_size = (input_size // self.scale_out_size, 1)
        stride = kernel_size

        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=kernel_size, stride=stride)
        else:  # avg_pool
            x = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride)

        # Reshape back to original format
        x = x.reshape(B, L, self.scale_out_size, D)

        return x

    def forward(self, features):
        """Forward pass through the reduction module.

        Args:
            features (list): List of tensors, each of shape [B, L, N, D]

        Returns:
            torch.Tensor: Final output tensor of shape [B, L, D]
        """
        assert len(features) == self.num_scales, "Number of input features must match num_scales"

        reduced_features = []

        for i, (feature, scale_size) in enumerate(zip(features, self.scales_size)):
            if self.method == 'linear':
                # For linear projection, transpose to [B*L, N, D]
                B, L, N, D = feature.shape
                x = feature.reshape(B * L, N*D)

                # Apply linear projection
                x = self.reduction_layers[i](x)

                # Reshape back to [B, L, scale_out_size, D]
                x = x.reshape(B, L, self.scale_out_size, D)

            elif self.method == 'max_pool':
                x = self._apply_pooling(feature, scale_size, pool_type='max')

            else:  # avg_pool
                x = self._apply_pooling(feature, scale_size, pool_type='avg')

            reduced_features.append(x)

        # Concatenate all reduced features along the N dimension
        # Shape: [B, L, num_scales * scale_out_size, D]
        x = torch.cat(reduced_features, dim=2)

        if self.method == 'linear':
            # Reshape to [B*L, num_scales*D]
            B, L, N, D = x.shape
            x = x.reshape(B * L, N * D)
            # Apply linear projection
            # Shape: [B*L, num_scales*D] -> [B*L, D]
            x = self.out_projection(x)
            # Reshape back to [B, L, D]
            x = x.reshape(B, L, -1)

        else:  # max_pool or avg_pool
            # Apply pooling along the num_scales dimension
            if self.method == 'max':
                x = F.max_pool2d(x, kernel_size=(self.num_scales, 1), stride=(self.num_scales, 1))
            else:  # avg_pool
                x = F.avg_pool2d(x, kernel_size=(self.num_scales, 1), stride=(self.num_scales, 1))
            # Reshape back and squeeze out the reduced dimension
            x = x.squeeze(2)  # [B, L, D]

        return x

class MultiscaleSpatialTemporalTransformer(nn.Module):
    def __init__(
            self,
            in_shape=(3,256,256), # input shape of img
            seq_length=5, # the length of the longitudinal data
            num_scales=3,  # the number of scale translation
            num_encoder_layers=1,  # the num of encoder layers at one scale 
            num_decoder_layers=1,  # the num of decoder layers at one scale 
            patch_embd_size=16,  # patch size for image embedding
            d_model=256,  # embedding dim
            num_heads=8,  # num of head for multi head attetnion
            dropout=0.1,  # dropout
            num_classes=None,  # output class (not used for tMSHF)
            OMEGA=True  # time weight matrix
    ):
        super().__init__()
        C,H,W = in_shape
        patch_nums_level = calculate_patches_per_scale(H=H, W=W, initial_patch_size=patch_embd_size, num_scales=num_scales)
        # multi-scale transformer encoder-decoder 
        self.scales = nn.ModuleList()
        for idx in range(num_scales):
            encoder_decoder = SpatialTemporalEncoderDecoder(
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                OMEGA=OMEGA,
                Npatch=patch_nums_level[idx],
            )
            self.scales.append(encoder_decoder)


        # init scale translation layers
        self.scale_transitions = nn.ModuleList()
        for _ in range(num_scales - 1):
            transition = MST_ScaleTransition(
                input_dim=d_model,
                patch_size=2
            )
            self.scale_transitions.append(transition)

        # reduction method for different scales
        self.features_reduction = ScaleFeatureReducing(num_scales=num_scales, scales_size=patch_nums_level, scale_out_size=1, method="avg",d_model=d_model)

        # out method
        self.norm = nn.LayerNorm(d_model)
        if num_classes is not None:
            self.head = nn.Sequential(
                nn.Linear(d_model*seq_length, d_model),
                nn.Linear(d_model, num_classes)
            )
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.head = None

    def forward(self, xt_seq_embd, t_seq, l_seq_embd):
        """
        Args:
            xt_seq_embd: Patch embedded and time-aware pos encoded sequence [B, L, N, D]
            t_seq: time points tensor [B, L]
            l_seq_embd: Label embedded and pos encoded sequence [B, L, D]
        Return:
            x: output image feature tokens [B, L, D]
        """
        # multi-scale translation
        features = []
        x_patch = xt_seq_embd
        x_label = l_seq_embd

        for i, (encoder_decoder, transition) in enumerate(zip(self.scales[:-1], self.scale_transitions)):

            # the encoder decoder outputs at this scale
            x_label, x_patch = encoder_decoder(x_label, x_patch, t_seq)  # encoder output, decoder output
            features.append(x_patch)
            # scale translation
            x_patch = transition(x_patch)

        # the last scale
        x_label, x_patch = self.scales[-1](x_label, x_patch, t_seq) # encoder output, decoder output
        features.append(x_patch)
        x = self.features_reduction(features)
        x = self.norm(x)
        return x

def calculate_time_dists(time_points, beta=1.0):
    """
    Calculate the time distance scaling matrix Ω_{i,j} according to equation (4)

    Args:
        time_points: Tensor of shape [L] containing timestamps
        beta: Scaling factor β in the equation

    Returns:
        Tensor of shape [L, L] containing Ω_{i,j} values
    """
    L = len(time_points)
    time_dists = torch.zeros((L, L))

    for i in range(L):
        for j in range(i + 1):  # Only calculate for j ≤ i as specified
            delta_t = time_points[i] - time_points[j]
            time_dists[i, j] = 1 / (1 + torch.exp(delta_t * beta))

    return time_dists


def calculate_patches_per_scale(H: int, W: int, initial_patch_size: int = 16, scale_patch_size: int = 2 ,scale_stride: int = 2, num_scales: int = 3):
    """
    compute the num of patches at each scale

    Args:
        H (int): the hight of input embedding
        W (int): the width of input embedding
        initial_patch_size (int): the size of input patch
        scale_patch_size (int): the patch size for the scale translation
        scale_stride (int): the stride for the scale translation
        num_scales (int): the number of scales

    Returns:
        List[int]: num of patches in each scale
    """
    patches_per_scale = []

    curr_H = H // initial_patch_size
    curr_W = W // initial_patch_size
    N = curr_H * curr_W
    patches_per_scale.append(N)

    for scale in range(num_scales - 1):
        patch_size = scale_patch_size
        stride = scale_stride

        # compute the new width and hight 
        new_H = ((curr_H - patch_size) // stride) + 1
        new_W = ((curr_W - patch_size) // stride) + 1
        N = new_H * new_W

        patches_per_scale.append(N)
        curr_H, curr_W = new_H, new_W

    return patches_per_scale


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



# Using example
if __name__ == "__main__":
    device = 'cuda'
    from seq_embedding import ImgTimeLabelEmbeddingModule  # seq embd

    # Initialize model parameters
    b = 2
    L = 6  # seq_length
    N = 16  # patch size
    d_model = 128  # Embedding dimension
    in_shape = (3,256,256)
    c,h,w = in_shape

    x_seq = torch.randn((b, L, c, h,w),device=device)
    t_seq = generate_nonuniform_time_seq(b, L).to(device)
    l_seq = torch.randint(low=0, high=1, size=(b, L), device=device)
    embd_config = {
        "in_channels": c, # channel of the input img
        "seq_length": L,  # Length of the longitudinal data
        "patch_embd_size": 16,  # Patch size for image embedding
        "d_model": d_model,  # Embedding dimension
    }
    model_config = {
        # Input settings
        "in_shape": in_shape,  # Input channels for Patch Embedding
        "seq_length": L,  # Length of the longitudinal data

        # Architecture settings
        "num_scales": 3,  # Number of scales in the network
        "num_encoder_layers": 1,  # Number of encoder layers per scale
        "num_decoder_layers": 1,  # Number of decoder layers per scale
        "patch_embd_size": 16,  # Patch size for image embedding

        # Model dimensions
        "d_model": d_model,  # Embedding dimension
        "num_heads": 8,  # Number of attention heads

        # Regularization
        "dropout": 0.1,  # Dropout rate

        # Output settings
        "num_classes": None,  # Number of output classes
        "OMEGA":True,
    }
    embd = ImgTimeLabelEmbeddingModule(**embd_config).to(device)
    model = MultiscaleSpatialTemporalTransformer(**model_config).to(device)
    xt_seq_embd, l_seq_embd = embd(x_seq=x_seq, t_seq=t_seq, l_seq=l_seq)
    out = model(xt_seq_embd=xt_seq_embd, t_seq=t_seq, l_seq_embd=l_seq_embd)
    print(out.shape)