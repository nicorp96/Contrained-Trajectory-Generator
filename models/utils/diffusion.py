import torch.nn as nn
import torch
from einops import rearrange
from typing import Optional, Literal


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1))

    def forward(self, x):
        return nn.functional.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c n -> b (h c) n", h=self.heads)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, n = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) n -> b h c n", h=self.heads), qkv)

        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b (h d) n")
        return self.to_out(out)


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=8):
        super().__init__()
        # TODO: Check this (GroupNorm -> RMSNorm
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            RMSNorm(out_channels),  # nn.GroupNorm(n_groups, out_channels)
            nn.Mish(),  # nn.SiLU()
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidual1DBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cond_dim,
        kernel_size=3,
        n_groups=8,
        cond_film=False,
    ):
        super().__init__()
        self.cond_film = cond_film
        self.out_channels = out_channels
        self.blocks = nn.ModuleList(
            [
                Conv1DBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                Conv1DBlock(
                    out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )

        # TODO: Check FiLM Modulation.
        cond_ch = out_channels
        if cond_film:
            cond_ch = out_channels * 2
        self.cond_encoder = nn.Sequential(
            nn.Mish(),  # nn.SiLU()
            nn.Linear(cond_dim, cond_ch),  # Rearrange("batch t -> batch t 1")
        )
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        out = self.blocks[0](x)
        cond_emb = self.cond_encoder(cond)
        cond_emb = rearrange(cond_emb, "b t -> b t 1")
        if self.cond_film:
            cond_emb = cond_emb.reshape(cond_emb.shape[0], 2, self.out_channels, 1)
            # scale, bias = cond_emb.chunk(2, dim=-1)
            # TODO: check FiLM before the first block
            scale = cond_emb[:, 0, ...]
            bias = cond_emb[:, 1, ...]
            out = scale * out + bias
        else:
            out = out + cond_emb
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class TransformerHistoryEncoder(nn.Module):
    """
    Encodes padded history into a single vector using TransformerEncoder + masked mean pooling.

    Inputs:
      hist: (B, K, state_dim)
      mask: (B, K) with 1=valid, 0=pad  (optional)

    Output:
      cond: (B, d_model)
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.in_proj = nn.Linear(state_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self, hist: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.in_proj(hist)  # (B, K, d_model)

        if mask is None:
            x = self.encoder(x)  # (B, K, d_model)
            cond = x.mean(dim=1)  # (B, d_model)
            return self.out_norm(cond)

        key_padding_mask = ~mask.bool()  # True = pad positions ignored by attention
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, K, d_model)

        # masked mean pool
        w = mask.float().unsqueeze(-1)  # (B, K, 1)
        cond = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return self.out_norm(cond)


class SinusoidalPosEmb1D(nn.Module):
    """Classic sinusoidal position encoding (not learned)."""

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        # positions: (L,)
        return self.pe[positions]  # (L, d_model)


class AttnPool1D(nn.Module):
    """Learned query attends over sequence to produce a single vector."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (B, K, d_model)
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, d_model)
        out, _ = self.attn(
            q, x, x, key_padding_mask=key_padding_mask, need_weights=False
        )
        return self.norm(out.squeeze(1))  # (B, d_model)


class TransformerHistoryEncoderPosEmb(nn.Module):
    """
    Encodes padded history.
    - Can return a pooled conditioning vector (B, d_model) or token memory (B, K, d_model).

    Inputs:
      hist: (B, K, state_dim)
      mask: (B, K) with 1=valid, 0=pad (optional)
    """

    def __init__(
        self,
        state_dim: int,
        d_model: int,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        ff_mult: int = 4,
        max_hist_len: int = 512,
        pooling: Literal["mean", "cls", "attn"] = "attn",
        return_tokens: bool = True,  # recommended for cross-attn
    ):
        super().__init__()
        self.in_proj = nn.Linear(state_dim, d_model)
        self.pos_emb = SinusoidalPosEmb1D(d_model, max_len=max_hist_len)

        self.pooling = pooling
        self.return_tokens = return_tokens

        if pooling == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)
        else:
            self.cls = None

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.out_norm = nn.LayerNorm(d_model)

        if pooling == "attn":
            self.attn_pool = AttnPool1D(d_model, n_heads=n_heads, dropout=dropout)
        else:
            self.attn_pool = None

    def forward(
        self,
        hist: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Returns:
          tokens: (B, K, d_model)  if return_tokens=True
          pooled: (B, d_model)
        """
        B, K, _ = hist.shape
        x = self.in_proj(hist)  # (B, K, d_model)

        # positional encoding
        pos = self.pos_emb(torch.arange(K, device=hist.device))  # (K, d_model)
        x = x + pos.unsqueeze(0)  # (B, K, d_model)

        key_padding_mask = None
        if mask is not None:
            # True means "ignore"
            key_padding_mask = ~mask.bool()  # (B, K)

        if self.pooling == "cls":
            cls = self.cls.expand(B, -1, -1)  # (B, 1, d_model)
            x = torch.cat([cls, x], dim=1)  # (B, 1+K, d_model)
            if key_padding_mask is not None:
                cls_pad = torch.zeros(B, 1, dtype=torch.bool, device=hist.device)
                key_padding_mask = torch.cat(
                    [cls_pad, key_padding_mask], dim=1
                )  # (B, 1+K)

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)  # (B, L, d_model)

        if self.pooling == "cls":
            pooled = x[:, 0]  # (B, d_model)
            tokens = x[:, 1:]  # (B, K, d_model)
            # original key_padding_mask (for K tokens) is without cls
            mem_key_padding_mask = (~mask.bool()) if mask is not None else None

        else:
            tokens = x  # (B, K, d_model)
            mem_key_padding_mask = key_padding_mask  # (B, K) if provided

            if self.pooling == "mean":
                if mask is None:
                    pooled = tokens.mean(dim=1)
                else:
                    w = mask.float().unsqueeze(-1)  # (B, K, 1)
                    pooled = (tokens * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
            elif self.pooling == "attn":
                pooled = self.attn_pool(tokens, key_padding_mask=mem_key_padding_mask)
            else:
                raise ValueError(f"Unknown pooling={self.pooling}")

        pooled = self.out_norm(pooled)
        tokens = self.out_norm(tokens)

        if self.return_tokens:
            return tokens, pooled, mem_key_padding_mask
        else:
            return pooled


class CrossAttnBlock(nn.Module):
    """
    Cross-attention from query tokens (traj) to memory tokens (history).
    Uses pre-norm + residual + FFN.
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.1, ff_mult: int = 4
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.drop = nn.Dropout(dropout)

        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        memory: torch.Tensor,  # (B, K, d_model)
        mem_key_padding_mask: Optional[torch.Tensor] = None,  # (B, K) True=pad
    ) -> torch.Tensor:
        q = self.norm_q(x)
        attn_out, _ = self.attn(
            q, memory, memory, key_padding_mask=mem_key_padding_mask, need_weights=False
        )
        x = x + self.drop(attn_out)
        x = x + self.ff(self.norm_ff(x))
        return x


def sincos_to_angle_lastdim(x, sin_idx, cos_idx):
    """x: [B,T,C] -> θ: [B,T]"""
    return torch.atan2(x[..., sin_idx], x[..., cos_idx])


def project_sincos_channels(x, pairs, eps=1e-8):
    """
    x: [B, C, T]
    pairs: list of (sin_ch, cos_ch)
    """
    for s_ch, c_ch in pairs:
        s = x[:, :, s_ch]
        c = x[:, :, c_ch]
        r = torch.sqrt(s * s + c * c + eps)
        x[:, :, s_ch] = (s / r).clamp_(-1.0, 1.0)
        x[:, :, c_ch] = (c / r).clamp_(-1.0, 1.0)
    return x
