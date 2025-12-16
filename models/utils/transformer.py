import torch
import torch.nn as nn

class AdaLayerNorm(nn.Module):
    def __init__(self, hidden_size, cond_size, zero_init=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)  # no affine here
        self.to_gamma_beta = nn.Linear(cond_size, 2 * hidden_size)
        if zero_init:
            nn.init.zeros_(self.to_gamma_beta.weight)
            nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, cond):  # x: [B,T,H], cond: [B,T,C]
        x = self.norm(x)
        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim=-1)   # [B,T,H] each
        # smooth, bounded gain for stability
        return x * (1.0 + torch.tanh(gamma)) + beta

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


# class TransformerBlock(nn.Module):
#     def __init__(self, d_model, heads, ff_hidden_dim, dropout=0.1):
#         super().__init__()
#         self.attn = nn.MultiheadAttention(
#             d_model, heads, dropout=dropout, batch_first=True
#         )
#         self.ff = nn.Sequential(
#             nn.Linear(d_model, ff_hidden_dim),
#             nn.GELU(),
#             nn.Linear(ff_hidden_dim, d_model),
#         )
#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, x, mask):
#         attn_output, _ = self.attn(x, x, x, attn_mask=mask)
#         x = self.norm1(x + self.dropout(attn_output))
#         ff_output = self.ff(x)
#         return self.norm2(x + self.dropout(ff_output))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden_dim, dropout=0.1):
        super().__init__()#
        self.ada1 = AdaLayerNorm(ff_hidden_dim, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, heads, dropout=dropout, batch_first=True
        )
        self.ada2 = AdaLayerNorm(ff_hidden_dim, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden_dim),
            nn.GELU(),
            nn.Linear(ff_hidden_dim, d_model),
        )
        # optional residual scalars (DeepNet-style), zero init for safety
        self.res_scale1 = nn.Parameter(torch.zeros(1))
        self.res_scale2 = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask, cond=None):
        h = self.ada1(x, cond)
        attn_output, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.res_scale1 * attn_output
        h = self.ada2(x, cond)
        x = x + self.res_scale2 * self.ff(h)
        return x