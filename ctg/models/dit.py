"""
Diffusion transformer (DiT) plan denoiser with adaLN-Zero conditioning.

The observation history is encoded by a small transformer and pooled into one
token that is prepended to the plan tokens; the goal and the diffusion step enter
every block through adaptive layer norm.
"""

import torch
import torch.nn as nn

from ctg.models.common import SinusoidalPosEmb, apply_null_goal


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimeEmbedding(nn.Module):
    """Diffusion step -> sinusoidal features -> MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.sin = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.Mish(), nn.Linear(dim, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.sin(t.reshape(-1)))


class GoalEmbedder(nn.Module):
    """Continuous goal [B, G] -> one 128-d token per goal entry -> self-attention -> [B, d_model]."""

    def __init__(self, goal_dim: int, d_model: int, token_dim: int = 128):
        super().__init__()
        self.goal_dim, self.token_dim = goal_dim, token_dim
        self.embed = nn.Linear(goal_dim, goal_dim * token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=2, batch_first=True)
        self.out = nn.Linear(goal_dim * token_dim, d_model)

    def forward(self, goal: torch.Tensor) -> torch.Tensor:
        tokens = self.embed(goal).reshape(-1, self.goal_dim, self.token_dim)
        tokens, _ = self.attn(tokens, tokens, tokens)
        return self.out(tokens.flatten(1))


class HistoryEncoder(nn.Module):
    """History [B, H, F] -> transformer encoder -> attention pooling -> [B, d_model]."""

    def __init__(self, history_dim: int, d_model: int, n_heads: int = 2, n_layers: int = 2,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.proj = nn.Linear(history_dim, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * div), torch.cos(position * div)
        self.register_buffer("pe", pe, persistent=False)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pool = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        x = self.proj(history) + self.pe[: history.shape[1]].unsqueeze(0)
        x = self.encoder(x)
        pooled, _ = self.pool(self.query.expand(x.shape[0], -1, -1), x, x, need_weights=False)
        return self.out_norm(self.pool_norm(pooled.squeeze(1)))


class DiTBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(approximate="tanh"),
                                 nn.Dropout(dropout), nn.Linear(4 * d_model, d_model))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))

    def forward(self, x, c, key_padding_mask=None):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift1, scale1)
        # note: the residual starts from the modulated h (as in all DiT runs so far),
        # whereas the original DiT adds the attention output to the unmodulated x
        x = h + gate1.unsqueeze(1) * self.attn(h, h, h, key_padding_mask=key_padding_mask)[0]
        return x + gate2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift2, scale2))


class DiTDenoiser(nn.Module):
    def __init__(self, plan_dim: int, goal_dim: int, history_dim: int, history_len: int = 2,
                 d_model: int = 256, n_heads: int = 4, n_blocks: int = 3, dropout: float = 0.1):
        super().__init__()
        self.plan_proj = nn.Linear(plan_dim, d_model)
        self.pos = SinusoidalPosEmb(d_model)
        self.time = TimeEmbedding(d_model)
        self.goal = GoalEmbedder(goal_dim, d_model)
        self.null_goal = nn.Parameter(torch.zeros(goal_dim))
        self.history = HistoryEncoder(history_dim, d_model, dropout=dropout, max_len=max(512, history_len))
        self.blocks = nn.ModuleList([DiTBlock(d_model, n_heads, dropout) for _ in range(n_blocks)])
        self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.final = nn.Linear(d_model, plan_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.time.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time.mlp[2].weight, std=0.02)
        for layer in [b.ada[-1] for b in self.blocks] + [self.final_ada[-1], self.final]:  # adaLN-Zero
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, t, goal, history, goal_drop=None, mask=None):
        """x [B, T, plan_dim]; mask [B, T] with True = padded step (ignored by attention)."""
        B, T, _ = x.shape
        x = self.plan_proj(x) + self.pos(torch.arange(T, device=x.device)).unsqueeze(0)
        c = self.time(t.expand(B) if t.ndim == 0 else t)
        c = c + self.goal(apply_null_goal(goal, self.null_goal, goal_drop))
        x = torch.cat([self.history(history).unsqueeze(1), x], dim=1)  # history token first
        if mask is not None:
            mask = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=mask.device), mask.bool()], dim=1)
        for block in self.blocks:
            x = block(x, c, mask)
        shift, scale = self.final_ada(c).chunk(2, dim=1)
        return self.final(modulate(self.final_norm(x), shift, scale))[:, 1:]
