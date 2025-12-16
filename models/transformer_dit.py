import einops
import torch
import torch.nn as nn
from typing import Optional

from models.utils.diffusion import SinusoidalPosEmb


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.Mish(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class ContinuousCondEmbedder(nn.Module):
    """Modified from DiscreteCondEmbedder to embed a continuous variable instead of a 1-hot vector."""

    def __init__(self, cond_dim: int, hidden_size: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.embedding = nn.Linear(
            cond_dim, int(cond_dim * 128)
        )  # 1 layer affine to transform attribute into embedding vector
        self.attn = nn.MultiheadAttention(128, num_heads=2, batch_first=True)
        self.linear = nn.Linear(128 * cond_dim, hidden_size)

    def forward(self, attr: torch.Tensor, mask: torch.Tensor = None):
        """
        attr: (batch_size, cond_dim)
        mask: (batch_size, cond_dim) 0 or 1, 0 means ignoring
        """
        emb = self.embedding(attr).reshape(
            (-1, self.cond_dim, 128)
        )  # (b, cond_dim, 128)
        if mask is not None:
            emb *= mask.unsqueeze(-1)  # (b, cond_dim, 128)
        emb, _ = self.attn(emb, emb, emb)  # (b, cond_dim, 128)
        return self.linear(
            einops.rearrange(emb, "b c d -> b (c d)")
        )  # (b, hidden_size)


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 6)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_mod(
            t
        ).chunk(6, dim=1)
        x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = x + gate_msa.unsqueeze(1) * self.attn(x, x, x)[0]
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class FinalLayer1D(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size)
        self.adaLN_mod = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, hidden_size * 2)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_mod(t).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiTTrj(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.cond_dim = config.get("cond_dim", None)
        self.input_dim = config["input_dim"]
        self.d_model = config["d_model"]
        self.n_heads = config["n_heads"]
        self.n_blocks = config["n_blocks"]
        self.dropout = config["dropout"]
        self.trj_proj = nn.Linear(self.input_dim, self.d_model)
        self.t_emb = TimeEmbedding(self.d_model)
        if self.cond_dim is not None:
            self.cond_proj = ContinuousCondEmbedder(
                cond_dim=self.cond_dim, hidden_size=self.d_model
            )
        self.pos_embd = SinusoidalPosEmb(self.d_model)
        self.pos_embd_cache = None
        self.blocks = nn.ModuleList(
            [
                DiTBlock(self.d_model, self.n_heads, self.dropout)
                for _ in range(self.n_blocks)
            ]
        )
        self.final_layer = FinalLayer1D(self.d_model, self.input_dim)
        self.init_weights()

    def init_weights(self):
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(_basic_init)

        nn.init.normal_(self.t_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_emb.mlp[2].weight, std=0.02)

        for block in self.blocks:
            nn.init.constant_(block.adaLN_mod[-1].weight, 0)
            nn.init.constant_(block.adaLN_mod[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_mod[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_mod[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x: (batch_size, traj_len, input_size)
        t: (batch_size,)
        cond: (batch_size, cond_dim)
        """
        B, T, _ = x.shape
        if self.pos_embd_cache is None or self.pos_embd_cache.shape[0] != x.shape[0]:
            self.pos_embd_cache = self.pos_embd(
                torch.arange(x.shape[1], device=x.device)
            )
        x = self.trj_proj(x) + self.pos_embd_cache[None,]
        t = t.expand(B).unsqueeze(1).float()
        t = self.t_emb(t)
        if cond is not None:
            assert (
                self.cond_proj is not None
            ), "Model is not conditional and cannot accept cond input."
            t += self.cond_proj(cond, mask)
        for block in self.blocks:
            x = block(x, t)
        x = self.final_layer(x, t)
        return x
