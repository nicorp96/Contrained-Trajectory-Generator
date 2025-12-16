import torch.nn as nn
import torch
from einops import rearrange


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


def apply_inpainting(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,  # [B,H,D] 1=known
    x_known: torch.Tensor | None = None,  # [B,H,D]
    noise: bool = False,
):
    """
    If train_mode=True: replace known entries with 0.
    If train_mode=False: replace known entries with their provided values.
    """
    # --- dense mask-based inpainting (optional)
    if (mask is not None) and (x_known is not None):
        m = mask.float()
        if noise:
            x = x * (1.0 - m)  # known -> 0
        else:
            x = x * (1.0 - m) + x_known * m  # clamp known values
    return x
