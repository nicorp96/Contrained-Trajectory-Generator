from einops import rearrange, repeat
import torch
import torch.nn as nn

from models.utils.diffusion import (
    SinusoidalPosEmb,
    ConditionalResidual1DBlock,
    Downsample1d,
    Upsample1d,
    Conv1DBlock,
    Residual,
    Attention,
    PreNorm,
    LinearAttention,
)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


class Diffusion1DUnetAtt(nn.Module):
    def __init__(self, config):
        super(Diffusion1DUnetAtt, self).__init__()
        self.config = config
        self.input_dim = self.config["input_dim"]
        self.cond_dim = self.config["cond_dim"]
        self.diffusion_t_embed_dim = self.config["diffusion_t_embed_dim"]
        self.down_dims = self.config["down_dims"]  # List [256, 512, 1024]
        self.kernel_size = self.config["kernel_size"]
        self.n_blocks = self.config["n_blocks"]
        self.cond_film = self.config["cond_film"]
        # TODO: Check the linear projection!
        self.d_model = self.config["d_model"]
        # self.seq_embd = nn.Linear(254, 512)
        self.traj_embd = nn.Linear(self.input_dim, self.d_model)
        self.init_conv = nn.Conv1d(self.d_model, self.d_model, 7, padding=3)
        self.cond_emb = nn.Sequential(
            nn.Linear(self.cond_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(self.diffusion_t_embed_dim),
            nn.Linear(self.diffusion_t_embed_dim, 4 * self.diffusion_t_embed_dim),
            nn.Mish(),
            nn.Linear(4 * self.diffusion_t_embed_dim, self.diffusion_t_embed_dim),
        )

        cond_dim_t = self.diffusion_t_embed_dim
        if self.cond_dim is not None:
            cond_dim_t += self.d_model

        dim_total = [self.d_model] + list(self.down_dims)
        start_dim = dim_total[0]
        middle_dim = self.down_dims[-1]
        self.modules_mid = nn.ModuleList(
            [
                ConditionalResidual1DBlock(
                    in_channels=middle_dim,
                    out_channels=middle_dim,
                    cond_dim=cond_dim_t,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_blocks,
                    cond_film=self.cond_film,
                ),
                PreNorm(
                    middle_dim,
                    Attention(
                        middle_dim, dim_head=32, heads=4
                    ),  # Check if better MultiHeadAttention from torch
                ),
                ConditionalResidual1DBlock(
                    in_channels=middle_dim,
                    out_channels=middle_dim,
                    cond_dim=cond_dim_t,
                    kernel_size=self.kernel_size,
                    n_groups=self.n_blocks,
                    cond_film=self.cond_film,
                ),
            ]
        )
        self.down_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(
            list(zip(dim_total[:-1], dim_total[1:]))
        ):
            is_last = idx >= (len(list(zip(dim_total[:-1], dim_total[1:]))) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidual1DBlock(
                            in_channels=in_dim,
                            out_channels=in_dim,
                            cond_dim=cond_dim_t,
                            kernel_size=self.kernel_size,
                            n_groups=self.n_blocks,
                            cond_film=self.cond_film,
                        ),
                        ConditionalResidual1DBlock(
                            in_channels=in_dim,
                            out_channels=in_dim,
                            cond_dim=cond_dim_t,
                            kernel_size=self.kernel_size,
                            n_groups=self.n_blocks,
                            cond_film=self.cond_film,
                        ),
                        Residual(PreNorm(in_dim, LinearAttention(in_dim))),
                        (
                            Downsample(in_dim, out_dim)
                            if not is_last
                            else nn.Conv1d(in_dim, out_dim, 3, padding=1)
                        ),
                    ]
                )
            )
        self.up_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(
            reversed(list(zip(dim_total[:-1], dim_total[1:])))
        ):
            is_last = idx == (len(list(zip(dim_total[:-1], dim_total[1:]))) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidual1DBlock(
                            in_channels=out_dim + in_dim,
                            out_channels=out_dim,
                            cond_dim=cond_dim_t,
                            kernel_size=self.kernel_size,
                            n_groups=self.n_blocks,
                            cond_film=self.cond_film,
                        ),
                        ConditionalResidual1DBlock(
                            in_channels=out_dim + in_dim,
                            out_channels=out_dim,
                            cond_dim=cond_dim_t,
                            kernel_size=self.kernel_size,
                            n_groups=self.n_blocks,
                            cond_film=self.cond_film,
                        ),
                        Residual(PreNorm(out_dim, LinearAttention(out_dim))),
                        (
                            Upsample(out_dim, in_dim)
                            if not is_last
                            else nn.Conv1d(out_dim, in_dim, 3, padding=1)
                        ),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1DBlock(
                start_dim * 2,
                start_dim,  # self.input_dim,
                kernel_size=self.kernel_size,
                n_groups=2,
            ),  # TODO: Check this: not yet good implemented
            nn.Conv1d(start_dim, start_dim, 1),
            # nn.Linear(512, 254),
        )
        self.traj_out = nn.Linear(self.d_model, self.input_dim)

    def forward(
        self, sample: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor
    ):
        bsz, seq_len, inp = sample.shape
        sample = self.traj_embd(sample)
        sample = rearrange(sample, " b h t -> b t h")
        sample_pad = nn.functional.pad(sample, pad=(0, 2), mode="constant", value=0)
        sample = self.init_conv(sample_pad)
        res = sample_pad.clone()
        timesteps = timesteps.expand(sample.size(0))
        cond_t = self.diffusion_step_encoder(timesteps)
        if condition is not None:
            condition = self.cond_emb(condition)
            cond_t = torch.cat([cond_t, condition], dim=1)
        X = sample
        h_s = []
        # TODO: Maybe Change this
        for idx, (resnet, resnet2, attn, downsample) in enumerate(self.down_modules):
            X = resnet(X, cond_t)
            h_s.append(X)
            X = resnet2(X, cond_t)
            X = attn(X)
            h_s.append(X)
            X = downsample(X)

        X = self.modules_mid[0](X, cond_t)
        X = self.modules_mid[1](X)
        X = self.modules_mid[2](X, cond_t)

        for idx, (resnet, resnet2, attn, upsample) in enumerate(self.up_modules):
            X = torch.cat([X, h_s.pop()], dim=1)
            # X = torch.cat([X, h_s.pop()], dim=1)
            X = resnet(X, cond_t)
            X = torch.cat([X, h_s.pop()], dim=1)
            X = resnet2(X, cond_t)
            X = attn(X)
            X = upsample(X)
        X = torch.cat([X, res], dim=1)
        X = self.final_conv(X)[:, :, :seq_len]
        X = self.traj_out(rearrange(X, "b t h -> b h t"))
        return X
