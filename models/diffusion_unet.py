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


class Diffusion1DUnet(nn.Module):
    def __init__(self, config):
        super(Diffusion1DUnet, self).__init__()
        self.config = config
        self.input_dim = self.config["input_dim"]
        self.cond_dim = self.config["cond_dim"]
        self.diffusion_t_embed_dim = self.config["diffusion_t_embed_dim"]
        self.down_dims = self.config["down_dims"]  # List [256, 512, 1024]
        self.kernel_size = self.config["kernel_size"]
        self.n_blocks = self.config["n_blocks"]
        self.cond_film = self.config["cond_film"]
        self.d_model = self.config["d_model"]
        # self.traj_embd = nn.Linear(self.input_dim, self.d_model)
        self.traj_embd = nn.Conv1d(self.input_dim, self.d_model, kernel_size=1)

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
        self.time_encoder = None
        if config["time_encoder"]:
            self.time_t_embed_dim = self.config["time_t_embed_dim"]
            self.time_encoder = nn.Sequential(
                SinusoidalPosEmb(self.time_t_embed_dim),
                nn.Linear(self.time_t_embed_dim, 4 * self.time_t_embed_dim),
                nn.Mish(),
                nn.Linear(4 * self.time_t_embed_dim, self.time_t_embed_dim),
            )
        cond_dim_t = self.diffusion_t_embed_dim
        if self.cond_dim is not None:
            cond_dim_t += self.d_model
            if config["time_encoder"]:
                cond_dim_t += self.time_t_embed_dim

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
                            out_channels=out_dim,
                            cond_dim=cond_dim_t,
                            kernel_size=self.kernel_size,
                            n_groups=self.n_blocks,
                            cond_film=self.cond_film,
                        ),
                        ConditionalResidual1DBlock(
                            in_channels=out_dim,
                            out_channels=out_dim,
                            cond_dim=cond_dim_t,
                            kernel_size=self.kernel_size,
                            n_groups=self.n_blocks,
                            cond_film=self.cond_film,
                        ),
                        Residual(PreNorm(out_dim, LinearAttention(out_dim))),
                        Downsample1d(out_dim) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.up_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(
            reversed(list(zip(dim_total[:-1], dim_total[1:])))
        ):
            is_last = idx >= (len(list(zip(dim_total[:-1], dim_total[1:]))) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidual1DBlock(
                            in_channels=out_dim * 2,
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
                        Upsample1d(in_dim) if not is_last else nn.Identity(),
                    ]
                )
            )
        self.final_conv = nn.Sequential(
            Conv1DBlock(
                start_dim,
                start_dim,
                kernel_size=self.kernel_size,
                n_groups=2,
            ),  # TODO: Check this: not yet good implemented
            nn.Conv1d(start_dim, start_dim, 1),
            # nn.Linear(512, 254),
        )
        # self.traj_out = nn.Linear(self.d_model, self.input_dim)
        self.traj_out = nn.Conv1d(self.d_model, self.input_dim, kernel_size=1)

    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        time: torch.Tensor = None,
    ):
        bsz, seq_len, inp = sample.shape
        # sample = self.traj_embd(sample)
        sample = rearrange(sample, " b h t -> b t h")
        # With conv 1D as projection
        sample = self.traj_embd(sample)
        # sample_pad = nn.functional.pad(sample, pad=(0, 2), mode="constant", value=0)
        timesteps = timesteps.expand(sample.size(0))
        cond_t = self.diffusion_step_encoder(timesteps)
        if condition is not None:
            condition = self.cond_emb(condition)
            cond_t = torch.cat([cond_t, condition], dim=1)
            if time is not None and self.time_encoder:
                cond_time = self.time_encoder(time)
                cond_t = torch.cat([cond_t, cond_time], dim=1)
        X = sample  # sample_pad
        h_s = []
        for idx, (resnet, resnet2, attn, downsample) in enumerate(self.down_modules):
            X = resnet(X, cond_t)
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
            X = resnet2(X, cond_t)
            X = attn(X)
            X = upsample(X)
        X = self.final_conv(X)[:, :, :seq_len]
        # With conv 1D as projection
        X = self.traj_out(X)
        X = rearrange(X, "b t h -> b h t")
        # X = self.traj_out(rearrange(X, "b t h -> b h t"))
        return X


class Diffusion1DUnetCtx(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_dim = config["input_dim"]
        self.cond_dim = config["cond_dim"]  # dim of your existing cond vector
        self.diffusion_t_embed_dim = config["diffusion_t_embed_dim"]
        self.down_dims = config["down_dims"]
        self.kernel_size = config["kernel_size"]
        self.n_blocks = config["n_blocks"]
        self.cond_film = config["cond_film"]
        self.d_model = config["d_model"]

        # ---------- NEW: context encoder config ----------
        self.ctx_in_dim = config["input_dim"]  # e.g., 6 (pos+vel)
        self.ctx_len = config.get("ctx_len", 4)  # K
        self.ctx_mode = config.get("ctx_mode", "conv")  # "flat" | "conv"

        # trajectory embedding
        self.traj_embd = nn.Conv1d(self.input_dim, self.d_model, kernel_size=1)

        # existing global condition (vector) encoder -> [B, d_model]
        self.cond_emb = nn.Sequential(
            nn.Linear(self.cond_dim, self.d_model),
            nn.SiLU(),
            nn.Linear(self.d_model, self.d_model),
        )

        # ---------- NEW: context sequence encoder -> [B, d_model] ----------
        if self.ctx_in_dim is not None:
            if self.ctx_mode == "flat":
                self.ctx_encoder = nn.Sequential(
                    nn.Linear(self.ctx_len * self.ctx_in_dim, self.d_model),
                    nn.SiLU(),
                    nn.Linear(self.d_model, self.d_model),
                )
            elif self.ctx_mode == "conv":
                # expects [B, K, C] -> permute to [B, C, K]
                self.ctx_encoder = nn.Sequential(
                    nn.Conv1d(
                        self.ctx_in_dim, self.d_model // 2, kernel_size=3, padding=1
                    ),
                    nn.SiLU(),
                    nn.Conv1d(
                        self.d_model // 2, self.d_model, kernel_size=3, padding=1
                    ),
                    nn.SiLU(),
                )
            else:
                raise ValueError("ctx_mode must be 'flat' or 'conv'")

        # diffusion timestep encoding
        # TODO: Check this is correct??!!!
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(self.diffusion_t_embed_dim),
            nn.Linear(self.diffusion_t_embed_dim, 4 * self.diffusion_t_embed_dim),
            nn.Mish(),
            nn.Linear(4 * self.diffusion_t_embed_dim, self.diffusion_t_embed_dim),
        )

        # ---------- total conditioning width for FiLM blocks ----------
        # time emb + cond_token + (optional) ctx_token
        cond_dim_t = self.diffusion_t_embed_dim + self.d_model
        if self.ctx_in_dim is not None:
            cond_dim_t += self.d_model

        dim_total = [self.d_model] + list(self.down_dims)
        start_dim = dim_total[0]
        middle_dim = self.down_dims[-1]

        self.modules_mid = nn.ModuleList(
            [
                ConditionalResidual1DBlock(
                    middle_dim,
                    middle_dim,
                    cond_dim_t,
                    self.kernel_size,
                    self.n_blocks,
                    self.cond_film,
                ),
                PreNorm(middle_dim, Attention(middle_dim, dim_head=32, heads=4)),
                ConditionalResidual1DBlock(
                    middle_dim,
                    middle_dim,
                    cond_dim_t,
                    self.kernel_size,
                    self.n_blocks,
                    self.cond_film,
                ),
            ]
        )

        self.down_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(zip(dim_total[:-1], dim_total[1:])):
            is_last = idx >= (len(dim_total) - 2)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidual1DBlock(
                            in_dim,
                            out_dim,
                            cond_dim_t,
                            self.kernel_size,
                            self.n_blocks,
                            self.cond_film,
                        ),
                        ConditionalResidual1DBlock(
                            out_dim,
                            out_dim,
                            cond_dim_t,
                            self.kernel_size,
                            self.n_blocks,
                            self.cond_film,
                        ),
                        Residual(PreNorm(out_dim, LinearAttention(out_dim))),
                        Downsample1d(out_dim) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.up_modules = nn.ModuleList([])
        for idx, (in_dim, out_dim) in enumerate(
            reversed(list(zip(dim_total[:-1], dim_total[1:])))
        ):
            is_last = idx >= (len(dim_total) - 2)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidual1DBlock(
                            out_dim * 2,
                            in_dim,
                            cond_dim_t,
                            self.kernel_size,
                            self.n_blocks,
                            self.cond_film,
                        ),
                        ConditionalResidual1DBlock(
                            in_dim,
                            in_dim,
                            cond_dim_t,
                            self.kernel_size,
                            self.n_blocks,
                            self.cond_film,
                        ),
                        Residual(PreNorm(in_dim, LinearAttention(in_dim))),
                        Upsample1d(in_dim) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1DBlock(start_dim, start_dim, kernel_size=self.kernel_size, n_groups=2),
            nn.Conv1d(start_dim, start_dim, 1),
        )
        self.traj_out = nn.Linear(self.d_model, self.input_dim)

    # ---------- helpers ----------
    def _encode_cond_vec(self, cond_vec: torch.Tensor) -> torch.Tensor:
        return self.cond_emb(cond_vec)  # [B, d_model]

    def _encode_context(self, context: torch.Tensor) -> torch.Tensor:
        """
        context: [B, K, C] with K == self.ctx_len and C == self.ctx_in_dim
        returns ctx_token: [B, d_model]
        """
        if self.ctx_in_dim is None or context is None:
            # learned null token could be used; zero is fine if you use CFG
            return (
                torch.zeros(context.size(0), self.d_model, device=context.device)
                if context is not None
                else None
            )

        if self.ctx_mode == "flat":
            B, K, C = context.shape
            assert K == self.ctx_len and C == self.ctx_in_dim, "context shape mismatch"
            x = context.reshape(B, K * C)
            return self.ctx_encoder(x)
        else:  # "conv"
            # [B,K,C] -> [B,C,K]
            x = rearrange(context, "b k c -> b c k")
            x = self.ctx_encoder(x)  # [B, d_model, K]
            x = x.mean(dim=-1)  # GAP over K -> [B, d_model]
            return x

    # ---------- forward ----------
    def forward(
        self,
        sample: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        time: torch.Tensor = None,
        **kwargs,
    ):
        """
        sample   : [B, T, input_dim]  (noisy sequence)
        timesteps: [B] or scalar
        condition: [B, cond_dim]      (your existing goal/start/etc. vector)
        context  : [B, K, ctx_in_dim] (last-K pos+vel), optional
        """
        B, T, _ = sample.shape

        # embed sequence and switch to [B, C, T] for convs  (FIXED)
        sample = rearrange(sample, "b t h -> b h t")  # [B, d_model, T]
        X = self.traj_embd(sample)  # [B, T, d_model]

        # encode time + cond + NEW context
        t_emb = self.diffusion_step_encoder(timesteps.expand(B))  # [B, diff_emb]
        cond_token = self._encode_cond_vec(condition)  # [B, d_model]
        if "context" in kwargs.keys():
            ctx_token = self._encode_context(kwargs["context"])  # [B, d_model]
            cond_t = torch.cat([t_emb, cond_token, ctx_token], dim=1)
        else:
            cond_t = torch.cat([t_emb, cond_token], dim=1)

        # UNet
        skips = []
        for res1, res2, attn, down in self.down_modules:
            X = res1(X, cond_t)
            X = res2(X, cond_t)
            X = attn(X)
            skips.append(X)
            X = down(X)

        X = self.modules_mid[0](X, cond_t)
        X = self.modules_mid[1](X)
        X = self.modules_mid[2](X, cond_t)

        for res1, res2, attn, up in self.up_modules:
            X = torch.cat([X, skips.pop()], dim=1)
            X = res1(X, cond_t)
            X = res2(X, cond_t)
            X = attn(X)
            X = up(X)

        X = self.final_conv(X)  # [B, d_model, T]
        X = X[:, :, :T]
        X = rearrange(X, "b h t -> b t h")
        return self.traj_out(X)  # [B, T, input_dim]
