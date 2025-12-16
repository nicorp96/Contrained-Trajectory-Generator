from dataclasses import dataclass
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import repeat, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import torchvision.models as models

from common.utils import set_torch_dict_to

# from models.diffusion_unet import Diffusion1DUnet
from models.utils.normalizer import DictNormalizer
from common.get_class import get_class_dict

# @dataclass
# class ConstraintOutputs:
#     gradE_x: torch.Tensor  # [B,N,D]
#     gradE_v: torch.Tensor  # [B,N,D] (optional, kept 0 here)
#     energy: torch.Tensor  # [B]


@dataclass
class ConstraintOutputs:
    grad_pos: (
        torch.Tensor
    )  # [B,N,3]  ∂E/∂position (in *normalized* space unless return_denorm_grads=True)
    grad_vel: torch.Tensor  # [B,N,3]  ∂E/∂velocity
    energy: torch.Tensor  # [B]      total scalar energy


class DictEncoder(nn.Module):
    def __init__(self, param_shapes: dict):
        super(DictEncoder, self).__init__()
        self.encoders = nn.ModuleDict()

        for name in param_shapes.keys():
            type = param_shapes[name]["type"]
            net = None
            if type == "resnet18":
                net = models.resnet18(pretrained=param_shapes.get("pretrained", True))
                modules = list(net.children())[
                    :-1
                ]  # remove the classification head (fc layer)
                net = nn.Sequential(*modules, nn.Flatten())  # flatten output to [B, D]
                output_dim = param_shapes.get(
                    "output_dim", 512
                )  # resnet18 final feature dim

            elif type == "resnet50":
                net = models.resnet50(pretrained=param_shapes.get("pretrained", True))
                modules = list(net.children())[:-1]
                net = nn.Sequential(*modules, nn.Flatten(), nn.Linear())
                output_dim = param_shapes.get(
                    "output_dim", 2048
                )  # resnet50 final feature dim
            elif type == "gru":
                net = nn.Sequential(
                    nn.Linear(
                        param_shapes[name]["d_obs"], param_shapes[name]["d_model"]
                    ),
                    nn.GRU(
                        param_shapes[name]["d_model"],
                        param_shapes[name]["d_model"] // 2,
                        num_layers=param_shapes[name]["n_layers"],
                        batch_first=True,
                        dropout=param_shapes[name]["dropout"],
                        bidirectional=True,
                    ),
                    nn.LayerNorm(param_shapes[name]["d_model"]),
                )
            elif type == "cnn":
                net = nn.Sequential(
                    nn.Conv1d(
                        in_channels=param_shapes[name]["in_channels"],
                        out_channels=param_shapes[name]["out_channels"],
                        kernel_size=param_shapes[name]["kernel_size"],
                        stride=param_shapes[name]["stride"],
                        padding=param_shapes[name]["padding"],
                    ),
                    nn.ReLU(),
                    nn.Flatten(),
                )
            elif type == "linear":
                net = nn.Sequential(
                    nn.Linear(
                        param_shapes[name]["d_obs"], param_shapes[name]["d_model"]
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        param_shapes[name]["d_model"], param_shapes[name]["d_model"]
                    ),
                )
            else:
                raise ValueError(f"Unknown encoder type: {type}")

            self.encoders[name] = net

    # def set_to_device(self, device: torch.device):
    #     for name, net in self.encoders.items():
    #         self.encoders[name] = net.to(device)

    def forward(self, observations: Dict):
        return {
            name: self.encoders[name](value) for name, value in observations.items()
        }


# class ConstraintAnalyzer(nn.Module):
#     """
#     Differentiable constraint energies + autograd gradients.
#     Works with normalized inputs but can denormalize internally for physically
#     meaningful thresholds (v_max, a_max). Then gradients are mapped back.
#     """
#
#     def __init__(self, cfg: Dict):
#         super().__init__()
#         # weights
#         w = cfg.get("weights", {})
#         self.w_goal = w.get("goal", 1.0)
#         self.w_dyn = w.get("dyn", 1.0)
#         self.w_spd = w.get("speed", 0.2)
#         self.w_acc = w.get("accel", 0.2)
#         self.w_jrk = w.get("jerk", 0.05)
#         self.w_term = w.get("term", 0.5)
#
#         # sub-weights inside goal term
#         gcfg = cfg.get("goal", {})
#         self.alpha_final = gcfg.get("alpha_final", 1.0)
#         self.alpha_path = gcfg.get("alpha_path", 0.1)
#         self.beta_pos = gcfg.get("beta_pos", 1.0)
#         self.beta_vel = gcfg.get("beta_vel", 0.2)
#
#         # physical limits (in *real* units)
#         lim = cfg.get("limits", {})
#         self.v_max = lim.get("v_max", 5.0)
#         self.a_max = lim.get("a_max", 4.0)
#
#         # stats for (de)normalization (pos/vel), shape [3]
#         stats = cfg.get("stats", None)
#         if stats is None:
#             raise ValueError(
#                 "ConstraintAnalyzer needs stats: mean_pos,std_pos, mean_vel,std_vel"
#             )
#         self.register_buffer(
#             "mean_pos", torch.tensor(stats["mean_pos"], dtype=torch.float32)
#         )
#         self.register_buffer(
#             "std_pos",
#             torch.clamp(torch.tensor(stats["std_pos"], dtype=torch.float32), min=1e-6),
#         )
#         self.register_buffer(
#             "mean_vel", torch.tensor(stats["mean_vel"], dtype=torch.float32)
#         )
#         self.register_buffer(
#             "std_vel",
#             torch.clamp(torch.tensor(stats["std_vel"], dtype=torch.float32), min=1e-6),
#         )
#
#         self.return_denorm_grads = cfg.get(
#             "return_denorm_grads", False
#         )  # usually False
#
#     def _denorm_pos(self, x_nm):  # [...,3]
#         return x_nm * self.std_pos + self.mean_pos
#
#     def _denorm_vel(self, v_nm):  # [...,3]
#         return v_nm * self.std_vel + self.mean_vel
#
#     def forward(
#         self,
#         state: Dict[
#             str, torch.Tensor
#         ],  # {"position":[B,N,3], "vel_ned":[B,N,3]}  (normalized)
#         environment: Dict[str, torch.Tensor],  # {"goal":[B,1,3]} (normalized)
#         extras: Optional[
#             Dict[str, torch.Tensor]
#         ] = None,  # {"dt":[B] or [B,N] or scalar}
#         mask: Optional[torch.Tensor] = None,  # [B,N] True for valid
#     ) -> ConstraintOutputs:
#
#         pos_nm = state["position"]  # [B,N,3]
#         vel_nm = state["vel_ned"]  # [B,N,3]
#         goal_nm = environment["goal"]  # [B,1,3] or [B,3]
#
#         B, N, _ = pos_nm.shape
#         if goal_nm.dim() == 3 and goal_nm.size(1) == 1:
#             goal_nm = goal_nm.expand(B, N, 3)
#         elif goal_nm.dim() == 2:
#             goal_nm = goal_nm[:, None, :].expand(B, N, 3)
#
#         # dt handling (supports scalar, [B], or [B,N])
#         if extras is not None and "dt" in extras:
#             dt = extras["dt"]
#             if dt.dim() == 0:
#                 dt = dt.view(1, 1).expand(B, N)
#             elif dt.dim() == 1:
#                 dt = dt[:, None].expand(B, N)
#             # else assume [B,N]
#         else:
#             dt = torch.ones(B, N, device=pos_nm.device)
#
#         # Create *denormalized* tensors for physically meaningful penalties
#         pos = self._denorm_pos(pos_nm).requires_grad_(True)  # [B,N,3]
#         vel = self._denorm_vel(vel_nm).requires_grad_(True)  # [B,N,3]
#         goal = self._denorm_pos(goal_nm)  # [B,N,3]
#         dtv = dt  # [B,N]
#
#         # mask
#         if mask is None:
#             mask = torch.ones(B, N, device=pos.device, dtype=torch.bool)
#         m = mask.float()
#
#         # ---------- finite differences ----------
#         # shift helpers
#         def shift(x, offset):
#             if offset > 0:
#                 pad = x[:, :1].expand(-1, offset, -1)
#                 return torch.cat([pad, x[:, :-offset]], dim=1)
#             elif offset < 0:
#                 pad = x[:, -1:].expand(-1, -offset, -1)
#                 return torch.cat([x[:, -offset:], pad], dim=1)
#             return x
#
#         # acceleration a_k = (v_k - v_{k-1}) / dt_k
#         v_prev = shift(vel, +1)
#         a = (vel - v_prev) / dtv.unsqueeze(-1)
#
#         # jerk j_k = (a_k - a_{k-1}) / dt_k
#         a_prev = shift(a, +1)
#         j = (a - a_prev) / dtv.unsqueeze(-1)
#
#         # dynamic consistency residual: r_k = p_{k+1}-p_k - v_{k+1}*dt_k
#         p_next = shift(pos, -1)
#         v_next = shift(vel, -1)
#         r_dyn = (p_next - pos) - v_next * dtv.unsqueeze(-1)
#
#         # ---------- energies ----------
#         # goal: final + mild path
#         e_goal_final = (pos[:, -1] - goal[:, -1]).pow(2).sum(dim=-1)  # [B]
#         e_goal_path = ((pos - goal).pow(2).sum(dim=-1) * m).sum(dim=1) / (
#             m.sum(dim=1) + 1e-8
#         )  # [B]
#         E_goal = self.alpha_final * e_goal_final + self.alpha_path * e_goal_path
#
#         # dynamic consistency
#         E_dyn = (r_dyn.pow(2).sum(dim=-1) * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
#
#         # speed limit (soft hinge)
#         speed = torch.linalg.norm(vel, dim=-1)  # [B,N]
#         spd_viol = torch.clamp(speed - self.v_max, min=0.0)
#         E_spd = (spd_viol.pow(2) * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
#
#         # acceleration limit (soft hinge)
#         accn = torch.linalg.norm(a, dim=-1)
#         acc_viol = torch.clamp(accn - self.a_max, min=0.0)
#         E_acc = (acc_viol.pow(2) * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
#
#         # jerk smoothness (L2)
#         E_jrk = (j.pow(2).sum(dim=-1) * m).sum(dim=1) / (m.sum(dim=1) + 1e-8)
#
#         # terminal rest near goal
#         E_term = self.beta_pos * (pos[:, -1] - goal[:, -1]).pow(2).sum(
#             dim=-1
#         ) + self.beta_vel * vel[:, -1].pow(2).sum(dim=-1)
#
#         # total energy
#         E = (
#             self.w_goal * E_goal
#             + self.w_dyn * E_dyn
#             + self.w_spd * E_spd
#             + self.w_acc * E_acc
#             + self.w_jrk * E_jrk
#             + self.w_term * E_term
#         )  # [B]
#         E_sum = E.sum()
#
#         # autograd grads wrt *denormalized* pos/vel
#         grad_pos_denorm, grad_vel_denorm = torch.autograd.grad(
#             E_sum, (pos, vel), retain_graph=False, create_graph=False
#         )
#
#         # map gradients back to *normalized* space if requested
#         if self.return_denorm_grads:
#             grad_pos_out = grad_pos_denorm
#             grad_vel_out = grad_vel_denorm
#         else:
#             # x_real = x_nm * std + mean  =>  dE/dx_nm = dE/dx_real * std
#             grad_pos_out = grad_pos_denorm * self.std_pos
#             grad_vel_out = grad_vel_denorm * self.std_vel
#
#         return ConstraintOutputs(
#             gradE_x=grad_pos_out.detach(),
#             gradE_v=grad_vel_out.detach(),
#             energy=E.detach(),
#         )


class ConstraintAnalyzer(nn.Module):
    """
    Penalize speeds above v_max and/or below v_min.
    Uses denormalized velocities for interpretable limits, then maps grads back.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        lim = cfg.get("limits", {})
        self.v_max = lim.get("v_max", 30.0)  # float or None
        self.v_min = lim.get("v_min", -30.0)  # float or None
        # TODO: is this necessary?: weights for the two penalties
        w = cfg.get("weights", {})
        self.w_max = w.get("over_speed", 1.0)
        self.w_min = w.get("under_speed", 1.0)
        # smooth hinge via softplus for stable grads
        self.softplus_beta = cfg.get("softplus_beta", 10.0)
        self.eps = 1e-8

        # optional time weighting window (e.g., downweight endpoints)
        self.time_weight = cfg.get(
            "time_weight", None
        )  # "middle", "flat", or custom tensor via setter
        self.state_normalizer = None
        self.obs_normalizer = None
        self.environment_normalizer = None

    def set_normalizer(
        self,
        state_normalizer: DictNormalizer,
        obs_normalizer: DictNormalizer,
        environment_normalizer: DictNormalizer,
    ):
        self.state_normalizer = state_normalizer
        self.obs_normalizer = obs_normalizer
        self.environment_normalizer = environment_normalizer

    def set_time_weight(self, w_t: torch.Tensor):
        """Optionally set a [N] or [1,N] or [B,N] time weight (will be broadcast)."""
        self.time_weight = w_t

    # def _denorm_vel(self, v_nm):  # [...,3]
    #     return v_nm * self.std_vel + self.mean_vel

    def forward(
        self,
        state: Dict[
            str, torch.Tensor
        ],  # {"position":[B,N,3], "vel_ned":[B,N,3]} (normalized)
        environment: Dict[str, torch.Tensor],  # unused here
        extras: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,  # [B,N] True for valid
    ) -> ConstraintOutputs:
        # TODO: Physical units -> require denormalization?
        # 1) Work on a detached copy of normalized vel so the guidance
        #    gradient does not backprop into the rest of the network.
        vel_nm_orig = state["vel_ned"]  # [B,N,3] normalized from model
        vel_nm = vel_nm_orig.detach().requires_grad_(True)  # leaf for autograd

        # 2) Build a shallow copy of the state dict with this new vel_ned
        state_for_unnorm = dict(state)
        state_for_unnorm["vel_ned"] = vel_nm

        # 3) Unnormalize via your dict normalizer (differentiable)
        state_unorm = self.state_normalizer.unnormalize(state_for_unnorm)
        vel = state_unorm["vel_ned"]
        B, N, _ = vel.shape
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=vel.device)
        m = mask.float()
        # time weights
        if (
            self.time_weight is None
            or isinstance(self.time_weight, str)
            and self.time_weight == "flat"
        ):
            w_t = torch.ones(B, N, device=vel.device)
        elif isinstance(self.time_weight, str) and self.time_weight == "middle":
            # cosine window: low at ends, high in middle
            t = torch.linspace(0, 1, N, device=vel.device)
            w = 0.5 - 0.5 * torch.cos(2 * torch.pi * t)  # [N]
            w_t = w[None, :].expand(B, N)
        else:
            w_t = self.time_weight
            while w_t.dim() < 2:
                w_t = w_t.unsqueeze(0)
            if w_t.size(0) == 1:
                w_t = w_t.expand(B, -1)

        speed = torch.linalg.norm(vel, dim=-1)  # [B,N]
        beta = self.softplus_beta
        per_traj_E = torch.zeros(B, device=vel.device)  # [B]
        E_terms = []

        # Over-speed penalty: softplus(speed - v_max)
        if self.v_max is not None:
            over = torch.nn.functional.softplus(speed - self.v_max, beta=beta)  # ~relu
            E_max = ((over**2) * m * w_t).sum(dim=1) / (m.sum(dim=1) + self.eps)  # [B]
            E_terms.append(self.w_max * E_max)
            per_traj_E = per_traj_E + self.w_max * E_max
        # Under-speed penalty: softplus(v_min - speed)
        if self.v_min is not None:  # and self.v_min > 0:
            under = torch.nn.functional.softplus(self.v_min - speed, beta=beta)
            E_min = ((under**2) * m * w_t).sum(dim=1) / (m.sum(dim=1) + self.eps)  # [B]
            E_terms.append(self.w_min * E_min)
            per_traj_E = per_traj_E + self.w_min * E_min

            # Early exit if no terms
        if (self.v_max is None) and (self.v_min is None):
            zeros = torch.zeros(B, device=vel.device)
            return ConstraintOutputs(
                grad_pos=torch.zeros(B, N, 3, device=vel.device),
                grad_vel=torch.zeros(B, N, 3, device=vel.device),
                energy=zeros,
            )

        E_scalar = per_traj_E.sum()  # scalar

        # 7) Gradient w.r.t. *normalized* velocity vel_nm
        if torch.is_grad_enabled() and vel_nm.requires_grad:
            (grad_v_nm,) = torch.autograd.grad(
                E_scalar,
                (vel_nm,),
                retain_graph=False,
                create_graph=False,
            )
        else:
            grad_v_nm = torch.zeros_like(vel_nm)
        # E_sum = E.sum()  # scalar
        # gradients w.r.t. *denormalized* vel
        # if torch.is_grad_enabled():
        #     (grad_vel_denorm,) = torch.autograd.grad(
        #         E_sum, (vel,), retain_graph=False, create_graph=False
        #     )
        # else:
        #     grad_vel_denorm = torch.zeros_like(vel)
        # TODO: map back to normalized space:
        # v_real = v_nm * std + mean  =>  dE/dv_nm = dE/dv_real * std
        # grad_vel_nm = grad_vel_denorm * self.std_vel

        return ConstraintOutputs(
            grad_pos=torch.zeros_like(
                vel
            ),  # no position constraint in this minimalist version
            grad_vel=grad_v_nm,  # torch.zeros_like(vel),
            energy=per_traj_E.detach(),  # .detach(),
        )


class ObsEncoder(nn.Module):
    def __init__(self, config: Dict):
        super(ObsEncoder, self).__init__()
        self.config = config
        self.dict_encoders = DictEncoder(config["param_shapes"])

    def forward(
        self, obs: Dict, mask: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        obs_encoded_dict = self.dict_encoders(obs)
        obs_encoded = torch.cat(
            [obs_encoded_dict[key] for key in obs_encoded_dict.keys()], dim=-1
        )
        # obs_encoded = obs_encoded * mask[..., None]
        return obs_encoded


class ConstraintsAttention(nn.Module):
    def __init__(self, config: Dict):
        super().__init__()
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        # project inputs to a common model dim
        self.q_proj = nn.Linear(config["d_state"], d_model)
        self.k_proj = nn.Linear(config["d_ctx"], d_model)
        self.v_proj = nn.Linear(config["d_ctx"], d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config["ff_mult"] * d_model),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["ff_mult"] * d_model, d_model),
        )
        self.out_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        state: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        state: [B, N, d_q]
        context: [B, M, d_ctx]
        mask: [B, N]
        returns: [B, N, d_model]
        """
        Q = self.q_proj(state)
        K = self.k_proj(context)
        V = self.v_proj(context)
        # key_padding_mask = ~mask
        Qn = self.ln1(Q)
        fused, _ = self.attention(Qn, K, V, need_weights=False)
        fused = Q + fused
        fused = self.out_ln(fused + self.ff(fused))
        # TODO: here is the cross attention to the hole trajectory, so we've got [B,N,d_model]. Is it correct?, how to deal with this?
        out = fused.mean(dim=1)
        return out


class DiffusionTrajectory(nn.Module):
    def __init__(self, config):
        super(DiffusionTrajectory, self).__init__()
        self.config = config
        # self.unet = Diffusion1DUnet(config)
        self.denoiser_net = get_class_dict(self.config)
        self.scheduler = None
        self.inference_steps = config.get("num_inference_steps", 12)
        self.guidance_scale = config.get("guidance_scale", 0.0)

    def set_scheduler(self, scheduler: Optional[DDIMScheduler]):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        ctx: torch.Tensor,
        mask: torch.Tensor = None,
        guidance_grad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = noise.device
        scheduler = self.scheduler
        model = self.denoiser_net
        guidance_scale = self.guidance_scale
        x = noise
        scheduler.set_timesteps(self.inference_steps, device=device)
        for t in scheduler.timesteps:
            t = t.to(device)
            eps = model(x, t, ctx)
            # Optional constraint guidance as noise adjustment
            # GUIDANCE WITH ENERGY GRADIENT
            if guidance_grad is not None and guidance_scale > 0.0:
                # Convert gradient on x to noise-space adjustment approximately
                eps = eps - guidance_scale * guidance_grad
            # One scheduler step
            x = scheduler.step(eps, t, x).prev_sample
            # keep padded steps at zero
            # x = x.masked_fill(~mask[..., None], 0.0)
        return x

    def forward(
        self, noise: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor
    ) -> torch.Tensor:
        return self.denoiser_net(noise, t, ctx)


class StructureHead(nn.Module):
    def __init__(self, d_ctx: int, d_state: int):  # param_state: Dict,
        super(StructureHead, self).__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_ctx),
            nn.Linear(d_ctx, 2 * d_ctx),
            nn.GELU(),
            nn.Linear(2 * d_ctx, d_ctx),
        )
        # TODO: check list for states
        self.ds = nn.Linear(d_ctx, d_state)
        # TODO: check timing head
        self.dT = nn.Sequential(nn.Linear(d_ctx, 1), nn.Sigmoid(), nn.Linear(1, 1))

        self.score = nn.Linear(d_ctx, 1)

    def forward(self, x: torch.Tensor, seq_len: int):
        h = self.net(x)
        x_s = repeat(x, "B D -> B N D", N=seq_len)  # expand for N if needed
        # TODO: it should be a Dictionary Output
        return self.ds(x_s), self.dT(h), torch.sigmoid(self.score(h))


class ConstraintCorrectionHead(nn.Module):
    def __init__(
        self,
        d_state: int,
        d_resid: int,
        d_obs: int,
        d_env: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_state + d_resid + d_obs + d_env + 1),  # +1 for E
            nn.Linear(d_state + d_resid + d_obs + d_env + 1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_state),
        )
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        state: torch.Tensor,  # [B, N, d_state]   (s)
        residuals: torch.Tensor,  # [B, N, d_resid]   (r_t)
        energy: torch.Tensor,  # [B, 1]            (E)
        obs_ctx: torch.Tensor,  # [B, d_obs]        (O)
        env_ctx: torch.Tensor,  # [B, d_env]        (G)
    ):
        B, N, _ = state.shape

        # broadcast globals over time
        E_exp = energy.unsqueeze(1).expand(B, N, 1)
        O_exp = obs_ctx.unsqueeze(1).expand(B, N, obs_ctx.size(-1))
        G_exp = env_ctx.unsqueeze(1).expand(B, N, env_ctx.size(-1))

        x = torch.cat([state, residuals, E_exp, O_exp, G_exp], dim=-1)  # [B, N, *]
        h = self.mlp[0:3](x)  # up to and including GELU
        delta_s_corr = self.mlp[3](h)  # [B, N, d_state]
        score = self.score_head(h.mean(dim=1))  # optional [B, 1] trajectory score

        return delta_s_corr, score


class TrajectorySoR(nn.Module):
    def __init__(self, config: Dict):
        super(TrajectorySoR, self).__init__()
        self.config = config
        self.scheduler = None
        self.obs_encoder = ObsEncoder(config["obs_encoder"])
        self.constraint_analyzer = ConstraintAnalyzer(config["constraint_analyzer"])
        self.constraints_attention = ConstraintsAttention(
            config["constraints_attention"]
        )
        self.diffusion_model = DiffusionTrajectory(config["diffusion_model"])
        self.structure_head = StructureHead(
            d_ctx=config["diffusion_model"]["cond_dim"],
            d_state=config["d_state"],
        )
        self.K_cycles = config["K_cycles"]
        self.guidance_eta = config.get("guidance_eta", 0.0)
        # TODO: check dimensions
        self.time_head = nn.Sequential(
            nn.LayerNorm(config["d_state"] * config["sequence_len"]),
            nn.Linear(config["d_state"] * config["sequence_len"], 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

        self.state_normalizer = None
        self.obs_normalizer = None
        self.environment_normalizer = None

    def set_scheduler(self, scheduler: Optional[DDIMScheduler]):
        self.scheduler = scheduler
        self.diffusion_model.set_scheduler(scheduler)

    def set_normalizer(
        self,
        state_normalizer: DictNormalizer,
        obs_normalizer: DictNormalizer,
        environment_normalizer: DictNormalizer,
    ):
        self.constraint_analyzer.set_normalizer(
            state_normalizer, obs_normalizer, environment_normalizer
        )
        self.state_normalizer = state_normalizer
        self.obs_normalizer = obs_normalizer
        self.environment_normalizer = environment_normalizer

    @torch.no_grad()
    def context_without_diff(
        self, states_gt_nm, observations_nm, env_nm, device: torch.device
    ):
        self.obs_encoder.eval()
        self.constraint_analyzer.eval()
        trajectory = torch.cat(
            [states_gt_nm[key] for key in states_gt_nm.keys()], dim=-1
        )
        trajectory = trajectory.to(device)
        observations = set_torch_dict_to(dict=observations_nm, device=device)
        states = set_torch_dict_to(dict=states_gt_nm, device=device)
        environment = set_torch_dict_to(dict=env_nm, device=device)
        obs_encoder = self.obs_encoder.to(device)
        obs_ctx = obs_encoder(observations, mask=None, device=device)
        constraint_out: ConstraintOutputs = self.constraint_analyzer(
            states, environment
        )
        # cond_ctx = torch.cat([obs_ctx, constraint_out], dim=-1) # here also use constraints_out
        cond_ctx = self.constraints_attention(trajectory, obs_ctx, None)
        return dict(cond_ctx=cond_ctx, constraint_out=constraint_out)

    def forward(
        self,
        states: Dict[str, torch.Tensor],
        observations: Dict[str, torch.Tensor],
        environment: Dict[str, torch.Tensor],
        device: torch.device,
    ):
        """
        states: Dict of tensors with keys like 'position', 'velocity', each of shape [B, N, D]
        observations: Dict of tensors with keys like 'camera', 'lidar', each of shape [B, ...]
        environment: Dict of tensors with keys like 'goal', 'obstacles', each of shape [B, ...]
        returns: Dict with updated states after K diffusion cycles
        """
        trajectory = torch.cat([states[key] for key in states.keys()], dim=-1)
        trajectory = trajectory.to(device)
        B, N, D = trajectory.shape
        outs = []
        # trajectory_for_attn = trajectory.clone()
        observations = set_torch_dict_to(dict=observations, device=device)
        states = set_torch_dict_to(dict=states, device=device)
        environment = set_torch_dict_to(dict=environment, device=device)
        obs_encoder = self.obs_encoder.to(device)
        obs_ctx = obs_encoder(observations, mask=None, device=device)

        for _ in range(self.K_cycles):
            constraint_out: ConstraintOutputs = self.constraint_analyzer(
                states, environment
            )
            # cond_ctx = torch.cat([obs_ctx, constraint_out], dim=-1) # here also use constraints_out
            cond_ctx = self.constraints_attention(trajectory, obs_ctx, None)
            noise = torch.randn(B, N, D, device=device)
            # TODO: Idea add noise by constraint violation and only remove noise using diffusion model. (Inpainting)
            pred = self.diffusion_model.sample(
                noise,
                ctx=cond_ctx,
                guidance_grad=torch.cat(
                    [constraint_out.grad_pos, constraint_out.grad_vel], dim=-1
                ),
            )
            # TODO: it should be a Dict??
            ds, dT, score = self.structure_head(cond_ctx, seq_len=N)
            pred_t = rearrange(pred, "B N D -> B (N D)")
            # TODO: Compute Time with states or it should be learned?
            head_time = F.softplus(self.time_head(pred_t)) + dT

            # TODO: prediction (B,N, S) but ds is (B, S) ? need to expand ds
            x = (
                pred[:, :, :3]
                + ds[:, :, :3]
                - self.guidance_eta * constraint_out.grad_pos
            )
            v = (
                pred[:, :, 3:6]
                + ds[:, :, 3:6]
                - self.guidance_eta * constraint_out.grad_vel
            )
            # TODO: Update only states better
            states["position"] = x
            states["vel_ned"] = v
            # TODO: dont use this!!!
            # trajectory[:, :, :3] = x
            # trajectory[:, :, 3:6] = v
            trajectory = torch.cat([x, v], dim=-1)
            # TODO: Check output
            outs.append(
                dict(
                    trajectory=trajectory,
                    time=head_time,
                    score=score,
                    energy=constraint_out.energy,
                )
            )
        return outs


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "obs_encoder": {
            "param_shapes": {
                "start_position": {
                    "type": "linear",
                    "d_obs": 3,
                    "d_model": 64,
                },
                "start_velocity": {
                    "type": "linear",
                    "d_obs": 3,
                    "d_model": 64,
                },
            }
        },
        "constraint_analyzer": {
            "d_state": 6,
            "d_res": 128,
        },
        "constraints_attention": {
            "d_model": 128,
            "n_heads": 4,
            "d_state": 6,
            "d_ctx": 128,
            "ff_mult": 4,
            "dropout": 0.1,
        },
        "diffusion_model": {
            "input_dim": 6,
            "cond_dim": 128,
            "time_t_embed_dim": 128,
            "time_encoder": False,
            "d_model": 128,
            "down_dims": [256, 512, 1024],
            "kernel_size": 3,
            "n_blocks": 8,
            "cond_film": True,
            "inference_steps": 12,
            "guidance_scale": 1.5,
            "diffusion_t_embed_dim": 128,
            "cfg_drop_prob": 0.1,
        },
        "d_state": 6,
        "K_cycles": 3,
        "guidance_eta": 0.1,
    }

    states_ig_nm = {
        "position": torch.randn(2, 128, 3).to(device),
        "vel_ned": torch.randn(2, 128, 3).to(device),
    }
    observations_nm = {
        "start_position": torch.randn(2, 1, 3).to(device),
        "start_velocity": torch.randn(2, 1, 3).to(device),
    }
    environments_nm = {
        "goal": torch.randn(2, 1, 3).to(device),
    }

    scheduler = DDIMScheduler(
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        set_alpha_to_one=True,
        num_train_timesteps=1000,
    )
    model = TrajectorySoR(config)
    model.set_scheduler(scheduler)
    outputs = model(states_ig_nm, observations_nm, environments_nm, device=device)
