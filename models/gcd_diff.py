from dataclasses import dataclass
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import repeat, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


from common.utils import dict_to_tensor_concat
from models.utils.diffusion import apply_inpainting
from models.utils.normalizer import DictNormalizer
from common.get_class import get_class_dict


@dataclass
class ConstraintOutputs:
    grad_pos: (
        torch.Tensor
    )  # [B,N,3]  ∂E/∂position (in *normalized* space unless return_denorm_grads=True)
    grad_vel: torch.Tensor  # [B,N,3]  ∂E/∂velocity
    energy: torch.Tensor  # [B]      total scalar energy


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

    def forward(
        self,
        states: Dict[
            str, torch.Tensor
        ],  # {"position":[B,N,3], "vel_ned":[B,N,3]} (normalized)
        actions: Dict[str, torch.Tensor],  # unused here
        environment: Dict[str, torch.Tensor],  # unused here
        extras: Optional[Dict[str, torch.Tensor]] = None,
        mask: Optional[torch.Tensor] = None,  # [B,N] True for valid
    ) -> ConstraintOutputs:
        # TODO: Physical units -> require denormalization?
        # 1) Work on a detached copy of normalized vel so the guidance
        #    gradient does not backprop into the rest of the network.
        vel_nm_orig = states["vel_ned"]  # [B,N,3] normalized from model
        vel_nm = vel_nm_orig.detach().requires_grad_(True)  # leaf for autograd

        # 2) Build a shallow copy of the state dict with this new vel_ned
        state_for_unnorm = dict(states)
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
        trajectory: Optional[torch.Tensor],
        context: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        state: [B, N, d_q]
        context: [B, M, d_ctx]
        mask: [B, N]
        returns: [B, N, d_model]
        """
        Q = self.q_proj(trajectory)
        K = self.k_proj(context).unsqueeze(1)
        V = self.v_proj(context).unsqueeze(1)
        # key_padding_mask = ~mask
        Qn = self.ln1(Q)
        # TODO: here is the cross attention to the hole trajectory, so we've got [B,N,d_model]. Is it correct?, how to deal with this?
        fused, _ = self.attention(Qn, K, V, need_weights=False)
        fused = Q + fused
        fused = self.out_ln(fused + self.ff(fused))
        out = fused.mean(dim=1)
        return out


class DiffusionTrajectory(nn.Module):
    def __init__(self, config):
        super(DiffusionTrajectory, self).__init__()
        self.config = config
        self.denoiser_net = get_class_dict(self.config)
        self.scheduler = None
        self.inference_steps = config.get("num_inference_steps", 12)
        self.guidance_scale = config.get("guidance_scale", 0.0)
        self.cfg_drop_prob = config.get("cfg_drop_prob", 0.0)
        self.cfg = True if "cfg_drop_prob" in config else False

    def set_scheduler(self, scheduler: Optional[DDIMScheduler]):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
        self,
        noise: torch.Tensor,
        ctx: torch.Tensor,
        traj_start: torch.Tensor,
    ) -> torch.Tensor:
        device = noise.device
        scheduler = self.scheduler
        # TODO: Use EMA model for denoiser network
        model = self.denoiser_net
        guidance_scale = self.guidance_scale
        x = noise
        mask = torch.zeros_like(noise, dtype=torch.bool, device=device)
        mask[:, 0:1, :] = torch.ones_like(traj_start, dtype=torch.bool, device=device)
        scheduler.set_timesteps(self.inference_steps, device=device)
        uncond = torch.zeros_like(ctx)
        x = apply_inpainting(x, mask, traj_start, noise=False)

        def guide_and_step(x_t, t_scalar):
            # batched uncond/cond forward
            traj_in = torch.cat([x_t, x_t], dim=0)
            cond_in = torch.cat([uncond, ctx], dim=0)
            # t_in = t_scalar.expand(noise.shape[0]).to(noise.device)
            t_in = t_scalar.to(noise.device)

            model_out = model(traj_in, t_in, cond_in)
            out_uncond, out_cond = model_out.chunk(2, dim=0)
            pred_type = scheduler.config.prediction_type
            guided = out_cond
            if guidance_scale == 1.0:
                guided = out_cond
            elif pred_type in ("epsilon", "v_prediction"):
                guided = out_uncond + guidance_scale * (out_cond - out_uncond)
            x_prev = scheduler.step(guided, t_scalar, x_t).prev_sample
            return x_prev

        for t in scheduler.timesteps:
            x = guide_and_step(x, t)
            x = apply_inpainting(x, mask, traj_start, noise=False)
        return x

    def compute_loss(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target)
        return loss

    def forward(
        self, input: torch.Tensor, ctx: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        scheduler = self.scheduler
        denoiser_net = self.denoiser_net
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (input.size(0),),
            device=device,
        )
        start_traj_cond = input[:, 0, :].unsqueeze(1)
        noise = torch.randn_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool, device=device)
        mask[:, 0:1, :] = torch.ones_like(
            start_traj_cond, dtype=torch.bool, device=device
        )
        # TODO: Apply Inpainting (start) should be zero at t=0
        input = apply_inpainting(input, mask, x_known=start_traj_cond, noise=True)
        states_noisy = scheduler.add_noise(input, noise, timesteps)
        # TODO: Apply Inpainting (start) should be known Trajectory
        states_noisy = apply_inpainting(
            states_noisy, mask, x_known=start_traj_cond, noise=False
        )
        if self.cfg:
            cfg_p = self.cfg_drop_prob
            if cfg_p > 0.0:
                B = ctx.size(0)
                drop_mask = (
                    torch.rand(B, 1, device=device) < cfg_p
                ).float()  # 1 = drop -> unconditional
                ctx = ctx * (1.0 - drop_mask)  # zeros become the "null" condition

        noise_pred = denoiser_net(states_noisy, timesteps, ctx)
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = input
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(input, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type {pred_type}")

        loss_diff = self.compute_loss(noise_pred, target)
        return loss_diff


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
        self.constraint_analyzer = ConstraintAnalyzer(config["constraint_analyzer"])
        self.constraints_attention = ConstraintsAttention(
            config["constraints_attention"]
        )
        self.diffusion_model = DiffusionTrajectory(config["diffusion_model"])
        # self.structure_head = StructureHead(
        #     d_ctx=config["diffusion_model"]["cond_dim"],
        #     d_state=config["d_state"],
        # )
        # self.K_cycles = config["K_cycles"]
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
    def conditional_sample(
        self,
        states: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        environment: Dict[str, torch.Tensor],
        device: torch.device,
    ):
        states_gt_nm = dict_to_tensor_concat(states, dim=2, device=device)
        action_gt_nm = dict_to_tensor_concat(actions, dim=2, device=device)
        env_nm = dict_to_tensor_concat(environment, dim=-1, device=device)
        # Trajectory: States and Actions: (s_0, a_0, s_1, s_2, ..., s_T, a_T)
        trajectory = torch.concatenate([action_gt_nm, states_gt_nm], dim=-1).to(
            device, dtype=torch.float32
        )
        current_traj_0 = trajectory[:, 0, :]
        cond = torch.cat([current_traj_0, env_nm], dim=-1).to(
            device, dtype=torch.float32
        )
        cond_ctx = self.constraints_attention(trajectory, cond)
        noise = torch.randn_like(trajectory).to(device, dtype=torch.float32)
        trajectory_pred = self.diffusion_model.sample(
            noise=noise, ctx=cond_ctx, traj_start=current_traj_0.unsqueeze(1)
        )
        return {"traj_pred": trajectory_pred, "trj_gt": trajectory}

    def forward(
        self,
        states: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        environment: Dict[str, torch.Tensor],
        device: torch.device,
    ):
        """
        states: Dict of tensors with keys like 'position', 'velocity', each of shape [B, N, D]
        actions: Dict of tensors with keys like 'velocity', 'acceleration each of shape [B, ...]
        environment: Dict of tensors with keys like 'goal', 'obstacles', 'camera', 'lidar', each of shape [B, ...]
        returns: Dict with updated states after K diffusion cycles
        """
        states_gt_nm = dict_to_tensor_concat(states, dim=2, device=device)
        action_gt_nm = dict_to_tensor_concat(actions, dim=2, device=device)
        env_nm = dict_to_tensor_concat(environment, dim=-1, device=device)
        # Trajectory: States and Actions: (s_0, a_0, s_1, s_2, ..., s_T, a_T)
        trajectory = torch.concatenate([action_gt_nm, states_gt_nm], dim=-1).to(
            device, dtype=torch.float32
        )
        current_traj_0 = trajectory[:, 0, :]
        cond = torch.cat([current_traj_0, env_nm], dim=-1).to(
            device, dtype=torch.float32
        )
        cond_ctx = self.constraints_attention(trajectory, cond)
        loss_diff = self.diffusion_model(trajectory, cond_ctx, device=device)
        loss_total = loss_diff  # + loss_cycle + loss_constraints
        return {
            "loss": loss_total,
            "loss_diff": loss_diff,
        }  # , "loss_cycle": loss_cycle, "loss_constraints": loss_constraints


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
