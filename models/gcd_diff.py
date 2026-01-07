from dataclasses import dataclass
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import repeat, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


from common.utils import dict_to_tensor_concat
from common.get_class import get_class_dict
from models.utils.diffusion import apply_inpainting
from models.utils.normalizer import DictNormalizer

from models.utils.constraints import ObstacleConstraint


@dataclass
class ConstraintOutputs:
    residual: (
        torch.Tensor
    )  # [B,N,3]  ∂E/∂position (in *normalized* space unless return_denorm_grads=True)
    # grad_energy: torch.Tensor  # [B,N,3]  ∂E/∂velocity
    energy: torch.Tensor  # [B]      total scalar energy


class ConstraintAnalyzer(nn.Module):
    """
    Penalize speeds above v_max and/or below v_min.
    Uses denormalized velocities for interpretable limits, then maps grads back.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.cfg = cfg
        self.constraints = {}
        for key in cfg.keys():
            if key == "constraint_obstacles":
                config_const = cfg[key]
                if "target" not in config_const.keys():
                    config_const.update(
                        {"target": "models.utils.constraints.ObstacleConstraintGrad"}
                    )
                constraint = get_class_dict(config_const)
                self.constraints["states"] = constraint
    def set_normalizer(
        self,
        state_normalizer: DictNormalizer,
        obs_normalizer: DictNormalizer,
        environment_normalizer: DictNormalizer,
    ):
        self.state_normalizer = state_normalizer
        self.obs_normalizer = obs_normalizer
        self.environment_normalizer = environment_normalizer

    def forward(
        self,
        states: Dict[str, torch.Tensor],
        actions: Dict[str, torch.Tensor],
        environment: Dict[str, torch.Tensor],
        extras: Optional[Dict[str, torch.Tensor]] = None,
        trajectory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> ConstraintOutputs:
        constraints = self.constraints
        energy = []
        residuals = []
        states = self.state_normalizer.unnormalize(states)
        for key in self.constraints.keys():
            if key == "states":
                # TODO: constraint should have a states names that are needed
                # value = torch.cat([states[k] for k in states_names_dict.keys()], dim=-1)
                value = torch.cat([states["desired_pos"] , states["current_pos"]], dim=-1)  # .detach().clone()
                dict_out = constraints[key](value)
                energy.append(dict_out["energy_t"])
                residuals.append(dict_out["residual_t"])
        energy_all = torch.stack(energy, dim=0).sum(dim=0)  # [B]
        residuals_all = torch.stack(residuals, dim=0).sum(dim=0)  # [B,H] (etc.)
        return ConstraintOutputs(
            residual=residuals_all,
            # grad_vel=grad_v_nm,
            energy=energy_all,
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
        self.guidance_scale = config["guidance"].get("guidance_scale", 0.0)
        self.cfg_drop_prob = config["guidance"].get("cfg_drop_prob", 0.0)
        self.cfg = True if "cfg_drop_prob" in config["guidance"] else False
        self.guidance_fuc = get_class_dict(self.config["guidance"])

    def set_scheduler(self, scheduler: Optional[DDIMScheduler]):
        self.scheduler = scheduler

    @torch.no_grad()
    def sample(
            self,
            noise: torch.Tensor,
            ctx: torch.Tensor,
            traj_start: torch.Tensor,
            **kwargs,
    ) -> torch.Tensor:
        device = noise.device
        scheduler = self.scheduler
        # TODO: Use EMA model for denoiser network
        model = self.denoiser_net
        # guidance_scale = self.guidance_scale
        guidance_fuc = self.guidance_fuc
        x = noise
        mask = torch.zeros_like(noise, dtype=torch.bool, device=device)
        # TODO: Not Hardcoding -> use action dim
        mask[:, 0:1, 2:] = torch.ones_like(
            traj_start[:, :1, 2:], dtype=torch.bool, device=device
        )
        scheduler.set_timesteps(self.inference_steps, device=device)
        x = apply_inpainting(x, mask, traj_start, noise=False)
        for t in scheduler.timesteps:
            x = guidance_fuc(x, t, ctx, scheduler, model, **kwargs)
            x = apply_inpainting(x, mask, traj_start, noise=False)
        return x


    def compute_loss(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target)
        return loss

    def forward(
        self,
        input: torch.Tensor,
        ctx: torch.Tensor,
        start_trj: torch.Tensor,
        device: torch.device,
        **kwargs,
    ) -> torch.Tensor:
        scheduler = self.scheduler
        denoiser_net = self.denoiser_net
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (input.size(0),),
            device=device,
        )
        start_traj_cond = start_trj.unsqueeze(1)
        noise = torch.randn_like(input)
        mask = torch.zeros_like(input, dtype=torch.bool, device=device)
        mask[:, 0:1, 2:] = torch.ones_like(
            start_traj_cond[:, :1, 2:], dtype=torch.bool, device=device
        )
        # TODO: Apply Inpainting (start) should be zero at t=0
        input = apply_inpainting(input, mask, x_known=start_traj_cond, noise=True)
        states_noisy = scheduler.add_noise(input, noise, timesteps)
        # TODO: Apply Inpainting (start) should be known Trajectory
        states_noisy = apply_inpainting(
            states_noisy, mask, x_known=start_traj_cond, noise=False
        )

        # if "constraints_out" in kwargs.keys():
        #     constraint: ConstraintOutputs = kwargs["constraints_out"]
        #     energy = ConstraintOutputs.energy
        #     (grad_v_nm,) = torch.autograd.grad(
        #         energy,
        #         states_noisy,
        #         retain_graph=False,
        #         create_graph=False,
        #     )
        if self.cfg:
            cfg_p = self.cfg_drop_prob
            drop_mask = torch.zeros_like(ctx).float()
            if cfg_p > 0.0:
                B = ctx.size(0)
                drop_mask = (
                    torch.rand(B, 1, device=device) < cfg_p
                ).float()  # 1 = drop -> unconditional
            elif cfg_p == 0.0:
                drop_mask = torch.ones_like(ctx).float()
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
        analyser = self.constraint_analyzer
        # Trajectory: States and Actions: (s_0, a_0, s_1, s_2, ..., s_T, a_T)
        trajectory = torch.concatenate([action_gt_nm, states_gt_nm], dim=-1).to(
            device, dtype=torch.float32
        )

        current_state_0 = states_gt_nm[:, 0, :].to(device, dtype=torch.float32)
        # cond = torch.cat([current_traj_0, env_nm], dim=-1).to(
        #     device, dtype=torch.float32
        # )
        current_traj_0 = torch.zeros_like(trajectory[:, 0, :])
        # TODO: use action dim not hardcoded
        cond = current_state_0
        current_traj_0[:, 2:] = current_state_0
        cond_ctx = self.constraints_attention(trajectory, cond)
        noise = torch.randn_like(trajectory).to(device, dtype=torch.float32)
        trajectory_pred = self.diffusion_model.sample(
            noise=noise,
            ctx=cond_ctx,
            traj_start=current_traj_0.unsqueeze(1),
            analyzer=analyser,
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

        current_state_0 = states_gt_nm[:, 0, :].to(device, dtype=torch.float32)
        # cond = torch.cat([current_traj_0, env_nm], dim=-1).to(
        #     device, dtype=torch.float32
        # )
        current_traj_0 = torch.zeros_like(trajectory[:, 0, :])
        # TODO: use action dim not hardcoded
        current_traj_0[:, 2:] = current_state_0
        cond = current_state_0
        cond_ctx = self.constraints_attention(trajectory, cond)
        loss_diff = self.diffusion_model(
            trajectory, cond_ctx, current_traj_0, device=device
        )
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
