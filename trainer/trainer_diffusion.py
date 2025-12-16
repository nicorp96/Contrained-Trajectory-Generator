import copy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import os
import torch
from torch.utils.data import DataLoader

from common.get_class import get_class_dict
from dataset.utils import get_ds_from_cfg
from trainer.base_trainer import BaseTrainer
from models.utils.ema import EMA
from models.utils.diffusion import project_sincos_channels
from common.helper import (
    pos_from_deltas,
    dt_from_interval_velocity,
)
from common.helper_reconstructor import TrajectoryReconstructor
from dataset.datatset_sequence_sampler import TrajectorySegmentDataset


class DiffusionTrainer(BaseTrainer):
    def __init__(self, config):
        super(DiffusionTrainer, self).__init__(config)
        self.num_inference_steps = self.config["model"].get("num_inference_steps", 100)

    def setup_model(self):
        self.model = get_class_dict(self.config["model"])
        assert torch.cuda.is_available()
        if self.from_checkpoint:
            self.model.load_state_dict(self.checkpoint["state_dicts"]["model_state"])
        self.model.to(self.config["training"]["device"])

        if self.config["training"]["use_ema"]:
            self.ema_model = copy.deepcopy(self.model)
            self.ema = EMA(self.ema_model, self.config["ema"])

    def setup_scheduler(self):
        super().setup_scheduler()
        scheduler_cfg = self.config["model"]["noise_scheduler"]
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=scheduler_cfg["num_train_timesteps"],
            beta_start=scheduler_cfg["beta_start"],
            beta_end=scheduler_cfg["beta_end"],
            beta_schedule=scheduler_cfg["beta_schedule"],
            clip_sample=scheduler_cfg["clip_sample"],
            set_alpha_to_one=scheduler_cfg["set_alpha_to_one"],
            steps_offset=scheduler_cfg["steps_offset"],
            prediction_type=scheduler_cfg["prediction_type"],
        )

    def setup_normalizer(self):
        if "path_stats" in self.config["dataset"]:
            state_normalizer, goal_normalizer = (
                self.train_dataset.dataset.get_stats_from_file()
            )
        else:
            state_normalizer, goal_normalizer = (
                self.train_loader.dataset.get_normalizer_fit()
            )
        self.normalizer_state = state_normalizer
        self.normalizer_goal = goal_normalizer
        self.normalizers.append(state_normalizer)
        self.normalizers.append(goal_normalizer)

    def compute_loss(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target)
        return loss

    def conditional_sample(self, trj, cond, use_ema=True, generator=None, **kwargs):
        raise NotImplementedError()

    def predict_trajectory(self, trj, cond, use_ema=False, **kwargs):
        raise NotImplementedError()

    def plotting(self, output, target, step, epoch, name="val", unorm=True):
        config = self.config
        output_dict = self.split_state_tensor(output, config["dataset"]["state_shapes"])
        target_unnormalized = target
        if unorm:
            output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        for key in output_dict.keys():
            output_np = output_unnormalized[key].detach().cpu().numpy()
            target_np = target_unnormalized[key].detach().cpu().numpy()  # [:, 1:, :]
            output_np[:, 0, :] = target_np[:, 0, :]
            if output_np.shape[2] == 3:
                batch_id = 0
                image_name = os.path.join(
                    f"{name}_{key}", f"ep{epoch}_batch_{batch_id}"
                )
                out = output_np[batch_id, :, :]
                tgt = target_np[batch_id, :, :]
                img_np = self.plot_3d_trajectory(
                    out,
                    tgt,
                    title=image_name,
                )
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )
                if key == "position" or key == "delta_pos" or key == "vel_ned":
                    img_x = self.plot_1d_seq(
                        output_np[0, :, 0], target_np[0, :, 0], title=image_name + "_x"
                    )
                    self.writer.add_image(
                        img_tensor=img_x,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_x", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )
                    img_y = self.plot_1d_seq(
                        output_np[0, :, 1], target_np[0, :, 1], title=image_name + "_y"
                    )
                    self.writer.add_image(
                        img_tensor=img_y,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_y", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )
                    img_z = self.plot_1d_seq(
                        output_np[0, :, 2], target_np[0, :, 2], title=image_name + "_z"
                    )
                    self.writer.add_image(
                        img_tensor=img_z,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_z", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )

            else:
                image_name = os.path.join(f"{name}_{key}", f"epoch_{epoch}")
                img_np = self.plot_1d_seq(
                    output_np[0, :, :], target_np[0, :, :], title=image_name
                )
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )


class DiffusionAnglesTrainer(DiffusionTrainer):
    """
    Description:
        ** Apply sin and cos for each angle as preprocessing step
        ** Input:
            1. Absolute Position x, y, z (NED)
            2. Angles: (for each angle sin, cos are applied)
        ** Condition:
            1. Delta Pos init: P_goal - P_init
            2. Sin-Cos gat angle
    """

    def __init__(self, config):
        super().__init__(config)
        self.angles_pairs = []
        idx = 0
        state_shapes = self.config["dataset"]["state_shapes"]
        for key, value in state_shapes.items():
            idx += value["shape"]
            if "rad" in key or "deg" in key:
                self.angles_pairs.append(((idx - 1), (idx)))
                idx += 1

    @torch.no_grad()
    def conditional_sample(
        self,
        trj,
        cond,
        use_ema=True,
        generator=None,
        guidance_scale=None,
        **kwargs,
    ):
        model = (
            self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        )
        model.eval()
        scheduler = self.noise_scheduler

        gs = (
            float(self.config["model"].get("guidance_scale", 1.0))
            if guidance_scale is None
            else float(guidance_scale)
        )

        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )

        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )

        uncond = torch.zeros_like(cond)

        def guide_and_step(x_t, t_scalar):
            # batched uncond/cond forward
            traj_in = torch.cat([x_t, x_t], dim=0)
            cond_in = torch.cat([uncond, cond], dim=0)
            t_in = t_scalar.expand(traj_in.shape[0]).to(trj.device)

            model_out = model(traj_in, t_in, cond_in)
            out_uncond, out_cond = model_out.chunk(2, dim=0)

            pred_type = scheduler.config.prediction_type
            if gs == 1.0:
                guided = out_cond
            elif pred_type in ("epsilon", "v_prediction"):
                guided = out_uncond + gs * (out_cond - out_uncond)
            elif pred_type == "sample":
                t_idx = (
                    t_scalar.long().item()
                    if t_scalar.ndim == 0
                    else int(t_scalar[0].long().item())
                )
                a_bar = scheduler.alphas_cumprod.to(trj.device)[t_idx]
                expand = (1,) * x_t.ndim
                sqrt_ab = a_bar.sqrt().view(*expand)
                sqrt_omb = (1.0 - a_bar).sqrt().view(*expand)
                eps_uncond = (x_t - sqrt_ab * out_uncond) / sqrt_omb
                eps_cond = (x_t - sqrt_ab * out_cond) / sqrt_omb
                eps_guided = eps_uncond + gs * (eps_cond - eps_uncond)
                guided = (x_t - sqrt_omb * eps_guided) / sqrt_ab  # x0
            else:
                raise ValueError(f"Unsupported prediction_type: {pred_type}")

            x_prev = scheduler.step(
                guided, t_scalar, x_t, generator=generator, **kwargs
            ).prev_sample
            return x_prev

        for t in scheduler.timesteps:
            t_scalar = t  # scalar (HF timesteps tensor element)
            trajectory_n = guide_and_step(trajectory_n, t_scalar)
            # keep your sin/cos normalization each step
            trajectory_n = project_sincos_channels(
                trajectory_n,
                pairs=self.angles_pairs,
            )

        # recover angle channels at the very end (unchanged from your code)
        trj_out = trajectory_n[:, :, : (self.angles_pairs[0][0])]
        angles = []
        for sin_ch, cos_ch in self.angles_pairs:
            angle = torch.atan2(trajectory_n[:, :, sin_ch], trajectory_n[:, :, cos_ch])
            angles.append(angle)
        angles = torch.stack(angles, dim=-1)
        trj_out = torch.cat([trj_out, angles], dim=-1)
        return trajectory_n, trj_out

    def predict_trajectory(self, trj, cond, use_ema=False, **kwargs):
        trajectory_n, trj_angle = self.conditional_sample(
            trj, cond, use_ema=use_ema, **kwargs
        )
        result = {"output": trajectory_n, "target": trj, "trj_angle": trj_angle}
        return result

    def train_step(self, batch):
        device = self.device
        states_dict, goals_dict = batch
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        goal_pos = goals_norm.pop("pip_x_y_z_km").to(device)
        # trajectory = states_norm_dict.pop("position")
        cond_goal = goal_pos - trajectory[:, 0, :3]
        trajectory[:, 0, :3] = cond_goal
        cond_goal_init = trajectory[:, 0, :]  # .unsqueeze(1)
        trajectory_wo_cond = trajectory[:, 1:, :]
        # goal_nm = torch.cat([goals_norm[key] for key in goals_norm.keys()], dim=1).to(
        #     device
        # )
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)

        noise = torch.randn(trajectory_wo_cond.shape, device=device)
        batch_size = trajectory_wo_cond.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        cond = torch.cat((goal_nm, cond_goal_init), dim=1)
        # ---------- Classifier-free dropout ----------
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (
                torch.rand(B, 1, device=device) < cfg_p
            ).float()  # 1 = drop -> unconditional
            cond = cond * (1.0 - drop_mask)  # zeros become the "null" condition
        # (no change to model dims; we don't append an extra bit)
        noisy_states = self.noise_scheduler.add_noise(
            trajectory_wo_cond, noise, timesteps
        )
        trj_noised_cond = (
            noisy_states  # torch.cat((cond_goal_init, noisy_states), dim=1)
        )
        pred = self.model(trj_noised_cond, timesteps, cond)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory_wo_cond
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                trajectory_wo_cond, noise, timesteps
            )
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = self.compute_loss(pred, target)
        return {"loss": loss}

    def val_step(self, batch):
        device = self.device
        use_ema = self.ema_model is not None
        states_dict, goals_dict = batch
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()], dim=2
        ).to(device)
        goal_pos = goals_norm.pop("pip_x_y_z_km").to(device)
        # trajectory = states_norm_dict.pop("position")
        cond_goal = goal_pos - trajectory[:, 0, :3]
        trajectory[:, 0, :3] = cond_goal
        cond_goal_init = trajectory[:, 0, :]  # .unsqueeze(1)
        trajectory_wo_cond = trajectory[:, 1:, :]
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        cond = torch.cat((goal_nm, cond_goal_init), dim=1)
        out = self.predict_trajectory(trajectory_wo_cond, cond, use_ema)
        loss = self.compute_loss(out["output"], out["target"])
        out["target_dict"] = states_dict
        out.update({"loss": loss})
        return out

    def plotting(self, output, target, step, epoch, name="val"):
        config = self.config
        output_dict = self.split_state_tensor(output, config["dataset"]["state_shapes"])
        target_unnormalized = target
        output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        for key in output_dict.keys():
            output_np = output_unnormalized[key].detach().cpu().numpy()
            target_np = target_unnormalized[key].detach().cpu().numpy()  # [:, 1:, :]
            if output_np.shape[2] == 3:
                batch_id = 0
                image_name = os.path.join(
                    f"{name}_{key}", f"ep{epoch}_batch_{batch_id}"
                )
                out = output_np[batch_id, :, :]
                tgt = target_np[batch_id, :, :]
                img_np = self.plot_3d_trajectory(
                    out,
                    tgt,
                    title=image_name,
                )
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )
                if key == "position" or key == "vel_ned":
                    img_x = self.plot_1d_seq(
                        output_np[0, :, 0], target_np[0, :, 0], title=image_name + "_x"
                    )
                    self.writer.add_image(
                        img_tensor=img_x,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_x", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )
                    img_y = self.plot_1d_seq(
                        output_np[0, :, 1], target_np[0, :, 1], title=image_name + "_y"
                    )
                    self.writer.add_image(
                        img_tensor=img_y,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_y", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )
                    img_z = self.plot_1d_seq(
                        output_np[0, :, 2], target_np[0, :, 2], title=image_name + "_z"
                    )
                    self.writer.add_image(
                        img_tensor=img_z,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_z", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )

            else:
                image_name = os.path.join(f"{name}_{key}", f"epoch_{epoch}")
                img_np = self.plot_1d_seq(
                    output_np[0, :, :], target_np[0, :, :], title=image_name
                )
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )


class DiffusionTimeAnglesTrainer(DiffusionAnglesTrainer):
    def __init__(self, config):
        super().__init__(config)

    def predict_trajectory(self, trj, cond, use_ema=False, **kwargs):
        trajectory_n, trj_angle = self.conditional_sample(
            trj, cond, use_ema=use_ema, **kwargs
        )
        result = {"output": trj_angle, "target": trj, "trj_sin_cos_angle": trajectory_n}
        return result

    @staticmethod
    def make_suffix_batch(X_full, T_full, K_suf=128):
        """
        X_full: (B, K_full, D) preprocessed on normalized time s=k/(K_full-1)
        T_full: (B,) total durations in seconds
        Returns:
          X_suf: (B, K_suf, D) resampled suffix
          s_start: (B, D) start state at chosen start_idx
          goal: (B, D) last state
          T_rem: (B,) remaining duration
          start_idx: (B,) chosen start indices (int)
        """
        B, K_full, D = X_full.shape
        device = X_full.device
        start_max = 200  # K_full - 1
        # random start index (ensure at least one step remains)
        start_idx = torch.randint(low=0, high=start_max, size=(B,), device=device)

        # normalized positions for the full grid
        # we exploit uniform spacing to avoid general interp: just index in [0, K_full-1]
        s0 = (start_idx.float() / (start_max)).unsqueeze(-1)

        tau_rem = 1.0 - s0

        # start state and goal (you can also slice only pos dims as your "goal")
        s_start = X_full[torch.arange(B, device=device), start_idx]  # (B, D)
        # target grid of normalized positions in the *full* sequence coordinates
        # idx_suf ∈ [s0, 1], shape (B, K_suf)
        base = torch.linspace(0, 1, steps=K_suf, device=device)  # (K_suf,)
        idx_suf = (s0[:, None] + (1 - s0)[:, None] * base[None, :]).squeeze(
            1
        )  # (B, K_suf)

        # convert normalized positions to float indices in [0, K_full-1]
        fidx = idx_suf * (start_max)
        i0 = torch.clamp(fidx.floor().long(), 0, start_max - 1)  # (B, K_suf)
        w = (fidx - i0.float()).unsqueeze(-1)  # (B, K_suf, 1)

        ar = torch.arange(B, device=device).unsqueeze(-1)  # (B,1)

        y0 = X_full[ar, i0]  # (B, K_suf, D)
        y1 = X_full[ar, i0 + 1]  # (B, K_suf, D)
        X_suf = y0 * (1 - w) + y1 * w  # (B, K_suf, D)

        # (optional) hard-clamp exact endpoints to avoid tiny numeric drift
        # X_suf[:, 0] = s_start
        return X_suf, s_start, tau_rem, start_idx

    def train_step(self, batch):
        device = self.device
        states_dict, goals_dict, T_d = batch
        T_d = T_d.to(device)
        states_norm_dict = self.normalizer_state(states_dict)
        T_nm = states_norm_dict.pop("duration_T")
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        trajectory_k_n, s_start, tau_rem, start_idx = self.make_suffix_batch(
            trajectory, T_d
        )
        T_rem = T_d * tau_rem
        T_rem_nm = self.normalizer_state.normalizers["duration_T"](T_rem)
        goal_pos = goals_norm.pop("pip_x_y_z_km").to(device)
        cond_goal_init = s_start
        cond_goal_init[:, :3] = goal_pos - s_start[:, :3]

        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)

        noise = torch.randn(trajectory_k_n.shape, device=device)
        batch_size = trajectory_k_n.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        cond = torch.cat((goal_nm, T_rem_nm, cond_goal_init), dim=1)
        # ---------- Classifier-free dropout ----------
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (
                torch.rand(B, 1, device=device) < cfg_p
            ).float()  # 1 = drop -> unconditional
            cond = cond * (1.0 - drop_mask)  # zeros become the "null" condition
        # (no change to model dims; we don't append an extra bit)
        noisy_states = self.noise_scheduler.add_noise(trajectory_k_n, noise, timesteps)
        trj_noised_cond = (
            noisy_states  # torch.cat((cond_goal_init, noisy_states), dim=1)
        )
        pred = self.model(trj_noised_cond, timesteps, cond)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory_k_n
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(trajectory_k_n, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = self.compute_loss(pred, target)
        return {"loss": loss}

    def val_step(self, batch):
        device = self.device
        use_ema = self.ema_model is not None
        states_dict, goals_dict, T_d = batch
        T_d = T_d.to(device)
        states_norm_dict = self.normalizer_state(states_dict)
        T_nm = states_norm_dict.pop("duration_T")
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()], dim=2
        ).to(device)
        trajectory_k_n, s_start, tau_rem, start_idx = self.make_suffix_batch(
            trajectory, T_d
        )
        T_rem = T_d * tau_rem
        T_rem_nm = self.normalizer_state.normalizers["duration_T"](T_rem)
        goal_pos = goals_norm.pop("pip_x_y_z_km").to(device)
        cond_goal_init = s_start
        cond_goal_init[:, :3] = goal_pos - s_start[:, :3]
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        cond = torch.cat((goal_nm, T_rem_nm, cond_goal_init), dim=1)
        out = self.predict_trajectory(trajectory_k_n, cond, use_ema)
        loss = self.compute_loss(out["trj_sin_cos_angle"], out["target"])
        config = self.config["dataset"]["state_shapes"]
        target_dict = self.split_state_tensor(trajectory_k_n, config, ["duration_T"])
        target_unnormalized_dict = self.normalizer_state.unnormalize(target_dict)
        target_unnormalized = torch.cat(
            [target_unnormalized_dict[key] for key in target_unnormalized_dict.keys()],
            dim=2,
        ).to(device)
        out["target_dict"] = target_unnormalized_dict
        out["target"] = target_unnormalized
        out.update({"loss": loss})
        return out

    def plotting(self, output, target, step, epoch, name="val"):
        config = self.config["dataset"]["state_shapes"]
        output_dict = self.split_state_tensor(output, config, ["duration_T"])
        output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        # target_unnormalized = self.normalizer_state.unnormalize(target)
        for key in output_dict.keys():
            output_np = output_unnormalized[key].detach().cpu().numpy()
            target_np = target[key].detach().cpu().numpy()  # [:, 1:, :]
            if output_np.shape[2] == 3:
                batch_id = 0
                image_name = os.path.join(
                    f"{name}_{key}", f"ep{epoch}_batch_{batch_id}"
                )
                out = output_np[batch_id, :, :]
                tgt = target_np[batch_id, :, :]
                img_np = self.plot_3d_trajectory(
                    out,
                    tgt,
                    title=image_name,
                )
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )
                if key == "position":
                    img_x = self.plot_1d_seq(
                        output_np[0, :, 0], target_np[0, :, 0], title=image_name + "_x"
                    )
                    self.writer.add_image(
                        img_tensor=img_x,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_x", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )
                    img_y = self.plot_1d_seq(
                        output_np[0, :, 1], target_np[0, :, 1], title=image_name + "_y"
                    )
                    self.writer.add_image(
                        img_tensor=img_y,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_y", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )
                    img_z = self.plot_1d_seq(
                        output_np[0, :, 2], target_np[0, :, 2], title=image_name + "_z"
                    )
                    self.writer.add_image(
                        img_tensor=img_z,
                        global_step=step,
                        dataformats="HWC",
                        tag=os.path.join(
                            f"{name}_{key}_z", f"ep{epoch}_batch_{batch_id}"
                        ),
                    )

            else:
                image_name = os.path.join(f"{name}_{key}", f"epoch_{epoch}")
                img_np = self.plot_1d_seq(
                    output_np[0, :, :], target_np[0, :, :], title=image_name
                )
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )


class DiffusionTimeGridAnglesTrainer(DiffusionAnglesTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.w_pos = 1.0
        self.w_vel = 1.0

    @torch.no_grad()
    def conditional_sample(
        self,
        trj,
        cond,
        use_ema=True,
        generator=None,
        guidance_scale=None,
        time=None,
        **kwargs,
    ):
        state_shapes = self.config["dataset"]["state_shapes"]
        model = (
            self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        )
        model.eval()
        scheduler = self.noise_scheduler

        gs = (
            float(self.config["model"].get("guidance_scale", 1.0))
            if guidance_scale is None
            else float(guidance_scale)
        )

        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )
        # trajectory_n[:, 0, :] = trj[:, 0, :]
        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )

        uncond = torch.zeros_like(cond)

        def guide_and_step(x_t, t_scalar, time):
            # batched uncond/cond forward
            traj_in = torch.cat([x_t, x_t], dim=0)
            cond_in = torch.cat([uncond, cond], dim=0)
            t_in = t_scalar.expand(traj_in.shape[0]).to(trj.device)
            time_in = time
            if time is not None:
                time_in = torch.cat([time, time], dim=0)
            model_out = model(traj_in, t_in, cond_in, time_in)
            out_uncond, out_cond = model_out.chunk(2, dim=0)
            # t_in = t_scalar.expand(x_t.shape[0]).to(x_t.device)
            # time_in = time
            # run separately to avoid doubling batch in memory
            # out_uncond = model(x_t, t_in, uncond, time_in)
            # out_cond = model(x_t, t_in, cond, time_in)
            pred_type = scheduler.config.prediction_type
            if gs == 1.0:
                guided = out_cond
            elif pred_type in ("epsilon", "v_prediction"):
                guided = out_uncond + gs * (out_cond - out_uncond)
            x_prev = scheduler.step(
                guided, t_scalar, x_t, generator=generator, **kwargs
            ).prev_sample
            return x_prev

        for t in scheduler.timesteps:
            t_scalar = t  # scalar (HF timesteps tensor element)
            trajectory_n = guide_and_step(trajectory_n, t_scalar, time)
            # keep your sin/cos normalization each step
            if "rad" in state_shapes:
                trajectory_n = project_sincos_channels(
                    trajectory_n,
                    pairs=self.angles_pairs,
                )

        # recover angle channels at the very end (unchanged from your code)
        if "rad" in state_shapes:
            trj_out = trajectory_n[:, :, : (self.angles_pairs[0][0])]
            angles = []
            for sin_ch, cos_ch in self.angles_pairs:
                angle = torch.atan2(
                    trajectory_n[:, :, sin_ch], trajectory_n[:, :, cos_ch]
                )
                angles.append(angle)
            angles = torch.stack(angles, dim=-1)
            trj_out = torch.cat([trj_out, angles], dim=-1)
            return trajectory_n, trj_out
        return trajectory_n

    def predict_trajectory(self, trj, cond, time=None, use_ema=False, **kwargs):
        output_cd = self.conditional_sample(
            trj, cond, time=time, use_ema=use_ema, **kwargs
        )

        if "rad" in self.config["dataset"]["state_shapes"]:
            trajectory_n, trj_angle = output_cd
            return {"output": trajectory_n, "target": trj, "trj_angle": trj_angle}

        return {"output": output_cd, "target": trj}

    def train_step(self, batch):
        device = self.device
        states_dict, goals_dict, extra_shapes = batch
        start_pos = states_dict["position"][:, 0, :3]
        goal_pos = goals_dict["pip_x_y_z_km"]
        states_norm_dict = self.normalizer_state(
            states_dict, start=start_pos, goal=goal_pos
        )
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        goal_pos = goals_norm.pop("pip_x_y_z_km").to(device)
        cond_init = trajectory[:, 0, 3:]
        cond_goal = trajectory[:, -1, :3]
        trajectory_wo_cond = trajectory

        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)

        noise = torch.randn(trajectory_wo_cond.shape, device=device)
        batch_size = trajectory_wo_cond.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        cond = torch.cat((goal_nm, cond_goal, cond_init), dim=1)
        # ---------- Classifier-free dropout ----------
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (
                torch.rand(B, 1, device=device) < cfg_p
            ).float()  # 1 = drop -> unconditional
            cond = cond * (1.0 - drop_mask)  # zeros become the "null" condition
        noisy_states = self.noise_scheduler.add_noise(
            trajectory_wo_cond, noise, timesteps
        )
        trj_noised_cond = noisy_states
        pred = self.model(trj_noised_cond, timesteps, cond)  # , T_d)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory_wo_cond
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                trajectory_wo_cond, noise, timesteps
            )
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = self.compute_loss(pred, target)
        return {"loss": loss}

    def val_step(self, batch):
        device = self.device
        use_ema = self.ema_model is not None
        states_dict, goals_dict, extra_shapes = batch
        # position = extra_shapes["position"].to(device)
        # T_d = extra_shapes["duration_T"].squeeze(-1).to(device)

        start_pos = states_dict["position"][:, 0, :3]
        goal_pos = goals_dict["pip_x_y_z_km"]
        states_norm_dict = self.normalizer_state(
            states_dict, start=start_pos, goal=goal_pos
        )
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        goal_pos = goals_norm.pop("pip_x_y_z_km").to(device)
        cond_init = trajectory[:, 0, 3:]
        cond_goal = trajectory[:, -1, :3]
        trajectory_wo_cond = trajectory
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        cond = torch.cat((goal_nm, cond_goal, cond_init), dim=1)
        out = self.predict_trajectory(trajectory_wo_cond, cond, use_ema=use_ema)
        loss = self.compute_loss(out["output"], out["target"])

        out["target_dict"] = states_dict
        out.update({"loss": loss})
        return out

    def extra_evaluation(
        self,
        batch,
        preds,
        step,
        epoch,
    ):
        device = self.device
        states_dict, goals_dict, extra_shapes = batch
        trajectory_reconstructor = TrajectoryReconstructor()
        pos_0 = extra_shapes["position"][:, :1, :].to(device)
        t_d = None
        pred_dict = self.split_state_tensor(
            preds, self.config["dataset"]["state_shapes"]
        )
        pred_unnormalized = self.normalizer_state.unnormalize(pred_dict)
        pred_unnormalized = torch.cat(
            [pred_unnormalized[key] for key in pred_unnormalized.keys()],
            dim=2,
        ).to(self.device)

        position = pred_unnormalized[:, :, :3]
        vel_ned = pred_unnormalized[:, :, 3:]
        if "time" in extra_shapes.keys():
            t_d = extra_shapes["time"].squeeze(-1).to(device)
            T = t_d[:, -1] - t_d[:, 0]
        else:
            t_d, dt_rec = trajectory_reconstructor.reconstruct_time(position, vel_ned)

        if "delta_pos" in states_dict.keys():
            dx = pred_unnormalized[:, :, :3]
            v = pred_unnormalized[:, :, 3:]
            dt_from_vel_pos = dt_from_interval_velocity(deltas=dx, vel_iv=v)
            abs_pos_pred = pos_from_deltas(dx, pos_0)
            abs_pos_pred = torch.cat([pos_0, abs_pos_pred], dim=1)
            t_pred = torch.cumsum(dt_from_vel_pos, dim=1)
            t_pred = torch.cat([t_d[:, :1], t_pred], dim=1)
            T_pred = t_pred[:, -1] - t_pred[:, 0]
            mse_T = torch.mean((T_pred - T) ** 2)
            self.writer.add_scalar("val/mse_T", mse_T, epoch)
            output_np = abs_pos_pred.detach().cpu().numpy()
            target_np = extra_shapes["position"].detach().cpu().numpy()
            name = "val"
            key = "position"
            batch_id = 0
            image_name = os.path.join(f"{name}_{key}", f"ep{epoch}_batch_{batch_id}")
            # TODO: do plotting in another function
            out = output_np[batch_id, :, :]
            tgt = target_np[batch_id, :, :]
            img_np = self.plot_3d_trajectory(
                out,
                tgt,
                title=image_name,
            )
            self.writer.add_image(
                tag=image_name,
                global_step=step,
                img_tensor=img_np,
                dataformats="HWC",
            )
            img_x = self.plot_1d_seq(
                output_np[0, :, 0], target_np[0, :, 0], title=image_name + "_x"
            )
            self.writer.add_image(
                img_tensor=img_x,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_x", f"ep{epoch}_batch_{batch_id}"),
            )
            img_y = self.plot_1d_seq(
                output_np[0, :, 1], target_np[0, :, 1], title=image_name + "_y"
            )
            self.writer.add_image(
                img_tensor=img_y,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_y", f"ep{epoch}_batch_{batch_id}"),
            )
            img_z = self.plot_1d_seq(
                output_np[0, :, 2], target_np[0, :, 2], title=image_name + "_z"
            )
            self.writer.add_image(
                img_tensor=img_z,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_z", f"ep{epoch}_batch_{batch_id}"),
            )

        if "acc_ned" in extra_shapes.keys():
            acc_true = extra_shapes["acc_ned"].squeeze(-1).to(device)
            acc_ned = trajectory_reconstructor.accel_from_pos(position, t_d)
            batch_id = 0
            image_name = os.path.join(f"val_acc_ned", f"ep{epoch}_batch_{batch_id}")
            out = acc_ned[batch_id, :, :].detach().cpu().numpy()
            tgt = acc_true[batch_id, :, :].detach().cpu().numpy()
            img_np = self.plot_3d_trajectory(
                out,
                tgt,
                title=image_name,
            )
            self.writer.add_image(
                tag=image_name,
                global_step=step,
                img_tensor=img_np,
                dataformats="HWC",
            )

    def compute_loss(self, output, target):
        # pos_idx, vel_idx = group_slices[0][1], group_slices[1][1]
        lp = torch.mean((output[:, :, :3] - target[:, :, :3]) ** 2)
        lv = torch.mean((output[:, :, 3:] - target[:, :, 3:]) ** 2)
        return self.w_pos * lp + self.w_vel * lv


class DiffusionSegTrainer(DiffusionTrainer):
    def __init__(self, config):
        super().__init__(config)
        # optional: let context influence the condition vector
        self.use_context_in_cond = bool(
            self.config["model"].get("use_context_in_cond", False)
        )

    def setup_dataloader(self):
        ds_cfg = self.config["dataset"]
        base_train, base_val = get_ds_from_cfg(ds_cfg)

        H = int(ds_cfg.get("segment_horizon", 16))
        K = int(ds_cfg.get("segment_context", 0))
        stride = ds_cfg.get("segment_stride", 4)  # H => non-overlap

        self.train_dataset = TrajectorySegmentDataset(
            base_train, horizon=H, context=K, stride=stride
        )
        self.validation_dataset = None
        if ds_cfg.get("val", False) and base_val is not None:
            self.validation_dataset = TrajectorySegmentDataset(
                base_val, horizon=H, context=K, stride=stride
            )

        bs = self.config["training"]["batch_size"]
        self.train_loader = DataLoader(self.train_dataset, batch_size=bs, shuffle=True)
        self.val_loader = (
            DataLoader(self.validation_dataset, batch_size=bs, shuffle=False)
            if self.validation_dataset is not None
            else None
        )

    def setup_normalizer(self):
        if "path_stats" in self.config["dataset"]:
            state_normalizer, goal_normalizer = (
                self.train_dataset.base.dataset.get_stats_from_file()
            )
        else:
            state_normalizer, goal_normalizer = (
                self.train_loader.base.dataset.get_normalizer_fit()
            )
        self.normalizer_state = state_normalizer
        self.normalizer_goal = goal_normalizer
        self.normalizers.append(state_normalizer)
        self.normalizers.append(goal_normalizer)

    @staticmethod
    def _concat_state_dict(d):
        # concatenate along feature dim (B, K, D_i) -> (B, K, sum D_i)
        return torch.cat([d[k] for k in d.keys()], dim=2)

    @staticmethod
    def _first_step(d):
        # returns the first time step of a dict of sequences (B, K, D) -> (B, D)
        out = []
        for k, v in d.items():
            if v.ndim >= 3:
                out.append(v[:, 0, :])
            else:
                # static features broadcast as-is
                out.append(v if v.ndim == 2 else v.unsqueeze(-1))
        return torch.cat(out, dim=1)

    @staticmethod
    def _goal_flat(goals_norm):
        # cat all goal tensors; if time-varying, take the first step
        parts = []
        for v in goals_norm.values():
            if v.ndim > 2:
                parts.append(v[:, 0, :])
            elif v.ndim == 2:
                parts.append(v)
            else:
                parts.append(v.unsqueeze(-1))
        return torch.cat(parts, dim=1)

    @staticmethod
    def _extract_total_T(states_dict):
        """
        Expect states_dict['duration_T'] to be:
          - (B, K, 1) cumulative time or
          - (B, 1) total
        Returns (B,)
        """
        T = states_dict["duration_T"]
        if T.ndim == 3:  # (B,K,1)
            return T[:, -1, 0]
        elif T.ndim == 2:  # (B,1) or (B,K)
            if T.size(1) == 1:  # (B,1)
                return T[:, 0]
            else:  # (B,K) -> last
                return T[:, -1]
        else:  # (B,)
            return T

    @staticmethod
    def _time_at_index(states_dict, idx=0):
        """Return time scalar at given step idx from states_dict['duration_T']."""
        T = states_dict["duration_T"]
        if T.ndim == 3:  # (B,K,1)
            return T[:, idx, 0]
        elif T.ndim == 2:  # (B,K) or (B,1)
            return T[:, idx]
        else:  # (B,)
            return T

    def _build_condition(self, goal_nm, T_rem_nm, s_start, goal_pos_key="pip_x_y_z_km"):
        cond_goal_init = s_start.clone()
        cond_goal_init[:, :3] = goal_nm.new_tensor(0)
        return torch.cat((goal_nm, T_rem_nm, cond_goal_init), dim=1)

    @torch.no_grad()
    def conditional_sample(
        self,
        trj,
        cond,
        use_ema=True,
        generator=None,
        guidance_scale=None,
        time=None,
        **kwargs,
    ):
        state_shapes = self.config["dataset"]["state_shapes"]
        model = (
            self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        )
        model.eval()
        scheduler = self.noise_scheduler

        gs = (
            float(self.config["model"].get("guidance_scale", 1.0))
            if guidance_scale is None
            else float(guidance_scale)
        )

        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )
        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )

        uncond = torch.zeros_like(cond)

        def guide_and_step(x_t, t_scalar, time):
            # batched uncond/cond forward
            traj_in = torch.cat([x_t, x_t], dim=0)
            cond_in = torch.cat([uncond, cond], dim=0)
            t_in = t_scalar.expand(traj_in.shape[0]).to(trj.device)
            time_in = time
            if time is not None:
                time_in = torch.cat([time, time], dim=0)
            model_out = model(traj_in, t_in, cond_in, time_in)
            out_uncond, out_cond = model_out.chunk(2, dim=0)
            pred_type = scheduler.config.prediction_type
            if gs == 1.0:
                guided = out_cond
            elif pred_type in ("epsilon", "v_prediction"):
                guided = out_uncond + gs * (out_cond - out_uncond)
            x_prev = scheduler.step(
                guided, t_scalar, x_t, generator=generator, **kwargs
            ).prev_sample
            return x_prev

        for t in scheduler.timesteps:
            t_scalar = t  # scalar (HF timesteps tensor element)
            trajectory_n = guide_and_step(trajectory_n, t_scalar, time)
            # keep your sin/cos normalization each step
            if "rad" in state_shapes:
                trajectory_n = project_sincos_channels(
                    trajectory_n,
                    pairs=self.angles_pairs,
                )
            # trajectory_n[:, 0, :] = trj[:, 0, :]

        # recover angle channels at the very end (unchanged from your code)
        if "rad" in state_shapes:
            trj_out = trajectory_n[:, :, : (self.angles_pairs[0][0])]
            angles = []
            for sin_ch, cos_ch in self.angles_pairs:
                angle = torch.atan2(
                    trajectory_n[:, :, sin_ch], trajectory_n[:, :, cos_ch]
                )
                angles.append(angle)
            angles = torch.stack(angles, dim=-1)
            trj_out = torch.cat([trj_out, angles], dim=-1)
            return trajectory_n, trj_out
        return trajectory_n

    def predict_trajectory(self, trj, cond, time=None, use_ema=False, **kwargs):
        output_cd = self.conditional_sample(
            trj, cond, time=time, use_ema=use_ema, **kwargs
        )

        if "rad" in self.config["dataset"]["state_shapes"]:
            trajectory_n, trj_angle = output_cd
            return {"output": trajectory_n, "target": trj, "trj_angle": trj_angle}

        return {"output": output_cd, "target": trj}

    def train_step(self, batch):
        device = self.device

        # Accept with/without context
        if isinstance(
            batch[0], tuple
        ):  # ((states_ctx, goals_ctx), (states_tgt, goals_tgt))
            (states_ctx, goals_ctx, *_), (states_dict, goals_dict, *_) = batch
            has_ctx = True
        else:
            states_dict, goals_dict, extra_full, states_full, _ = batch
            states_ctx = goals_ctx = None
            has_ctx = False

        start = extra_full["position"][:, 0, :]
        goal = goals_dict["pip_x_y_z_km"]
        # Normalize and DROP any time channels
        states_norm = self.normalizer_state(states_dict, start=start, goal=goal)
        states_full_norm = self.normalizer_state(states_full, start=start, goal=goal)
        _ = states_norm.pop("duration_T", None)  # ignore time if present
        goals_norm = self.normalizer_goal(goals_dict)

        # Target segment to denoise (B,H,D)
        trajectory_k_n = self._concat_state_dict(states_norm).to(device)
        full_trajectory_k_n = self._concat_state_dict(states_full_norm).to(device)
        # Start state (normalized)
        s_start = self._first_step(states_norm).to(device)

        # Goal conditioning (time-free):
        # - keep your delta xyz trick (goal_xyz - start_xyz) in normalized space
        goal_pos_nm = goals_norm.pop("pip_x_y_z_km").to(device)
        if goal_pos_nm.ndim > 2:
            goal_pos_nm = goal_pos_nm[:, 0, :]

        # s_start_delta = s_start.clone()
        # s_start_delta = goal_pos_nm - s_start[:, :3]
        s_start_delta = full_trajectory_k_n[:, -1, :3]

        # - append the rest of goal features (time-varying -> take first step)
        goal_nm_rest = self._goal_flat(goals_norm).to(device)

        # Final condition: [goal_features, delta_xyz_from_start]
        # cond = torch.cat((goal_nm_rest, s_start_delta), dim=1)
        cond = torch.cat((goal_nm_rest, s_start, s_start_delta), dim=1)

        # (optional) append a context summary (still time-free)
        if has_ctx and self.use_context_in_cond:
            ctx_nm = self.normalizer_state(states_ctx)
            _ = ctx_nm.pop("duration_T", None)
            ctx_last = torch.cat([v[:, -1, :] for v in ctx_nm.values()], dim=1)
            cond = torch.cat((cond, ctx_last.to(device)), dim=1)

        # Diffusion noise & loss
        noise = torch.randn_like(trajectory_k_n, device=device)
        B = trajectory_k_n.size(0)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (B,), device=device
        ).long()

        # Classifier-free guidance dropout
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            drop = (torch.rand(B, 1, device=device) < cfg_p).float()
            cond = cond * (1.0 - drop)

        noisy = self.noise_scheduler.add_noise(trajectory_k_n, noise, timesteps)
        pred = self.model(noisy, timesteps, cond)

        pt = self.noise_scheduler.config.prediction_type
        if pt == "epsilon":
            target = noise
        elif pt == "sample":
            target = trajectory_k_n
        elif pt == "v_prediction":
            target = self.noise_scheduler.get_velocity(trajectory_k_n, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pt}")

        loss = self.compute_loss(pred, target)
        return {"loss": loss}

    # ---------- VAL (time-free, mirrors train)
    def val_step(self, batch):
        device = self.device

        if isinstance(batch[0], tuple):
            (states_ctx, goals_ctx, *_), (states_dict, goals_dict, *_) = batch
            has_ctx = True
        else:
            states_dict, goals_dict, extra, states_full, _ = batch
            states_ctx = goals_ctx = None
            has_ctx = False

        start = extra["position"][:, 0, :]
        goal = goals_dict["pip_x_y_z_km"]
        # Normalize and DROP any time channels
        states_norm = self.normalizer_state(states_dict, start=start, goal=goal)
        states_full_norm = self.normalizer_state(states_full, start=start, goal=goal)
        _ = states_norm.pop("duration_T", None)
        goals_norm = self.normalizer_goal(goals_dict)

        trajectory_k_n = self._concat_state_dict(states_norm).to(device)
        full_trajectory_k_n = self._concat_state_dict(states_full_norm).to(device)
        s_start = self._first_step(states_norm).to(device)

        goal_pos_nm = goals_norm.pop("pip_x_y_z_km").to(device)
        if goal_pos_nm.ndim > 2:
            goal_pos_nm = goal_pos_nm[:, 0, :]
        # s_start_delta = s_start.clone()
        # s_start_delta = goal_pos_nm - s_start[:, :3]
        s_start_delta = full_trajectory_k_n[:, -1, :3]
        # - append the rest of goal features (time-varying -> take first step)
        goal_nm_rest = self._goal_flat(goals_norm).to(device)
        # Final condition: [goal_features, delta_xyz_from_start]
        # cond = torch.cat((goal_nm_rest, s_start_delta), dim=1)
        cond = torch.cat((goal_nm_rest, s_start, s_start_delta), dim=1)

        if has_ctx and self.use_context_in_cond:
            ctx_nm = self.normalizer_state(states_ctx)
            _ = ctx_nm.pop("duration_T", None)
            ctx_last = torch.cat([v[:, -1, :] for v in ctx_nm.values()], dim=1)
            cond = torch.cat((cond, ctx_last.to(device)), dim=1)

        use_ema = self.ema_model is not None
        out = self.predict_trajectory(trajectory_k_n, cond, use_ema=use_ema)
        loss = self.compute_loss(out["output"], out["target"])
        out["target_dict"] = states_dict
        out.update({"loss": loss})
        return out

    @torch.no_grad()
    def rollout_trajectory(
        self,
        states_dict,
        goals_dict,
        start_pos,
        max_iters=128,
        batch_idx=1,
        use_ema=True,
    ):
        device = self.device
        config = self.config
        sequence_len = config["dataset"].get("segment_horizon", 16)
        input_dim = config["model"]["input_dim"]
        goal = goals_dict["pip_x_y_z_km"][(batch_idx - 1) : (batch_idx), :].to(device)

        states_norm = self.normalizer_state(states_dict, start=start_pos, goal=goal)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory_k_n = self._concat_state_dict(states_norm).to(device)
        s_start = self._first_step(states_norm).to(device)
        goal_pos_nm = goals_norm.pop("pip_x_y_z_km").to(device)
        goal_pos_nm = goal_pos_nm[(batch_idx - 1) : (batch_idx), :]
        if goal_pos_nm.ndim > 2:
            goal_pos_nm = goal_pos_nm[:, 0, :]
        goal_nm_rest = self._goal_flat(goals_norm).to(device)
        goal_nm_rest = goal_nm_rest[(batch_idx - 1) : (batch_idx)]
        B, L, C = trajectory_k_n.shape
        shape = (B, sequence_len, input_dim)
        chunks = []
        done = torch.zeros(B, dtype=torch.bool, device=device)
        steps = torch.zeros(B, dtype=torch.long, device=device)
        final_err = torch.empty(B, dtype=torch.float32, device=device)
        steps_inf = 2
        tol_norm = 0.02
        for it in range(max_iters):
            # build condition: [goal_feats, (goal_xyz - curr_xyz), (optional ctx_last)]
            s_start_delta = trajectory_k_n[:, -1, :3]
            # s_start_delta = goal_pos_nm - s_start[:, :3]
            # cond = torch.cat((goal_nm_rest, s_curr), dim=1)
            cond = torch.cat((goal_nm_rest, s_start, s_start_delta), dim=1)
            # sample an H-step chunk (normalized)
            trajectory_n = torch.randn(
                size=shape,
                dtype=goal_nm_rest.dtype,
                device=device,
                generator=None,
            )

            trajectory_gen = self.conditional_sample(
                trajectory_n,
                cond,
                use_ema=use_ema,
                guidance_scale=self.config["model"]["guidance_scale"],
            )

            # append first `advance` steps to the rollout
            seg_take = trajectory_gen[:, :steps_inf]  # (B, adv, Dn)
            chunks.append(seg_take)

            # update current (last appended)
            s_start = seg_take[:, -1, :]  # (B, Dn)

            # update counters and check goal (normalized space)
            err = torch.linalg.norm(goal_pos_nm - s_start[:, :3], dim=1)  # (B,)
            newly_done = (err <= tol_norm) & (~done)
            done = done | newly_done
            steps = (
                steps + (~done).long() * steps_inf
            )  # rough count; can also track per-sample precisely

            # store last error; will be overwritten until final
            final_err = err

            if done.all():
                break

        rollout_n = (
            torch.cat(chunks, dim=1) if chunks else trajectory_n[:, :0]
        ).to()  # (B, T*, Dn)
        rollout_n = self.split_state_tensor(
            rollout_n, config["dataset"]["state_shapes"]
        )
        trajectory_gen_dict_unorm = self.normalizer_state.unnormalize(rollout_n)
        # trajectory_gen_dict_unorm["position"][:, 0, :3] = start_pos[:, :]
        # trajectory_gen_dict_unorm["vel_ned"][:, 0, :3] = states_dict["vel_ned"][:, 0, :]

        success = final_err <= tol_norm
        return {
            "success": success,
            "trajectory_gen": trajectory_gen_dict_unorm,
        }

    def extra_evaluation(
        self,
        batch,
        preds,
        step,
        epoch,
    ):
        device = self.device
        batch_idx = 1  # np.random.randint(1,B)
        _, _, extra_shapes, states_dict, goals_dict = batch
        trajectory_reconstructor = TrajectoryReconstructor()

        if "delta_pos" in states_dict.keys():
            states_dict["delta_pos"] = states_dict["delta_pos"][
                (batch_idx - 1) : batch_idx, :, :
            ]
            states_dict["vel_ned"] = states_dict["vel_ned"][
                (batch_idx - 1) : batch_idx, :, :
            ]
            true_trj = states_dict["delta_pos"].shape[1] // 2
            pos_0 = states_dict["delta_pos"][:1, :1, :].to(device)
            rollout = self.rollout_trajectory(
                states_dict, goals_dict, start_pos=pos_0, max_iters=true_trj
            )
            # Unormalize delta x, y, z
            pred_unnormalized = rollout["trajectory_gen"]
            dx = pred_unnormalized[:, :, :3]
            v = pred_unnormalized[:, :, 3:]
            # dt_from_vel_pos = dt_from_interval_velocity(deltas=dx, vel_iv=v)
            abs_pos_pred = pos_from_deltas(dx, pos_0)
            abs_pos_pred = torch.cat([pos_0, abs_pos_pred], dim=1)
            # t, dt = trajectory_reconstructor.reconstruct_time(abs_pos_pred, v, t0=0.0)
            output_np = abs_pos_pred.detach().cpu().numpy()
            target_np = extra_shapes["position"].detach().cpu().numpy()
            name = "val"
            key = "position"
            batch_id = 0
            image_name = os.path.join(f"{name}_{key}", f"ep{epoch}_batch_{batch_id}")
            # TODO: do plotting in another function
            out = output_np[batch_id, :, :]
            tgt = target_np[batch_id, :, :]
            img_np = self.plot_3d_trajectory(
                out,
                tgt,
                title=image_name,
            )
            self.writer.add_image(
                tag=image_name,
                global_step=step,
                img_tensor=img_np,
                dataformats="HWC",
            )
            img_x = self.plot_1d_seq(
                output_np[0, :, 0], target_np[0, :, 0], title=image_name + "_x"
            )
            self.writer.add_image(
                img_tensor=img_x,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_x", f"ep{epoch}_batch_{batch_id}"),
            )
            img_y = self.plot_1d_seq(
                output_np[0, :, 1], target_np[0, :, 1], title=image_name + "_y"
            )
            self.writer.add_image(
                img_tensor=img_y,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_y", f"ep{epoch}_batch_{batch_id}"),
            )
            img_z = self.plot_1d_seq(
                output_np[0, :, 2], target_np[0, :, 2], title=image_name + "_z"
            )
            self.writer.add_image(
                img_tensor=img_z,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_z", f"ep{epoch}_batch_{batch_id}"),
            )
        elif "position" in states_dict.keys():
            states_dict["position"] = states_dict["position"][
                (batch_idx - 1) : batch_idx, :, :
            ]
            states_dict["vel_ned"] = states_dict["vel_ned"][
                (batch_idx - 1) : batch_idx, :, :
            ]
            true_trj = states_dict["position"].shape[1] // 2
            pos_0 = states_dict["position"][:1, 0, :].to(device)
            rollout = self.rollout_trajectory(
                states_dict, goals_dict, start_pos=pos_0, max_iters=true_trj
            )
            # Unormalize delta x, y, z
            pred_dict = rollout["trajectory_gen"]
            pred_unnormalized = self.normalizer_state.unnormalize(pred_dict)
            pred_unnormalized = torch.cat(
                [pred_unnormalized[key] for key in pred_unnormalized.keys()],
                dim=2,
            ).to(self.device)
            output_np = pred_unnormalized[:, :, :3].detach().cpu().numpy()
            target_np = states_dict["position"][:, 1:, :].detach().cpu().numpy()
            name = "rollout"
            key = "position"
            batch_id = 0
            image_name = os.path.join(f"{name}_{key}", f"ep{epoch}_batch_{batch_id}")
            # TODO: do plotting in another function
            out = output_np[batch_id, :, :]
            tgt = target_np[batch_id, :, :]
            img_np = self.plot_3d_trajectory(
                out,
                tgt,
                title=image_name,
            )
            self.writer.add_image(
                tag=image_name,
                global_step=step,
                img_tensor=img_np,
                dataformats="HWC",
            )
            img_x = self.plot_1d_seq(
                output_np[0, :, 0], target_np[0, :, 0], title=image_name + "_x"
            )
            self.writer.add_image(
                img_tensor=img_x,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_x", f"ep{epoch}_batch_{batch_id}"),
            )
            img_y = self.plot_1d_seq(
                output_np[0, :, 1], target_np[0, :, 1], title=image_name + "_y"
            )
            self.writer.add_image(
                img_tensor=img_y,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_y", f"ep{epoch}_batch_{batch_id}"),
            )
            img_z = self.plot_1d_seq(
                output_np[0, :, 2], target_np[0, :, 2], title=image_name + "_z"
            )
            self.writer.add_image(
                img_tensor=img_z,
                global_step=step,
                dataformats="HWC",
                tag=os.path.join(f"{name}_{key}_z", f"ep{epoch}_batch_{batch_id}"),
            )
