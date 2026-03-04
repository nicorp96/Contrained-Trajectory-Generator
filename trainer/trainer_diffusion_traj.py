import copy

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from einops import repeat
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.get_class import get_class_dict
from common.inpainting import apply_inpainting, make_inpainting_mask
from dataset.utils import get_ds_from_cfg
from global_parameters import ConfigGlobalP
from models.utils.ema import EMA
from models.utils.guidance_robot import BaseGuidance, CFGGuidance
from trainer.base_trainer import BaseTrainer

cfg_gp = ConfigGlobalP()


class DiffusionTrainer(BaseTrainer):
    def __init__(self, config):
        super(DiffusionTrainer, self).__init__(config)
        self.num_inference_steps = self.config["model"].get("num_inference_steps", 100)
        self.goal_indices = config.get("goal_indices", None)
        self.start_indices = config.get("start_indices", None)
        self.inpainting = config.get("inpainting", True)

    def setup_model(self):
        self.model = get_class_dict(self.config["model"])
        self.guidance_algo: BaseGuidance = get_class_dict(
            self.config["model"]["guidance"]
        )
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
            state_normalizer, goal_normalizer, obs_normalizer = (
                self.train_dataset.dataset.get_stats_from_file()
            )
        else:
            state_normalizer, goal_normalizer, obs_normalizer = (
                self.train_loader.get_normalizer_fit()
            )
        self.normalizer_state = state_normalizer
        self.normalizer_goal = goal_normalizer
        self.normalizer_obs = obs_normalizer
        self.normalizers.append(state_normalizer)
        self.normalizers.append(goal_normalizer)
        self.normalizers.append(obs_normalizer)
        for normalizer in self.normalizers:
            normalizer.set_device(self.device)

    def compute_loss(self, output, target, mask_valid=None):
        # output, target: [B, M, D]
        # mask_valid:     [B, M]  (True/1 = real token)
        per_elem = (output - target) ** 2  # [B, M, D]
        per_token = per_elem.mean(dim=-1)  # [B, M]

        if mask_valid is not None:
            mask = mask_valid.float()
            loss = (per_token * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = per_token.mean()

        return loss

    def conditional_sample(
        self, trj, cond, use_ema=True, generator=None, mask_inpt=None, **kwargs
    ):
        raise NotImplementedError()

    def predict_trajectory(self, trj, cond, use_ema=False, **kwargs):
        mask_inp = make_inpainting_mask(
            trajectory=trj,
            start_indices=self.start_indices,
            goal_indices=self.goal_indices,
        )
        output_cd = self.conditional_sample(
            trj, cond, use_ema=use_ema, mask_inpt=mask_inp, **kwargs
        )
        return {"output": output_cd, "target": trj}

    def plotting(self, value_dict, step, epoch, name="val", unorm=True):
        config = self.config
        output = value_dict["output"]
        target = value_dict["target_dict"]
        output_dict = self.split_state_tensor(output, config["dataset"]["state_shapes"])
        # goal_pos = value_dict["goal_pos"].unsqueeze(1).detach().cpu().numpy()
        # goal_pos[:, :, -1] = goal_pos[:, :, -1] * (-1.0)
        target_unnormalized = target
        if unorm:
            output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        for key in output_dict.keys():
            output_np = output_unnormalized[key].detach().cpu().numpy()
            target_np = target_unnormalized[key].detach().cpu().numpy()
            if key == "position":
                output_np = output_np * cfg_gp.M_TO_KM
                target_np = target_np * cfg_gp.M_TO_KM
                output_np[:, :, -1] = output_np[:, :, -1] * (-1.0)
                target_np[:, :, -1] = target_np[:, :, -1] * (-1.0)
            if output_np.shape[2] == 3:
                batch_id = 0
                image_name = os.path.join(
                    f"{name}_{key}", f"ep{epoch}_batch_{batch_id}"
                )
                out = output_np[batch_id, :, :]
                tgt = target_np[batch_id, :, :]
                img_np = self.plot_3d_trajectory(out, tgt, title=image_name, key=key)
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )
                # if output_np.ndim == 3: # key == "position" or key == "delta_pos" or key == "vel_ned":
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


class DiffusionTrajFullTrainer(DiffusionTrainer):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def conditional_sample(
        self, trj, cond, use_ema=True, generator=None, mask_inpt=None, **kwargs
    ):
        inpt = self.inpainting
        guidance_algo = self.guidance_algo
        # device = trj.device
        # guidance_scale = (
        #     None if "guidance_scale" not in kwargs.keys() else kwargs["guidance_scale"]
        # )
        model = (
            self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        )
        model.eval()
        scheduler = self.noise_scheduler

        # gs = (
        #     float(self.config["model"].get("guidance_scale", 1.0))
        #     if guidance_scale is None
        #     else float(guidance_scale)
        # )

        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )
        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )
        if inpt:
            trajectory_n = apply_inpainting(trajectory_n, mask_inpt, trj, noise=False)
        for k in scheduler.timesteps:
            # t_scalar = k  # scalar (HF timesteps tensor element)
            trajectory_n = guidance_algo(
                input_noised=trajectory_n,
                k=k,
                cond=cond,
                scheduler=scheduler,
                model=model,
                **kwargs,
            )
            # trajectory_n = guide_and_step(trajectory_n, t_scalar)
            if inpt:
                trajectory_n = apply_inpainting(
                    trajectory_n, mask_inpt, trj, noise=False
                )
        return trajectory_n

    def train_step(self, batch):
        device = self.device
        scheduler = self.noise_scheduler
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
        cond_init = trajectory[:, 0, 3:]
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)

        noise = torch.randn(trajectory.shape, device=device)
        batch_size = trajectory.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=device,
        ).long()
        noisy_states = scheduler.add_noise(trajectory, noise, timesteps)
        if self.inpainting:
            mask_inp = make_inpainting_mask(
                trajectory=trajectory,
                start_indices=self.start_indices,
                goal_indices=self.goal_indices,
            )
            noisy_states = apply_inpainting(
                noisy_states, mask_inp, x_known=trajectory, noise=False
            )
        cond = torch.cat((goal_nm, cond_init), dim=1)
        # ---------- Classifier-free dropout ----------
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (
                torch.rand(B, 1, device=device) < cfg_p
            ).float()  # 1 = drop -> unconditional
            cond = cond * (1.0 - drop_mask)  # zeros become the "null" condition

        trj_noised_cond = noisy_states
        pred = self.model(trj_noised_cond, timesteps, cond)  # , T_d)
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = self.compute_loss(pred, target)
        return {"loss": loss}

    def val_step(self, batch):
        device = self.device
        use_ema = self.ema_model is not None
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
        cond_init = trajectory[:, 0, 3:]
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        cond = torch.cat((goal_nm, cond_init), dim=1)
        out = self.predict_trajectory(trajectory, cond, use_ema=use_ema)
        loss = self.compute_loss(out["output"], out["target"])
        out["goal_pos"] = goal_pos
        out["target_dict"] = states_dict
        out.update({"loss": loss})
        return out


class DiffusionTrajectoryPadHistTrainer(DiffusionTrainer):
    def __init__(self, config):
        super().__init__(config)

    @torch.no_grad()
    def conditional_sample(
        self,
        trj,
        cond,
        use_ema=True,
        generator=None,
        guidance_scale=None,
        time=None,
        mask_inpt=None,
        **kwargs,
    ):
        inpt = self.inpainting
        model = (
            self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        )
        guidance_algo = self.guidance_algo
        model.eval()
        scheduler = self.noise_scheduler
        kwargs["goal"] = trj[:, -1:, :3]
        kwargs["normalizer_state"] = self.normalizer_state
        kwargs["state_shapes"] = self.config["dataset"]["state_shapes"]

        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )

        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )
        if inpt:
            trajectory_n = apply_inpainting(trajectory_n, mask_inpt, trj, noise=False)
        for k in scheduler.timesteps:
            trajectory_n = guidance_algo(
                input_noised=trajectory_n,
                k=k,
                cond=cond,
                scheduler=scheduler,
                model=model,
                **kwargs,
            )
            if inpt:
                trajectory_n = apply_inpainting(
                    trajectory_n, mask_inpt, trj, noise=False
                )
        return trajectory_n

    def train_step(self, batch):
        device = self.device
        scheduler = self.noise_scheduler

        states_dict, goals_dict, extra = batch

        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)

        B, L, D = trajectory.shape
        if "mask" in extra.keys():
            mask_valid = extra.pop("mask").to(device=device)
        seq_len_L = (
            extra.pop("seq_len_L")
            if "seq_len_L" in extra.keys()
            else self.config["dataset"].get("segment_horizon", 16)
        )

        mask_valid = None
        mask_key_att = None  # ~mask_valid
        history_dict_nm = self.normalizer_obs(extra)
        trajectory_history = torch.cat(
            [history_dict_nm[key].to(device) for key in history_dict_nm.keys()],
            dim=2,
        )
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)

        noise = torch.randn(trajectory.shape, device=device)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()
        cond = goal_nm
        noisy_states = scheduler.add_noise(trajectory, noise, timesteps)
        if self.inpainting:
            mask_inp = make_inpainting_mask(
                trajectory=trajectory,
                start_indices=self.start_indices,
                goal_indices=self.goal_indices,
            )
            noisy_states = apply_inpainting(
                noisy_states, mask_inp, x_known=trajectory, noise=False
            )
        # ---------- Classifier-free dropout ----------
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (
                torch.rand(B, 1, device=device) < cfg_p
            ).float()  # 1 = drop -> unconditional
            cond = cond * (1.0 - drop_mask)  # zeros become the "null" condition
        # (no change to model dims; we don't append an extra bit)
        pred = self.model(
            noisy_states,
            timesteps,
            cond,
            mask=mask_key_att,
            state_hist=trajectory_history,
        )
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            target = scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")
        loss = self.compute_loss(pred, target, mask_valid)
        return {"loss": loss}

    def val_step(self, batch):
        device = self.device
        use_ema = self.ema_model is not None
        states_dict, goals_dict, extra = batch

        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        if "mask" in extra.keys():
            mask_valid = extra.pop("mask").to(device=device)
        seq_len_L = (
            extra.pop("seq_len_L")
            if "seq_len_L" in extra.keys()
            else self.config["dataset"].get("segment_horizon", 16)
        )

        mask_valid = None
        history_dict_nm = self.normalizer_obs(extra)
        trajectory_history = torch.cat(
            [history_dict_nm[key].to(device) for key in history_dict_nm.keys()],
            dim=2,
        )
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        cond = goal_nm
        out = self.predict_trajectory(
            trajectory,
            cond,
            use_ema=use_ema,
            state_hist=trajectory_history,
        )
        loss = self.compute_loss(out["output"], out["target"], mask_valid)
        out["target_dict"] = states_dict
        out.update({"loss": loss})
        out.update({"seq_len_L": seq_len_L})
        return out

    def plotting(self, value_dict, step, epoch, name="val"):
        config = self.config["dataset"]["state_shapes"]
        output_dict = self.split_state_tensor(value_dict["output"], config)
        # # TODO: Test only (remove)
        output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        target = value_dict["target_dict"]
        # seq_len_L = value_dict["seq_len_L"]
        # TODO: Remove when not needed
        # start_pos = value_dict["start_pos"]
        # target_unnormalized = self.normalizer_state.unnormalize(target)
        for key in output_dict.keys():
            output_np = output_unnormalized[key].detach().cpu().numpy()
            target_np = target[key].detach().cpu().numpy()
            batch_id = np.random.randint(0, output_np.shape[0])
            seq_orig = output_np.shape[1]
            seq_len_L_i = seq_orig  # seq_len_L[batch_id]

            if key == "tcp_pose":
                image_name = os.path.join(
                    f"{name}_{key}", f"ep{epoch}_batch_{batch_id}"
                )
                out = output_np[batch_id, :seq_len_L_i, :3]
                tgt = target_np[batch_id, :seq_len_L_i, :3]
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

                # img_x = self.plot_1d_seq(
                #     output_np[0, :seq_len_L_i, 0],
                #     target_np[0, :seq_len_L_i, 0],
                #     title=image_name + "_x",
                # )
                # self.writer.add_image(
                #     img_tensor=img_x,
                #     global_step=step,
                #     dataformats="HWC",
                #     tag=os.path.join(f"{name}_{key}_x", f"ep{epoch}_batch_{batch_id}"),
                # )
                # img_y = self.plot_1d_seq(
                #     output_np[0, :seq_len_L_i, 1],
                #     target_np[0, :seq_len_L_i, 1],
                #     title=image_name + "_y",
                # )
                # self.writer.add_image(
                #     img_tensor=img_y,
                #     global_step=step,
                #     dataformats="HWC",
                #     tag=os.path.join(f"{name}_{key}_y", f"ep{epoch}_batch_{batch_id}"),
                # )
                # img_z = self.plot_1d_seq(
                #     output_np[0, :seq_len_L_i, 2],
                #     target_np[0, :seq_len_L_i, 2],
                #     title=image_name + "_z",
                # )
                # self.writer.add_image(
                #     img_tensor=img_z,
                #     global_step=step,
                #     dataformats="HWC",
                #     tag=os.path.join(f"{name}_{key}_z", f"ep{epoch}_batch_{batch_id}"),
                # )

            for axis_idx in range(output_np.shape[-1]):
                image_name = os.path.join(
                    f"{name}_{key}_ax_{axis_idx}", f"epoch_{epoch}"
                )
                img_np = self.plot_1d_seq(
                    output_np[batch_id, :seq_len_L_i, axis_idx],
                    target_np[batch_id, :seq_len_L_i, axis_idx],
                    title=image_name,
                )
                self.writer.add_image(
                    tag=image_name,
                    global_step=step,
                    img_tensor=img_np,
                    dataformats="HWC",
                )


class DiffusionTrajectoryPadHPhyITrainer(DiffusionTrajectoryPadHistTrainer):
    def __init__(self, config):
        super(DiffusionTrajectoryPadHPhyITrainer, self).__init__(config)
        self.lambda_phys = config.get("lambda_phys", 0.4)
        self.loss_type = config.get("lambda_phys", "huber")
        self.small_value = 1e-12
        self.max_val_vel = config.get("max_val_vel", 1400)
        self.min_val_vel = config.get("min_val_vel", 1.0)

    @staticmethod
    def residual_phys(
        x0_hat,
        pos_idx=(0, 1, 2),
        vel_idx=(3, 4, 5),
        dt=0.1,
        loss_type="",
        mask=None,
    ):
        """
        x0_hat: (B, T, D) denoised trajectory estimate
        """
        device = x0_hat.device
        pos_idx = torch.as_tensor(pos_idx, device=device, dtype=torch.long)
        vel_idx = torch.as_tensor(vel_idx, device=device, dtype=torch.long)
        p = x0_hat[:, :, pos_idx]  # (B,T,3)
        v = x0_hat[:, :, vel_idx]  # (B,T,3)

        dp = p[:, 1:] - p[:, :-1]  # (B,T-1,3)
        vbar = 0.5 * (v[:, 1:] + v[:, :-1])  # (B,T-1,3)

        if not torch.is_tensor(dt):
            dt = torch.tensor(dt, device=device, dtype=x0_hat.dtype)
        while dt.ndim < dp.ndim:
            dt = dt.unsqueeze(-1)

        r = dp - vbar * dt
        denom = torch.tensor(1.0, device=device, dtype=x0_hat.dtype)
        if mask is not None:
            pair_mask = mask[:, 1:] * mask[:, :-1]  # (B,T-1)
            pair_mask = pair_mask.unsqueeze(-1)  # (B,T-1,1)
            r = r * pair_mask

            # normalize by number of valid pairs (important!)
            denom = pair_mask.sum().clamp_min(1.0)

        if loss_type == "huber":
            loss = F.huber_loss(
                r,
                torch.zeros_like(r),
                delta=0.1,
                reduction="sum",  # <-- sum, not mean
            )
        elif loss_type == "charbonnier":
            loss = torch.sqrt(r.pow(2) + 1e-4).sum()
        else:
            loss = r.pow(2).sum()

        return loss / denom

    @staticmethod
    def box_constraint_loss(x0_hat, value_min, value_max, idx=(0, 1, 2)):
        # x_min, x_max can be scalars, [D], or broadcastable to [B,T,D]
        device = x0_hat.device
        value_max = torch.tensor(value_max, device=device, dtype=torch.float32)
        value_min = torch.tensor(value_min, device=device, dtype=torch.float32)
        idx = torch.as_tensor(idx, device=device, dtype=torch.long)
        val = torch.norm(x0_hat[:, :, idx], p=2, dim=-1)
        low = F.relu(value_min - val)
        high = F.relu(val - value_max)
        return (low**2 + high**2).mean()

    def compute_physic_loss(self, x_t, denoiser_pred, k, d_t, mask=None):
        """
        For diffusers-style schedulers: x_t = sqrt(alphas_cumprod)*x0 + sqrt(1-alpha_cumprod)*eps
        => x0 = (x_t - sqrt(1-a)*eps) / sqrt(a)
        """
        scheduler = self.noise_scheduler
        normalizer_state = self.normalizer_state
        state_shapes = self.config["dataset"]["state_shapes"]
        pred_type = scheduler.config.prediction_type
        x_0_hat = None
        d_t = d_t.to(x_t.device)
        if pred_type == "epsilon":
            a = scheduler.alphas_cumprod[k].to(x_t.device)
            if a.ndim == 0:
                a = a.view(1)
            if a.ndim < x_t.ndim:
                a = a.unsqueeze(-1)
            sqrt_a = a.sqrt()
            sqrt_one_minus_a = (1.0 - a).sqrt()
            x_0_hat = (x_t - sqrt_one_minus_a.unsqueeze(1) * denoiser_pred) / (
                sqrt_a.unsqueeze(1) + self.small_value
            )
        elif pred_type == "sample":
            x_0_hat = denoiser_pred

        elif pred_type == "v_prediction":
            alphas_cumprod = scheduler.alphas_cumprod.to(
                device=x_t.device, dtype=x_t.dtype
            )

            a = alphas_cumprod[k]  # [B]
            sqrt_a = torch.sqrt(a)  # [B]
            sqrt_one_minus_a = torch.sqrt(1.0 - a)  # [B]

            if sqrt_a.ndim < x_t.ndim:
                sqrt_a = sqrt_a.unsqueeze(-1)
                sqrt_one_minus_a = sqrt_one_minus_a.unsqueeze(-1)

            x_0_hat = (
                sqrt_a.unsqueeze(1) * x_t
                - sqrt_one_minus_a.unsqueeze(1) * denoiser_pred
            )
        x_0_hat_dict = self.split_state_tensor(x_0_hat, state_shapes)
        x_0_hat_unorm_dict = normalizer_state.unnormalize(x_0_hat_dict)
        x_0_hat_phy = torch.cat(
            [x_0_hat_unorm_dict[key] for key in x_0_hat_unorm_dict.keys()],
            dim=2,
        ).to(x_0_hat.device)
        # Unormalized x_0_hat to -> physic space
        loss_res = self.residual_phys(
            x_0_hat_phy,
            pos_idx=(0, 1, 2),
            vel_idx=(3, 4, 5),
            dt=d_t,
            loss_type=self.loss_type,
            mask=mask,
        )
        loss_constraints = self.box_constraint_loss(
            x_0_hat_phy, self.min_val_vel, self.max_val_vel, idx=(3, 4, 5)
        )
        return loss_res + loss_constraints

    def train_step(self, batch):
        device = self.device
        scheduler = self.noise_scheduler
        lambda_phys = self.lambda_phys
        states_dict, goals_dict, extra = batch
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)

        B, L, D = trajectory.shape
        # TODO: Remove if nor needed
        mask_valid = extra.pop("mask").to(device=device)
        seq_len_L = extra.pop("seq_len_L")
        d_t = extra.pop("delta_t")[:, 0, :]
        mask_valid = None
        mask_key_att = None  # ~mask_valid
        history_dict_nm = self.normalizer_obs(extra)
        trajectory_history = torch.cat(
            [history_dict_nm[key] for key in history_dict_nm.keys()],
            dim=2,
        ).to(device)
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)

        noise = torch.randn(trajectory.shape, device=device)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()
        cond = goal_nm
        noisy_states = scheduler.add_noise(trajectory, noise, timesteps)
        if self.inpainting:
            mask_inp = make_inpainting_mask(
                trajectory=trajectory,
                start_indices=self.start_indices,
                goal_indices=self.goal_indices,
            )
            noisy_states = apply_inpainting(
                noisy_states, mask_inp, x_known=trajectory, noise=False
            )
        # ---------- Classifier-free dropout ----------
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (
                torch.rand(B, 1, device=device) < cfg_p
            ).float()  # 1 = drop -> unconditional
            cond = cond * (1.0 - drop_mask)  # zeros become the "null" condition
        # (no change to model dims; we don't append an extra bit)
        pred = self.model(
            noisy_states,
            timesteps,
            cond,
            mask=mask_key_att,
            state_hist=trajectory_history,
        )
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            target = scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_diff = self.compute_loss(pred, target, mask_valid)
        loss_phys = self.compute_physic_loss(
            noisy_states, pred, timesteps, d_t, mask_valid
        )

        a = scheduler.alphas_cumprod[timesteps].to(device)
        w = a.mean()  # scalar weight (simple). You can also do per-sample weighting.
        weighted_loss = w * loss_phys
        loss = loss_diff + lambda_phys * weighted_loss  # .mean()
        return {"loss": loss, "diff": loss_diff, "phys": loss_phys}

    def val_step(self, batch):
        device = self.device
        use_ema = self.ema_model is not None
        states_dict, goals_dict, extra = batch

        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        # TODO: Remove if nor needed
        mask_valid = extra.pop("mask").to(device)
        d_t = extra.pop("delta_t")
        seq_len_L = extra.pop("seq_len_L")
        mask_valid = None
        history_dict_nm = self.normalizer_obs(extra)
        trajectory_history = torch.cat(
            [history_dict_nm[key] for key in history_dict_nm.keys()],
            dim=2,
        ).to(device)

        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        # mask_inp = torch.zeros_like(trajectory, dtype=torch.bool, device=device)
        # mask_inp[:, :1, :] = torch.ones_like(
        #     trajectory[:, :1, :], dtype=torch.bool, device=device
        # )
        # mask_inp[:, -1:, :3] = torch.ones_like(
        #     trajectory[:, -1:, :3], dtype=torch.bool, device=device
        # )

        cond = goal_nm
        out = self.predict_trajectory(
            trajectory,
            cond,
            use_ema=use_ema,
            state_hist=trajectory_history,
        )
        loss = self.compute_loss(out["output"], out["target"], mask_valid)
        out["target_dict"] = states_dict
        out.update({"loss": loss})
        out.update({"seq_len_L": seq_len_L})
        return out


class DiffusionTrajectoryPadHistRSTrainer(DiffusionTrajectoryPadHistTrainer):
    def __init__(self, config):
        super(DiffusionTrajectoryPadHistRSTrainer, self).__init__(config)

    # TODO: Added for resampling: Remove if not needed.
    @staticmethod
    def dist_to_minus1_1(d, eps=1e-8):
        # d: (B,) positive
        d_max = 160078.11
        d_min = 357.55267
        d01 = (d - d_min) / (d_max - d_min + eps)
        d01 = d01.clamp(0.0, 1.0)
        return 2.0 * d01 - 1.0

    @staticmethod
    def dist_to_01(d, eps=1e-8):
        d_max = 160078.11
        d_min = 357.55267
        return ((d - d_min) / (d_max - d_min + eps)).clamp(0.0, 1.0)

    # @staticmethod
    # def sigma_scale(d01, s_min=0.1, s_max=1.0):
    #     return (s_min + (s_max - s_min) * d01).clamp(s_min, s_max)
    @staticmethod
    def scale_with_dist(d, eps=1e-8):
        d_max = 160078.11
        d_min = 357.55267
        d_01 = (d - d_min) / (d_max - d_min + eps)
        if d_01.min() == 0.0:
            return d_01.clamp(0.0001, 1.0)
        return d_01

    @staticmethod
    def dist_to_01_log(d):
        d_max = torch.tensor(160078.11)
        d_min = torch.tensor(357.55267)
        log_d_min = torch.log(d_min)
        log_d_max = torch.log(d_max)
        dlog = torch.log(d + 1e-6)
        d01 = (dlog - log_d_min) / (log_d_max - log_d_min + 1e-8)
        return d01.clamp(0.0, 1.0)

    @staticmethod
    def sample_timesteps_by_distance(d, T, gamma=1.5, t_min=0):
        """
        d: (B,) distances (any positive scale)
        T: number of diffusion steps (int)
        gamma: strength of bias (higher -> more bias towards small t for small d)
        t_min: optionally avoid t=0 if your code expects 1..T-1
        returns: t (B,) int64 in [t_min, T-1]
        """
        B = d.shape[0]
        device = d.device
        # Normalize distances to [0,1] within the batch (robust + simple)
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)  # 0=short, 1=long
        # For short d_norm~0 -> prefer small t. For long d_norm~1 -> nearly uniform.
        # Create a per-sample exponent that controls bias of u**p:
        p = 1.0 + gamma * (1.0 - d_norm)  # short -> larger p -> more mass near 0
        u = torch.rand(B, device=device)
        t_float = u**p  # bias towards 0 when p>1
        t = (t_float * (T - 1 - t_min) + t_min).long()
        return t

    @staticmethod
    def effective_timestep(t_base, d, w_min=0.3, w_max=1.0):
        """
        t_base: (B,) int64 in [0, T-1]
        d: (B,) distances
        w_min: minimum warp factor for shortest distances
        w_max: maximum warp factor for longest distances (usually 1.0)
        returns: t_eff (B,) int64
        """
        # normalize d to [0,1] in the batch
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-8)
        # short (0) -> w_min, long (1) -> w_max
        w = w_min + (w_max - w_min) * d_norm
        t_eff = torch.clamp((w * t_base.float()).long(), min=0)
        return t_eff

    @torch.no_grad()
    def conditional_sample(
        self,
        trj,
        cond,
        use_ema=True,
        generator=None,
        guidance_scale=None,
        time=None,
        mask_inpt=None,
        **kwargs,
    ):
        model = (
            self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        )
        guidance_algo = self.guidance_algo
        inp = self.inpainting
        model.eval()
        scheduler = self.noise_scheduler
        kwargs["goal"] = trj[:, -1:, :3]
        kwargs["normalizer_state"] = self.normalizer_state
        kwargs["state_shapes"] = self.config["dataset"]["state_shapes"]
        dist = kwargs["dist"]
        B, L, D = trj.shape
        # scale_noise = repeat(
        #     self.scale_with_dist(dist).to(device=trj.device), "B -> B L D", L=L, D=D
        # )
        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )
        # trajectory_n = self.scale_model.scale_noise(trajectory_n, dist.unsqueeze(-1))

        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError(
                "cond is required for CFG (set guidance_scale=1.0 to disable)."
            )
        if inp:
            trajectory_n = apply_inpainting(trajectory_n, mask_inpt, trj, noise=False)
        for k in scheduler.timesteps:
            # k = self.effective_timestep(k, dist, w_min=0.4, w_max=1.0)
            trajectory_n = guidance_algo(
                input_noised=trajectory_n,
                k=k,
                cond=cond,
                scheduler=scheduler,
                model=model,
                **kwargs,
            )
            if inp:
                trajectory_n = apply_inpainting(
                    trajectory_n, mask_inpt, trj, noise=False
                )
        return trajectory_n

    def train_step(self, batch):
        device = self.device
        scheduler = self.noise_scheduler
        states_dict, goals_dict, extra = batch
        # pos_start = states_dict["position"][:, :1, :]
        # states_dict["position"] = states_dict["position"]  - pos_start
        # TODO: Test only (remove)
        # states_dict["position"] = states_dict["position"]
        # goals_dict["pip_x_y_z_km"] = goals_dict["pip_x_y_z_km"]
        # extra["position"] = extra["position"][:, :, [1, 0, 2]]
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        # TODO: Test only (remove)
        # trajectory = torch.cat([trajectory[:, :, 1:2], trajectory[:, :, 3:]], dim=-1)
        B, L, D = trajectory.shape
        mask_valid = extra.pop("mask").to(device=device)
        seq_len_L = extra.pop("seq_len_L")
        # dist = torch.norm(
        #     states_dict["position"][:, -1:, :] - states_dict["position"][:, :, :],
        #     p=2,
        #     dim=-1,
        # ).to(device=device)
        dist = torch.norm(
            states_dict["position"][:, -1, :] - states_dict["position"][:, 0, :],
            p=2,
            dim=-1,
        ).to(device=device)
        # TODO: Test only (remove)
        dist = self.dist_to_01(dist)
        mask_valid = None
        mask_key_att = None  # ~mask_valid
        history_dict_nm = self.normalizer_obs(extra)
        trajectory_history = torch.cat(
            [history_dict_nm[key] for key in history_dict_nm.keys()],
            dim=2,
        ).to(device)
        # TODO: Test only (remove)
        # trajectory_history = torch.cat(
        #     [trajectory_history[:, :, 1:2], trajectory_history[:, :, 3:]], dim=-1
        # )
        # mask = repeat(
        #     extra["mask"], "B C -> B C d", d=D
        # )  # Variable length, where padding = true
        # goal_pos = goals_norm.pop("pip_x_y_z_km").to(device)
        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        # TODO: Test only (remove)
        # goal_nm = torch.cat([goal_nm[:, 1:2], goal_nm[:, 3:]], dim=1)
        # dist_scale = self.scale_with_dist(dist)
        # scale_noise = repeat(dist_scale.to(device=device), "B -> B L D", L=L, D=D)
        noise = torch.randn(trajectory.shape, device=device)
        # noise = self.scale_model.scale_noise(noise, dist.unsqueeze(-1))
        # noise = torch.randn(trajectory.shape, device=device)
        timesteps = torch.randint(
            0,
            scheduler.config.num_train_timesteps,
            (B,),
            device=device,
        ).long()

        # timesteps = self.effective_timestep(
        #     t_base=timesteps, d=dist, w_min=0.6, w_max=1.0
        # )
        # timesteps = self.sample_timesteps_by_distance(
        #     d=dist,
        #     T=scheduler.config.num_train_timesteps,
        # )
        # dist = self.dist_to_minus1_1(dist)
        # dist_c = self.dist_to_01_log(dist)
        # cond = torch.cat([dist_c.unsqueeze(1), goal_nm], dim=-1)
        cond = goal_nm
        noisy_states = scheduler.add_noise(trajectory, noise, timesteps)
        if self.inpainting:
            mask_inp = make_inpainting_mask(
                trajectory=trajectory,
                start_indices=self.start_indices,
                goal_indices=self.goal_indices,
            )
            noisy_states = apply_inpainting(
                noisy_states, mask_inp, x_known=trajectory, noise=False
            )
        # ---------- Classifier-free dropout ----------
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (
                torch.rand(B, 1, device=device) < cfg_p
            ).float()  # 1 = drop -> unconditional
            cond = cond * (1.0 - drop_mask)  # zeros become the "null" condition
        # (no change to model dims; we don't append an extra bit)
        pred = self.model(
            noisy_states,
            timesteps,
            cond,
            mask=mask_key_att,
            state_hist=trajectory_history,
            dist=dist,
        )
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        elif pred_type == "v_prediction":
            target = scheduler.get_velocity(trajectory, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        # loss = self.compute_loss_weighted(pred, target, dist_c, mask_valid)
        loss = self.compute_loss(pred, target, mask_valid)
        #         + self.scale_model.reg(
        #     dist.unsqueeze(1), lam=1e-3
        # ))
        return {"loss": loss}

    def val_step(self, batch):
        device = self.device
        use_ema = self.ema_model is not None
        states_dict, goals_dict, extra = batch
        # TODO: Test only (remove)
        # states_dict["position"] = states_dict["position"][:, :, [1, 0, 2]]
        # goals_dict["pip_x_y_z_km"] = goals_dict["pip_x_y_z_km"][:, [1, 0, 2]]
        # extra["position"] = extra["position"][:, :, [1, 0, 2]]
        # pos_start = states_dict["position"][:, :1, :]
        # states_dict["position"] = states_dict["position"] - pos_start
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        trajectory = torch.cat(
            [states_norm_dict[key] for key in states_norm_dict.keys()],
            dim=2,
        ).to(device)
        # TODO: Test only (remove)
        # trajectory = torch.cat([trajectory[:, :, 1:2], trajectory[:, :, 3:]], dim=-1)
        # dist = torch.norm(
        #     states_dict["position"][:, -1:, :] - states_dict["position"][:, :, :],
        #     p=2,
        #     dim=-1,
        # ).to(device=device)
        dist = torch.norm(
            states_dict["position"][:, -1, :] - states_dict["position"][:, 0, :],
            p=2,
            dim=-1,
        ).to(device=device)
        dist = self.dist_to_01(dist)
        mask_valid = extra.pop("mask").to(device)
        seq_len_L = extra.pop("seq_len_L")
        mask_valid = None
        history_dict_nm = self.normalizer_obs(extra)
        trajectory_history = torch.cat(
            [history_dict_nm[key] for key in history_dict_nm.keys()],
            dim=2,
        ).to(device)
        # TODO: Test only (remove)
        # trajectory_history = torch.cat(
        #     [trajectory_history[:, :, 1:2], trajectory_history[:, :, 3:]], dim=-1
        # )
        B, M, D = trajectory.shape

        goal_l = []
        for value in goals_norm.values():
            vlu = value
            if value.ndim > 2:
                vlu = value[:, 0, :]
            goal_l.append(vlu)
        goal_nm = torch.cat(goal_l, dim=1).to(device)
        # TODO: Test only (remove)
        # goal_nm = torch.cat([goal_nm[:, 1:2], goal_nm[:, 3:]], dim=1)
        # dist_c = self.dist_to_minus1_1(dist)
        # dist_scale = self.scale_with_dist(dist)
        # dist_c = self.dist_to_01_log(dist)
        # cond = torch.cat([dist_c.unsqueeze(1), goal_nm], dim=-1)
        cond = goal_nm
        out = self.predict_trajectory(
            trajectory,
            cond,
            use_ema=use_ema,
            state_hist=trajectory_history,
            dist=dist,
        )
        loss = self.compute_loss(out["output"], out["target"], mask_valid)
        out["target_dict"] = states_dict
        # TODO: Test only (remove)
        # out["output"] = torch.cat(
        #     [
        #         out["target"][:, :, 0].unsqueeze(-1),
        #         out["output"],
        #         out["target"][:, :, 2].unsqueeze(-1),
        #     ],
        #     dim=-1,
        # )
        out.update({"loss": loss})
        out.update({"seq_len_L": seq_len_L})
        # out.update({"pos_start": pos_start})
        return out

    def compute_loss_weighted(
        self, output, target, d01, mask_valid=None, w_min=0.4, w_max=1.0
    ):
        """
        output, target: [B, M, D]
        d01:            [B] in [0,1]  (0=short, 1=long)
        mask_valid:     [B, M] (True/1 = real token)

        w_min/w_max: short gets weight w_max, long gets weight w_min
        """
        per_elem = (output - target) ** 2  # [B, M, D]
        per_token = per_elem.mean(dim=-1)  # [B, M]

        # per-sample weights: short (0) -> w_max, long (1) -> w_min
        w = w_min + (w_max - w_min) * (1.0 - d01)  # [B]
        per_token = per_token * w[:, None]  # broadcast to [B, M]

        if mask_valid is not None:
            mask = mask_valid.float()
            loss = (per_token * mask).sum() / mask.sum().clamp_min(1.0)
        else:
            loss = per_token.mean()

        return loss


class DiffusionTrajectoryChunk(DiffusionTrajectoryPadHistTrainer):
    def __init__(self, config):
        super().__init__(config)

    def setup_dataloader(self):
        ds_cfg = self.config["dataset"]
        base_train, base_val = get_ds_from_cfg(ds_cfg)

        H = int(ds_cfg.get("segment_horizon", 16))
        K = int(ds_cfg.get("segment_context", 4))
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
            state_normalizer, goal_normalizer, normalizer_obs = (
                self.train_dataset.base.get_stats_from_file()
            )
        else:
            state_normalizer, goal_normalizer = (
                self.train_loader.base.get_normalizer_fit()
            )
        self.normalizer_state = state_normalizer
        self.normalizer_goal = goal_normalizer
        self.normalizer_obs = normalizer_obs
        self.normalizers.append(state_normalizer)
        self.normalizers.append(goal_normalizer)
        self.normalizers.append(normalizer_obs)
