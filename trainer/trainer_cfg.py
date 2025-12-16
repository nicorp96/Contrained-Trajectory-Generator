import copy

import numpy as np
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import os
import torch

from common.get_class import get_class_dict
from trainer.base_trainer import BaseTrainer
from models.utils.ema import EMA
from models.utils.diffusion import sincos_to_angle_lastdim, project_sincos_channels


class DiffusionTrainer(BaseTrainer):
    def __init__(self, config):
        super(DiffusionTrainer, self).__init__(config)
        self.num_inference_steps = self.config["model"].get("num_inference_steps", 100)

    def setup_model(self):
        self.model = get_class_dict(self.config["model"])
        assert torch.cuda.is_available()
        if self.from_checkpoint:
            self.model.load_state_dict(self.checkpoint["state_dict"]["model_state"])
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
        state_normalizer, goal_normalizer = (
            self.train_dataset.dataset.get_normalizer_fit()
        )
        self.normalizer_state = state_normalizer
        self.normalizer_goal = goal_normalizer
        if self.from_checkpoint:
            self.normalizer_state.load_state_dict(
                self.checkpoint["state_dicts"]["normalizer_state"]
            )
            self.normalizer_goal.load_state_dict(
                self.checkpoint["state_dicts"]["normalizer_goal"]
            )
        self.normalizers.append(state_normalizer)
        self.normalizers.append(goal_normalizer)

    def compute_loss(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target)
        return loss

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
        model = self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        model.eval()
        scheduler = self.noise_scheduler

        gs = float(self.config["model"].get("guidance_scale", 1.0)) if guidance_scale is None else float(guidance_scale)

        # x_T ~ N(0, I)
        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )

        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError("cond is required for CFG (set guidance_scale=1.0 to disable).")

        # build the unconditional conditioning vector (zeros = NULL token)
        uncond = torch.zeros_like(cond)

        # fast path: no CFG
        if gs == 1.0:
            for t in scheduler.timesteps:
                # make a [B]-shaped timestep tensor for safety with batched forward
                t_in = t.expand(trajectory_n.shape[0]).to(trj.device)
                model_out = model(trajectory_n, t_in, cond)
                trajectory_n = scheduler.step(model_out, t, trajectory_n, generator=generator, **kwargs).prev_sample
            return trajectory_n

        # guided path
        for t in scheduler.timesteps:
            # one batched forward for [uncond; cond]
            traj_in = torch.cat([trajectory_n, trajectory_n], dim=0)
            cond_in = torch.cat([uncond, cond], dim=0)
            t_in = t.expand(traj_in.shape[0]).to(trj.device)

            model_out = model(traj_in, t_in, cond_in)
            out_uncond, out_cond = model_out.chunk(2, dim=0)

            pred_type = scheduler.config.prediction_type

            if pred_type in ("epsilon", "v_prediction"):
                guided = out_uncond + gs * (out_cond - out_uncond)
                model_out_guided = guided

            elif pred_type == "sample":
                # Convert x0->eps, guide in eps space, convert back to x0
                # t is a scalar index into alphas_cumprod
                t_idx = t.long().item() if t.ndim == 0 else int(t[0].long().item())
                a_bar = scheduler.alphas_cumprod.to(trj.device)[t_idx]  # scalar
                # broadcast to sample shape
                expand = (1,) * trajectory_n.ndim
                sqrt_ab = a_bar.sqrt().view(*expand)
                sqrt_omb = (1.0 - a_bar).sqrt().view(*expand)

                eps_uncond = (trajectory_n - sqrt_ab * out_uncond) / sqrt_omb
                eps_cond = (trajectory_n - sqrt_ab * out_cond) / sqrt_omb
                eps_guided = eps_uncond + gs * (eps_cond - eps_uncond)
                x0_guided = (trajectory_n - sqrt_omb * eps_guided) / sqrt_ab

                model_out_guided = x0_guided

            else:
                raise ValueError(f"Unsupported prediction_type: {pred_type}")

            trajectory_n = scheduler.step(model_out_guided, t, trajectory_n, generator=generator, **kwargs).prev_sample

        return trajectory_n

    def predict_trajectory(self, trj, cond, use_ema=False, **kwargs):
        n_sample = self.conditional_sample(trj, cond, use_ema=use_ema, **kwargs)
        result = {"output": n_sample, "target": trj}
        # TODO: Apply unnormalize
        return result

    def train_step(self, batch):
        device = self.device
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
            drop_mask = (torch.rand(B, 1, device=device) < cfg_p).float()  # 1 = drop -> unconditional
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
        out.update({"loss": loss})
        return out

    def plotting(self, output, target, step, epoch, name="val"):
        config = self.config
        output_dict = self.split_state_tensor(output, config["dataset"]["state_shapes"])
        target_dict = self.split_state_tensor(target, config["dataset"]["state_shapes"])
        output_unnormalized = self.normalizer_state.unnormalize(output_dict)
        target_unnormalized = self.normalizer_state.unnormalize(target_dict)
        for key in output_dict.keys():
            output_np = output_unnormalized[key].detach().cpu().numpy()
            target_np = target_unnormalized[key].detach().cpu().numpy()
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
            idx += value
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
        model = self.ema_model if (use_ema and hasattr(self, "ema_model")) else self.model
        model.eval()
        scheduler = self.noise_scheduler

        gs = float(self.config["model"].get("guidance_scale", 1.0)) if guidance_scale is None else float(guidance_scale)

        trajectory_n = torch.randn(
            size=trj.shape, dtype=trj.dtype, device=trj.device, generator=generator
        )

        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)

        if cond is None:
            raise ValueError("cond is required for CFG (set guidance_scale=1.0 to disable).")

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
                t_idx = t_scalar.long().item() if t_scalar.ndim == 0 else int(t_scalar[0].long().item())
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

            x_prev = scheduler.step(guided, t_scalar, x_t, generator=generator, **kwargs).prev_sample
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
        # TODO: Apply unnormalize
        return result

    def train_step(self, batch):
        device = self.device
        # Select Start Index for trajectories
        states_dict, goals_dict = batch
        start_idx = np.random.randint(0, 230)
        states_norm_dict = self.normalizer_state(states_dict)
        goals_norm = self.normalizer_goal(goals_dict)
        # Trajectorie lenght to predict is then 255 - start_idx
        trajectory = torch.cat(
            [states_norm_dict[key][:,start_idx:,:] for key in states_norm_dict.keys()], dim=2
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
        cfg_p = float(self.config["model"].get(" ", 0.0))
        if cfg_p > 0.0:
            B = cond.size(0)
            drop_mask = (torch.rand(B, 1, device=device) < cfg_p).float()  # 1 = drop -> unconditional
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

        # ---- CFG validation sweep ----
        val_cfg = self.config.get("validation", {})
        scales = val_cfg.get(
            "guidance_scales",
            [self.config["model"].get("guidance_scale", 1.0)]
        )
        # Use a fixed init noise to compare scales apples-to-apples
        g = torch.Generator(device=device)
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
            target_np = target_unnormalized[key].detach().cpu().numpy()[:, 1:, :]
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