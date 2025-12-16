import copy
from lib2to3.fixer_util import consuming_calls

from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F

from common.get_class import get_class_dict
from common.utils import safe_norm
from trainer.base_trainer import BaseTrainer
from models.utils.ema import EMA


@dataclass
class LossWeights:
    # TODO: name correctly the weights
    fape: float = 1.0
    vel: float = 0.25
    jerk: float = 0.05
    constraints: float = 0.5
    timing: float = 0.2
    conf: float = 0.1


def fape_like(
    pred_x: torch.Tensor, gt_x: torch.Tensor, mask: torch.Tensor = None
) -> torch.Tensor:
    """Root-anchored position error."""
    root_pred = pred_x[:, :1]
    root_gt = gt_x[:, :1]
    pred_centered = pred_x - root_pred
    gt_centered = gt_x - root_gt
    err = safe_norm(pred_centered - gt_centered, dim=-1)  # [B,N]
    return err.mean()
    # return (err * mask).sum() / (mask.sum() + 1e-6)


def jerk_loss(x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Second time derivative magnitude."""
    ddx = torch.diff(x, n=2, dim=1)
    pad = torch.zeros(x.size(0), 2, x.size(-1), device=x.device)
    ddx = torch.cat([ddx, pad], dim=1)
    mag = safe_norm(ddx, dim=-1)
    return mag.mean()
    # return (mag * mask).sum() / (mask.sum() + 1e-6)


class SORTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)
        self.weights = LossWeights(
            fape=config["training"]["loss_weights"]["fape"],
            vel=config["training"]["loss_weights"]["vel"],
            jerk=config["training"]["loss_weights"]["jerk"],
            constraints=config["training"]["loss_weights"]["constraints"],
            timing=config["training"]["loss_weights"]["timing"],
            conf=config["training"]["loss_weights"]["conf"],
        )
        self.model.set_scheduler(self.noise_scheduler)
        self.model.set_normalizer(
            state_normalizer=self.normalizer_state,
            obs_normalizer=self.normalizer_obs,
            environment_normalizer=self.normalizer_env,
        )

    def setup_model(self):
        self.model = get_class_dict(self.config["model"])
        if self.checkpoint is not None:
            self.model.diffusion_model.load_state_dict(
                self.checkpoint["state_dicts"]["model_state_diff"]
            )
        # TODO: Check if load only diffusion model needed!!
        self.model.to(self.device)
        if self.config["training"]["use_ema"]:
            self.ema_model = copy.deepcopy(self.model)
            self.ema = EMA(self.ema_model, self.config["ema"])

    def setup_scheduler(self):
        super().setup_scheduler()
        scheduler_cfg = self.config["model"]["diffusion_model"]["noise_scheduler"]
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
        # if "path_stats" in self.config["dataset"]:
        list_normalizer = self.train_dataset.dataset.get_stats_from_file()
        # else:
        #     list_normalizer = self.train_loader.dataset.get_normalizer_fit()
        self.normalizer_state = list_normalizer[0]
        self.normalizer_obs = list_normalizer[1]
        self.normalizer_env = list_normalizer[2]

    def compute_loss(self, output, target):
        loss = torch.nn.functional.mse_loss(output, target)
        return loss

    def compute_cycle_loss(
        self, outputs, states_gt_nm, observations_gt_nm, env_gt_nm, extras
    ):
        """
        Compute the cycle consistency loss.
        """
        w = self.weights
        total = torch.tensor(0.0, device=self.device)
        K = len(outputs)
        for i, out in enumerate(outputs):
            w_K = (i + 1) / K  # weigh later outputs more
            trajectory = out["trajectory"]
            x, v, T, score, energy = (
                trajectory[:, :, :3],
                trajectory[:, :, 3:6],
                out["time"],
                out["score"],
                out["energy"],
            )
            gt_x = states_gt_nm["position"]
            # gt_v = states_gt_nm["vel_ned"]
            # positions
            f = fape_like(x, gt_x)
            # velocity MSE (masked)
            # vloss = F.mse_loss(v, gt_v, reduction="none").mean(dim=-1)
            # vloss = (vloss * mask).sum() / (mask.sum() + 1e-6)
            # smoothness
            j = jerk_loss(x)  # , mask)
            # constraint proxy (already aggregated per-batch inside analyzer)
            cons = energy.mean()
            # timing supervision (optional)
            gt_T = extras.get("T", None)
            gt_T = gt_T.to(T.device) if gt_T is not None else None
            t = torch.tensor(0.0, device=T.device)
            if gt_T is not None:
                t = F.mse_loss(T.squeeze(-1), gt_T)
            # TODO: Check confidence calibration
            # confidence calibration: low energy ↔ high score
            # conf = F.mse_loss(
            #     score.squeeze(-1).mean(dim=1), torch.exp(-energy.detach())
            # )

            total = total + w_K * (
                w.fape * f
                # + w.vel * vloss
                + w.jerk * j
                + w.constraints * cons
                + w.timing * t
                # + w.conf * conf
            )
        return total

    def initial_guess(self, states_gt_nm, observations_nm, env_nm):
        """
        Provide an initial guess for the states. Only
        """

        return states_gt_nm, observations_nm, env_nm

    def train_step_with_sampling(self, batch):
        """Alternative train step with sampling (not used currently)."""
        device = self.device
        model = self.model
        scheduler = self.noise_scheduler
        states_gt_nm, observations_nm, env_nm, extras = (
            self.normalizer_state(batch["states"]),
            self.normalizer_obs(batch["observations"]),
            self.normalizer_env(batch["environment"]),
            batch["extras"],
        )
        B, N, D = batch["states"]["position"].shape
        C = self.config["model"]["diffusion_model"]["cond_dim"]
        # TODO: train without condition?, then train with the condition and constraints
        cond = torch.zeros((B, C), device=device)
        states_gt_nm_vec = torch.cat(
            [states_gt_nm[key] for key in states_gt_nm.keys()], dim=2
        ).to(device)

        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (states_gt_nm_vec.size(0),),
            device=self.device,
        )
        noise = torch.rand_like(states_gt_nm_vec)
        states_noisy = scheduler.add_noise(states_gt_nm_vec, noise, timesteps)
        noise_pred = model.diffusion_model(states_noisy, timesteps, cond)
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = states_gt_nm_vec
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                states_gt_nm_vec, noise, timesteps
            )
        else:
            raise ValueError(f"Unsupported prediction_type {pred_type}")

        loss_diff = self.compute_loss(noise_pred, target)
        return {"loss": loss_diff}  # add Score and other metrics later

    def train_step(self, batch):
        device = self.device
        model = self.model
        scheduler = self.noise_scheduler
        states_gt_nm, observations_nm, env_nm, extras = (
            self.normalizer_state(batch["states"]),
            self.normalizer_obs(batch["observations"]),
            self.normalizer_env(batch["environment"]),
            batch["extras"],
        )
        # states_gt_nm = self.normalizer_state(states)
        # observations_nm = self.normalizer_observation(observations)
        # env_nm = self.normalizer_env(env)

        states_ig_nm, observations_nm, env_nm = self.initial_guess(
            states_gt_nm, observations_nm, env_nm
        )
        outputs = model(
            states_ig_nm, observations_nm, env_nm, device=device
        )  # List of dict with traj, score, energy
        loss_cycle = self.compute_cycle_loss(
            outputs, states_gt_nm, observations_nm, env_nm, extras
        )
        states_gt_nm_vec = torch.cat(
            [states_gt_nm[key] for key in states_gt_nm.keys()], dim=2
        ).to(device)

        with torch.no_grad():
            ctx_dict = model.context_without_diff(
                states_gt_nm, observations_nm, env_nm, device
            )
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (states_gt_nm_vec.size(0),),
            device=self.device,
        )
        noise = torch.rand_like(states_gt_nm_vec)
        states_noisy = scheduler.add_noise(states_gt_nm_vec, noise, timesteps)
        noise_pred = model.diffusion_model(
            states_noisy, timesteps, ctx_dict["cond_ctx"]
        )
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = states_gt_nm_vec
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(
                states_gt_nm_vec, noise, timesteps
            )
        else:
            raise ValueError(f"Unsupported prediction_type {pred_type}")

        loss_diff = self.compute_loss(noise_pred, target)

        loss = loss_cycle + 1.0 * loss_diff
        return {"loss": loss}  # add Score and other metrics later

    def val_step(self, batch):
        device = self.device
        model = self.model
        states_gt_nm, observations_nm, env_nm, extras = (
            self.normalizer_state(batch["states"]),
            self.normalizer_obs(batch["observations"]),
            self.normalizer_env(batch["environment"]),
            batch["extras"],
        )
        trajectory = torch.cat(
            [states_gt_nm[key] for key in states_gt_nm.keys()], dim=-1
        )
        trajectory = trajectory.to(device)
        if not (self.config["training"].get("train_with_sampling", False)):
            states_ig_nm, observations_nm, env_nm = self.initial_guess(
                states_gt_nm, observations_nm, env_nm
            )
            outputs = model(
                states_ig_nm, observations_nm, env_nm, device=device
            )  # List of dict with traj, score, energy
            loss_cycle = self.compute_cycle_loss(
                outputs, states_gt_nm, observations_nm, env_nm, extras
            )
            out = {
                "output": outputs[-1]["trajectory"],
                "target": trajectory,
                "loss": loss_cycle,
            }
        else:
            B, N, D = trajectory.shape
            C = self.config["model"]["diffusion_model"]["cond_dim"]
            # TODO: train without condition?, then train with the condition and constraints
            cond_ctx = torch.zeros((B, C), device=device)
            noise = torch.randn(B, N, D, device=device)
            pred = self.model.diffusion_model.sample(
                noise, ctx=cond_ctx, guidance_grad=None
            )
            loss = self.compute_loss(pred, trajectory)
            out = {
                "output": pred,
                "target": trajectory,
                "loss": loss,
            }
        return out

    # TODO: Modify plotting for all states
    def plotting(self, output, target, step, epoch, name="val"):
        state_slices = self.config["dataset"]["state_shapes"]
        c_sl = 0
        for key, sl in state_slices.items():
            batch_id = 0
            image_name = os.path.join(f"{name}_{key}", f"ep{epoch}_batch_{batch_id}")
            pred_slice = (
                output[batch_id, :, c_sl : (sl["shape"] + c_sl)].detach().cpu().numpy()
            )
            target_slice = (
                target[batch_id, :, c_sl : (sl["shape"] + c_sl)].detach().cpu().numpy()
            )
            img_np = self.plot_3d_trajectory(pred_slice, target_slice, title=image_name)
            self.writer.add_image(
                image_name, img_np, global_step=step, dataformats="HWC"
            )
            c_sl += sl["shape"]

    def compute_state_metrics(self, preds, targets):
        """Compute per-variable metrics."""
        metrics = {}
        state_slices = self.config["dataset"]["state_shapes"]
        c_sl = 0
        for name, sl in state_slices.items():
            pred_slice = preds[:, :, c_sl : (sl["shape"] + c_sl)]
            target_slice = targets[:, :, c_sl : (sl["shape"] + c_sl)]
            mse = torch.mean((pred_slice - target_slice) ** 2).item()
            mae = torch.mean(torch.abs(pred_slice - target_slice)).item()
            metrics[f"{name}_mse"] = mse
            metrics[f"{name}_mae"] = mae
            c_sl += sl["shape"]
        return metrics

    def save_checkpoint(self):
        epoch_step = self.epoch
        global_step = self.global_step
        checkpoint_path = os.path.join(
            os.getcwd(),
            self.config["training"]["output_dir"],
            self.config["name"],
            self.current_run_dir,
            self.config["training"]["checkpoint_path"],
        )
        payload = {
            "config": self.config,
            "state_dicts": dict(),
            "global_step": global_step,
            "epoch": epoch_step,
        }
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        payload["state_dicts"] = {
            "model_state": self.model.state_dict(),
            "model_state_diff": self.model.diffusion_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(
            payload,
            f"{checkpoint_path}/checkpoint_{epoch_step}.pth",
        )
