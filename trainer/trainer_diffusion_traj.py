import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common.get_class import get_class_dict
from common.inpainting import apply_inpainting, make_inpainting_mask
from common.utils import split_trajectory
from dataset.utils import base_dataset
from models.utils.ema import EMA
from models.utils.encoders_robot import (
    ResNet18Encoder,
    HFVisionEncoder,
    freeze_all_parameters,
    unfreeze_resnet_last_n,
    unfreeze_hf_last_n,
)
from models.utils.guidance_robot import BaseGuidance, to_physical
from models.utils.noise_scheduler import build_noise_scheduler
from models.utils.robot_constraints import RobotConstraints
from trainer.base_trainer import BaseTrainer


class ObservationEncoder(nn.Module):
    """Encodes camera entries of an observation-history dict; other keys pass through."""

    def __init__(self, encoders: nn.ModuleDict):
        super().__init__()
        self.encoders = encoders

    def train(self, mode=True):
        """Only encoders with trainable parameters enter train mode: a frozen
        BatchNorm must keep its statistics, or features differ between training
        (batch statistics) and inference (running statistics)."""
        super().train(mode)
        for enc in self.encoders.values():
            if not any(p.requires_grad for p in enc.parameters()):
                enc.eval()
        return self

    def forward(self, obs_dict):
        encoded = {}
        for key, value in obs_dict.items():
            if key in self.encoders:
                # [B, L, C, H, W] -> [B, L, F]
                B, L, C, H, W = value.shape
                feat = self.encoders[key](value.reshape(B * L, C, H, W))
                if feat.ndim == 4:
                    feat = feat.flatten(1)
                encoded[key] = feat.reshape(B, L, -1).float()
            else:
                encoded[key] = value
        return encoded


def build_image_encoders(config, for_inference=False) -> ObservationEncoder:
    """Image encoders from config['model']['image_encoders'] (keyed by camera name)."""
    encoder_cfg = config["model"].get("image_encoders", {})
    encoder_dict = nn.ModuleDict()

    for key, cfg in encoder_cfg.items():
        enc_type = cfg["type"].lower()
        if enc_type == "resnet18":
            encoder = ResNet18Encoder(
                pretrained=cfg.get("pretrained", True),
                norm=cfg.get("norm", "batch"),
                pool=cfg.get("pool", "avg"),
                num_keypoints=cfg.get("num_keypoints", 32),
                center_crop=cfg.get("center_crop", True),
            )
        elif enc_type in {"dinov2", "hf", "transformer"}:
            encoder = HFVisionEncoder(model_name=cfg["name"], pool=cfg.get("pool", "cls"))
        elif enc_type == "none":
            encoder = nn.Identity()
        else:
            raise ValueError(f"Unsupported image encoder type: {enc_type}")

        if enc_type != "none":
            freeze_all_parameters(encoder)
            if cfg.get("trainable", False) and not for_inference:
                train_last_n = cfg.get("train_last_n", None)
                if train_last_n is None:
                    for p in encoder.parameters():
                        p.requires_grad_(True)
                elif enc_type == "resnet18":
                    unfreeze_resnet_last_n(encoder, n=train_last_n)
                else:
                    unfreeze_hf_last_n(encoder, n=train_last_n)
        encoder_dict[key] = encoder
    return ObservationEncoder(encoder_dict)


class DiffusionTrainer(BaseTrainer):
    """
    Goal- and history-conditioned trajectory diffusion for ManiSkill demos.

    batch = (states, goals, obs) from dataset.dataset_maniskill:
      states: future trajectory from t (diffused)        -> x0 [B, L, D]
      goals:  goal quantities (CFG condition)            -> cond [B, G]
      obs:    observation history (+ cameras) and mask   -> state_hist [B, H, F]
    """

    def __init__(self, config):
        self.num_inference_steps = config["model"].get("num_inference_steps", 100)
        self.goal_indices = config.get("goal_indices", None)
        self.start_indices = config.get("start_indices", None)
        self.inpainting = config.get("inpainting", True)
        self.ema_encoders = None
        self.constraints = None
        super().__init__(config)

    # ------------------------------------------------------------------ setup
    def prepare_all_accelerator(self):
        super().prepare_all_accelerator()
        # DDP refuses modules without trainable parameters (frozen encoders)
        if any(p.requires_grad for p in self.image_encoder_dict.parameters()):
            self.image_encoder_dict = self.accelerator.prepare(self.image_encoder_dict)

    def setup_model(self):
        model_cfg = self.config["model"]
        self.model = get_class_dict(model_cfg).to(self.device)
        self.guidance_algo: BaseGuidance = get_class_dict(model_cfg["guidance"])
        self.image_encoder_dict = build_image_encoders(self.config).to(self.device)
        if "constraints" in self.config:
            self.constraints = RobotConstraints(
                self.config["constraints"], self.config["dataset"]["state_shapes"]
            )

        if self.config["training"]["use_ema"]:
            self.ema_model = copy.deepcopy(self.model)
            self.ema = EMA(self.ema_model, self.config["ema"])
            # trainable encoders get their own EMA so sampling uses matching weights
            if any(p.requires_grad for p in self.image_encoder_dict.parameters()):
                self.ema_encoders = EMA(
                    copy.deepcopy(self.image_encoder_dict), self.config["ema"]
                )

    def setup_optimizer(self):
        training_config = self.config["training"]
        optimizer_type = training_config["optimizer"].lower()
        learning_rate = training_config["learning_rate"]
        encoder_lr = training_config.get("image_encoder_lr", learning_rate)

        param_groups = [
            {
                "params": [p for p in self.model.parameters() if p.requires_grad],
                "lr": learning_rate,
            }
        ]
        encoder_params = [
            p for p in self.image_encoder_dict.parameters() if p.requires_grad
        ]
        if encoder_params:
            param_groups.append({"params": encoder_params, "lr": encoder_lr})

        if optimizer_type == "adam":
            self.optimizer = optim.Adam(param_groups)
        elif optimizer_type == "sgd":
            self.optimizer = optim.SGD(param_groups)
        elif optimizer_type == "adamw":
            self.optimizer = optim.AdamW(param_groups)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def setup_scheduler(self):
        super().setup_scheduler()
        self.noise_scheduler = build_noise_scheduler(self.config["model"]["noise_scheduler"])

    def setup_normalizer(self):
        if "path_stats" not in self.config["dataset"]:
            raise ValueError("dataset.path_stats (normalizer statistics) is required")
        base = base_dataset(self.train_dataset)
        state_normalizer, goal_normalizer, obs_normalizer = (
            base.get_normalizers_from_file()
        )
        self.normalizer_state = state_normalizer
        self.normalizer_goal = goal_normalizer
        self.normalizer_obs = obs_normalizer
        self.normalizers = [state_normalizer, goal_normalizer, obs_normalizer]
        for normalizer in self.normalizers:
            normalizer.to(self.device)
            normalizer.set_device(self.device)

    def set_train_mode(self):
        self.model.train()
        self.image_encoder_dict.train()  # frozen encoders stay in eval mode

    def set_eval_mode(self):
        self.model.eval()
        self.image_encoder_dict.eval()

    def update_ema(self):
        super().update_ema()
        if self.ema_encoders is not None:
            self.ema_encoders.step(self.accelerator.unwrap_model(self.image_encoder_dict))

    # ------------------------------------------------------------------ checkpoint
    def checkpoint_state_dicts(self):
        sd = super().checkpoint_state_dicts()
        sd["image_encoders_state"] = self.accelerator.unwrap_model(
            self.image_encoder_dict
        ).state_dict()
        sd["ema_image_encoders_state"] = (
            self.ema_encoders.model.state_dict() if self.ema_encoders is not None else None
        )
        sd["normalizer"] = {
            "normalizer_state": self.normalizer_state.state_dict(),
            "normalizer_goal": self.normalizer_goal.state_dict(),
            "normalizer_obs": self.normalizer_obs.state_dict(),
        }
        return sd

    def load_checkpoint_into_state(self):
        super().load_checkpoint_into_state()
        sd = self.checkpoint["state_dicts"]
        if sd.get("image_encoders_state") is not None:
            self.accelerator.unwrap_model(self.image_encoder_dict).load_state_dict(
                sd["image_encoders_state"]
            )
        if self.ema_encoders is not None and sd.get("ema_image_encoders_state"):
            self.ema_encoders.model.load_state_dict(sd["ema_image_encoders_state"])

    # ------------------------------------------------------------------ batch
    def _prepare_batch(self, batch, use_ema_encoders=False):
        device = self.device
        states_dict, goals_dict, extra = batch
        extra = dict(extra)
        mask = extra.pop("mask", None)
        extra.pop("seq_len_L", None)

        states_norm = self.normalizer_state(states_dict)
        trajectory = torch.cat([states_norm[k] for k in states_norm], dim=2).to(device)

        goals_norm = self.normalizer_goal(goals_dict)
        goal = torch.cat(
            [v[:, 0, :] if v.ndim > 2 else v for v in goals_norm.values()], dim=1
        ).to(device)

        encoders = (
            self.ema_encoders.model
            if (use_ema_encoders and self.ema_encoders is not None)
            else self.image_encoder_dict
        )
        history = encoders(self.normalizer_obs(extra))
        history = torch.cat(
            [history[k].to(device).float() for k in history], dim=2
        )

        mask = mask.to(device).bool() if mask is not None else None
        mask_valid = mask if self.config["training"].get("mask_padding", True) else None
        return {
            "trajectory": trajectory.float(),
            "goal": goal.float(),
            "history": history,
            "mask": mask,
            "mask_valid": mask_valid,
            "states_raw": states_dict,
        }

    def _known_mask(self, trajectory):
        if not self.inpainting:
            return None
        return make_inpainting_mask(
            trajectory=trajectory,
            start_indices=self.start_indices,
            goal_indices=self.goal_indices,
        )

    def _diffusion_forward(self, prep):
        scheduler = self.noise_scheduler
        trajectory = prep["trajectory"]
        B = trajectory.shape[0]
        device = trajectory.device

        noise = torch.randn_like(trajectory)
        timesteps = torch.randint(
            0, scheduler.config.num_train_timesteps, (B,), device=device
        ).long()
        noisy = scheduler.add_noise(trajectory, noise, timesteps)
        known = self._known_mask(trajectory)
        if known is not None:
            noisy = apply_inpainting(noisy, known, x_known=trajectory, noise=False)

        # classifier-free dropout: the model swaps the goal for a learned null token
        cfg_p = float(self.config["model"].get("cfg_drop_prob", 0.0))
        cond_drop = torch.rand(B, device=device) < cfg_p if cfg_p > 0.0 else None

        pred = self.model(
            noisy,
            timesteps,
            prep["goal"],
            cond_drop=cond_drop,
            state_hist=prep["history"],
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
        return {
            "pred": pred,
            "target": target,
            "noisy": noisy,
            "timesteps": timesteps,
            "known": known,
        }

    def compute_loss(self, output, target, mask_valid=None, mask_known=None):
        """
        MSE over real (non-padded) steps and over entries that are not inpainted.
        output, target: [B, L, D]; mask_valid: [B, L] (True = real); mask_known: [B, L, D]
        """
        w = torch.ones_like(output, dtype=torch.float32)
        if mask_valid is not None:
            w = w * mask_valid.unsqueeze(-1).float()
        if mask_known is not None:
            w = w * (~mask_known.bool()).float()
        err = (output.float() - target.float()) ** 2
        return (err * w).sum() / w.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ steps
    def train_step(self, batch):
        prep = self._prepare_batch(batch)
        fwd = self._diffusion_forward(prep)
        loss = self.compute_loss(fwd["pred"], fwd["target"], prep["mask_valid"], fwd["known"])
        return {"loss": loss}

    def guidance_kwargs(self, state_hist, mask_known):
        return {
            "state_hist": state_hist,
            "mask_known": mask_known,
            "constraints": self.constraints,
            "normalizer_state": self.normalizer_state,
            "state_shapes": self.config["dataset"]["state_shapes"],
        }

    @torch.no_grad()
    def conditional_sample(
        self, trj, cond, use_ema=True, state_hist=None, mask_known=None, generator=None
    ):
        model = (
            self.ema_model
            if (use_ema and self.ema_model is not None)
            else self.accelerator.unwrap_model(self.model)
        )
        model.eval()
        scheduler = self.noise_scheduler
        scheduler.set_timesteps(self.num_inference_steps, device=trj.device)
        kwargs = self.guidance_kwargs(state_hist, mask_known)

        x = torch.randn(trj.shape, dtype=trj.dtype, device=trj.device, generator=generator)
        if mask_known is not None:
            x = apply_inpainting(x, mask_known, trj, noise=False)
        for k in scheduler.timesteps:
            x = self.guidance_algo(
                input_noised=x, k=k, cond=cond, scheduler=scheduler, model=model, **kwargs
            )
            if mask_known is not None:
                x = apply_inpainting(x, mask_known, trj, noise=False)
        x = self.guidance_algo.finalize(x, **kwargs)
        if mask_known is not None:
            x = apply_inpainting(x, mask_known, trj, noise=False)
        return x

    def val_step(self, batch):
        use_ema = self.ema_model is not None
        prep = self._prepare_batch(batch, use_ema_encoders=use_ema)
        trajectory = prep["trajectory"]
        known = self._known_mask(trajectory)
        output = self.conditional_sample(
            trajectory, prep["goal"], use_ema=use_ema, state_hist=prep["history"], mask_known=known
        )
        loss = self.compute_loss(output, trajectory, prep["mask_valid"])
        out = {
            "output": output,
            "target": trajectory,
            "loss": loss,
            "target_dict": prep["states_raw"],
            "mask": prep["mask"],
        }
        if self.constraints is not None:
            shapes = self.config["dataset"]["state_shapes"]
            pred_phys = to_physical(output, self.normalizer_state, shapes)
            gt_phys = {k: v.to(self.device).float() for k, v in prep["states_raw"].items()}
            metrics = {
                f"constraints_pred/{k}": v
                for k, v in self.constraints.metrics(pred_phys, prep["mask"]).items()
            }
            # residuals of the demonstrations: calibrates dt / tolerances
            metrics.update(
                {
                    f"constraints_gt/{k}": v
                    for k, v in self.constraints.metrics(gt_phys, prep["mask"]).items()
                }
            )
            out["metrics"] = metrics
        return out

    # ------------------------------------------------------------------ plotting
    def plotting(self, value_dict, step, epoch, name="val"):
        shapes = self.config["dataset"]["state_shapes"]
        with torch.no_grad():
            output_unnormalized = self.normalizer_state.unnormalize(
                split_trajectory(value_dict["output"].float(), shapes)
            )
        target = value_dict["target_dict"]
        mask = value_dict.get("mask")
        batch_id = np.random.randint(0, value_dict["output"].shape[0])
        seq_len = (
            int(mask[batch_id].sum().item())
            if mask is not None
            else value_dict["output"].shape[1]
        )
        for key in output_unnormalized.keys():
            output_np = output_unnormalized[key][batch_id, :seq_len].detach().cpu().numpy()
            target_np = target[key][batch_id, :seq_len].detach().cpu().numpy()

            if key == "tcp_pose":
                image_name = os.path.join(f"{name}_{key}", f"ep{epoch}")
                img_np = self.plot_3d_trajectory(
                    output_np[:, :3], target_np[:, :3], title=image_name
                )
                self.writer.add_image(
                    tag=image_name, global_step=step, img_tensor=img_np, dataformats="HWC"
                )
            for axis_idx in range(output_np.shape[-1]):
                image_name = os.path.join(f"{name}_{key}_ax_{axis_idx}", f"epoch_{epoch}")
                img_np = self.plot_1d_seq(
                    output_np[:, axis_idx], target_np[:, axis_idx], title=image_name
                )
                self.writer.add_image(
                    tag=image_name, global_step=step, img_tensor=img_np, dataformats="HWC"
                )


class DiffusionTrajectoryPhysTrainer(DiffusionTrainer):
    """
    Adds a physics-informed loss on the denoised estimate x0_hat:
        loss = loss_diffusion + lambda * sum_b w_b * E_constraints(x0_hat_b) / sum_b w_b
    with w_b = alpha_bar(t_b), so nearly pure-noise samples (meaningless x0_hat)
    contribute almost nothing. Requires a `constraints` section in the config.

    config:
      physics_loss:
        lambda: 0.1
    """

    def __init__(self, config):
        self.lambda_phys = float(config.get("physics_loss", {}).get("lambda", 0.1))
        super().__init__(config)
        if self.constraints is None:
            raise ValueError(f"{type(self).__name__} needs a `constraints` config section")

    def x0_from_prediction(self, pred, x_t, timesteps):
        scheduler = self.noise_scheduler
        a_bar = scheduler.alphas_cumprod.to(x_t.device)[timesteps].view(-1, 1, 1)
        sqrt_ab, sqrt_1mab = a_bar.sqrt(), (1.0 - a_bar).sqrt()
        pred_type = scheduler.config.prediction_type
        if pred_type == "epsilon":
            return (x_t - sqrt_1mab * pred) / sqrt_ab
        if pred_type == "v_prediction":
            return sqrt_ab * x_t - sqrt_1mab * pred
        if pred_type == "sample":
            return pred
        raise ValueError(f"Unsupported prediction type {pred_type}")

    def train_step(self, batch):
        prep = self._prepare_batch(batch)
        fwd = self._diffusion_forward(prep)
        loss_diff = self.compute_loss(
            fwd["pred"], fwd["target"], prep["mask_valid"], fwd["known"]
        )

        x0_hat = self.x0_from_prediction(
            fwd["pred"].float(), fwd["noisy"].float(), fwd["timesteps"]
        )
        if fwd["known"] is not None:
            x0_hat = torch.where(fwd["known"], prep["trajectory"], x0_hat)
        phys = to_physical(
            x0_hat, self.normalizer_state, self.config["dataset"]["state_shapes"]
        )
        # padded steps are never penalised, independent of mask_padding
        terms = self.constraints.energy(phys, prep["mask"])

        w = self.noise_scheduler.alphas_cumprod.to(x0_hat.device)[fwd["timesteps"]].float()
        loss_phys = (w * terms["total"]).sum() / w.sum().clamp_min(1e-8)
        loss = loss_diff + self.lambda_phys * loss_phys

        out = {"loss": loss, "diff": loss_diff.detach(), "phys": loss_phys.detach()}
        for name, value in terms.items():
            if name != "total":
                out[f"phys_{name}"] = value.mean().detach()
        return out


# Names used by existing configs
class DiffusionTrajectoryPadHistTrainer(DiffusionTrainer):
    pass


class DiffusionTrajectoryPadHistRSTrainer(DiffusionTrainer):
    pass


class DiffusionTrajectoryPadHPhyITrainer(DiffusionTrajectoryPhysTrainer):
    pass
