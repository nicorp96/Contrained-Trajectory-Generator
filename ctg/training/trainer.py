"""
Training of a goal- and history-conditioned plan denoiser on demonstrations.

Everything is built from the Hydra config: dataset and split (cfg.data), camera
encoders (cfg.encoder), denoiser (cfg.model), noise scheduler and sampling
(cfg.diffusion), physical constraints (cfg.physics) and optimization (cfg.training).
Widths of plan, goal and history are derived from the data layout and the encoders.

Loss: MSE between prediction and target over real (not padded) and not inpainted
plan entries, plus optionally physics.loss_weight * constraint energy of the denoised
estimate. The energy only counts for low-noise samples (k <= physics.loss_start * T),
weighted by alpha_bar(k): at high noise the estimate is necessarily poor, and limits on
velocities and accelerations magnify its errors by 1/dt and 1/dt^2.
"""

import copy
import logging
import os
from typing import Dict, Optional

import torch
import tqdm
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ctg.data.layout import concat, split, total_width
from ctg.data.normalization import LayoutNormalizer, PlanCodec, load_stats_file
from ctg.data.splits import make_splits
from ctg.diffusion.inpainting import known_mask
from ctg.diffusion.sampler import DiffusionSampler
from ctg.diffusion.scheduler import predicted_x0, training_target
from ctg.models.encoders import CameraEncoders
from ctg.training.ema import EMA

log = logging.getLogger(__name__)


def channel_indices(layout, fields) -> list:
    """Plan channels of the given fields (for inpainting)."""
    out, start = [], 0
    for name, spec in layout.items():
        if name in (fields or ()):
            out.extend(range(start, start + spec.width))
        start += spec.width
    return out


def build_components(cfg: DictConfig, dataset, pretrained_encoders: Optional[bool] = None) -> Dict:
    """Normalizers, encoders, denoiser, scheduler, sampler and constraints for a dataset layout.
    Shared by the trainer and the policy, so both build identical networks.
    pretrained_encoders: None keeps cfg.encoder.pretrained; the policy passes False because
    its weights come from the checkpoint."""
    plan, goal, history = dataset.plan_layout, dataset.goal_layout, dataset.history_layout
    normalizers = {"plan": LayoutNormalizer(plan), "goal": LayoutNormalizer(goal),
                   "history": LayoutNormalizer(history, dataset.cameras)}
    encoder = None
    if dataset.cameras:
        overrides = {} if pretrained_encoders is None else {"pretrained": pretrained_encoders}
        encoder = instantiate(cfg.encoder, **overrides)
    encoders = CameraEncoders(dataset.cameras, encoder)
    history_dim = total_width(history) + sum(encoders.feature_dims.values())
    model = instantiate(cfg.model, plan_dim=total_width(plan), goal_dim=total_width(goal),
                        history_dim=history_dim, history_len=dataset.history_len)
    noise_scheduler = instantiate(cfg.diffusion.scheduler)
    constraints = instantiate(cfg.physics.constraints) if cfg.physics.get("constraints") else None
    sampler = DiffusionSampler(
        noise_scheduler, cfg.diffusion.num_inference_steps, cfg.diffusion.guidance_scale, constraints,
        cfg.physics.guidance_eta, cfg.physics.guidance_grad_clip, cfg.physics.guidance_start, cfg.physics.project_final,
    )
    return dict(normalizers=normalizers, encoders=encoders, model=model, noise_scheduler=noise_scheduler,
                constraints=constraints, sampler=sampler, codec=PlanCodec(plan, normalizers["plan"]))


class Trainer:
    def __init__(self, cfg: DictConfig, run_dir: str):
        self.cfg, self.run_dir = cfg, run_dir
        t = cfg.training
        self.accelerator = Accelerator(mixed_precision=t.mixed_precision, gradient_accumulation_steps=t.gradient_accumulation_steps)
        self.device = self.accelerator.device
        set_seed(t.seed)

        dataset = instantiate(cfg.data.dataset)
        self.dataset = dataset
        self.train_set, self.val_set = make_splits(dataset, cfg.data.val_ratio, cfg.data.split_seed, cfg.data.samples_per_episode)
        parts = build_components(cfg, dataset)
        stats = load_stats_file(cfg.data.stats)
        for norm in parts["normalizers"].values():
            norm.load_stats(stats)
            norm.to(self.device)
        self.normalizers, self.codec = parts["normalizers"], parts["codec"]
        self.encoders, self.model = parts["encoders"].to(self.device), parts["model"].to(self.device)
        self.noise_scheduler, self.constraints, self.sampler = parts["noise_scheduler"], parts["constraints"], parts["sampler"]
        self.physics_weight = float(cfg.physics.loss_weight) if self.constraints is not None else 0.0
        self.physics_max_step = cfg.physics.loss_start * self.noise_scheduler.config.num_train_timesteps
        self.start_dims = channel_indices(dataset.plan_layout, cfg.diffusion.inpainting.start_fields)
        self.goal_dims = channel_indices(dataset.plan_layout, cfg.diffusion.inpainting.goal_fields)

        ema_args = OmegaConf.to_container(t.ema)
        self.ema = EMA(copy.deepcopy(self.model), **ema_args)
        self.ema_encoders = EMA(copy.deepcopy(self.encoders), **ema_args) if self.encoders.is_trainable else None

        groups = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": t.learning_rate}]
        if self.encoders.is_trainable:
            groups.append({"params": [p for p in self.encoders.parameters() if p.requires_grad], "lr": t.encoder_learning_rate})
        self.optimizer = torch.optim.AdamW(groups, weight_decay=t.weight_decay)

        loader_args = dict(batch_size=t.batch_size, num_workers=t.num_workers, pin_memory=True,
                           persistent_workers=t.num_workers > 0)
        train_loader = DataLoader(self.train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(self.val_set, shuffle=False, **loader_args) if self.val_set is not None else None
        steps_per_epoch = len(train_loader) // t.gradient_accumulation_steps
        self.lr_scheduler = get_scheduler(t.lr_schedule, self.optimizer, num_warmup_steps=t.warmup_steps,
                                          num_training_steps=steps_per_epoch * t.epochs)

        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(self.model, self.optimizer, train_loader)
        self.val_loader = self.accelerator.prepare(val_loader) if val_loader is not None else None
        if self.encoders.is_trainable:  # DDP refuses modules without trainable parameters
            self.encoders = self.accelerator.prepare(self.encoders)

        self.epoch, self.global_step = 0, 0
        if t.get("resume"):
            self.load_checkpoint(t.resume)
        self.writer = SummaryWriter(os.path.join(run_dir, "tensorboard")) if self.accelerator.is_main_process else None
        log.info("plan %s | goal %s | history %s + cameras %s | %d train / %d val episodes",
                 list(dataset.plan_layout), list(dataset.goal_layout), list(dataset.history_layout),
                 list(dataset.cameras), len(self.train_set), len(self.val_set or []))

    # ------------------------------------------------------------------ batches
    def _prepare(self, batch, use_ema: bool = False) -> Dict[str, torch.Tensor]:
        plan = self.codec.encode({k: v.to(self.device) for k, v in batch["plan"].items()}).float()
        goal = concat(self.normalizers["goal"]({k: v.to(self.device) for k, v in batch["goal"].items()}),
                      self.dataset.goal_layout).float()
        encoders = self.ema_encoders.model if (use_ema and self.ema_encoders is not None) else self.encoders
        hist = encoders(self.normalizers["history"]({k: v.to(self.device) for k, v in batch["history"].items()}))
        keys = list(self.dataset.history_layout) + list(self.dataset.cameras)
        history = torch.cat([hist[k].float() for k in keys], dim=-1)
        mask = batch["mask"].to(self.device).bool()
        return {"plan": plan, "goal": goal, "history": history, "mask": mask, "time": batch["time"].to(self.device),
                "loss_mask": mask if self.cfg.training.mask_padding else None}

    def _known(self, plan: torch.Tensor) -> Optional[torch.Tensor]:
        return known_mask(plan.shape, self.start_dims, self.goal_dims, device=plan.device)

    @staticmethod
    def masked_mse(pred, target, loss_mask=None, known=None) -> torch.Tensor:
        """MSE over real (loss_mask) and not inpainted (known) entries."""
        w = torch.ones_like(pred, dtype=torch.float32)
        if loss_mask is not None:
            w = w * loss_mask.unsqueeze(-1).float()
        if known is not None:
            w = w * (~known).float()
        return (((pred.float() - target.float()) ** 2) * w).sum() / w.sum().clamp_min(1.0)

    # ------------------------------------------------------------------ steps
    def train_step(self, batch) -> Dict[str, torch.Tensor]:
        b = self._prepare(batch)
        x0, B = b["plan"], b["plan"].shape[0]
        noise = torch.randn_like(x0)
        k = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (B,), device=self.device).long()
        known = self._known(x0)
        x_t = torch.where(known, x0, self.noise_scheduler.add_noise(x0, noise, k)) if known is not None \
            else self.noise_scheduler.add_noise(x0, noise, k)
        p = self.cfg.diffusion.goal_drop_prob
        goal_drop = torch.rand(B, device=self.device) < p if p > 0 else None
        pred = self.model(x_t, k, b["goal"], b["history"], goal_drop=goal_drop)
        loss = self.masked_mse(pred, training_target(self.noise_scheduler, x0, noise, k), b["loss_mask"], known)
        out = {"loss": loss, "diffusion": loss.detach()}
        if self.physics_weight > 0:
            x0_hat = predicted_x0(self.noise_scheduler, pred.float(), x_t.float(), k)
            if known is not None:
                x0_hat = torch.where(known, x0, x0_hat)
            terms = self.constraints.energy(self.codec.decode(x0_hat), b["mask"], b["time"])  # padding never penalised
            w = self.noise_scheduler.alphas_cumprod.to(self.device)[k].float() * (k <= self.physics_max_step).float()
            weighted = {n: (w * v).sum() / w.sum().clamp_min(1e-8) for n, v in terms.items()}
            out.update({"loss": loss + self.physics_weight * weighted["total"], "physics": weighted["total"].detach()},
                       **{f"physics_{n}": v.detach() for n, v in weighted.items() if n != "total"})
        return out

    def train(self):
        t = self.cfg.training
        while self.epoch < t.epochs:
            self.model.train()
            self.encoders.train()
            total = 0.0
            for batch in tqdm.tqdm(self.train_loader, desc=f"epoch {self.epoch}", leave=False,
                                   mininterval=t.progress_interval, disable=not self.accelerator.is_main_process):
                with self.accelerator.accumulate(self.model):
                    out = self.train_step(batch)
                    self.accelerator.backward(out["loss"])
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                if self.accelerator.sync_gradients:  # EMA and LR schedule advance once per optimizer step
                    self.ema.step(self.accelerator.unwrap_model(self.model))
                    if self.ema_encoders is not None:
                        self.ema_encoders.step(self.accelerator.unwrap_model(self.encoders))
                    self.lr_scheduler.step()
                loss = self.accelerator.reduce(out["loss"].detach(), reduction="mean").item()
                total += loss
                if self.writer is not None:
                    for name, value in out.items():
                        self.writer.add_scalar(f"train/{name}", float(value.detach()), self.global_step)
                self.global_step += 1
            if self.writer is not None:
                self.writer.add_scalar("train/epoch_loss", total / max(1, len(self.train_loader)), self.epoch)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.epoch)
            if self.val_loader is not None and self.epoch % t.val_every == 0:
                for name, value in self.validate().items():
                    if self.writer is not None:
                        self.writer.add_scalar(f"val/{name}", value, self.epoch)
            if self.epoch % t.save_every == 0 or self.epoch == t.epochs - 1:
                self.save_checkpoint()
            self.epoch += 1
        if self.writer is not None:
            self.writer.close()

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Sampled plans (EMA weights) vs. demonstrations: loss, per-field MSE, constraint metrics."""
        self.encoders.eval()
        sums, n = {}, 0
        for batch in self.val_loader:
            b = self._prepare(batch, use_ema=True)
            known = self._known(b["plan"])
            plan = self.sampler.sample(self.ema.model, b["plan"].shape, b["goal"], b["history"], self.codec,
                                       known, b["plan"], b["time"])
            metrics = {"loss": self.masked_mse(plan, b["plan"], b["loss_mask"]).item()}
            pred, true = split(plan, self.dataset.plan_layout), split(b["plan"], self.dataset.plan_layout)
            for name in pred:
                metrics[f"{name}_mse"] = torch.mean((pred[name] - true[name]) ** 2).item()
            if self.constraints is not None:
                for prefix, x in (("constraints_pred", self.codec.decode(plan)),
                                  ("constraints_demo", {k: v.to(self.device).float() for k, v in batch["plan"].items()})):
                    metrics.update({f"{prefix}/{k}": v for k, v in self.constraints.metrics(x, b["mask"], b["time"]).items()})
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + value
            n += 1
        return {k: v / max(1, n) for k, v in sums.items()}

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            unwrap = self.accelerator.unwrap_model
            payload = {
                "config": OmegaConf.to_container(self.cfg, resolve=True),
                "epoch": self.epoch, "global_step": self.global_step,
                "model": unwrap(self.model).state_dict(), "ema_model": self.ema.model.state_dict(),
                "encoders": unwrap(self.encoders).state_dict(),
                "ema_encoders": self.ema_encoders.model.state_dict() if self.ema_encoders is not None else None,
                "normalizers": {k: v.state_dict() for k, v in self.normalizers.items()},
                "optimizer": self.optimizer.state_dict(), "lr_scheduler": self.lr_scheduler.state_dict(),
            }
            path = os.path.join(self.run_dir, "checkpoints", f"epoch_{self.epoch:05d}.pt")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            self.accelerator.save(payload, path)
            log.info("saved %s", path)
        self.accelerator.wait_for_everyone()

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        unwrap = self.accelerator.unwrap_model
        unwrap(self.model).load_state_dict(ckpt["model"])
        self.ema.model.load_state_dict(ckpt["ema_model"])
        unwrap(self.encoders).load_state_dict(ckpt["encoders"])
        if self.ema_encoders is not None and ckpt.get("ema_encoders") is not None:
            self.ema_encoders.model.load_state_dict(ckpt["ema_encoders"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        self.epoch, self.global_step = ckpt["epoch"] + 1, ckpt["global_step"]
        for ema in (self.ema, self.ema_encoders):
            if ema is not None:
                ema.optimization_step = ckpt["global_step"]
        log.info("resumed from %s at epoch %d", path, self.epoch)
