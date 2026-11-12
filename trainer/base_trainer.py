from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler as hf_get_scheduler
from accelerate import Accelerator
from accelerate.utils import set_seed
import tqdm
import yaml

from common.utils import normalized_width, split_state_tensor
from dataset.utils import get_ds_from_cfg
from models.utils.ema import EMA
from global_parameters import ConfigGlobalP

cfg_global_p = ConfigGlobalP()


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


class BaseTrainer:
    def __init__(self, config):
        self.config = config
        self._output_dir = None
        self._saving_thread = None
        self.normalizers = []

        self.train_dataset = None
        self.validation_dataset = None
        self.train_loader = None
        self.val_loader = None

        self.checkpoint = None
        self.scheduler = None
        self.from_checkpoint = self.config["training"].get("from_ckp", False)
        self.current_run_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
        mp = self.config["training"].get("mixed_precision", "fp16")
        gas = int(self.config["training"].get("gradient_accumulation_steps", 1))
        self.accelerator = Accelerator(
            mixed_precision=mp,
            gradient_accumulation_steps=gas,
        )
        self.device = self.accelerator.device
        set_seed(int(self.config["training"].get("seed", 0)))

        self.epoch = 0
        self.global_step = 0

        self.ema: EMA = None
        self.ema_model = None

        # Writer/logger are created only on main proc
        self.writer = None
        self.logger = None

        # ---- Build pieces (unprepared) ----
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_normalizer()

        # ---- Prepare with accelerator (model/opt/loaders) ----
        # IMPORTANT: scheduler is NOT prepared
        self.prepare_all_accelerator()

        # ---- After prepare: logging/writer only on main ----
        self.setup_logging()
        self.setup_writer()

        # ---- Load checkpoint AFTER prepare (so wrappers exist) ----
        if self.from_checkpoint:
            self.load_checkpoint_into_state()

        # quick sanity info
        if self.accelerator.is_main_process:
            self.logger.info(
                f"Accelerate: device={self.device}, "
                f"num_processes={self.accelerator.num_processes}, "
                f"mixed_precision={self.accelerator.mixed_precision}, "
                f"grad_accum={self.accelerator.gradient_accumulation_steps}"
            )

    # -------------------------
    # Abstract hooks
    # -------------------------
    def prepare_all_accelerator(self):
        self.model, self.optimizer, self.train_loader, self.val_loader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        )

    def setup_normalizer(self):
        raise NotImplementedError(
            "setup_normalizer() must be implemented in subclasses"
        )

    def setup_model(self):
        raise NotImplementedError("setup_model() must be implemented in subclasses")

    def train_step(self, batch):
        """Must return dict with key 'loss' at minimum."""
        raise NotImplementedError("train_step() must be implemented in subclasses")

    def train_step_with_sampling(self, batch):
        raise NotImplementedError(
            "train_step_with_sampling() must be implemented in subclasses"
        )

    def compute_loss(self, output, target):
        raise NotImplementedError("compute_loss() must be implemented in subclasses")

    def val_step(self, batch):
        """Must return dict with keys: loss, output, target (at minimum).
        Optional: 'metrics' (dict of python floats averaged over batches)."""
        raise NotImplementedError("val_step() must be implemented in subclasses")

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def update_ema(self):
        if self.ema is not None:
            self.ema.step(self.accelerator.unwrap_model(self.model))

    # -------------------------
    # Setup helpers
    # -------------------------
    def setup_writer(self):
        if not self.accelerator.is_main_process:
            return
        log_dir = os.path.join(
            cfg_global_p.LOGS_DIR,
            self.config["training"]["log_dir"],
            self.config["name"],
            self.current_run_dir,
        )
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def setup_logging(self):
        # Only main process writes log files
        self.logger = logging.getLogger(f"Trainer[{self.accelerator.process_index}]")
        self.logger.setLevel(logging.INFO)

        # Clear handlers to avoid duplicates in notebooks/restarts
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler: only main to avoid spam
        if self.accelerator.is_main_process:
            console_handler = TqdmLoggingHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File logging
            log_dir = os.path.join(
                cfg_global_p.LOGS_DIR,
                self.config["training"]["output_dir"],
                self.config["name"],
                self.current_run_dir,
                "files",
            )
            os.makedirs(log_dir, exist_ok=True)

            log_file = os.path.join(log_dir, "training.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # dump config
            config_output_path = os.path.join(log_dir, "config.yaml")
            with open(config_output_path, "w") as f:
                yaml.dump(self.config, f)

    def _loader_kwargs(self):
        num_workers = int(self.config["training"].get("num_workers", 4))
        return dict(
            batch_size=self.config["training"]["batch_size"],
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=num_workers > 0,
        )

    def setup_dataloader(self):
        dataset_dict = self.config["dataset"]
        train_dataset, validation_dataset = get_ds_from_cfg(dataset_dict)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.train_loader = DataLoader(
            train_dataset, shuffle=True, drop_last=True, **self._loader_kwargs()
        )
        self.val_loader = None
        if validation_dataset is not None:
            self.val_loader = DataLoader(
                validation_dataset, shuffle=False, **self._loader_kwargs()
            )

    def setup_optimizer(self):
        training_config = self.config["training"]
        optimizer_type = training_config["optimizer"]
        learning_rate = training_config["learning_rate"]

        params = self.model.parameters()
        if optimizer_type.lower() == "adam":
            self.optimizer = optim.Adam(params, lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            self.optimizer = optim.SGD(params, lr=learning_rate)
        elif optimizer_type.lower() == "adamw":
            self.optimizer = optim.AdamW(params, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    def setup_scheduler(self):
        scheduler_config = self.config["training"].get("scheduler", None)
        if scheduler_config is None:
            self.scheduler = None
            return

        name = scheduler_config["name"].lower()

        if name == "step":
            # will be stepped per-optimizer-step
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1),
            )
            return

        if name in [
            "linear",
            "cosine",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ]:
            # Scheduler is stepped per optimizer step; with gradient accumulation,
            # "optimizer steps" are fewer than "batches".
            steps_per_epoch = len(self.train_loader) // max(
                1, self.accelerator.gradient_accumulation_steps
            )
            total_steps = steps_per_epoch * int(self.config["training"]["num_epochs"])

            self.scheduler = hf_get_scheduler(
                name,
                optimizer=self.optimizer,
                num_warmup_steps=scheduler_config.get("num_warmup_steps", 0),
                num_training_steps=total_steps,
            )
            return

        raise ValueError(f"Unsupported scheduler type: {name}")

    # -------------------------
    # Checkpointing
    # -------------------------
    def _checkpoint_path(self, ckpt_name: str | None = None):
        ckpt_dir = os.path.join(
            cfg_global_p.LOGS_DIR,
            self.config["training"]["output_dir"],
            self.config["name"],
            self.current_run_dir,
            self.config["training"]["checkpoint_path"],
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        if ckpt_name is None:
            ckpt_name = f"checkpoint_{self.epoch}.pth"
        return os.path.join(ckpt_dir, ckpt_name)

    def load_checkpoint_file(self):
        # ckpt_name may be absolute or relative to LOGS_DIR/checkpoint_path
        ckpt_name = self.config["training"]["ckpt_name"]
        checkpoint_path = (
            ckpt_name
            if os.path.isabs(ckpt_name)
            else os.path.join(
                cfg_global_p.LOGS_DIR,
                self.config["training"]["checkpoint_path"],
                ckpt_name,
            )
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    def load_checkpoint_into_state(self):
        # IMPORTANT: load AFTER prepare(), and load into UNWRAPPED model
        ckpt = self.load_checkpoint_file()
        self.checkpoint = ckpt
        sd = ckpt["state_dicts"]

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(sd["model_state"], strict=True)

        if sd.get("optimizer_state") is not None:
            self.optimizer.load_state_dict(sd["optimizer_state"])
        if self.ema is not None and sd.get("ema_state") is not None:
            self.ema.model.load_state_dict(sd["ema_state"], strict=True)
            self.ema.optimization_step = int(ckpt.get("global_step", 0))
        if self.scheduler is not None and sd.get("scheduler_state") is not None:
            self.scheduler.load_state_dict(sd["scheduler_state"])

        # resume after the saved epoch
        self.epoch = int(ckpt.get("epoch", -1)) + 1
        self.global_step = int(ckpt.get("global_step", self.global_step))

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Loaded checkpoint: epoch={self.epoch}, global_step={self.global_step}"
            )

    def checkpoint_state_dicts(self):
        return {
            "model_state": self.accelerator.unwrap_model(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "ema_state": (self.ema.model.state_dict() if self.ema is not None else None),
            "scheduler_state": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
        }

    def save_checkpoint(self):
        # sync before saving
        self.accelerator.wait_for_everyone()

        payload = {
            "config": self.config,
            "state_dicts": self.checkpoint_state_dicts(),
            "global_step": int(self.global_step),
            "epoch": int(self.epoch),
        }

        path = self._checkpoint_path()
        # accelerator.save is safe in distributed; still only write once
        if self.accelerator.is_main_process:
            self.accelerator.save(payload, path)
            self.logger.info(f"Saved checkpoint: {path}")

        self.accelerator.wait_for_everyone()

    # -------------------------
    # Training / Validation
    # -------------------------
    def train(self):
        self.set_train_mode()

        func_train_step = (
            self.train_step_with_sampling
            if self.config["training"].get("train_with_sampling", False)
            else self.train_step
        )

        num_epochs = int(self.config["training"]["num_epochs"])
        tqdm_interval = float(self.config["training"].get("tqdm_interval_sec", 0.5))
        val_every = int(self.config["training"]["val_every"])
        save_every = int(self.config["training"]["save_every"])

        while self.epoch < num_epochs:
            # only main shows progress bar
            pbar = tqdm.tqdm(
                self.train_loader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=tqdm_interval,
                disable=not self.accelerator.is_main_process,
            )

            total_loss_local = 0.0

            for batch in pbar:
                with self.accelerator.accumulate(self.model):
                    out = func_train_step(batch)
                    loss = out["loss"]

                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

                # EMA and LR schedule advance once per real optimizer step
                if self.accelerator.sync_gradients:
                    self.update_ema()
                    if self.scheduler is not None and not isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step()

                # gather loss for metrics (global mean across processes)
                loss_mean = (
                    self.accelerator.gather_for_metrics(loss.detach()).mean().item()
                )
                total_loss_local += loss_mean

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss_mean)
                    if self.writer is not None:
                        self.writer.add_scalar(
                            "Loss/train_step", loss_mean, self.global_step
                        )
                        for key, value in out.items():
                            if key != "loss" and torch.is_tensor(value) and value.ndim == 0:
                                self.writer.add_scalar(
                                    f"Loss/train_{key}", value.item(), self.global_step
                                )

                self.global_step += 1

            # epoch-end logging
            avg_loss = total_loss_local / max(1, len(self.train_loader))

            if self.accelerator.is_main_process and self.writer is not None:
                self.writer.add_scalar("Loss/train", avg_loss, self.epoch)

            # Validation
            if self.val_loader is not None and self.epoch % val_every == 0:
                val_loss = self.validate()

                # ReduceLROnPlateau steps on val
                if self.scheduler is not None and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)

                if self.accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("Loss/val", val_loss, self.epoch)

            # Save checkpoint (always keep the last epoch)
            if self.epoch % save_every == 0 or self.epoch == num_epochs - 1:
                self.save_checkpoint()

            if (
                self.scheduler is not None
                and self.accelerator.is_main_process
                and self.writer is not None
            ):
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.writer.add_scalar("LearningRate", current_lr, self.global_step)

            self.epoch += 1

        if self.accelerator.is_main_process and self.writer is not None:
            self.writer.close()

    def validate(self):
        self.set_eval_mode()

        tqdm_interval = float(self.config["training"].get("tqdm_interval_sec", 0.5))
        pbar = tqdm.tqdm(
            self.val_loader,
            desc=f"Validating {self.epoch}",
            leave=False,
            mininterval=tqdm_interval,
            disable=not self.accelerator.is_main_process,
        )

        val_loss_sum = 0.0
        n_batches = 0
        extra_metrics = {}

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                out_dict = self.val_step(batch)
                loss = out_dict["loss"]

                # global mean loss across processes
                loss_mean = (
                    self.accelerator.gather_for_metrics(loss.detach()).mean().item()
                )
                val_loss_sum += loss_mean
                n_batches += 1

                for key, value in out_dict.get("metrics", {}).items():
                    extra_metrics[key] = extra_metrics.get(key, 0.0) + float(value)

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss_mean, refresh=False)

                preds = out_dict["output"]
                targets = out_dict["target"]

                # gather predictions/targets for global metrics
                preds_g = self.accelerator.gather_for_metrics(preds.detach())
                targets_g = self.accelerator.gather_for_metrics(targets.detach())
                all_preds.append(preds_g.float().cpu())
                all_targets.append(targets_g.float().cpu())

                # Plot one batch per validation on the main process
                if (
                    batch_idx == 0
                    and self.config["training"].get("plotting", False)
                    and self.accelerator.is_main_process
                ):
                    self.plotting(out_dict, batch_idx, self.epoch)

        # compute metrics on gathered arrays (now global)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = self.compute_state_metrics(all_preds, all_targets)
        metrics.update({k: v / max(1, n_batches) for k, v in extra_metrics.items()})
        if self.accelerator.is_main_process and self.writer is not None:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f"val/{metric_name}", value, self.epoch)

        self.set_train_mode()
        return val_loss_sum / max(1, n_batches)

    # -------------------------
    # Utilities
    # -------------------------
    def plotting(self, out_dict, batch_idx, epoch, name="val"):
        output = out_dict["output"]
        target = out_dict["target"]

        output_np = output.detach().float().cpu().numpy()
        target_np = target.detach().float().cpu().numpy()
        for i, (trj_out, trj_true) in enumerate(zip(output_np, target_np)):
            if i == 5:
                break
            image_name = os.path.join("val_dir", "trajectory")
            img_np = self.plot_3d_trajectory(trj_out, trj_true, title=image_name)
            if self.writer is not None:
                self.writer.add_image(
                    image_name, img_np, global_step=i, dataformats="HWC"
                )

    @staticmethod
    def plot_1d_seq(out, target, title):
        """
        Vectorized plot of predicted vs. target sequence over time.

        Args:
            out: [T, D] torch.Tensor or np.ndarray
            target: [T, D] same shape as `out`
        Returns:
            img_np: [H, W, 3] RGB image for TensorBoard
        """
        out_np = out.detach().cpu().numpy() if hasattr(out, "detach") else np.array(out)
        target_np = (
            target.detach().cpu().numpy()
            if hasattr(target, "detach")
            else np.array(target)
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        T = out_np.shape[0]
        x = np.arange(T)

        ax.plot(x[:, None], out_np, "-", color="red", label="pred")
        ax.plot(x[:, None], target_np, "--", color="blue", label="target")

        ax.set_title(title)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_np = buf[:, :, :3]
        plt.close(fig)
        return img_np

    @staticmethod
    def split_state_tensor(state_tensor: torch.Tensor, param_shapes: dict) -> dict:
        return split_state_tensor(state_tensor, param_shapes)

    def compute_state_metrics(self, preds, targets, ignore_keys=()):
        """Per-key / per-channel MSE and MAE in normalized space."""
        metrics = {}
        param_shapes = self.config["dataset"]["state_shapes"]
        start = 0
        for name, spec in param_shapes.items():
            end = start + normalized_width(spec)
            if name not in ignore_keys:
                pred_slice = preds[:, :, start:end]
                target_slice = targets[:, :, start:end]
                for axis_idx in range(pred_slice.shape[-1]):
                    diff = pred_slice[:, :, axis_idx] - target_slice[:, :, axis_idx]
                    metrics[f"{name}_ax_{axis_idx}_mse"] = torch.mean(diff**2).item()
                    metrics[f"{name}_ax_{axis_idx}_mae"] = torch.mean(diff.abs()).item()
                metrics[f"{name}_mse"] = torch.mean((pred_slice - target_slice) ** 2).item()
                metrics[f"{name}_mae"] = torch.mean(
                    torch.abs(pred_slice - target_slice)
                ).item()
            start = end
        return metrics

    @staticmethod
    def plot_3d_trajectory(trj_out, trj_true, title):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(
            trj_out[:, 0],
            trj_out[:, 1],
            trj_out[:, 2],
            "-",
            color="red",
            label="Output",
            linewidth=2,
        )
        ax.plot(
            trj_true[:, 0],
            trj_true[:, 1],
            trj_true[:, 2],
            "--",
            color="blue",
            label="Target",
            linewidth=2,
        )

        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        ax.set_zlabel("Z", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.legend()

        ax.view_init(elev=20, azim=120)
        plt.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_np = buf[:, :, :3]

        plt.close(fig)
        return img_np
