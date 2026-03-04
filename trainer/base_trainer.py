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
import tqdm
import yaml

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
        self.from_checkpoint = self.config["training"].get("from_ckp", False)
        self.current_run_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
        mp = self.config["training"].get("mixed_precision", "fp16")
        gas = int(self.config["training"].get("gradient_accumulation_steps", 1))
        self.accelerator = Accelerator(
            mixed_precision=mp,
            gradient_accumulation_steps=gas,
        )
        self.device = self.accelerator.device

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
        self.model, self.optimizer, self.train_loader, self.val_loader = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader
            )
        )

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
        """Must return dict with keys: loss, output, target (at minimum)."""
        raise NotImplementedError("val_step() must be implemented in subclasses")

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
        # Only main process writes log files; all procs can have console logs if you want,
        # but it’s usually cleaner to restrict to main.
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

    def setup_dataloader(self):
        dataset_dict = self.config["dataset"]
        train_dataset, validation_dataset = get_ds_from_cfg(dataset_dict)

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
            pin_memory=True,
        )

        self.val_loader = None
        if dataset_dict.get("val", False):
            self.val_loader = DataLoader(
                validation_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
                pin_memory=True,
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
            # If you step scheduler per optimizer step, num_training_steps should reflect that.
            # With gradient accumulation, "optimizer steps" are fewer than "batches".
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
        # NOTE: decide where ckpt_name comes from; here we keep your structure
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
        # load a checkpoint dict from the path specified in config
        checkpoint_path = os.path.join(
            cfg_global_p.LOGS_DIR,
            self.config["training"]["checkpoint_path"],
            self.config["training"]["ckpt_name"],
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location="cpu")

    def load_checkpoint_into_state(self):
        # IMPORTANT: load AFTER prepare(), and load into UNWRAPPED model
        ckpt = self.load_checkpoint_file()
        self.checkpoint = ckpt

        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.load_state_dict(ckpt["state_dicts"]["model_state"], strict=True)

        if (
            "optimizer_state" in ckpt["state_dicts"]
            and ckpt["state_dicts"]["optimizer_state"] is not None
        ):
            self.optimizer.load_state_dict(ckpt["state_dicts"]["optimizer_state"])

        # EMA (optional)
        if self.ema is not None and ckpt["state_dicts"].get("ema_state") is not None:
            # NOTE: depends on how your EMA class stores state
            self.ema.model.load_state_dict(
                ckpt["state_dicts"]["ema_state"], strict=False
            )

        # restore counters if you want
        self.epoch = int(ckpt.get("epoch", self.epoch))
        self.global_step = int(ckpt.get("global_step", self.global_step))

        if self.accelerator.is_main_process:
            self.logger.info(
                f"Loaded checkpoint: epoch={self.epoch}, global_step={self.global_step}"
            )

    def save_checkpoint(self):
        # sync before saving
        self.accelerator.wait_for_everyone()

        payload = {
            "config": self.config,
            "state_dicts": {
                "model_state": self.accelerator.unwrap_model(self.model).state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "ema_state": (
                    self.ema.model.state_dict() if self.ema is not None else None
                ),
            },
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
        self.model.train()

        func_train_step = (
            self.train_step_with_sampling
            if self.config["training"].get("train_with_sampling", False)
            else self.train_step
        )

        num_epochs = int(self.config["training"]["num_epochs"])
        tqdm_interval = float(self.config["training"].get("tqdm_interval_sec", 0.5))

        for _ in range(num_epochs):
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

                    # EMA should run after optimizer step; on all procs is fine
                    if self.ema is not None:
                        self.ema.step(self.model)

                    # step scheduler ONLY when we actually stepped optimizer
                    if self.scheduler is not None and self.accelerator.sync_gradients:
                        # ReduceLROnPlateau handled after val
                        if not isinstance(
                            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            self.scheduler.step()

                # gather loss for metrics (global mean across processes)
                loss_detached = loss.detach()
                loss_mean = (
                    self.accelerator.gather_for_metrics(loss_detached).mean().item()
                )

                total_loss_local += loss_mean

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss_mean)

                    if self.writer is not None:
                        self.writer.add_scalar(
                            "Loss/train_step", loss_mean, self.global_step
                        )

                self.global_step += 1

            # epoch-end logging
            avg_loss = total_loss_local / max(1, len(self.train_loader))

            if self.accelerator.is_main_process and self.writer is not None:
                self.writer.add_scalar("Loss/train", avg_loss, self.epoch)

            # Validation
            do_val = (self.epoch % int(self.config["training"]["val_every"]) == 0) and (
                self.val_loader is not None
            )
            if do_val:
                val_loss = self.validate()

                # ReduceLROnPlateau steps on val
                if self.scheduler is not None and isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_loss)

                if self.accelerator.is_main_process and self.writer is not None:
                    self.writer.add_scalar("Loss/val", val_loss, self.epoch)

            # Save checkpoint
            if self.epoch % int(self.config["training"]["save_every"]) == 0:
                self.save_checkpoint()

            # LR logging (only meaningful if scheduler exists)
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
        self.model.eval()

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

                if self.accelerator.is_main_process:
                    pbar.set_postfix(loss=loss_mean, refresh=False)

                preds = out_dict["output"]
                targets = out_dict["target"]

                # gather predictions/targets for global metrics
                preds_g = self.accelerator.gather_for_metrics(preds.detach())
                targets_g = self.accelerator.gather_for_metrics(targets.detach())
                all_preds.append(preds_g.cpu())
                all_targets.append(targets_g.cpu())

                # Plotting only on main process
                if (
                    self.config["training"].get("plotting", False)
                    and self.accelerator.is_main_process
                ):
                    self.plotting(out_dict, batch_idx, self.epoch)

        # compute metrics on gathered arrays (now global)
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = self.compute_state_metrics(all_preds, all_targets)
        if self.accelerator.is_main_process and self.writer is not None:
            for metric_name, value in metrics.items():
                self.writer.add_scalar(f"val/{metric_name}", value, self.epoch)

        self.model.train()
        return val_loss_sum / max(1, n_batches)

    # -------------------------
    # Your existing utilities
    # -------------------------
    def plotting(self, out_dict, batch_idx, epoch, name="val"):
        # NOTE: your original signature was weird (output, target, step, epoch)
        # Here we expect out_dict to contain output/target
        output = out_dict["output"]
        target = out_dict["target"]

        output_np = output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
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
        # Plot all dimensions in one go using transpose
        T = out_np.shape[0]
        x = np.arange(T)

        # Stack predictions and targets to distinguish lines
        ax.plot(
            x[:, None], out_np, "-", color="red", label="pred"
        )  # Dashed lines for predictions
        ax.plot(
            x[:, None], target_np, "--", color="blue", label="target"
        )  # Solid lines for targets

        ax.set_title(title)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        img_np = buf[:, :, :3]
        plt.close()
        return img_np

    @staticmethod
    def split_state_tensor(state_tensor: torch.Tensor, param_shapes: dict) -> dict:
        split_dict = {}
        start = 0
        for name in param_shapes.keys():
            end = start + param_shapes[name]["shape"]
            split_dict[name] = state_tensor[:, :, start:end].detach().cpu()
            start = end
        return split_dict

    def compute_state_metrics(self, preds, targets, ignore_keys=[""]):
        metrics = {}
        param_shapes = self.config["dataset"]["state_shapes"]
        start = 0
        for name in param_shapes.keys():
            if name not in ignore_keys:
                end = start + param_shapes[name]["shape"]
                pred_slice = preds[:, :, start:end]
                target_slice = targets[:, :, start:end]
                for axis_idx in range(pred_slice.shape[-1]):
                    mse_i = torch.mean(
                        (pred_slice[:, :, axis_idx] - target_slice[:, :, axis_idx]) ** 2
                    ).item()
                    mae_i = torch.mean(
                        torch.abs(
                            pred_slice[:, :, axis_idx] - target_slice[:, :, axis_idx]
                        )
                    ).item()
                    metrics[f"{name}_ax_{axis_idx}_mse"] = mse_i
                    metrics[f"{name}_ax_{axis_idx}_mae"] = mae_i
                mse = torch.mean((pred_slice - target_slice) ** 2).item()
                mae = torch.mean(torch.abs(pred_slice - target_slice)).item()
                metrics[f"{name}_mse"] = mse
                metrics[f"{name}_mae"] = mae
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
