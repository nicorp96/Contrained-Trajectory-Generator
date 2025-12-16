from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    get_scheduler as hf_get_scheduler,
)  # If using Hugging Face schedulers
import tqdm
import yaml

from dataset.utils import get_ds_from_cfg
from models.utils.ema import EMA

torch.autograd.set_detect_anomaly(True)


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
        self.normalizers = [None]
        self.train_dataset = None
        self.validation_dataset = None
        self.checkpoint = None
        self.from_checkpoint = self.config["training"]["from_ckp"]
        self.current_run_dir = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.device = self.config["training"]["device"]
        self.epoch = 0
        self.global_step = 0
        self.ema: EMA = None
        self.ema_model = None
        if self.from_checkpoint:
            self.load_checkpoint()
        log_dir = os.path.join(
            os.getcwd(),
            self.config["training"]["log_dir"],
            self.config["name"],
            self.current_run_dir,
        )
        self.writer = SummaryWriter(log_dir=log_dir)
        self.setup_model()
        self.setup_dataloader()
        self.setup_optimizer()
        self.setup_logging()
        self.setup_scheduler()
        self.setup_normalizer()

    def setup_normalizer(self):
        raise NotImplementedError("setup_model() must be implemented in subclasses")

    def setup_model(self):
        raise NotImplementedError("setup_model() must be implemented in subclasses")

    def load_checkpoint(self):
        checkpoint_path = os.path.join(
            os.getcwd(),
            self.config["training"]["checkpoint_path"],
            self.config["training"]["ckpt_name"],
        )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # self.epoch = self.checkpoint["epoch"]
        # self.global_step = self.checkpoint["global_step"]
        # self.config = self.checkpoint["config"]
        # self.current_run_dir = self.config["training"]["ckpt_name"].split("/")[0]
        print(f"Loaded checkpoint from step {self.epoch}")

    def setup_logging(self):
        # Set up logging to both console and file
        log_dir = os.path.join(
            self.config["training"]["output_dir"],
            self.config["name"],
            self.current_run_dir,
            "files",
        )
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger("Trainer")
        self.logger.setLevel(logging.INFO)
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        # Console: use TQDM-safe logging
        console_handler = TqdmLoggingHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File logging (normal)
        log_file = os.path.join(log_dir, "training.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        config_output_path = os.path.join(
            log_dir,
            "config.yaml",
        )
        with open(config_output_path, "w") as f:
            yaml.dump(self.config, f)

    def setup_optimizer(self):
        training_config = self.config["training"]
        optimizer_type = training_config["optimizer"]
        learning_rate = training_config["learning_rate"]
        params = self.model.parameters()

        if optimizer_type.lower() == "adam":
            optimizer = optim.Adam(params, lr=learning_rate)
        elif optimizer_type.lower() == "sgd":
            optimizer = optim.SGD(params, lr=learning_rate)
        elif optimizer_type.lower() == "adamw":
            optimizer = optim.AdamW(params, lr=learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        self.optimizer = optimizer
        if self.from_checkpoint:
            self.optimizer.load_state_dict(
                self.checkpoint["state_dicts"]["optimizer_state"]
            )

    def setup_dataloader(self):
        dataset_dict = self.config["dataset"]
        train_dataset, validation_dataset = get_ds_from_cfg(dataset_dict)
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["training"]["batch_size"],
            shuffle=True,
        )
        self.val_loader = None
        if dataset_dict["val"]:
            self.val_loader = DataLoader(
                validation_dataset,
                batch_size=self.config["training"]["batch_size"],
                shuffle=False,
            )

    def setup_scheduler(self):
        scheduler_config = self.config["training"].get("scheduler", None)
        if scheduler_config is None:
            self.scheduler = None
            return

        name = scheduler_config["name"]

        if name.lower() == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get("step_size", 10),
                gamma=scheduler_config.get("gamma", 0.1),
            )
        elif name.lower() in [
            "linear",
            "cosine",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ]:
            self.scheduler = hf_get_scheduler(
                name,
                optimizer=self.optimizer,
                num_warmup_steps=scheduler_config.get("num_warmup_steps", 0),
                num_training_steps=len(self.train_loader)
                * self.config["training"][
                    "num_epochs"
                ],  # // self.config["training"].get("gradient_accumulation_steps", 1)
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {name}")

    def train_step(self, batch):
        raise NotImplementedError("train_step() must be implemented in subclasses")

    def train_step_with_sampling(self, batch):
        pass

    def compute_loss(self, output, target):
        raise NotImplementedError("compute_loss() must be implemented in subclasses")

    def val_step(self, batch):
        raise NotImplementedError("train_step() must be implemented in subclasses")

    def save_net_graph(self, batch):
        self.writer.add_graph(self.model, batch)

    def train(self):
        self.model.train()
        func_train_step = (
            self.train_step_with_sampling
            if self.config["training"].get("train_with_sampling", False)
            else self.train_step
        )
        for local_epoch_idx in range(self.config["training"]["num_epochs"]):
            with tqdm.tqdm(
                self.train_loader,
                desc=f"Training epoch {self.epoch}",
                leave=False,
                mininterval=self.config["training"]["tqdm_interval_sec"],
            ) as t_epoch:
                total_loss = 0
                for batch_idx, batch in enumerate(t_epoch):
                    train_step_dict = func_train_step(batch)
                    loss = train_step_dict["loss"]
                    # Backpropagation
                    raw_loss_cpu = loss.item()
                    t_epoch.set_postfix(loss=raw_loss_cpu)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.ema is not None:
                        self.ema.step(self.model)
                    if self.scheduler is not None:
                        if isinstance(
                            self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                        ):
                            # ReduceLROnPlateau needs validation loss at end of epoch
                            pass
                        else:
                            self.scheduler.step()
                    self.writer.add_scalar(
                        "Loss/train_step", raw_loss_cpu, self.global_step
                    )
                    self.global_step += 1
                    total_loss += raw_loss_cpu
                    # self.logger.info(
                    #     f"|Epoch {self.epoch}|----Training Loss: {raw_loss_cpu}---- LR: {self.optimizer.param_groups[0]['lr']} ---- Global Step: {self.global_step} ---"
                    # )

                # Log loss to TensorBoard
                average_loss = (
                    total_loss / len(self.train_loader)
                    if len(self.train_loader) > 0
                    else 0.0
                )
                self.writer.add_scalar("Loss/train", average_loss, self.epoch)

                # Validate and log validation loss to TensorBoard
                if (
                    self.epoch % self.config["training"]["val_every"] == 0
                    and self.val_loader is not None
                ):

                    val_loss = self.validate()
                    if self.scheduler is not None and isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_loss)
                    self.writer.add_scalar("Loss/val", val_loss, self.epoch)
                # Save checkpoint periodically
                if self.epoch % self.config["training"]["save_every"] == 0:
                    self.save_checkpoint()

                if self.scheduler is not None:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("LearningRate", current_lr, self.global_step)
            self.epoch += 1
        self.writer.close()
        # self.save_net_graph(batch)

    def validate(self):
        """Validation loop (similar to training but without backpropagation)."""
        self.model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0
        with torch.no_grad():
            with tqdm.tqdm(
                self.val_loader,
                desc=f"Validating {self.epoch}",
                leave=False,
                mininterval=self.config["training"]["tqdm_interval_sec"],
            ) as val_loader:
                for batch_idx, batch in enumerate(val_loader):
                    out_dict = self.val_step(batch)
                    loss = out_dict["loss"]
                    val_loss += loss.item()
                    val_loss_cpu = loss.item()
                    val_loader.set_postfix(loss=val_loss_cpu, refresh=False)
                    preds = out_dict["output"]
                    targets = out_dict["target"]
                    all_preds.append(preds)
                    all_targets.append(targets)
                    if self.config["training"]["plotting"]:
                        # Plotting the output and target trajectories
                        self.plotting(
                            out_dict["output"],
                            out_dict["target"],
                            batch_idx,
                            self.epoch,
                        )
        all_preds = torch.cat(all_preds, dim=0)  # shape: [B, T, D]
        all_targets = torch.cat(all_targets, dim=0)  # shape: [B, T, D]
        metrics = self.compute_state_metrics(all_preds, all_targets)
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f"val/{metric_name}", value, self.epoch)

        self.model.train()
        return val_loss / len(self.val_loader)

    def compute_state_metrics(self, preds, targets):
        """Compute per-variable metrics."""
        metrics = {}
        state_slices = self.config["dataset"]["state_shapes"]
        for name, sl in state_slices.items():
            pred_slice = preds[:, sl]
            target_slice = targets[:, sl]
            mse = torch.mean((pred_slice - target_slice) ** 2).item()
            mae = torch.mean(torch.abs(pred_slice - target_slice)).item()
            metrics[f"{name}_mse"] = mse
            metrics[f"{name}_mae"] = mae
        return metrics

    def save_checkpoint(self):
        epoch_step = self.epoch
        global_step = self.global_step
        checkpoint_path = os.path.join(
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
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(
            payload,
            f"{checkpoint_path}/checkpoint_{epoch_step}.pth",
        )

    def plotting(self, output, target, step, epoch, name="val"):
        output_np = output.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        for i, (trj_out, trj_true) in enumerate(zip(output_np, target_np)):
            if i == 5:
                break
            image_name = os.path.join("val_dir", "trajectory")
            img_np = self.plot_3d_trajectory(trj_out, trj_true, title=image_name)
            self.writer.add_image(image_name, img_np, global_step=i, dataformats="HWC")

    @staticmethod
    def split_state_tensor(state_tensor: torch.Tensor, param_shapes: dict) -> dict:
        split_dict = {}
        start = 0
        for name in param_shapes.keys():
            end = start + param_shapes[name]["shape"]
            split_dict[name] = state_tensor[:, :, start:end].detach().cpu()
            start = end
        return split_dict

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
    def plot_2d_plane(out, target, title, axis):
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

        # Stack predictions and targets to distinguish lines
        ax.plot(
            out_np[:, axis[0]], out_np[:, axis[1]], "-", color="red", label="pred"
        )  # Dashed lines for predictions
        ax.plot(
            target_np[:, axis[0]],
            target_np[:, axis[1]],
            "--",
            color="blue",
            label="target",
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

        # Render the canvas
        fig.canvas.draw()

        # Get RGBA buffer from canvas
        buf = np.asarray(fig.canvas.buffer_rgba())  # <-- modern way

        # buf is (H, W, 4) because of alpha channel (RGBA), we drop the alpha
        img_np = buf[:, :, :3]  # Keep only RGB

        plt.close(fig)
        return img_np
