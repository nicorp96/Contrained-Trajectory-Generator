import torch.nn as nn
from einops import repeat
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import numpy as np
import h5py
import torch


def _stats_for_group(args):
    filename, group_name, batch_size = args
    # open per-process
    with h5py.File(filename, "r") as h5:
        grp = h5[group_name]
        # infer dim
        first_key = next(iter(grp.keys()))
        dim = grp[first_key].shape[-1]
        n_total = 0
        mean = np.zeros(dim, dtype=np.float64)
        M2 = np.zeros(dim, dtype=np.float64)

        def update(x_np):
            nonlocal n_total, mean, M2
            if x_np.size == 0:
                return
            x_np = np.asarray(x_np, dtype=np.float64)
            b = x_np.shape[0]
            n_total += b
            delta = x_np - mean
            mean += delta.sum(axis=0) / n_total
            delta2 = x_np - mean
            M2 += (delta * delta2).sum(axis=0)

        for key in grp.keys():
            dset = grp[key]
            n = dset.shape[0]
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                update(dset[s:e])

        var = M2 / max(n_total - 1, 1)
        std = np.sqrt(np.maximum(var, 1e-12)).astype(np.float32)
        mean = mean.astype(np.float32)

        return group_name, mean, std, n_total


class Normalizer(nn.Module):
    def __init__(self, size, name, factor=1.0, method="standard", eps=1e-8):
        super().__init__()
        self.method = method
        self.eps = eps
        self.size = size
        self.name = name
        self.start = None
        self.goal = None
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("std", torch.ones(size))
        self.register_buffer("min", torch.zeros(size))
        self.register_buffer("max", torch.ones(size))
        self.fitted = False
        self.mean = None
        self.std = None
        self.min = None
        self.max = None
        self.factor = factor

    def fit(self, data_list):
        """data_list: List of tensors to compute statistics over."""
        data = torch.cat(data_list, dim=0)
        if self.method == "standard":
            self.mean = data.mean(dim=0)
            self.std = data.std(dim=0, unbiased=False)
        elif self.method == "minmax":
            self.min = data.min(dim=0).values
            self.max = data.max(dim=0).values
        elif self.method == "none":
            print("Without normalization")
        else:
            raise ValueError("Unsupported method")
        self.fitted = True

    def set_stats_from_dict(self, dict):
        self.mean = torch.tensor(dict["mean"])
        self.std = torch.tensor(dict["std"])
        self.min = torch.tensor(dict["min"])
        self.max = torch.tensor(dict["max"])
        self.fitted = True

    @staticmethod
    def get_mask(x, ch: int = 1, thresh: float = 1e-6):
        """
        Returns a boolean mask of shape (B, T, C) where True means 'apply normalization'.
        Only channel `ch` is conditionally masked off (set to False) per batch; all other
        channels remain True.
        """
        B, T, C = x.shape
        device = x.device

        # Compute batch-wise validity based on the selected channel over time
        data_y = x[:, 1:, ch]  # (B, T)
        all_small = (data_y < thresh).all(dim=1)  # (B,), bool
        valid = ~all_small  # (B,), bool
        # Start with "apply normalization everywhere"
        mask = torch.ones((B, T, C), dtype=torch.bool, device=device)
        # For the selected channel, apply the batch validity
        # If valid[b] == False -> mask[b, :, ch] == False (skip normalization only on that channel)
        mask[:, :, ch] = valid[:, None].expand(B, T)

        return mask

    def forward(self, x, **kwargs):
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted yet.")
        if self.method == "standard":
            if self.mean.ndim == 1:
                mask = self.get_mask(x)
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
                x_temp = (x - self.mean) / (self.std + self.eps)
                # return mask * x_temp + (~mask) * x
                return x_temp
                # return torch.where(mask, (x - self.mean) / (self.std + self.eps), x)
            return (x - self.mean) / (self.std + self.eps)
        elif self.method == "minmax":
            return 2 * (x - self.min) / (self.max - self.min + self.eps) - 1
        elif self.method == "path_len":
            seg_len = np.linalg.norm(np.diff(x, axis=0), axis=1)
            D = seg_len.sum()
            x_rel = x - x[:, :1, :]
            return (x_rel) / (D + self.eps)
        elif self.method == "goal_n":
            B, L, C = x.shape
            assert "start" in kwargs.keys()
            assert "goal" in kwargs.keys()
            # self.start = x[:, 0, :]
            # self.goal = x[:, -1, :]
            self.start = kwargs["start"].to(x.device)
            self.goal = kwargs["goal"].to(x.device)
            D = torch.norm(self.goal - self.start, p=2, dim=1)
            D = repeat(D, "B -> B L C", C=C, L=L)
            x_rel = x
            if self.name == "position":
                x_rel = x - self.start.unsqueeze(1)
            return (x_rel * self.factor) / (D + self.eps)
        elif self.method == "none":
            return x
        return None

    def unnormalize(self, x):
        if self.method == "standard":
            if self.mean.ndim == 1:
                device = x.device
                self.std = self.std.to(device)
                self.mean = self.mean.to(device)
                mask = self.get_mask(x)
                x_temp = x * (self.std + self.eps) + self.mean
                return x_temp
                # return mask * x_temp + (~mask) * x
                # return torch.where(mask, x * (self.std + self.eps) + self.mean, x)
            return x * (self.std + self.eps) + self.mean
        elif self.method == "minmax":
            return 0.5 * (x + 1) * (self.max - self.min + self.eps) + self.min
        elif self.method == "path_len":
            seg_len = np.linalg.norm(np.diff(x, axis=0), axis=1)
            D = seg_len.sum()
            x_rel = x - x[:, :1, :]
            return x_rel / (D + self.eps)
        elif self.method == "goal_n":
            if self.start is None or self.goal is None:
                raise ValueError("Start and Goal were not computed!")
            B, L, C = x.shape
            device = x.device
            D = torch.norm(self.goal - self.start, p=2, dim=1)
            D = repeat(D, "B -> B L C", C=C, L=L).to(device)
            if self.name == "position":
                return self.start.unsqueeze(1) + (x * D) / self.factor
            return (x * D) / self.factor
        elif self.method == "none":
            return x
        return None

    def log_stats(self, writer: SummaryWriter, tag: str, global_step: int = 0):
        if self.method == "standard":
            writer.add_histogram(f"{tag}/mean", self.mean, global_step)
            writer.add_histogram(f"{tag}/std", self.std, global_step)
        elif self.method == "minmax":
            writer.add_histogram(f"{tag}/min", self.min, global_step)
            writer.add_histogram(f"{tag}/max", self.max, global_step)


class AnglesSinCos(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x, **kwargs):
        """x: [B, dim]"""
        if x.shape[-1] != self.size:
            raise ValueError(
                f"Input tensor must have last dimension of size {self.size}."
            )
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def fit(self, x):
        pass

    def unnormalize(self, x):
        return x

    def log_stats(self, writer: SummaryWriter, tag: str, global_step: int = 0):
        pass

    def set_stats_from_dict(self, dict):
        pass


class DictNormalizer(nn.Module):
    def __init__(self, param_shapes: dict):
        super().__init__()
        self.normalizers = nn.ModuleDict()
        for name, value in param_shapes.items():
            if "image" in name.lower():
                # Skip image normalization
                self.normalizers[name] = nn.Identity()

            elif "deg" in name or "rad" in name:
                # Use AnglesSinCos for angle representations
                self.normalizers[name] = AnglesSinCos(size=value["shape"])
            else:
                # Use standard Normalizer for other parameters
                self.normalizers[name] = Normalizer(
                    size=value["shape"],
                    name=name,
                    method=value["method_norm"],
                    factor=value.get("factor", 1.0),
                )

    @torch.no_grad()
    def fit(
        self, hdf5_data: h5py.File, batch_size: int = 131072, processes: int = None
    ):
        filename = hdf5_data.filename
        group_names = list(self.normalizers.keys())
        # TODO: go back
        # run workers (spawn is safest)
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=processes) as pool:
            for group_name, mean, std, _count in pool.imap_unordered(
                _stats_for_group, [(filename, g, batch_size) for g in group_names]
            ):
                norm = self.normalizers[group_name]
                # push stats into buffers
                # common: normalizer has .mean/.std buffers
                norm.register_buffer("mean", torch.from_numpy(mean))
                norm.register_buffer("std", torch.from_numpy(std))
                norm.mean = torch.from_numpy(mean)
                norm.std = torch.from_numpy(std)

    def set_stats_from_json(self, stats_json_dict):
        for key in self.normalizers:
            norm = self.normalizers[key]
            stats = stats_json_dict[key]
            norm.register_buffer("mean", torch.tensor(stats["mean"]))
            norm.register_buffer("std", torch.tensor(stats["std"]))
            norm.mean = torch.tensor(stats.get("mean", 0.0))
            norm.std = torch.tensor(stats.get("std", 0.0))
            norm.min = torch.tensor(stats.get("min", 0.0))
            norm.max = torch.tensor(stats.get("max", 0.0))
            norm.fitted = True

    def set_stats_from_dict(self, stats_dict):
        for key in stats_dict:
            if key in self.normalizers.keys():
                self.normalizers[key].set_stats_from_dict(stats_dict[key])

    def forward(self, x: dict, **kwargs):
        return {
            name: self.normalizers[name](tensor, **kwargs) for name, tensor in x.items()
        }

    def unnormalize(self, x: dict):
        return {
            name: self.normalizers[name].unnormalize(tensor)
            for name, tensor in x.items()
        }

    def log_all_stats(self, writer: SummaryWriter, global_step: int = 0):
        for name, normalizer in self.normalizers.items():
            normalizer.log_stats(
                writer, tag=f"norm_stats/{name}", global_step=global_step
            )
