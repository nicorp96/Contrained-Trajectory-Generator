import torch.nn as nn
from einops import repeat, rearrange
from torch.utils.tensorboard import SummaryWriter
import multiprocessing as mp
import numpy as np
import h5py
import torch
from transformers import AutoImageProcessor, AutoModel

# Example names:
#   "voltron:r3m"
#   "voltron:mvp"
#   "voltron:voltron"
# Provided by the voltron-robotics repo. :contentReference[oaicite:5]{index=5}
# from voltron import load  # depends on how you install their package


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


class BaseNormalizer(nn.Module):
    def __init__(self, size, name, method="standard", eps=1e-8):
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

    def fit(self, data_list):
        pass

    def set_stats_from_dict(self, dict):
        raise NotImplementedError

    def forward(self, x, **kwargs):
        raise NotImplementedError

    def unnormalize(self, x):
        pass

    def log_stats(self, writer: SummaryWriter, tag: str, global_step: int = 0):
        pass

    def set_device(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.min = self.min.to(device)
        self.max = self.max.to(device)


class Normalizer(BaseNormalizer):
    def __init__(self, size, name, method="standard", eps=1e-8):
        super().__init__(size, name, method, eps)

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

    @torch.no_grad()
    def forward(self, x, **kwargs):
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted yet.")
        if self.method == "standard":
            if self.mean.ndim == 1:
                self.mean = self.mean.to(x.device)
                self.std = self.std.to(x.device)
                x_temp = (x - self.mean) / (self.std + self.eps)
                return x_temp
            return (x - self.mean) / (self.std + self.eps)
        elif self.method == "minmax":
            self.max = self.max.to(x.device)
            self.min = self.min.to(x.device)
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

    @torch.no_grad()
    def unnormalize(self, x):
        if self.method == "standard":
            if self.mean.ndim == 1:
                device = x.device
                self.std = self.std.to(device)
                self.mean = self.mean.to(device)
                x_temp = x * (self.std + self.eps) + self.mean
                return x_temp
            return x * (self.std + self.eps) + self.mean
        elif self.method == "minmax":
            self.max = self.max.to(x.device)
            self.min = self.min.to(x.device)
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


class AnglesSinCos(BaseNormalizer):
    def __init__(self, size, name, method="sincos", eps=1e-8):
        super().__init__(size, name, method, eps)

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


class ImageNormalizerEncoder(BaseNormalizer):
    """
    HF-native vision encoder + feature normalization.

    method:
      - "none": return raw features
      - "standard": (feat-mean)/std
      - "tanh_standard": tanh((feat-mean)/std)  -> bounded (-1, 1), no min/max
      - "layernorm_tanh": tanh(LayerNorm(feat)) -> no fitting, bounded (-1, 1)
    """

    def __init__(
        self,
        size,
        name,
        method="tanh_standard",
        eps=1e-8,
        encoder_name="facebook/dinov2-base",  # good default if you want strong generic features
        pool="cls",  # "cls" or "mean"
        freeze_encoder=True,
        flatten_time=True,
    ):
        super().__init__(size=0, name=name, method=method, eps=eps)
        self.encoder_name = encoder_name
        self.pool = pool
        self.freeze_encoder = freeze_encoder
        self.flatten_time = flatten_time

        self.processor = AutoImageProcessor.from_pretrained(encoder_name, use_fast=True)
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.eval()

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        # infer feature dim
        with torch.no_grad():
            dev = next(self.encoder.parameters()).device
            dummy = torch.zeros(1, 3, 224, 224, device=dev)
            feat = self._encode_bchw(dummy)
            self.size = feat.shape[-1]

        self.register_buffer("feat_mean", torch.zeros(self.size))
        self.register_buffer("feat_std", torch.ones(self.size))
        self.fitted = True

        # for "layernorm_tanh"
        self._ln = nn.LayerNorm(self.size)

    def set_device(self, device):
        self._ln = self._ln.to(device)
        self.encoder = self.encoder.to(device)

    def _to_bchw_float(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept:
          - [B,C,H,W] or [B,H,W,C]
          - optionally [B,T,C,H,W] or [B,T,H,W,C]
        Return:
          - [B,C,H,W] float in [0,1] (if uint8 was given), or float as-is.
        """
        if x.ndim == 5:
            # [B, T, ...]
            if self.flatten_time:
                B, T = x.shape[:2]
                x = x.reshape(B * T, *x.shape[2:])
            else:
                raise ValueError(
                    "flatten_time=False not supported in this simple version."
                )

        if x.ndim != 4:
            raise ValueError(
                f"Expected 4D (or 5D with time) image tensor, got {tuple(x.shape)}"
            )

        # BHWC -> BCHW
        if x.shape[1] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 3, 1, 2).contiguous()

        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()

        return x

    @torch.inference_mode()
    def _encode_bchw(self, x_bchw: torch.Tensor) -> torch.Tensor:
        """
        x_bchw: [B,3,H,W] float
        returns: [B,D]
        """
        dev = next(self.encoder.parameters()).device
        x_bchw = x_bchw.to(dev)

        # HF processors can accept torch tensors directly as "images"
        inputs = self.processor(images=x_bchw, return_tensors="pt")
        inputs = {k: v.to(dev) for k, v in inputs.items()}

        out = self.encoder(**inputs)

        # common outputs:
        # - ViT/DINO: last_hidden_state [B, N, D]
        # - Some models may expose pooler_output [B, D]
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            feat = out.pooler_output
        else:
            tokens = out.last_hidden_state  # [B, N, D]
            if self.pool == "cls":
                feat = tokens[:, 0]
            elif self.pool == "mean":
                feat = tokens.mean(dim=1)
            else:
                raise ValueError(f"Unknown pool={self.pool}")

        return feat

    def fit(self, data_list, batch_size=64):
        """
        data_list: iterable of image tensors (any of the supported shapes).
        Computes mean/std of *features* (not pixels).
        """
        pass

    def set_stats_from_dict(self, dct):
        dev = next(self.encoder.parameters()).device
        self.feat_mean = torch.tensor(dct["mean"], device=dev, dtype=torch.float16)
        self.feat_std = torch.tensor(dct["std"], device=dev, dtype=torch.float16)
        self.fitted = True

    @torch.no_grad()
    def forward(self, x, **kwargs):
        B, L, C, H, W = x.shape
        x = self._to_bchw_float(x)
        feat = self._encode_bchw(x)
        feat = rearrange(feat, "(B L) D -> B L D", B=B)
        if self.method == "none":
            return feat

        if self.method == "standard":
            if not self.fitted:
                raise RuntimeError("Need feature mean/std. Call fit() or load stats.")
            return (feat - self.feat_mean) / (self.feat_std + self.eps)

        if self.method == "tanh_standard":
            if not self.fitted:
                raise RuntimeError("Need feature mean/std. Call fit() or load stats.")
            z = (feat - self.feat_mean) / (self.feat_std + self.eps)
            return torch.tanh(z)  # -> (-1, 1), no min/max

        if self.method == "layernorm_tanh":
            # no fitting needed; per-sample normalization
            return torch.tanh(self._ln(feat))

        raise ValueError(f"Unsupported method={self.method}")

    @torch.no_grad()
    def unnormalize(self, x):
        # Only meaningful for reversing standardization (not pixels).
        if self.method == "standard":
            return x * (self.feat_std + self.eps) + self.feat_mean
        if self.method == "tanh_standard":
            # tanh is not exactly invertible in a stable way -> don't pretend
            return x
        return x


class DictNormalizer(nn.Module):
    def __init__(self, param_shapes: dict):
        super().__init__()
        self.normalizers = nn.ModuleDict()
        for name, value in param_shapes.items():
            if "camera" in name.lower():
                # TODO: apply image features extractor and normalizer instead of identity
                self.normalizers[name] = ImageNormalizerEncoder(
                    size=value["shape"],  # can be ignored / overwritten internally
                    name=name,
                    method=value["method_norm"],
                    encoder_name=value.get("encoder_name", "facebook/dinov2-base"),
                    freeze_encoder=value.get("freeze_encoder", True),
                    pool=value.get("pool", "cls"),
                    flatten_time=value.get("flatten_time", True),
                )

            elif "deg" in name or "rad" in name:
                # Use AnglesSinCos for angle representations
                self.normalizers[name] = AnglesSinCos(
                    size=value["shape"],
                    name=name,
                    method=value["method_norm"],
                )
            else:
                # Use standard Normalizer for other parameters
                self.normalizers[name] = Normalizer(
                    size=value["shape"],
                    name=name,
                    method=value["method_norm"],
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
            if "camera" in key.lower():
                continue
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

    def set_device(self, device):
        for key in self.normalizers.keys():
            self.normalizers[key].set_device(device)

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
