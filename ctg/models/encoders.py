"""Camera image encoders: ResNet18 with optional GroupNorm and spatial-softmax keypoints."""

from typing import Callable, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T


class SpatialSoftmax(nn.Module):
    """Feature map -> expected image coordinates of `num_keypoints` learned keypoints
    (Levine et al. 2016, as in robomimic / Diffusion Policy). Keeps *where* things are,
    which global average pooling discards."""

    def __init__(self, in_channels: int, num_keypoints: int = 32):
        super().__init__()
        self.keypoints = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.output_dim = 2 * num_keypoints

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # [B, C, H, W] -> [B, 2K] in [-1, 1]
        _, _, H, W = feat.shape
        attn = torch.softmax(self.keypoints(feat).flatten(2), dim=-1)  # [B, K, H*W]
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device, dtype=attn.dtype),
            torch.linspace(-1, 1, W, device=feat.device, dtype=attn.dtype),
            indexing="ij",
        )
        x = (attn * xs.reshape(1, 1, -1)).sum(-1)
        y = (attn * ys.reshape(1, 1, -1)).sum(-1)
        return torch.stack((x, y), dim=-1).flatten(1)


def replace_batchnorm_with_groupnorm(module: nn.Module, channels_per_group: int = 16) -> nn.Module:
    """BatchNorm couples samples and differs between train/eval; GroupNorm does not."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.GroupNorm(max(child.num_features // channels_per_group, 1), child.num_features))
        else:
            replace_batchnorm_with_groupnorm(child, channels_per_group)
    return module


class ResNet18Encoder(nn.Module):
    """
    ResNet18 on float images in [0, 1] -> feature vector of `output_dim`.

    norm:        "group" (BatchNorm replaced by GroupNorm) or "batch" (torchvision)
    pool:        "spatial_softmax" (2 * num_keypoints) or "avg" (512)
    center_crop: True applies torchvision's resize 256 / crop 224 (cuts the image border)
    trainable:   False freezes all weights; train_last_n unfreezes only the last n stages
    """

    def __init__(self, pretrained: bool = True, norm: str = "group", pool: str = "spatial_softmax",
                 num_keypoints: int = 32, center_crop: bool = False, trainable: bool = True,
                 train_last_n: Optional[int] = None):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT
        transforms = weights.transforms()
        self.preprocess = transforms if center_crop else T.Normalize(mean=transforms.mean, std=transforms.std)
        backbone = tvm.resnet18(weights=weights if pretrained else None)
        backbone.fc = nn.Identity()
        if norm == "group":
            replace_batchnorm_with_groupnorm(backbone)
        elif norm != "batch":
            raise ValueError(f"unknown norm {norm!r}")
        if pool == "spatial_softmax":
            self.pool = SpatialSoftmax(512, num_keypoints)
            self.output_dim = self.pool.output_dim
        elif pool == "avg":
            self.pool = None
            self.output_dim = 512
        else:
            raise ValueError(f"unknown pool {pool!r}")
        self.backbone = backbone
        self._set_trainable(trainable, train_last_n)

    def _set_trainable(self, trainable: bool, train_last_n: Optional[int]):
        self.requires_grad_(False)
        if not trainable:
            return
        if train_last_n is None:
            self.requires_grad_(True)
            return
        for stage in ["layer1", "layer2", "layer3", "layer4"][-train_last_n:]:
            getattr(self.backbone, stage).requires_grad_(True)
        if self.pool is not None:
            self.pool.requires_grad_(True)

    @property
    def is_trainable(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Convolutional trunk up to layer4: [B, 512, h, w]."""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        return b.layer4(b.layer3(b.layer2(b.layer1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        return self.backbone(x) if self.pool is None else self.pool(self.features(x))


class CameraEncoders(nn.Module):
    """One encoder per camera. Encodes the camera entries of a history dict
    ([B, H, C, h, w] -> [B, H, F]); other entries pass through unchanged."""

    def __init__(self, cameras: Sequence[str] = (), encoder: Optional[Callable[[], nn.Module]] = None):
        super().__init__()
        if cameras and encoder is None:
            raise ValueError("cameras given but no encoder")
        self.encoders = nn.ModuleDict({cam: encoder() for cam in cameras})

    @property
    def feature_dims(self) -> Dict[str, int]:
        return {cam: enc.output_dim for cam, enc in self.encoders.items()}

    @property
    def is_trainable(self) -> bool:
        return any(p.requires_grad for p in self.parameters())

    def train(self, mode: bool = True):
        """Frozen encoders stay in eval mode: a frozen BatchNorm must keep its running
        statistics, otherwise features differ between training and inference."""
        super().train(mode)
        for enc in self.encoders.values():
            if not any(p.requires_grad for p in enc.parameters()):
                enc.eval()
        return self

    def forward(self, history: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = dict(history)
        for cam, enc in self.encoders.items():
            B, H = history[cam].shape[:2]
            out[cam] = enc(history[cam].flatten(0, 1)).reshape(B, H, -1).float()
        return out
