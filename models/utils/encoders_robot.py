import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from transformers import AutoModel, AutoImageProcessor


class SpatialSoftmax(nn.Module):
    """
    Feature map -> expected 2D image coordinates of `num_keypoints` learned keypoints
    (Levine et al. 2016; as in robomimic / Diffusion Policy). Keeps *where* things
    are in the image, which global average pooling discards.
    """

    def __init__(self, in_channels, num_keypoints=32):
        super().__init__()
        self.keypoints = nn.Conv2d(in_channels, num_keypoints, kernel_size=1)
        self.output_dim = 2 * num_keypoints

    def forward(self, feat):  # [B, C, H, W] -> [B, 2K] in [-1, 1]
        B, _, H, W = feat.shape
        logits = self.keypoints(feat).flatten(2)  # [B, K, H*W]
        attn = torch.softmax(logits, dim=-1)
        ys, xs = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device, dtype=attn.dtype),
            torch.linspace(-1, 1, W, device=feat.device, dtype=attn.dtype),
            indexing="ij",
        )
        x = (attn * xs.reshape(1, 1, -1)).sum(-1)
        y = (attn * ys.reshape(1, 1, -1)).sum(-1)
        return torch.stack((x, y), dim=-1).flatten(1)


def replace_batchnorm_with_groupnorm(module, channels_per_group=16):
    """BatchNorm couples samples and differs between train/eval; GroupNorm does not."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            groups = max(child.num_features // channels_per_group, 1)
            setattr(module, name, nn.GroupNorm(groups, child.num_features))
        else:
            replace_batchnorm_with_groupnorm(child, channels_per_group)
    return module


class ResNet18Encoder(nn.Module):
    """
    ResNet18 image encoder on float images in [0, 1].

    norm:       "batch" (torchvision) or "group" (BatchNorm replaced by GroupNorm)
    pool:       "avg" (512-d global feature) or "spatial_softmax" (2 * num_keypoints)
    center_crop: True resizes to 256 and crops 224 (torchvision classification
                preprocessing, cuts the image border); False only normalizes.
    The defaults reproduce the original encoder so older checkpoints still load.
    """

    def __init__(self, pretrained=True, norm="batch", pool="avg", num_keypoints=32, center_crop=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.DEFAULT
        transforms = weights.transforms()
        self.preprocess = (
            transforms if center_crop else T.Normalize(mean=transforms.mean, std=transforms.std)
        )
        model = tvm.resnet18(weights=weights if pretrained else None)
        model.fc = nn.Identity()
        if norm == "group":
            replace_batchnorm_with_groupnorm(model)
        elif norm != "batch":
            raise ValueError(f"Unknown norm={norm}")
        if pool == "spatial_softmax":
            model.avgpool = nn.Identity()
            model.flatten = nn.Identity()
            self.pool = SpatialSoftmax(512, num_keypoints)
            self.output_dim = self.pool.output_dim
        elif pool == "avg":
            self.pool = None
            self.output_dim = 512
        else:
            raise ValueError(f"Unknown pool={pool}")
        self.backbone = model

    def features(self, x):
        """Convolutional trunk up to layer4: [B, 512, h, w]."""
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        return b.layer4(b.layer3(b.layer2(b.layer1(x))))

    def forward(self, x):
        x = self.preprocess(x)  # [B, C, H, W]
        if self.pool is None:
            return self.backbone(x)  # [B, 512]
        return self.pool(self.features(x))  # [B, 2K]


class HFVisionEncoder(nn.Module):
    def __init__(self, model_name, pool="cls"):
        super().__init__()
        self.model_name = model_name
        self.pool = pool
        self.processor = AutoImageProcessor.from_pretrained(model_name, use_fast=True)
        self.backbone = AutoModel.from_pretrained(model_name)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            out_dim = self._infer_dim(dummy)
        self.output_dim = out_dim

    def _infer_dim(self, x):
        feat = self.forward(x)
        return feat.shape[-1]

    def forward(self, x):
        dev = next(self.backbone.parameters()).device
        x = x.to(dev)

        # inputs are already float in [0, 1] (see ImageNormalizer)
        inputs = self.processor(images=x, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        out = self.backbone(**inputs)

        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output

        tokens = out.last_hidden_state
        if self.pool == "cls":
            return tokens[:, 0]
        elif self.pool == "mean":
            return tokens.mean(dim=1)
        else:
            raise ValueError(f"Unknown pool={self.pool}")


def freeze_all_parameters(module):
    for p in module.parameters():
        p.requires_grad_(False)


def unfreeze_resnet_last_n(encoder, n=1):
    """Unfreeze the last n residual stages (and the pooling head) of a ResNet18Encoder."""
    layers = ["layer1", "layer2", "layer3", "layer4"]
    for layer_name in layers[-n:]:
        for p in getattr(encoder.backbone, layer_name).parameters():
            p.requires_grad_(True)
    if encoder.pool is not None:
        for p in encoder.pool.parameters():
            p.requires_grad_(True)


def unfreeze_hf_last_n(encoder, n=1):
    # common HF transformer layout
    backbone = encoder.backbone if hasattr(encoder, "backbone") else encoder.encoder

    # try common transformer block containers
    candidate_paths = [
        ("encoder", "layer"),  # BERT/ViT-like
        ("encoder", "layers"),  # some variants
        ("layers",),  # some custom models
        ("layer",),  # fallback
    ]

    blocks = None
    for path in candidate_paths:
        obj = backbone
        ok = True
        for attr in path:
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
            else:
                ok = False
                break
        if ok:
            blocks = obj
            break

    if blocks is None:
        # fallback: unfreeze everything if structure unknown
        for p in backbone.parameters():
            p.requires_grad_(True)
        return

    for block in list(blocks)[-n:]:
        for p in block.parameters():
            p.requires_grad_(True)
