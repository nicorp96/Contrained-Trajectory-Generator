"""Image encoders: trainability, normalization layers, pooling and train/eval modes."""

import pytest
import torch
import torch.nn as nn

from models.utils.encoders_robot import (
    ResNet18Encoder,
    SpatialSoftmax,
    replace_batchnorm_with_groupnorm,
)
from trainer.trainer_diffusion_traj import build_image_encoders


def n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def config(**camera):
    camera = {"type": "resnet18", "pretrained": False, **camera}
    return {"model": {"image_encoders": {"base_camera": camera}}}


def encoder(cfg, **kw):
    return build_image_encoders(cfg, **kw).encoders["base_camera"]


# ---------------------------------------------------------------- trainability
def test_train_last_n_unfreezes_only_last_stage():
    enc = encoder(config(trainable=True, train_last_n=1))
    trainable = {n.split(".")[1] for n, p in enc.named_parameters() if p.requires_grad}
    assert trainable == {"layer4"}


def test_train_last_n_includes_spatial_softmax_head():
    enc = encoder(config(trainable=True, train_last_n=1, pool="spatial_softmax"))
    assert all(p.requires_grad for p in enc.pool.parameters())
    assert not any(p.requires_grad for p in enc.backbone.layer3.parameters())


def test_trainable_without_train_last_n_trains_everything():
    enc = encoder(config(trainable=True))
    assert n_trainable(enc) == sum(p.numel() for p in enc.parameters())


@pytest.mark.parametrize("cfg", [config(), config(trainable=False)])
def test_frozen_encoder_has_no_trainable_parameters(cfg):
    assert n_trainable(encoder(cfg)) == 0


def test_inference_build_is_frozen():
    assert n_trainable(encoder(config(trainable=True), for_inference=True)) == 0


# ---------------------------------------------------------------- normalization
def test_groupnorm_replaces_every_batchnorm():
    enc = ResNet18Encoder(pretrained=False, norm="group")
    assert not any(isinstance(m, nn.BatchNorm2d) for m in enc.modules())
    gns = [m for m in enc.modules() if isinstance(m, nn.GroupNorm)]
    assert gns and all(m.num_channels % m.num_groups == 0 for m in gns)


def test_replace_batchnorm_small_channel_count_uses_one_group():
    module = replace_batchnorm_with_groupnorm(nn.Sequential(nn.BatchNorm2d(8)))
    assert isinstance(module[0], nn.GroupNorm) and module[0].num_groups == 1


@pytest.mark.parametrize("kw", [{"norm": "layer"}, {"pool": "max"}])
def test_invalid_options_raise(kw):
    with pytest.raises(ValueError):
        ResNet18Encoder(pretrained=False, **kw)


# ---------------------------------------------------------------- outputs
def test_default_encoder_keeps_original_behaviour():
    enc = ResNet18Encoder(pretrained=False).eval()
    assert enc.output_dim == 512
    assert enc(torch.rand(2, 3, 128, 128)).shape == (2, 512)


def test_spatial_softmax_encoder_output():
    enc = ResNet18Encoder(pretrained=False, norm="group", pool="spatial_softmax",
                          num_keypoints=16, center_crop=False).eval()
    out = enc(torch.rand(2, 3, 128, 128))
    assert enc.output_dim == 32 and out.shape == (2, 32)
    assert out.abs().max() <= 1.0


def test_no_center_crop_sees_the_image_border():
    enc = ResNet18Encoder(pretrained=False, center_crop=False).eval()
    img = torch.zeros(1, 3, 128, 128)
    border = img.clone()
    border[..., :4] = 1.0  # change only the left border columns
    assert not torch.allclose(enc(img), enc(border))


def test_center_crop_ignores_the_image_border():
    enc = ResNet18Encoder(pretrained=False, center_crop=True).eval()
    img = torch.zeros(1, 3, 128, 128)
    border = img.clone()
    border[..., :4] = 1.0
    assert torch.allclose(enc(img), enc(border), atol=1e-5)


def test_spatial_softmax_finds_the_peak():
    ss = SpatialSoftmax(in_channels=1, num_keypoints=1)
    nn.init.constant_(ss.keypoints.weight, 1.0)
    nn.init.zeros_(ss.keypoints.bias)
    feat = torch.zeros(1, 1, 5, 5)
    feat[0, 0, 4, 0] = 50.0  # bottom-left corner
    x, y = ss(feat)[0].tolist()
    assert x == pytest.approx(-1.0, abs=1e-3) and y == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------- train/eval modes
def test_frozen_encoder_stays_in_eval_mode_during_training():
    cfg = {"model": {"image_encoders": {
        "base_camera": {"type": "resnet18", "pretrained": False, "trainable": False},
        "hand_camera": {"type": "resnet18", "pretrained": False, "trainable": True},
    }}}
    encoders = build_image_encoders(cfg).train()
    assert encoders.training
    assert not encoders.encoders["base_camera"].training
    assert encoders.encoders["hand_camera"].training
    encoders.eval()
    assert not any(m.training for m in encoders.modules())


def test_frozen_batchnorm_statistics_do_not_change_in_training():
    encoders = build_image_encoders(config(trainable=False)).train()
    bn = encoders.encoders["base_camera"].backbone.bn1
    before = bn.running_mean.clone()
    encoders({"base_camera": torch.rand(2, 1, 3, 64, 64)})
    assert torch.equal(bn.running_mean, before)
