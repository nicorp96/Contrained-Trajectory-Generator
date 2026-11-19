"""ctg/models/encoders.py: trainability, normalization layers, pooling and train/eval modes."""

import pytest
import torch
import torch.nn as nn

from functools import partial

from ctg.models.encoders import CameraEncoders, ResNet18Encoder, SpatialSoftmax, replace_batchnorm_with_groupnorm


def n_trainable(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def encoder(**kw):
    return ResNet18Encoder(pretrained=False, **kw)


# ---------------------------------------------------------------- trainability
def test_train_last_n_unfreezes_only_last_stage():
    enc = encoder(trainable=True, train_last_n=1)
    trainable = {n.split(".")[0] for n, p in enc.backbone.named_parameters() if p.requires_grad}
    assert trainable == {"layer4"}


def test_train_last_n_includes_spatial_softmax_head():
    enc = encoder(trainable=True, train_last_n=1, pool="spatial_softmax")
    assert all(p.requires_grad for p in enc.pool.parameters())
    assert not any(p.requires_grad for p in enc.backbone.layer3.parameters())


def test_trainable_without_train_last_n_trains_everything():
    enc = encoder(trainable=True)
    assert n_trainable(enc) == sum(p.numel() for p in enc.parameters())


def test_frozen_encoder_has_no_trainable_parameters():
    enc = encoder(trainable=False)
    assert n_trainable(enc) == 0 and not enc.is_trainable


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
def test_average_pooling_encoder_output():
    enc = ResNet18Encoder(pretrained=False, pool="avg").eval()
    assert enc.output_dim == 512
    assert enc(torch.rand(2, 3, 128, 128)).shape == (2, 512)


def test_spatial_softmax_encoder_output():
    enc = ResNet18Encoder(pretrained=False, norm="group", pool="spatial_softmax",
                          num_keypoints=16, center_crop=False).eval()
    out = enc(torch.rand(2, 3, 128, 128))
    assert enc.output_dim == 32 and out.shape == (2, 32)
    assert out.abs().max() <= 1.0


def test_no_center_crop_sees_the_image_border():
    enc = ResNet18Encoder(pretrained=False, pool="avg", center_crop=False).eval()
    img = torch.zeros(1, 3, 128, 128)
    border = img.clone()
    border[..., :4] = 1.0  # change only the left border columns
    assert not torch.allclose(enc(img), enc(border))


def test_center_crop_ignores_the_image_border():
    enc = ResNet18Encoder(pretrained=False, pool="avg", center_crop=True).eval()
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
    trainable = iter([False, True])
    encoders = CameraEncoders(["base_camera", "hand_camera"],
                              lambda: ResNet18Encoder(pretrained=False, trainable=next(trainable))).train()
    assert encoders.is_trainable
    assert encoders.training
    assert not encoders.encoders["base_camera"].training
    assert encoders.encoders["hand_camera"].training
    encoders.eval()
    assert not any(m.training for m in encoders.modules())


def test_frozen_batchnorm_statistics_do_not_change_in_training():
    encoders = CameraEncoders(["base_camera"], partial(encoder, norm="batch", trainable=False)).train()
    bn = encoders.encoders["base_camera"].backbone.bn1
    before = bn.running_mean.clone()
    encoders({"base_camera": torch.rand(2, 1, 3, 64, 64)})
    assert torch.equal(bn.running_mean, before)


def test_camera_encoders_encode_only_cameras():
    encoders = CameraEncoders(["base_camera"], partial(encoder, num_keypoints=4)).eval()
    assert encoders.feature_dims == {"base_camera": 8}
    out = encoders({"base_camera": torch.rand(2, 3, 3, 32, 32), "qpos": torch.ones(2, 3, 9)})
    assert out["base_camera"].shape == (2, 3, 8) and torch.equal(out["qpos"], torch.ones(2, 3, 9))


def test_camera_encoders_without_cameras():
    encoders = CameraEncoders()
    assert encoders.feature_dims == {} and not encoders.is_trainable
    with pytest.raises(ValueError, match="no encoder"):
        CameraEncoders(["base_camera"])
