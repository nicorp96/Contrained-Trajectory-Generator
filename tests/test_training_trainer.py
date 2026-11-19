"""ctg/training/trainer.py end to end on the synthetic demonstrations (seconds on a CPU)."""

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ctg.data.layout import make_layout
from ctg.training.trainer import Trainer, channel_indices

from conftest import train_tiny


def scalars(run_dir):
    acc = EventAccumulator(str(run_dir / "tensorboard"))
    acc.Reload()
    return set(acc.Tags()["scalars"])


def test_channel_indices():
    layout = make_layout({"a": {"size": 2}, "b": {"size": 1, "norm": "sincos"}, "c": {"size": 3}})
    assert channel_indices(layout, ["b", "c"]) == [2, 3, 4, 5, 6]
    assert channel_indices(layout, None) == []


def test_masked_mse():
    pred, target = torch.ones(1, 2, 2), torch.zeros(1, 2, 2)
    assert Trainer.masked_mse(pred, target).item() == 1.0
    pred[0, 1] = 3.0
    assert Trainer.masked_mse(pred, target, loss_mask=torch.tensor([[True, False]])).item() == 1.0
    known = torch.tensor([[[False, False], [True, True]]])
    assert Trainer.masked_mse(pred, target, known=known).item() == 1.0


def test_e9_checkpoint_and_logs(e9_checkpoint):
    ckpt = torch.load(e9_checkpoint, map_location="cpu", weights_only=False)
    assert {"config", "epoch", "global_step", "model", "ema_model", "encoders", "normalizers", "optimizer",
            "lr_scheduler"} <= set(ckpt)
    # 3 training episodes x 2 windows = 6 samples = 1 full batch of 4 per epoch
    assert ckpt["epoch"] == 1 and ckpt["global_step"] == 2
    assert ckpt["config"]["name"] == "e9" and ckpt["ema_encoders"] is None
    tags = scalars(e9_checkpoint.parents[1])
    assert {"train/loss", "train/physics", "train/physics_kinematic", "val/loss", "val/qpos_mse",
            "val/constraints_pred/joint_limits_violation_rate",
            "val/constraints_demo/kinematic_violation_rate"} <= tags


def test_resume_continues_from_the_checkpoint(e9_checkpoint, demo_h5, demo_stats, tmp_path):
    trainer = train_tiny(demo_h5, demo_stats, tmp_path, "experiment=e9", "training.epochs=3",
                         f"training.resume={e9_checkpoint}")
    assert trainer.epoch == 3 and trainer.ema.optimization_step == trainer.global_step
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == ["epoch_00002.pt"]


@pytest.mark.parametrize("extra", [
    ["experiment=e9_inpainting", "model=dit", "model.d_model=32", "model.n_blocks=1", "physics.loss_weight=0.1"],
    ["experiment=e9_resampled_time", "training.mask_padding=true"],
    ["experiment=e9_resampled_arclength", "physics.guidance_eta=0.5", "physics.project_final=true"],
    ["experiment=e6a"],  # normalized fields, classifier-free guidance
], ids=["inpainting-dit", "resampled-time", "resampled-arclength-guided", "cfg-normalized"])
def test_variants_train_and_validate(extra, demo_h5, demo_stats, tmp_path):
    trainer = train_tiny(demo_h5, demo_stats, tmp_path, *extra, "training.epochs=1")
    assert (tmp_path / "checkpoints" / "epoch_00000.pt").exists()
    metrics = trainer.validate()
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_inpainted_entries_are_exact(demo_h5, demo_stats, tmp_path):
    trainer = train_tiny(demo_h5, demo_stats, tmp_path, "experiment=e9_inpainting", "training.epochs=1")
    batch = next(iter(trainer.val_loader))
    b = trainer._prepare(batch, use_ema=True)
    known = trainer._known(b["plan"])
    plan = trainer.sampler.sample(trainer.ema.model, b["plan"].shape, b["goal"], b["history"], trainer.codec,
                                  known, b["plan"], b["time"])
    assert torch.equal(plan[known], b["plan"][known])


def test_image_pipeline_with_trainable_encoder(demo_h5, demo_stats, tmp_path):
    trainer = train_tiny(demo_h5, demo_stats, tmp_path, "experiment=e8", "data.dataset.cameras=[base_camera]",
                         "encoder.pretrained=false", "encoder.num_keypoints=4", "training.epochs=1",
                         "training.gradient_accumulation_steps=2")
    assert trainer.ema_encoders is not None and len(trainer.optimizer.param_groups) == 2
    ckpt = torch.load(tmp_path / "checkpoints" / "epoch_00000.pt", map_location="cpu", weights_only=False)
    assert ckpt["ema_encoders"] is not None
