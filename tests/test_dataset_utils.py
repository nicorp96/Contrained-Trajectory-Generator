"""dataset/utils.py: train/val split, samples_per_episode repetition and unwrapping."""

import pytest
from torch.utils.data import Dataset

import dataset.utils as utils
from dataset.utils import RepeatDataset, base_dataset, get_ds_from_cfg


class Episodes(Dataset):
    """Stand-in for the ManiSkill dataset: item i is episode i."""

    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if not 0 <= idx < self.n:
            raise IndexError(idx)
        return idx


@pytest.fixture
def fake_dataset(monkeypatch):
    monkeypatch.setattr(utils, "get_class_dict", lambda cfg: Episodes(cfg["n"]))


def test_repeat_dataset_length_and_items():
    r = RepeatDataset(Episodes(3), 4)
    assert len(r) == 12
    assert [r[i] for i in range(12)] == [0, 1, 2] * 4
    assert r[-1] == 2


@pytest.mark.parametrize("idx", [12, -13])
def test_repeat_dataset_out_of_range(idx):
    with pytest.raises(IndexError):
        RepeatDataset(Episodes(3), 4)[idx]


def test_repeat_dataset_rejects_zero():
    with pytest.raises(ValueError):
        RepeatDataset(Episodes(3), 0)


def test_default_is_unchanged(fake_dataset):
    train, val = get_ds_from_cfg({"n": 10, "val": True, "val_ratio": 0.2, "split_seed": 0})
    assert len(train) == 8 and len(val) == 2
    assert not isinstance(train, RepeatDataset)


def test_repetition_after_split_keeps_validation_separate(fake_dataset):
    cfg = {"n": 10, "val": True, "val_ratio": 0.2, "split_seed": 0}
    train1, val1 = get_ds_from_cfg(cfg)
    train_k, val_k = get_ds_from_cfg({**cfg, "samples_per_episode": 5})
    assert list(val_k) == list(val1)  # same validation episodes, not repeated
    assert len(train_k) == 5 * len(train1)
    assert set(train_k) == set(train1)  # only training episodes
    assert not set(train_k) & set(val_k)  # no leakage


def test_repetition_without_validation(fake_dataset):
    train, val = get_ds_from_cfg({"n": 4, "samples_per_episode": 3})
    assert val is None and len(train) == 12


def test_base_dataset_unwraps_subset_and_repeat(fake_dataset):
    train, _ = get_ds_from_cfg({"n": 10, "val": True, "val_ratio": 0.2, "samples_per_episode": 2})
    assert isinstance(base_dataset(train), Episodes)
    assert isinstance(base_dataset(Episodes(2)), Episodes)
