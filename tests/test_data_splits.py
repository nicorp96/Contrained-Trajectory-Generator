"""ctg/data/splits.py"""

import pytest
import torch
from torch.utils.data import Dataset, Subset

from ctg.data.splits import RepeatDataset, base_dataset, make_splits, split_episodes


class Items(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return idx


def test_repeat_dataset_length_and_items():
    ds = RepeatDataset(Items(3), 4)
    assert len(ds) == 12
    assert [ds[i] for i in range(12)] == [0, 1, 2] * 4
    assert ds[-1] == 2


@pytest.mark.parametrize("idx", [12, -13])
def test_repeat_dataset_out_of_range(idx):
    with pytest.raises(IndexError):
        RepeatDataset(Items(3), 4)[idx]


def test_repeat_dataset_rejects_zero():
    with pytest.raises(ValueError, match="repeats"):
        RepeatDataset(Items(3), 0)


def test_split_is_seeded_disjoint_and_complete():
    train, val = split_episodes(10, 0.2, seed=0)
    assert len(val) == 2 and sorted(train + val) == list(range(10))
    assert split_episodes(10, 0.2, seed=0) == (train, val)
    assert split_episodes(10, 0.2, seed=1) != (train, val)


def test_split_matches_torch_random_split():
    """The episode split of all earlier runs (torch.utils.data.random_split)."""
    old_train, old_val = torch.utils.data.random_split(Items(10), [8, 2], generator=torch.Generator().manual_seed(0))
    assert split_episodes(10, 0.2, seed=0) == (list(old_train.indices), list(old_val.indices))


def test_no_validation():
    assert split_episodes(4, 0.0) == ([0, 1, 2, 3], [])
    train, val = make_splits(Items(4), val_ratio=0)
    assert val is None and len(train) == 4


def test_repetition_after_split_keeps_validation_separate():
    train, val = make_splits(Items(10), 0.2, seed=0, samples_per_episode=3)
    assert isinstance(train, RepeatDataset) and len(train) == 24
    assert isinstance(val, Subset) and len(val) == 2
    assert not set(train[i] for i in range(len(train))) & set(val[i] for i in range(len(val)))


def test_base_dataset_unwraps():
    items = Items(10)
    train, _ = make_splits(items, 0.2, samples_per_episode=2)
    assert base_dataset(train) is items and base_dataset(items) is items
