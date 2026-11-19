"""Hydra configs outside of a training run (stats, evaluation, tests)."""

from pathlib import Path
from typing import Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def compose_config(overrides: Sequence[str] = (), config_name: str = "train") -> DictConfig:
    """The composed training config, e.g. compose_config(["experiment=e8"])."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name=config_name, overrides=list(overrides))
