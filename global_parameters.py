import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


@dataclass
class ConfigGlobalP:
    # Override per machine with CTG_ROOT_DIR / CTG_DATA_DIR / CTG_LOGS_DIR
    ROOT_DIR: Path = field(
        default_factory=lambda: _env_path(
            "CTG_ROOT_DIR", str(Path(__file__).resolve().parent)
        )
    )
    DATA_DIR: Path = field(
        default_factory=lambda: _env_path("CTG_DATA_DIR", "/mnt/data_nrp/dataset")
    )
    LOGS_DIR: Path = field(
        default_factory=lambda: _env_path(
            "CTG_LOGS_DIR", "/mnt/data_nrp/constrained_trajectory_generator/logging"
        )
    )
