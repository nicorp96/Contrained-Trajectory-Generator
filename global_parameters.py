from dataclasses import dataclass
from pathlib import Path


@dataclass
class ConfigGlobalP:
    ROOT_DIR: Path = Path("/home/nrodriguez/Documents/research-2/Contrained-Trajectory-Generator")
    DATA_DIR: Path = Path("/mnt/data_nrp/dataset")
    LOGS_DIR: Path = Path("/mnt/data_nrp/constrained_trajectory_generator/logging")