import argparse
import os

from common.get_class import get_class_dict
from common.utils import load_config
import torch

BASE_PATH_PRJ = "/home/nrodriguez/Documents/research-2/Contrained-Trajectory-Generator/"


def main(args):
    try:
        config_path = os.path.join(BASE_PATH_PRJ, "configs", args.config + ".yaml")
        config = load_config(config_path)
        trainer = get_class_dict(config)
        if torch.cuda.is_available():
            trainer.train()
        else:
            print("CUDA is not available. Using CPU.")
    except KeyboardInterrupt:
        print("Saving last checkpoint...")
        trainer.save_checkpoint()
        trainer.writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AR Trajectory Predictor")
    parser.add_argument(
        "-c",
        "--config",
        help="Name of config file",
        default="transformer_trj_less_p",
    )
    args = parser.parse_args()
    main(args)
