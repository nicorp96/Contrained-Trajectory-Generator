import argparse
import os
import torch

from common.get_class import get_class_dict
from common.utils import load_config
from global_parameters import ConfigGlobalP

cfg_global_p = ConfigGlobalP()

def main(args):
    try:
        config_path = os.path.join(cfg_global_p.ROOT_DIR, "configs", args.config + ".yaml")
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
    parser = argparse.ArgumentParser(description="Train Trajectory Generator")
    parser.add_argument(
        "-c",
        "--config",
        help="Name of config file",
        default="diffusion",
    )
    args = parser.parse_args()
    main(args)
