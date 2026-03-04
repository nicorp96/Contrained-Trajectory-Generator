import argparse
import os

from common.get_class import get_class_dict
from common.utils import load_config
from global_parameters import ConfigGlobalP

cfg_global_p = ConfigGlobalP()


def main(args):
    trainer = None
    try:
        config_path = os.path.join(
            cfg_global_p.ROOT_DIR, "configs", args.config + ".yaml"
        )
        config = load_config(config_path)
        trainer = get_class_dict(config)
        trainer.train()

    except KeyboardInterrupt:
        print("KeyboardInterrupt: attempting to save checkpoint...")

        if trainer is not None:
            # Only save if trainer exists
            try:
                trainer.save_checkpoint()
            except Exception as e:
                print(f"Failed to save checkpoint: {e}")

    finally:
        # Close TB writer safely (it may be None if you only create it on main process)
        if trainer is not None and getattr(trainer, "writer", None) is not None:
            try:
                trainer.writer.close()
            except Exception as e:
                print(f"Failed to close writer: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Trajectory Generator")
    parser.add_argument(
        "-c", "--config", help="Name of config file", default="diffusion"
    )
    args = parser.parse_args()
    main(args)
