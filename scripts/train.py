"""
Train a plan denoiser.

  python scripts/train.py                          # reference pipeline (E7b, object state)
  python scripts/train.py experiment=e8            # from images
  python scripts/train.py physics=panda physics.loss_weight=0.1
  python scripts/train.py -m physics.loss_weight=0,0.1,1   # sweep

Runs are written to ${paths.logs}/runs/<name>/<timestamp>/ (config, log, TensorBoard,
checkpoints).
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from ctg.training.trainer import Trainer


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    trainer = Trainer(cfg, HydraConfig.get().runtime.output_dir)
    try:
        trainer.train()
    except KeyboardInterrupt:
        trainer.save_checkpoint()


if __name__ == "__main__":
    main()
