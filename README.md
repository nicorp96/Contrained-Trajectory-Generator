# Constrained Trajectory Generator

A research project for generating constrained trajectories using diffusion models, specifically designed for robotic manipulation tasks in environments like ManiSkill.

## Overview

This project implements diffusion-based trajectory generation for robotic tasks, focusing on constrained motion planning. It uses transformer and U-Net architectures with diffusion models to predict future trajectories given current states and goals, enabling safe and efficient robot control.

Key features:
- Diffusion model-based trajectory prediction
- Support for constrained environments (e.g., peg insertion)
- Integration with ManiSkill simulation environments
- Configurable policies and trainers
- Evaluation scripts with Accelerator support

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Constrained-Trajectory-Generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure ManiSkill environments are set up:
   ```bash
   pip install mani-skill
   ```

## Project Structure

```
├── common/                 # Utility functions and helpers
├── configs/                # YAML configuration files
├── dataset/                # Dataset classes for different environments
├── environment/            # Custom environment wrappers and recorders
├── models/                 # Diffusion model architectures
│   ├── policies/           # Policy classes (base and diffusion-based)
│   └── utils/              # Model utilities (normalizers, guidance, etc.)
├── scripts/                # Additional scripts
├── trainer/                # Training classes for different models
├── global_parameters.py    # Global configuration parameters
├── run_evaluation.py       # Evaluation script
├── run_training.py         # Training script
└── requirements.txt        # Python dependencies
```

## Usage

### Training

To train a model, use the training script with a configuration file:

```bash
python run_training.py -c diffusion_traj_dp
```

Available configs:
- `diffusion_traj_dp`: Diffusion trajectory with padding
- `diffusion_traj_pad`: Trajectory padding variant
- `diffusion_traj_resampling`: With resampling
- `gc_dit_diff_guidance`: Guided DiT diffusion
- `gc_dit_diff`: DiT diffusion
- `gc_unet_diff_guidance`: Guided U-Net diffusion
- `gc_unet_diff`: U-Net diffusion

### Evaluation

To evaluate a trained policy:

```bash
python run_evaluation.py -c diffusion_traj_dp
```

The evaluation script uses Accelerator for device management and runs the policy in the ManiSkill environment.

## Configuration

Configurations are defined in YAML files in the `configs/` directory. Each config specifies:

- Model architecture and hyperparameters
- Training parameters (batch size, learning rate, etc.)
- Dataset settings
- Noise scheduler configuration
- Guidance and normalization settings

Example config structure:
```yaml
name: "diffusion_traj_padding"
target: trainer.trainer_diffusion_traj.DiffusionTrajectoryPadHistTrainer

model:
  target: models.diffusion_transformer.DiTTrjCC
  input_dim: 17
  cond_dim: 7
  # ... model parameters

training:
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 5501
  # ... training parameters

dataset:
  target: dataset.dataset_maniskill.TrajectoryPadDatasetHDF5
  path: "mani_skill_data/demos/PegInsertionSide-ExtendedIMG/motionplanning/trajectory.h5"
  # ... dataset parameters
```

## Policies

The project includes policy classes for trajectory generation:

- `BasePolicy`: Abstract base class for policies
- `DiffPolicy`: Diffusion-based policy that samples trajectories and extracts actions

Policies can be instantiated from checkpoints and used for inference in robotic control loops.

## Datasets

Supports various dataset formats:
- `TrajectoryPadDatasetHDF5`: HDF5-based trajectories with padding
- `SimpleTrajDataset`: Simple trajectory datasets
- `BaseDataset`: Base dataset class for ManiSkill environments

## Models

Implemented diffusion models:
- `DiTTrjCC`: Diffusion Transformer for trajectories
- `DiffusionUNet`: U-Net based diffusion model
- `DiffusionUNet1DAtt`: 1D attention U-Net
- `TransformerDiT`: DiT transformer variant

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file.

[//]: # (## Citation)

[//]: # (If you use this code in your research, please cite:)

[//]: # ()
[//]: # (```)

[//]: # (@misc{constrained_trajectory_generator,)

[//]: # (  title={Constrained Trajectory Generator},)

[//]: # (  author={Your Name},)

[//]: # (  year={2024},)

[//]: # (  url={https://github.com/your-repo/constrained-trajectory-generator})

[//]: # (})
```

## Contact

For questions or issues, please open an issue on the GitHub repository.
