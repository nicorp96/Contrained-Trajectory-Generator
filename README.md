# Constrained Trajectory Generator

Diffusion models that generate robot plans from demonstrations, with the option to make
those plans **physically consistent**: joint limits, velocity and acceleration limits,
kinematic consistency between positions and velocities, and action bounds.

The testbed is ManiSkill's `PegInsertionSide-v1` with a Franka Panda: insert a peg into a
box hole with 3 mm clearance. A denoiser (Diffusion Policy's 1D U-Net or a diffusion
transformer) learns plans conditioned on the hole pose (goal) and the recent observation
history. The policy then executes the plans closed-loop in simulation.

Each of these options can be switched on independently, to study what helps:

| Option | What it does | Config |
|---|---|---|
| Physics loss | Penalizes constraint violations of the denoised plan during training | `physics=panda physics.loss_weight=0.1` |
| Physics guidance | Pushes every late denoising step down the constraint energy | `physics.guidance_eta=1.0` |
| Projection | Clamps the final plan into the joint, velocity and action limits | `physics.project_final=true` |
| Inpainting | Fixes known plan entries (e.g. the current joint state at step 0) | `diffusion.inpainting.start_fields=[qpos,qvel]` |
| Resampling | Plans cover the rest of the episode, uniform in time or in joint-space path length | `experiment=e9_resampled_time` / `e9_resampled_arclength` |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                      # the ctg package
export CTG_DATA_DIR=/path/to/data     # demonstrations, statistics
export CTG_LOGS_DIR=/path/to/logs     # runs: checkpoints, TensorBoard, Hydra logs
```

ManiSkill rendering needs a GPU with Vulkan. Training uses one CUDA GPU via
[Accelerate](https://huggingface.co/docs/accelerate) (bf16 by default).

### Data

```bash
D=$CTG_DATA_DIR/mani_skill_data/demos/PegInsertionSide-v1/motionplanning
# 1. motion-planning demonstrations (1000 episodes)
python -m mani_skill.utils.download_demo PegInsertionSide-v1 -o $CTG_DATA_DIR/mani_skill_data/demos
# 2. replay with joint-delta control and state + camera observations
python -m mani_skill.trajectory.replay_trajectory --traj-path $D/trajectory.h5 \
    -c pd_joint_delta_pos -o state_dict+rgb --save-traj -b physx_cpu --num-envs 16
# 3. normalization statistics (training episodes only)
python scripts/compute_stats.py                    # state data -> stats_state.json
python scripts/compute_stats.py data=peg_image     # image data -> its stats.json
```

The image experiments need a top camera at 256x256. It is re-rendered from the stored env
states, with the actions unchanged:

```bash
python scripts/replay_cameras.py --src $D/trajectory.state_dict+rgb.pd_joint_delta_pos.physx_cpu.h5 \
    --out-dir ${D}_top256 --resolution 256 --cameras top
python scripts/camera_visibility.py --h5 $D/trajectory.h5 --out visibility/   # optional: what each camera sees
```

## Usage

```bash
# train (Hydra: one run folder per call under $CTG_LOGS_DIR/runs/<name>/<time>)
python scripts/train.py                                  # E7b, the reference pipeline
python scripts/train.py experiment=e9                    # joint-space plans + Panda constraints
python scripts/train.py experiment=e9 physics.loss_weight=0.1 model=dit
python scripts/train.py -m experiment=e9_resampled_time,e9_resampled_arclength   # sweep

# evaluate a checkpoint closed-loop (100 seeded episodes, optional videos)
python scripts/evaluate.py $RUN/checkpoints/epoch_01000.pt --episodes 100 --video-dir videos/
python scripts/evaluate.py $CKPT physics.guidance_eta=1.0 physics.project_final=true
python scripts/evaluate.py $CKPT --physics panda       # constraints from configs/physics/

tensorboard --logdir $CTG_LOGS_DIR/runs
```

Each checkpoint stores its fully resolved config, so evaluation rebuilds exactly the trained
networks. Overrides on the evaluation command line change only sampling and physics.

## Configuration

`configs/train.yaml` composes one option per group. An experiment changes several groups at
once:

```
configs/
├── train.yaml          defaults = E7b; paths from CTG_DATA_DIR / CTG_LOGS_DIR
├── data/               dataset, plan / goal / history fields, cameras, normalization, split
├── encoder/            camera encoders (none, ResNet18 trained end to end, frozen ResNet18)
├── model/              denoiser: unet, dit
├── diffusion/          noise scheduler, sampling steps, classifier-free guidance, inpainting
├── physics/            constraints and their use (none, panda)
├── training/           budget and optimization (default, image, legacy)
└── experiment/         every run of the study as a named override set
```

Fields are declared once, in the data config, and all network widths follow from them:

```yaml
dataset:
  plan:                              # what the denoiser generates
    actions: {size: 8, norm: none}
    qpos:    {size: 9, norm: minmax}  # norm: none | standard | minmax | sincos
  goal:
    box_hole_pose: {size: 7, norm: none}
  history:                           # last history_len observations
    qvel: {size: 9, norm: standard}
  cameras: [base_camera, hand_camera]
```

Set a field to `null` to remove one inherited from another option.

## Experiments

Success rate over 100 seeded episodes on PegInsertionSide-v1. For reference, ManiSkill's own
Diffusion Policy reaches 61–74 % on the same demonstrations.

| Experiment | Change | Success |
|---|---|---|
| `fix2` | images, frozen ResNet18, DiT, 430-step plans | 0 |
| `fix3` | trainable ResNet18 (GroupNorm, spatial softmax), 64-step plans | 2 |
| `fix4` | + top camera, 256x256 | 6 |
| `state` | object state instead of cameras | 0 |
| `e1` | state, reference training budget (~1e8 samples) | 13 |
| `e2` | 16-step plans, 2 history steps | 15 |
| `e3` | plan only the actions | 9 |
| `e4` | no previous actions in the history | 8 |
| `e5` | U-Net instead of DiT | 32 |
| `e6a` | DDPM 100/100 instead of DDIM | 44 |
| `e6b` | no classifier-free guidance | 38 |
| `e6c` | loss also on padded steps | 39 |
| `e6d` | no normalization | 52 |
| `e7a` | no normalization + DDPM | 52 |
| **`e7b`** | **no normalization, DDPM, no CFG, loss on padding (= defaults)** | **59** |
| `e8` | E7b from images (base, hand, top camera), ~5e7 samples | 18 |
| `e9` | E7b planning actions + joint positions/velocities, Panda constraints measured | – |
| `e9_inpainting` | E9 with the current joint state fixed at the first plan step | – |
| `e9_resampled_time` | E9, plans = rest of the episode, uniform in time | – |
| `e9_resampled_arclength` | E9, plans uniform in joint-space path length | – |

The E9 family is the base for the physics study and has not been trained yet. Resampled plans
span the whole remaining episode with non-uniform time stamps. Their actions cannot be executed
step by step, so these experiments are compared on validation metrics (plan error and
constraint violations), not on closed-loop success.

Checkpoints from before this restructuring (runs up to E8) load with the code at tag
`v0-pre-refactor`.

## Code

```
ctg/
├── config.py                 compose the Hydra config outside of a run (stats, evaluation, tests)
├── data/
│   ├── layout.py             FieldSpec, layouts, DataSpec (what a policy sees and plans)
│   ├── normalization.py      per-field normalizers, images, PlanCodec (tensor <-> physical fields)
│   ├── maniskill.py          DemoDataset (fixed-step windows), ResampledDemoDataset
│   ├── resampling.py         TimeResampler, ArcLengthResampler (linear / cubic / pchip, slerp)
│   ├── splits.py             seeded episode split, repetition per epoch
│   └── stats.py              normalization statistics from the training episodes
├── models/
│   ├── unet.py               UNetDenoiser around Diffusion Policy's ConditionalUnet1D
│   ├── conditional_unet1d.py vendored (see THIRD_PARTY_NOTICES.md)
│   ├── dit.py                DiTDenoiser (adaLN-Zero, history token)
│   ├── encoders.py           ResNet18 camera encoders, spatial softmax
│   └── common.py             null goal (classifier-free guidance), sinusoidal embedding
├── diffusion/
│   ├── scheduler.py          DDPM / DDIM, training target, clean-sample estimate
│   ├── sampler.py            sampling with CFG, physics guidance, inpainting, projection
│   └── inpainting.py         known-entry masks
├── physics/constraints.py    JointConstraints: residuals, energy, metrics, projection
├── training/
│   ├── trainer.py            build_components (shared with the policy), Trainer
│   └── ema.py
├── policy/                   DiffusionPolicy (receding horizon), observation history
└── envs/                     PegInsertionSide-Extended (extra cameras), evaluation,
                              camera re-rendering, camera visibility analysis
scripts/                      train, evaluate, compute_stats, replay_cameras, camera_visibility
tests/                        one test file per module
```

How a training step works:

1. The dataset returns physical fields: plan `[L, d]`, goal, history, padding mask and plan time stamps.
2. The normalizers and the camera encoders turn them into the plan tensor `x0 [B, L, D]`,
   the goal vector and the history features.
3. Noise is added at a random diffusion step and the denoiser predicts it. The loss is the
   MSE over plan entries that are not inpainted, optionally plus `loss_weight` times the
   constraint energy of the decoded clean-sample estimate. That energy is weighted by ᾱ(t),
   so steps that are almost pure noise barely count.
4. Validation samples complete plans with the EMA weights and reports the per-field error.
   With constraints configured, it also reports the violation rates of predicted and
   demonstrated plans.

`JointConstraints` works on physical, unnormalized plans. It skips any term whose fields are
not planned. With resampled plans it takes the time step per sample from the plan time stamps.

## Tests

```bash
CTG_DATA_DIR=... pytest --cov=ctg
```

Tests use a small synthetic demonstration file (`tests/conftest.py`), including end-to-end
training, resuming, the policy and closed-loop evaluation runs. Simulator tests are skipped
without ManiSkill. Tests on the real demonstrations are skipped without `CTG_DATA_DIR`.

## Licenses

Code: see `LICENSE`. Vendored code: `THIRD_PARTY_NOTICES.md`. Every dependency in
`requirements.txt` allows commercial use (the licenses are listed there).
