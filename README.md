# PPO Agent for Super Mario Bros

A custom implementation of **Proximal Policy Optimization (PPO)** applied to the Super Mario Bros environment, built with PyTorch. This project is part of a Final Year Project at Macau University of Science and Technology, conducting a comparative study between PPO and Double DQN algorithms.

---

## Project Structure

```
mario_PPO/
├── src/
│   ├── __init__.py        # Package initializer
│   ├── env.py             # Environment setup, frame preprocessing, reward shaping
│   └── ppo.py             # PPO agent: MarioNet (CNN Actor-Critic) + update logic
├── train.py               # Main training script
├── test.py                # Load a trained model and run a visual demo
├── plot_training.py       # Generate training curve plots from CSV logs
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Algorithm Overview

This project implements PPO from scratch using PyTorch. Key components include:

- **MarioNet**: A CNN-based Actor-Critic network. Three convolutional layers extract spatial features from stacked grayscale frames, followed by separate Actor (policy) and Critic (value) heads.
- **Generalized Advantage Estimation (GAE)**: Computes stable advantage estimates across 4 parallel environments.
- **Clipped Surrogate Objective**: Prevents excessively large policy updates, improving training stability.
- **Value Function Clipping**: Stabilizes Critic network updates.
- **Returns Normalization**: Normalizes GAE returns before Critic updates to prevent value loss explosion.
- **Dynamic Entropy Regularization**: Adaptively strengthens entropy bonus when policy entropy falls below a target threshold, preventing premature convergence.
- **Linear Learning Rate Decay**: Linearly anneals learning rate from 2.5×10⁻⁴ to 0 over training.
- **Orthogonal Weight Initialization**: Applied to all layers for better early-stage gradient flow.

### Reward Shaping

Raw game scores are compressed using a square-root transformation to stabilize training:

```
r_shaped = sign(r) × (√(|r| + 1) − 1) + 0.001 × r
```

This preserves the relative ordering of rewards while reducing scale:
- Stage clear (+1000) → +31.6
- Death penalty (−15) → −3.0
- Movement reward (+1~+10) → +0.4~+2.3

---

## Environment Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3) **or** a machine with CUDA GPU
- [Anaconda](https://www.anaconda.com/) or Miniconda

### Installation

**1. Create and activate a Conda environment**

```bash
conda create -n mario python=3.11
conda activate mario
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

> **Note:** The pinned versions of `gym==0.26.2` and `nes-py==8.2.1` are required for compatibility. Do not upgrade these packages as newer versions conflict with the Mario environment on both Apple Silicon and standard setups.

---

## Training

```bash
python train.py
```

Training configuration (editable at the top of `train.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOTAL_TIMESTEPS` | 5,000,000 | Total environment steps |
| `NUM_ENVS` | 4 | Parallel environments |
| `NUM_STEPS` | 512 | Steps collected per rollout |
| `LEARNING_RATE` | 2.5×10⁻⁴ → 0 | Linear decay schedule |
| `GAMMA` | 0.99 | Discount factor |
| `GAE_LAMBDA` | 0.95 | GAE smoothing parameter |
| `ENT_COEF` | 0.02 | Base entropy coefficient |
| `VF_COEF` | 0.25 | Value loss coefficient |
| `CLIP_COEF` | 0.2 | PPO clipping epsilon |

### Monitoring Training

Open a second terminal and run:

```bash
tensorboard --logdir=./logs/
```

Then visit `http://localhost:6006` in your browser.

### Plotting Training Curves

```bash
python plot_training.py
```

This reads all CSV logs from `./logs/` and saves `training_curves.png`. You can also specify a file directly:

```bash
python plot_training.py logs/mario_ppo_1774935759_log.csv
```

---

## Running a Demo

```bash
python test.py
```

Loads `mario_models/mario_ppo_best.pt` and opens a game window showing the agent playing in real time. Press `Ctrl+C` to quit.

---

## Preprocessing Pipeline

```
RGB Frame (240×256×3)
    → Grayscale (240×256×1)
    → Resize to 84×84
    → Frame Skip ×4 (action repeated for 4 frames, rewards accumulated)
    → Stack 4 consecutive frames → (84×84×4)
    → Normalize pixel values to [0, 1] (inside network forward pass)
```

Stacking 4 frames gives the agent temporal context to perceive velocity and direction from static image inputs.

---

## Hardware & Software

| Item | Detail |
|------|--------|
| Device | MacBook Pro (Apple M1) |
| OS | macOS |
| Python | 3.11 |
| PyTorch | 2.5.1 (MPS backend) |
| gym | 0.26.2 |
| gym-super-mario-bros | 7.4.0 |
| nes-py | 8.2.1 |
| stable-baselines3 | 1.8.0 |

---

## Results

Training was conducted for **5,000,000 environment steps** across 4 parallel environments on Apple M1 (MPS backend).

| Metric | Value |
|--------|-------|
| Total Episodes | 8,063 |
| **First Stage Clear** | **Episode 32 (Step 20,480)** |
| Total Stage Clears | 6,012 |
| Overall Clear Rate | 74.6% |
| Final Stage Clear Rate (last 1M steps) | **94.6%** |
| Best Mean Reward (100-ep) | 1,348.8 |
| Final Mean Reward (100-ep) | 1,330.2 |

The agent learned to reliably complete World 1-1 within the first 2 million steps, with the clear rate stabilizing above 90% in the final training phase. Due to training on `SuperMarioBros-v0` (full game mode), the agent progresses to World 1-2 after clearing 1-1 but has limited performance there, as the policy specializes on the 1-1 layout.

> Full comparison with the Double DQN baseline will be included in the final project report.

---

## References

- Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
- Mnih et al., "Human-level control through deep reinforcement learning," Nature, 2015.
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning," AAAI, 2016.
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
