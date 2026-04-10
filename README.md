# 🍄 PPO Agent for Super Mario Bros

A PyTorch implementation of **Proximal Policy Optimization (PPO)** trained on the Super Mario Bros NES environment. This project is part of a Final Year Project at **Macau University of Science and Technology**, conducting a comparative study between PPO and Double DQN algorithms on classic Atari-style game environments.

---

## 📺 Demo

<!-- Replace the path below with your actual GIF file once uploaded -->
![Mario PPO Demo](figures/mario_ppo_best_run.gif)

> The agent achieves a **74.6% overall clear rate** on World 1-1, reaching **94.6%** in the final training phase.

---

## 📁 Project Structure

```
mario_PPO/
├── src/
│   ├── __init__.py          # Package initializer
│   ├── env.py               # Environment wrappers, preprocessing, reward shaping
│   └── ppo.py               # MarioNet (CNN Actor-Critic) + PPO update logic
├── figures/                 # Training curve plots and demo GIFs
├── train.py                 # Main training script
├── test.py                  # Load a saved model and run a visual demo
├── eval_100ep_1_1.py        # 100-episode evaluation on World 1-1
├── retro_eval.py            # Evaluation on retro environments
├── plot_training.py         # Generate training curves from CSV logs
├── record_video.py          # Record gameplay video to file
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 🧠 Algorithm Overview

PPO is implemented from scratch using PyTorch. Key components:

**Network Architecture — MarioNet**
- Three convolutional layers extract spatial features from stacked 84×84 grayscale frames
- Shared CNN trunk feeds into separate Actor (policy) and Critic (value) heads
- Orthogonal weight initialization for stable early-stage gradients

**PPO Techniques**
- **Generalized Advantage Estimation (GAE)** — stable advantage estimates across 4 parallel environments
- **Clipped Surrogate Objective** — prevents large policy updates (`ε = 0.2`)
- **Value Function Clipping** — stabilizes Critic updates
- **Returns Normalization** — prevents value loss explosion
- **Dynamic Entropy Regularization** — adaptively boosts entropy bonus when policy entropy falls below target threshold, preventing premature convergence
- **Linear Learning Rate Decay** — anneals from 2.5×10⁻⁴ → 0 over training

**Reward Shaping**

Raw game scores are compressed with a square-root transformation to stabilize training:

```
r_shaped = sign(r) × (√(|r| + 1) − 1) + 0.001 × r
```

| Event | Raw Reward | Shaped Reward |
|-------|-----------|--------------|
| Stage clear | +1000 | +31.6 |
| Death | −15 | −3.0 |
| Movement | +1 ~ +10 | +0.4 ~ +2.3 |

---

## ⚙️ Environment Setup

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3) **or** a CUDA-capable GPU machine
- [Anaconda](https://www.anaconda.com/) or Miniconda

### Installation

```bash
# 1. Create and activate a Conda environment
conda create -n mario python=3.11
conda activate mario

# 2. Install dependencies
pip install -r requirements.txt
```

> **Note:** `gym==0.26.2` and `nes-py==8.2.1` are pinned for compatibility. Do not upgrade — newer versions conflict with the Mario environment.

---

## 🚀 Training

```bash
python train.py
```

Key hyperparameters (editable at the top of `train.py`):

| Parameter | Value | Description |
|-----------|-------|-------------|
| `TOTAL_TIMESTEPS` | 5,000,000 | Total environment steps |
| `NUM_ENVS` | 4 | Parallel environments |
| `NUM_STEPS` | 512 | Steps per rollout |
| `LEARNING_RATE` | 2.5×10⁻⁴ → 0 | Linear decay |
| `GAMMA` | 0.99 | Discount factor |
| `GAE_LAMBDA` | 0.95 | GAE smoothing |
| `ENT_COEF` | 0.02 | Base entropy coefficient |
| `VF_COEF` | 0.25 | Value loss coefficient |
| `CLIP_COEF` | 0.2 | PPO clipping epsilon |

### Monitor Training

```bash
tensorboard --logdir=./logs/
# Then open http://localhost:6006
```

### Plot Training Curves

```bash
python plot_training.py
# Or specify a file directly:
python plot_training.py logs/mario_ppo_xxxxx_log.csv
```

---

## 🎮 Running a Demo

```bash
python test.py
```

Loads the saved model from `mario_models/` and opens a game window showing the agent playing in real time. Press `Ctrl+C` to quit.

---

## 🎬 Recording Video

```bash
python record_video.py
```

Records a gameplay session to a video file saved in the project directory.

---

## 🔬 Preprocessing Pipeline

```
RGB Frame (240×256×3)
    ↓  Grayscale
(240×256×1)
    ↓  Resize
(84×84×1)
    ↓  Frame Skip ×4  (action repeated 4 frames, rewards accumulated)
    ↓  Stack 4 consecutive frames
(84×84×4)  →  Normalize to [0,1] inside network forward pass
```

Stacking 4 frames gives the agent temporal context — it can perceive velocity and direction from static image inputs.

---

## 📊 Results

Training was conducted for **5,000,000 environment steps** across 4 parallel environments on Apple M1 (MPS backend).

| Metric | Value |
|--------|-------|
| Total Episodes | 8,063 |
| **First Stage Clear** | **Episode 32 (Step 20,480)** |
| Total Stage Clears | 6,012 |
| Overall Clear Rate | 74.6% |
| Final Clear Rate (last 1M steps) | **94.6%** |
| Best Mean Reward (100-ep window) | 1,348.8 |
| Final Mean Reward (100-ep window) | 1,330.2 |

The agent learned to reliably clear World 1-1 within the first 2 million steps, with the clear rate stabilizing above 90% in the final phase. Training was conducted on `SuperMarioBros-v0` (full game mode), so the agent progresses to World 1-2 after clearing 1-1 — performance there is limited as the policy specializes on the 1-1 layout.

> Full comparison with the Double DQN baseline will be included in the final project report.

---

## 🖥️ Hardware & Software

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

## 📚 References

- Schulman et al., ["Proximal Policy Optimization Algorithms"](https://arxiv.org/abs/1707.06347), arXiv:1707.06347, 2017
- Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, 2015
- van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning," AAAI, 2016
- [gym-super-mario-bros](https://github.com/Kautenja/gym-super-mario-bros)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
