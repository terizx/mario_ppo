# eval_100ep_1_1.py
#
# PPO 标准评估脚本 — 强制只评估 1-1 关
#
# 修正说明：
#   原 eval_100ep.py 使用 SuperMarioBros-v0，通关1-1后游戏会继续进入1-2，
#   导致 score / x_pos / reward 都是1-1+1-2混合的结果，数据不纯。
#
#   本脚本改用 SuperMarioBros-1-1-v0，通关1-1后 done=True，
#   保证每个 episode 的数据 100% 是1-1关的纯净结果，可直接用于论文对比。
#
# 用法：python eval_100ep_1_1.py

import os
import csv
import torch
import numpy as np

import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from src.ppo import PPOAgent

# ── 配置 ──────────────────────────────────────────────────────────
MODEL_PATH   = "./mario_models/mario_ppo_step4423680.pt"
OUTPUT_CSV   = "./eval_100ep_1_1.csv"
NUM_EPISODES = 100
OBS_SHAPE    = (4, 84, 84)
DEVICE       = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# ── 单独构建只有1-1的环境（不用 create_vec_env，避免用v0）─────────

import cv2

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._flag_get_seen = False

    def reset(self, **kwargs):
        self._flag_get_seen = False
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done = False
        info = {}
        for _ in range(self._skip):
            result = self.env.step(action)
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = terminated or truncated
            else:
                obs, reward, done, info = result
            total_reward += reward
            if info.get('flag_get', False):
                self._flag_get_seen = True
            if done:
                break
        if done:
            info['flag_get'] = self._flag_get_seen
            self._flag_get_seen = False
        else:
            info['flag_get'] = False
        return obs, total_reward, done, info


class GrayResizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._process(obs)

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result
        return self._process(obs), reward, done, info

    def _process(self, frame):
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


def make_1_1_env():
    """
    关键修正：使用 SuperMarioBros-1-1-v0
    通关1-1后 done=True，不会继续进入1-2
    """
    env = gym_super_mario_bros.make(
        'SuperMarioBros-1-1-v0',       # ← 只有1-1关
        apply_api_compatibility=True
    )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = GrayResizeWrapper(env)
    return env


def create_1_1_vec_env():
    env = DummyVecEnv([make_1_1_env])
    env = VecFrameStack(env, n_stack=4, channels_order='last')
    return env


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[Eval]  Model    : {MODEL_PATH}")
    print(f"[Eval]  Device   : {DEVICE}")
    print(f"[Eval]  Env      : SuperMarioBros-1-1-v0  (纯1-1关，通关即done)")
    print(f"[Eval]  Policy   : Deterministic (argmax)")
    print(f"[Eval]  Episodes : {NUM_EPISODES}")
    print()

    env = create_1_1_vec_env()
    action_dim = env.action_space.n
    agent = PPOAgent(OBS_SHAPE, action_dim, device=DEVICE)
    agent.load(MODEL_PATH)
    agent.network.eval()

    results = []
    obs = env.reset()

    ep           = 0
    ep_reward    = 0.0
    ep_length    = 0
    ep_max_x     = 0
    ep_coins     = 0
    ep_score     = 0
    ep_flag      = False
    ep_time_left = 400

    while ep < NUM_EPISODES:
        state_t = torch.FloatTensor(obs).to(DEVICE)
        with torch.no_grad():
            logits, _ = agent.network(state_t)
            action = torch.argmax(logits, dim=1).cpu().numpy()

        obs, reward, done, info = env.step(action)

        reward_val = float(np.array(reward).flatten()[0])
        done_val   = bool(np.array(done).flatten()[0])

        ep_reward += reward_val
        ep_length += 1

        try:
            env_info = info[0] if isinstance(info, (list, tuple)) else {}
            if isinstance(env_info, dict):
                x = env_info.get('x_pos', 0)
                if x > ep_max_x:
                    ep_max_x = x
                ep_coins     = env_info.get('coins', ep_coins)
                ep_score     = env_info.get('score', ep_score)
                ep_time_left = env_info.get('time',  ep_time_left)
                if env_info.get('flag_get', False):
                    ep_flag = True
        except (IndexError, TypeError):
            pass

        if done_val:
            results.append({
                'episode':   ep + 1,
                'reward':    round(ep_reward, 2),
                'length':    ep_length,
                'x_pos':     ep_max_x,
                'coins':     ep_coins,
                'score':     ep_score,
                'flag_get':  int(ep_flag),
                'time_left': ep_time_left,
            })
            status = "CLEAR ✓" if ep_flag else f"FAIL  x={ep_max_x}"
            print(f"  Ep {ep+1:>3}/100 | reward={ep_reward:>8.2f} | "
                  f"steps={ep_length:>4} | x={ep_max_x:>4} | "
                  f"coins={ep_coins} | score={ep_score:>6} | {status}")

            ep           += 1
            ep_reward     = 0.0
            ep_length     = 0
            ep_max_x      = 0
            ep_coins      = 0
            ep_score      = 0
            ep_flag       = False
            ep_time_left  = 400
            obs = env.reset()

    env.close()

    # ── 写 CSV ────────────────────────────────────────────────────
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'episode', 'reward', 'length', 'x_pos', 'coins',
            'score', 'flag_get', 'time_left'
        ])
        writer.writeheader()
        writer.writerows(results)

    # ── 统计摘要 ──────────────────────────────────────────────────
    rewards    = np.array([r['reward']   for r in results])
    lengths    = np.array([r['length']   for r in results])
    x_pos_arr  = np.array([r['x_pos']    for r in results])
    coins_arr  = np.array([r['coins']    for r in results])
    scores_arr = np.array([r['score']    for r in results])
    flags      = np.array([r['flag_get'] for r in results])

    clear_eps   = [r for r in results if r['flag_get']]
    clear_lens  = np.array([r['length']    for r in clear_eps]) if clear_eps else np.array([])
    clear_times = np.array([r['time_left'] for r in clear_eps]) if clear_eps else np.array([])

    q25, q75 = np.percentile(rewards, [25, 75])

    print()
    print("=" * 60)
    print("  100-Episode Evaluation  (PPO · 1-1 Only · Deterministic)")
    print("=" * 60)
    print(f"  Clear Rate             : {flags.mean()*100:.1f}%  ({int(flags.sum())}/100)")
    print(f"  Mean Reward ± std      : {rewards.mean():.2f} ± {rewards.std():.2f}")
    print(f"  Median Reward          : {np.median(rewards):.2f}")
    print(f"  Mean X Position ± std  : {x_pos_arr.mean():.1f} ± {x_pos_arr.std():.1f}")
    print(f"  Mean Coins ± std       : {coins_arr.mean():.2f} ± {coins_arr.std():.2f}")
    print(f"  Mean Score ± std       : {scores_arr.mean():.1f} ± {scores_arr.std():.1f}")
    print(f"  Mean Steps ± std       : {lengths.mean():.1f} ± {lengths.std():.1f}")
    if len(clear_lens) > 0:
        print(f"  Mean Clear Steps       : {clear_lens.mean():.1f} ± {clear_lens.std():.1f}")
    if len(clear_times) > 0:
        print(f"  Mean Time Left (clear) : {clear_times.mean():.1f} ± {clear_times.std():.1f}")
    print(f"  Min / Max Reward       : {rewards.min():.2f} / {rewards.max():.2f}")
    print(f"  Reward IQR             : {q75-q25:.2f}  (Q25={q25:.2f}, Q75={q75:.2f})")
    print("=" * 60)
    print(f"  Saved: {OUTPUT_CSV}")
    print()
  