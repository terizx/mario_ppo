# retro_eval.py
#
# 回溯评估脚本：加载 mario_models/ 里所有 checkpoint，
# 每个跑 EVAL_EPISODES 局，输出类似 DDQN eval_log 格式的 CSV。
#
# 用法：python retro_eval.py
#
# 输出：eval_log_retro.csv
# 列说明：
#   step            — 该 checkpoint 对应的训练步数
#   eval_reward     — 本次 EVAL_EPISODES 局的平均 reward
#   eval_best_x     — 本次的平均 x_pos
#   eval_clear      — 本次通关局数（满分=EVAL_EPISODES）
#   eval_clear_rate — 本次通关率（%）
#   best_eval_clear — 历史最佳通关局数（累计追踪）
#   best_eval_x     — 历史最佳平均 x_pos
#   best_eval_reward— 历史最佳平均 reward

import os
import re
import csv
import glob
import torch
import numpy as np
from src.env import create_vec_env
from src.ppo import PPOAgent

# ── 配置 ──────────────────────────────────────────────────────────
MODEL_DIR    = "./mario_models/"
OUTPUT_CSV   = "./eval_log_retro.csv"
EVAL_EPISODES = 10        # 每个 checkpoint 跑多少局（对标组员的10局）
OBS_SHAPE    = (4, 84, 84)
DEVICE       = ("cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu")


def extract_step(filename):
    """从文件名 mario_ppo_step{N}.pt 提取步数，返回 int"""
    m = re.search(r'step(\d+)', filename)
    return int(m.group(1)) if m else None


def eval_checkpoint(agent, env, n_episodes):
    """
    用 argmax 确定性策略跑 n_episodes 局。
    返回 (mean_reward, mean_x, n_clear)
    """
    rewards, x_positions, clears = [], [], []
    obs = env.reset()

    ep = 0
    ep_reward = 0.0
    ep_max_x  = 0
    ep_flag   = False

    while ep < n_episodes:
        state_t = torch.FloatTensor(obs).to(DEVICE)
        with torch.no_grad():
            logits, _ = agent.network(state_t)
            action = torch.argmax(logits, dim=1).cpu().numpy()

        obs, reward, done, info = env.step(action)

        ep_reward += float(np.array(reward).flatten()[0])
        done_val   = bool(np.array(done).flatten()[0])

        try:
            env_info = info[0] if isinstance(info, (list, tuple)) else {}
            if isinstance(env_info, dict):
                x = env_info.get('x_pos', 0)
                if x > ep_max_x:
                    ep_max_x = x
                if env_info.get('flag_get', False):
                    ep_flag = True
        except (IndexError, TypeError):
            pass

        if done_val:
            rewards.append(ep_reward)
            x_positions.append(ep_max_x)
            clears.append(1 if ep_flag else 0)

            ep        += 1
            ep_reward  = 0.0
            ep_max_x   = 0
            ep_flag    = False
            obs = env.reset()

    return (round(float(np.mean(rewards)), 1),
            round(float(np.mean(x_positions)), 1),
            int(sum(clears)))


# ── 主流程 ────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. 找所有 step checkpoint，按步数排序
    pattern = os.path.join(MODEL_DIR, "mario_ppo_step*.pt")
    ckpt_files = glob.glob(pattern)

    if not ckpt_files:
        print(f"[Error] No checkpoint files found in {MODEL_DIR}")
        print("        Expected filenames like: mario_ppo_step40960.pt")
        exit(1)

    # 解析步数并排序
    ckpts = []
    for f in ckpt_files:
        step = extract_step(os.path.basename(f))
        if step is not None:
            ckpts.append((step, f))
    ckpts.sort(key=lambda x: x[0])

    print(f"[Info]  Found {len(ckpts)} checkpoints in {MODEL_DIR}")
    print(f"[Info]  Step range: {ckpts[0][0]:,} → {ckpts[-1][0]:,}")
    print(f"[Info]  Eval episodes per checkpoint: {EVAL_EPISODES}")
    print(f"[Info]  Device: {DEVICE}")
    print(f"[Info]  Output: {OUTPUT_CSV}")
    print(f"[Info]  Estimated time: ~{len(ckpts) * EVAL_EPISODES * 5 // 60} min")
    print()

    # 2. 建环境（只创建一次，反复使用）
    env = create_vec_env(num_envs=1, n_stack=4, render_mode=None)
    action_dim = env.action_space.n

    # 3. 建 agent（只建一次，每次只换权重）
    agent = PPOAgent(OBS_SHAPE, action_dim, device=DEVICE)
    agent.network.eval()

    # 4. 追踪历史最佳
    best_clear  = 0
    best_x      = 0.0
    best_reward = 0.0

    results = []

    # 5. 逐 checkpoint 评估
    for idx, (step, ckpt_path) in enumerate(ckpts):
        agent.load(ckpt_path)

        mean_reward, mean_x, n_clear = eval_checkpoint(agent, env, EVAL_EPISODES)
        clear_rate = round(n_clear / EVAL_EPISODES * 100, 1)

        # 更新历史最佳
        if n_clear > best_clear or (n_clear == best_clear and mean_x > best_x):
            best_clear  = n_clear
            best_x      = mean_x
            best_reward = mean_reward

        row = {
            'step':             step,
            'eval_reward':      mean_reward,
            'eval_best_x':      mean_x,
            'eval_clear':       n_clear,
            'eval_clear_rate':  clear_rate,
            'best_eval_clear':  best_clear,
            'best_eval_x':      best_x,
            'best_eval_reward': best_reward,
        }
        results.append(row)

        print(f"  [{idx+1:>3}/{len(ckpts)}] step={step:>8,} | "
              f"reward={mean_reward:>7.1f} | "
              f"x={mean_x:>6.1f} | "
              f"clear={n_clear}/{EVAL_EPISODES} ({clear_rate}%) | "
              f"best_clear={best_clear}/{EVAL_EPISODES}")

    env.close()

    # 6. 写 CSV
    fieldnames = ['step', 'eval_reward', 'eval_best_x', 'eval_clear',
                  'eval_clear_rate', 'best_eval_clear', 'best_eval_x', 'best_eval_reward']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 60)
    print(f"  Done. Results saved to: {OUTPUT_CSV}")
    print(f"  Best checkpoint: clear={best_clear}/{EVAL_EPISODES}, "
          f"x={best_x}, reward={best_reward}")
    print("=" * 60)
    print()
   
