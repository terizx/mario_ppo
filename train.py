# train.py

import os
import time
import csv
import numpy as np
import torch
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from src.env import create_vec_env
from src.ppo import PPOAgent

# ==========================================
# 超参数
# ==========================================
NUM_ENVS  = 4
NUM_STEPS = 512

TOTAL_TIMESTEPS = 5_000_000

LEARNING_RATE = 2.5e-4   # 线性衰减至 0
GAMMA         = 0.99
GAE_LAMBDA    = 0.95

DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

SAVE_DIR         = "./mario_models/"
LOG_DIR          = "./logs/"
FINAL_MODEL_NAME = "mario_ppo_final.pt"
BEST_MODEL_NAME  = "mario_ppo_best.pt"


def compute_gae(rewards, values, dones, next_value, gamma, gae_lambda):
    T, N = rewards.shape
    advantages   = np.zeros((T, N), dtype=np.float32)
    last_gae_lam = np.zeros(N, dtype=np.float32)

    for t in reversed(range(T)):
        if t == T - 1:
            next_non_terminal = 1.0 - dones[t]
            next_val          = next_value
        else:
            next_non_terminal = 1.0 - dones[t + 1]
            next_val          = values[t + 1]

        delta = rewards[t] + gamma * next_val * next_non_terminal - values[t]
        advantages[t] = last_gae_lam = (
            delta + gamma * gae_lambda * next_non_terminal * last_gae_lam
        )

    returns = advantages + values
    return advantages, returns


if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    run_name = f"mario_ppo_{int(time.time())}"
    writer   = SummaryWriter(log_dir=os.path.join(LOG_DIR, run_name))

    csv_path = os.path.join(LOG_DIR, f"{run_name}_log.csv")
    csv_file = open(csv_path, 'w', newline='', buffering=1)
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'episode', 'step', 'ep_reward', 'ep_length', 'ep_v',
        'policy_loss', 'value_loss', 'entropy', 'approx_kl', 'clip_fraction',
        'mean_reward_100', 'mean_length_100', 'mean_v_100',
        'x_pos', 'coins', 'score', 'flag_get'
    ])

    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)

    print(f"{'='*60}")
    print(f"  Device            : {DEVICE}")
    print(f"  Num Envs          : {NUM_ENVS}")
    print(f"  Num Steps         : {NUM_STEPS}")
    print(f"  Total Steps Target: {TOTAL_TIMESTEPS:,}")
    print(f"  Num Updates       : {num_updates}")
    print(f"  LR Schedule       : {LEARNING_RATE} → 0 (linear decay)")
    print(f"  CSV Log           : {csv_path}")
    print(f"  TensorBoard       : tensorboard --logdir={LOG_DIR}")
    print(f"{'='*60}")

    env        = create_vec_env(num_envs=NUM_ENVS, n_stack=4)
    obs_shape  = (4, 84, 84)
    action_dim = env.action_space.n
    agent      = PPOAgent(obs_shape, action_dim, lr=LEARNING_RATE,
                          gamma=GAMMA, device=DEVICE)

    print("[Start]   Training from scratch.\n")

    obs = env.reset()

    ep_rewards    = np.zeros(NUM_ENVS, dtype=np.float32)
    ep_coins      = np.zeros(NUM_ENVS, dtype=np.float32)
    ep_lengths    = np.zeros(NUM_ENVS, dtype=np.int32)
    ep_value_sums = np.zeros(NUM_ENVS, dtype=np.float32)
    ep_max_x      = np.zeros(NUM_ENVS, dtype=np.float32)

    episode_count       = 0
    best_reward         = -np.inf
    first_clear_episode = None
    first_clear_step    = None

    recent_rewards = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    recent_values  = deque(maxlen=100)
    recent_flags   = deque(maxlen=100)

    clear_coins_list  = []
    clear_steps_list  = []

    last_loss_dict = {
        'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0,
        'approx_kl': 0.0, 'clip_fraction': 0.0
    }

    print(f"[Train]   Starting — {num_updates} updates total.\n")

    try:
        for update in range(1, num_updates + 1):
            start_time = time.time()

            # 学习率线性衰减
            frac   = 1.0 - (update - 1.0) / num_updates
            cur_lr = LEARNING_RATE * frac
            agent.set_lr(cur_lr)

            # ── Rollout ───────────────────────────────────────────
            buffer_obs, buffer_actions, buffer_logprobs = [], [], []
            buffer_rewards, buffer_dones, buffer_values = [], [], []

            for step in range(NUM_STEPS):
                action, log_prob, value = agent.select_action(obs)
                next_obs, reward, done, info = env.step(action)

                buffer_obs.append(obs)
                buffer_actions.append(action)
                buffer_logprobs.append(log_prob)
                buffer_rewards.append(reward)
                buffer_dones.append(done)
                buffer_values.append(value)

                obs = next_obs
                reward_arr = np.array(reward).flatten()
                done_arr   = np.array(done).flatten()
                value_arr  = np.array(value).flatten()

                ep_rewards    += reward_arr
                ep_lengths    += 1
                ep_value_sums += value_arr

                for i in range(NUM_ENVS):
                    try:
                        env_info = info[i] if isinstance(info, (list, tuple)) else {}
                        if isinstance(env_info, dict):
                            ep_coins[i] = env_info.get('coins', 0)
                            current_x = env_info.get('x_pos', 0)
                            if current_x > ep_max_x[i]:
                                ep_max_x[i] = current_x
                    except (IndexError, TypeError):
                        pass

                    if done_arr[i]:
                        flag = 0
                        final_score = 0
                        final_x = ep_max_x[i]

                        try:
                            env_info = info[i] if isinstance(info, (list, tuple)) else {}
                            if isinstance(env_info, dict):
                                flag = 1 if env_info.get('flag_get', False) else 0
                                final_score = env_info.get('score', 0)
                                tmp_x = env_info.get('x_pos', ep_max_x[i])
                                if tmp_x > final_x:
                                    final_x = tmp_x
                        except (IndexError, TypeError):
                            pass

                        ep_v = ep_value_sums[i] / ep_lengths[i] if ep_lengths[i] > 0 else 0.0

                        recent_rewards.append(ep_rewards[i])
                        recent_lengths.append(ep_lengths[i])
                        recent_values.append(ep_v)
                        recent_flags.append(flag)

                        mean_reward_100 = float(np.mean(recent_rewards)) if recent_rewards else 0.0
                        mean_length_100 = float(np.mean(recent_lengths)) if recent_lengths else 0.0
                        mean_v_100      = float(np.mean(recent_values))  if recent_values  else 0.0

                        total_steps = update * NUM_STEPS * NUM_ENVS
                        csv_writer.writerow([
                            episode_count,
                            total_steps,
                            round(float(ep_rewards[i]), 2),
                            int(ep_lengths[i]),
                            round(float(ep_v), 4),
                            round(last_loss_dict['policy_loss'], 6),
                            round(last_loss_dict['value_loss'], 6),
                            round(last_loss_dict['entropy'], 6),
                            round(last_loss_dict['approx_kl'], 6),
                            round(last_loss_dict['clip_fraction'], 4),
                            round(mean_reward_100, 2),
                            round(mean_length_100, 1),
                            round(mean_v_100, 4),
                            int(final_x),
                            int(ep_coins[i]),
                            int(final_score),
                            flag
                        ])

                        if flag == 1:
                            clear_coins_list.append(float(ep_coins[i]))
                            clear_steps_list.append(total_steps)
                            if first_clear_episode is None:
                                first_clear_episode = episode_count + 1
                                first_clear_step    = total_steps
                                print(f"\n{'!'*60}")
                                print(f"  [CLEAR!] First 1-1 clear!")
                                print(f"  Episode : {first_clear_episode}")
                                print(f"  Step    : {first_clear_step:,}")
                                print(f"{'!'*60}\n")

                        episode_count += 1
                        ep_rewards[i]    = 0.0
                        ep_coins[i]      = 0.0
                        ep_lengths[i]    = 0
                        ep_value_sums[i] = 0.0
                        ep_max_x[i]      = 0.0

            # ── GAE ───────────────────────────────────────────────
            _, _, next_value = agent.select_action(obs)

            b_rewards    = np.array(buffer_rewards)
            b_values     = np.array(buffer_values).squeeze(-1)
            b_dones      = np.array(buffer_dones)
            b_next_value = next_value.squeeze(-1)

            advantages, returns = compute_gae(
                b_rewards, b_values, b_dones, b_next_value, GAMMA, GAE_LAMBDA
            )

            rollouts = {
                'states':     np.concatenate(buffer_obs),
                'actions':    np.array(buffer_actions).flatten(),
                'log_probs':  np.array(buffer_logprobs).flatten(),
                'returns':    returns.flatten(),
                'advantages': advantages.flatten(),
                'values':     b_values.flatten(),
            }

            # ── 更新网络 ──────────────────────────────────────────
            loss_dict      = agent.update(rollouts)
            last_loss_dict = loss_dict

            # ── TensorBoard ───────────────────────────────────────
            total_steps = update * NUM_STEPS * NUM_ENVS

            if recent_rewards:
                smooth_reward_100 = float(np.mean(recent_rewards))
                smooth_length_100 = float(np.mean(recent_lengths))
                smooth_flag_rate  = float(np.mean(recent_flags)) * 100 if recent_flags else 0.0

                writer.add_scalar("Episode/Mean_Reward_100", smooth_reward_100, total_steps)
                writer.add_scalar("Episode/Mean_Length_100", smooth_length_100, total_steps)
                writer.add_scalar("Episode/Clear_Rate",      smooth_flag_rate,  total_steps)
                writer.add_scalar("Train/Value_Loss",        loss_dict['value_loss'],    total_steps)
                writer.add_scalar("Train/Policy_Loss",       loss_dict['policy_loss'],   total_steps)
                writer.add_scalar("Train/Entropy",           loss_dict['entropy'],       total_steps)
                writer.add_scalar("Train/Approx_KL",         loss_dict['approx_kl'],     total_steps)
                writer.add_scalar("Train/Clip_Fraction",     loss_dict['clip_fraction'], total_steps)
                writer.add_scalar("Train/LR",                cur_lr,                     total_steps)

            # ── 终端输出 ──────────────────────────────────────────
            elapsed = max(time.time() - start_time, 1e-5)
            fps     = int(NUM_STEPS * NUM_ENVS / elapsed)

            if recent_rewards:
                print(
                    f"Upd {update:>4}/{num_updates} | "
                    f"Step {total_steps:>8,} | "
                    f"Ep {episode_count:>5} | "
                    f"FPS {fps:>3} | "
                    f"Return {smooth_reward_100:>7.1f} | "
                    f"Ent {loss_dict['entropy']:>6.4f} | "
                    f"VLoss {loss_dict['value_loss']:>5.3f} | "
                    f"Clear {smooth_flag_rate:>5.1f}%"
                )

                if smooth_reward_100 > best_reward:
                    best_reward = smooth_reward_100
                    agent.save(os.path.join(SAVE_DIR, BEST_MODEL_NAME))
            else:
                print(
                    f"Upd {update:>4}/{num_updates} | "
                    f"Step {total_steps:>8,} | "
                    f"FPS {fps:>3} | "
                    f"No episodes finished yet"
                )

            if update % 20 == 0:
                ckpt = os.path.join(SAVE_DIR, f"mario_ppo_step{total_steps}.pt")
                agent.save(ckpt)

            if DEVICE == "mps" and update % 50 == 0:
                torch.mps.empty_cache()

    except KeyboardInterrupt:
        print("\n[Stop]    Training interrupted by user.")

    finally:
        csv_file.close()
        final_path = os.path.join(SAVE_DIR, FINAL_MODEL_NAME)
        agent.save(final_path)
        writer.close()
        env.close()

        print(f"\n{'='*60}")
        print(f"[Done]  Training finished.")
        print(f"[Done]  Final model : {final_path}")
        print(f"[Done]  Best model  : {os.path.join(SAVE_DIR, BEST_MODEL_NAME)}")
        print(f"[Done]  CSV log     : {csv_path}")
        print(f"{'='*60}")
        print(f"\n--- 论文数据摘要 ---")

        if first_clear_episode:
            print(f"  首次通关 Episode : {first_clear_episode}")
            print(f"  首次通关 Step    : {first_clear_step:,}")
        else:
            print(f"  首次通关         : 训练内未发生")

        if clear_coins_list:
            print(f"  总通关次数       : {len(clear_coins_list)}")
            print(f"  通关时平均金币   : {np.mean(clear_coins_list):.2f}")
            clear_rate_total = len(clear_coins_list) / episode_count * 100
            print(f"  总体通关率       : {clear_rate_total:.2f}%")
        else:
            print(f"  通关数据         : 无")

        print(f"{'='*60}")
        print(f"[Tip]   查看曲线 : tensorboard --logdir={LOG_DIR}")
