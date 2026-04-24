# record_video_1-1.py
#
# 录制 PPO best model 跑 World 1-1 完整通关的 GIF / MP4
# 强制锁定在 SuperMarioBros-1-1-v0，不会跑到 1-2
#
# 依赖安装：
#   pip install imageio imageio-ffmpeg pillow
#
# 用法：
#   python record_video_1-1.py                   # 录制1局，输出 GIF + MP4
#   python record_video_1-1.py --episodes 3      # 录制3局
#   python record_video_1-1.py --no-gif          # 只输出 MP4
#   python record_video_1-1.py --fps 30          # 调整帧率（默认30）

import os
import argparse
import torch
import numpy as np
from src.env import create_vec_env
from src.ppo import PPOAgent

# ── 配置 ──────────────────────────────────────────────────────────
MODEL_PATH = "./mario_models/mario_ppo_best.pt"
OUTPUT_DIR = "./recordings/"
OBS_SHAPE  = (4, 84, 84)

# ★ 固定关卡：只录 World 1-1，通关后直接 reset 回 1-1，不进 1-2
LEVEL      = "SuperMarioBros-1-1-v0"

DEVICE     = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
# ─────────────────────────────────────────────────────────────────


def record(num_episodes=1, fps=30, save_gif=True, save_mp4=True, output_dir=OUTPUT_DIR):
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "请先安装 imageio：pip install imageio imageio-ffmpeg pillow"
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"[Record]  Model  : {MODEL_PATH}")
    print(f"[Record]  Level  : {LEVEL}  ← locked to World 1-1 only")
    print(f"[Record]  Device : {DEVICE}")
    print(f"[Record]  Policy : Deterministic (argmax)")
    print(f"[Record]  FPS    : {fps}")
    print(f"[Record]  Output : {output_dir}")
    print()

    # 传入 level=LEVEL，锁定 1-1，通关后 reset 回 1-1 而不是进 1-2
    env = create_vec_env(
        num_envs=1,
        n_stack=4,
        render_mode="rgb_array",
        level=LEVEL,            
    )

    action_dim = env.action_space.n
    agent = PPOAgent(OBS_SHAPE, action_dim, device=DEVICE)
    agent.load(MODEL_PATH)
    agent.network.eval()

    for ep_idx in range(num_episodes):
        print(f"[Record]  Recording episode {ep_idx + 1}/{num_episodes} ...")
        frames      = []
        obs         = env.reset()
        ep_reward   = 0.0
        ep_length   = 0
        ep_flag     = False
        ep_max_x    = 0
        ep_score    = 0
        step_count  = 0
        max_steps   = 5000   

        while step_count < max_steps:
        
            raw = env.render(mode="rgb_array")
            if raw is not None:
                if isinstance(raw, (list, tuple)):
                    raw = raw[0]
                if isinstance(raw, np.ndarray) and raw.ndim == 3:
                    frames.append(raw.astype(np.uint8))

        
            state_t = torch.FloatTensor(obs).to(DEVICE)
            with torch.no_grad():
                logits, _ = agent.network(state_t)
                action = torch.argmax(logits, dim=1).cpu().numpy()

            obs, reward, done, info = env.step(action)

            ep_reward  += float(np.array(reward).flatten()[0])
            ep_length  += 1
            step_count += 1

            try:
                env_info = info[0] if isinstance(info, (list, tuple)) else info
                if isinstance(env_info, dict):
                    x = env_info.get("x_pos", 0)
                    if x > ep_max_x:
                        ep_max_x = x
                    ep_score = env_info.get("score", ep_score)
                    if env_info.get("flag_get", False):
                        ep_flag = True
            except (IndexError, TypeError):
                pass

            if bool(np.array(done).flatten()[0]):
                for _ in range(45):
                    raw = env.render(mode="rgb_array")
                    if raw is not None:
                        if isinstance(raw, (list, tuple)):
                            raw = raw[0]
                        if isinstance(raw, np.ndarray) and raw.ndim == 3:
                            frames.append(raw.astype(np.uint8))
                break

        status = "CLEAR ✓" if ep_flag else f"FAIL  (max_x={ep_max_x})"
        print(f"  → reward={ep_reward:.2f} | score={ep_score} | steps={ep_length} | {status}")
        print(f"  → Captured {len(frames)} frames")

        if not frames:
            print("  [Warning] No frames captured, skipping save.")
            continue

        
        if num_episodes == 1:
            base_name = "mario_ppo_1-1_best_run"
        else:
            base_name = f"mario_ppo_1-1_ep{ep_idx + 1:02d}"

        if save_mp4:
            mp4_path = os.path.join(output_dir, f"{base_name}.mp4")
            writer = imageio.get_writer(
                mp4_path,
                fps=fps,
                codec="libx264",
                quality=8,
                pixelformat="yuv420p",  
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            file_mb = os.path.getsize(mp4_path) / 1024 / 1024
            print(f"  → MP4 saved : {mp4_path}  ({file_mb:.1f} MB)")

        if save_gif:
            gif_path = os.path.join(output_dir, f"{base_name}.gif")
            gif_frames = frames[::2]
            gif_fps    = max(fps // 2, 10)
            imageio.mimsave(gif_path, gif_frames, fps=gif_fps, loop=0)
            file_mb = os.path.getsize(gif_path) / 1024 / 1024
            print(f"  → GIF saved : {gif_path}  ({file_mb:.1f} MB)")

        print()

    env.close()
    print("[Record]  Done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record PPO Mario gameplay — locked to World 1-1"
    )
    parser.add_argument("--episodes", type=int,  default=1,
                        help="录制局数 (default: 1)")
    parser.add_argument("--fps",      type=int,  default=30,
                        help="输出帧率 (default: 30)")
    parser.add_argument("--no-gif",   action="store_true",
                        help="不输出 GIF")
    parser.add_argument("--no-mp4",   action="store_true",
                        help="不输出 MP4")
    parser.add_argument("--output",   type=str,  default=OUTPUT_DIR,
                        help="输出目录 (default: ./recordings/)")
    args = parser.parse_args()

    record(
        num_episodes=args.episodes,
        fps=args.fps,
        save_gif=not args.no_gif,
        save_mp4=not args.no_mp4,
        output_dir=args.output,
    )
