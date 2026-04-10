
import os
import glob
import time
import torch
import numpy as np
from src.env import create_vec_env
from src.ppo import PPOAgent

OBS_SHAPE = (4, 84, 84)
DEVICE    = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def get_latest_model(dir_path="./mario_models/"):
    if not os.path.exists(dir_path):
        return None
    files = glob.glob(os.path.join(dir_path, "*.pt"))
    return max(files, key=os.path.getctime) if files else None


if __name__ == '__main__':
    model_path = os.path.join("./mario_models/", "mario_ppo_step4423680.pt")

    if not os.path.exists(model_path):
        model_path = get_latest_model("./mario_models/")

    if model_path is None:
        print("\n[Error]   No .pt model file found. Please run train.py first.")
    else:
        print(f"\n[Load]    Loading model: {model_path}")

        env        = create_vec_env(num_envs=1, n_stack=4, render_mode='human')
        action_dim = env.action_space.n
        agent      = PPOAgent(OBS_SHAPE, action_dim, device=DEVICE)
        agent.load(model_path)

        obs = env.reset()
        print("[Demo]    Running demo... (Press Ctrl+C to quit)\n")

        try:
            while True:
                state_tensor = torch.FloatTensor(obs).to(DEVICE)
                with torch.no_grad():
                    logits, _ = agent.network(state_tensor)
                    action    = torch.argmax(logits, dim=1).cpu().numpy()

                obs, reward, done, info = env.step(action)
                env.render()
                time.sleep(0.02)

        except KeyboardInterrupt:
            print("\n[Done]    Demo finished.")
            env.close()
