import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
import cv2
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack


class SkipFrame(gym.Wrapper):
 
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip         = skip
        self._flag_get_seen = False  

    def reset(self, **kwargs):
        self._flag_get_seen = False   
        return self.env.reset(**kwargs)

    def step(self, action):
        total_reward = 0.0
        done         = False
        info         = {}

        for _ in range(self._skip):
            step_result = self.env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

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


class MarioPreprocessingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._process_frame(obs)

    def step(self, action):
        step_result = self.env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        reward = self._shape_reward(reward)
        return self._process_frame(obs), reward, done, info

    def _shape_reward(self, reward):
      
        return np.sign(reward) * (np.sqrt(abs(reward) + 1) - 1) + 0.001 * reward

    def _process_frame(self, frame):
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

    def render(self, mode='human', **kwargs):
        return self.env.render()


def make_mario_env(render_mode=None, level='SuperMarioBros-v0'):
    """
    创建马里奥环境
    
    Args:
        render_mode: 渲染模式，None 或 'human'
        level: 关卡ID，默认 'SuperMarioBros-v0' (完整游戏)
               可选 'SuperMarioBros-1-1-v0' (仅1-1局)
    """
    if render_mode:
        env = gym_super_mario_bros.make(
            level,  # 使用传入的 level 参数
            apply_api_compatibility=True,
            render_mode=render_mode
        )
    else:
        env = gym_super_mario_bros.make(
            level,  # 使用传入的 level 参数
            apply_api_compatibility=True
        )
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = SkipFrame(env, skip=4)
    env = MarioPreprocessingWrapper(env)
    return env


def create_vec_env(num_envs=1, n_stack=4, render_mode=None, level='SuperMarioBros-v0'):
    """
    创建向量化环境
    
    Args:
        num_envs: 并行环境数量
        n_stack: 帧堆叠数量
        render_mode: 渲染模式
        level: 关卡ID，默认完整游戏，可指定 'SuperMarioBros-1-1-v0'
    """
    # 使用 lambda 捕获 level 参数
    env_fns = [lambda lvl=level: make_mario_env(render_mode=render_mode, level=lvl)
               for _ in range(num_envs)]
    
    if num_envs > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)
    
    env = VecFrameStack(env, n_stack=n_stack, channels_order='last')
    return env
