import numpy as np
import torch
import gymnasium as gym
import ale_py

from collections import deque
from gymnasium.wrappers import FrameStack
from gymnasium.wrappers.frame_stack import LazyFrames
from stable_baselines3.common.atari_wrappers import (
    AtariWrapper,
    FireResetEnv,
)

from tqdm import tqdm

gym.register_envs(ale_py)

class LazyFramesToNumpyWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def observation(self, observation):
        if isinstance(observation, LazyFrames):
            return np.array(observation)
        return observation

def make_env(game, render='rgb_array'):
    env = gym.make(game, render_mode=render)
    env = AtariWrapper(env, terminal_on_life_loss=False, frame_skip=4)
    env = FrameStack(env, num_stack=4)
    env = LazyFramesToNumpyWrapper(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    return env

class ReplayBuffer:

    def __init__(self, capacity, device) -> None:
        self.capacity = capacity
        self._buffer =  np.zeros((capacity,), dtype=object)
        self._position = 0
        self._size = 0
        self.device = device

    def store(self, experience: tuple) -> None:
        idx = self._position % self.capacity
        self._buffer[idx] = experience
        self._position += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        buffer = self._buffer[0:min(self._position-1, self.capacity-1)]
        batch = np.random.choice(buffer, size=[batch_size], replace=True)
        return (
            self.transform(batch, 0, shape=(batch_size, 4, 84, 84), dtype=torch.float32),
            self.transform(batch, 1, shape=(batch_size, 1), dtype=torch.int64),
            self.transform(batch, 2, shape=(batch_size, 1), dtype=torch.float32),
            self.transform(batch, 3, shape=(batch_size, 4, 84, 84), dtype=torch.float32),
            self.transform(batch, 4, shape=(batch_size, 1), dtype=torch.bool)
        )
        
    def transform(self, batch, index, shape, dtype):
        batched_values = np.array([val[index] for val in batch]).reshape(shape)
        batched_values = torch.as_tensor(batched_values, dtype=dtype, device=self.device)
        return batched_values

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        return self._buffer[index]

    def __setitem__(self, index, value: tuple):
        self._buffer[index] = value


def load_buffer(preload, capacity, game, *, device):
    env = make_env(game)
    buffer = ReplayBuffer(capacity,device=device)

    observation, _ = env.reset()
    for _ in tqdm(range(preload)):    
        action = env.action_space.sample()

        observation_prime, reward, terminated, truncated, _ = env.step(action)
        buffer.store((
            observation.squeeze(), 
            action, 
            reward, 
            observation_prime.squeeze(), 
            terminated or truncated))
        observation = observation_prime

        done = terminated or truncated
        if done:
            observation, _ = env.reset()
       
    return buffer, env


class MetricTracker:
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def add_step_reward(self, reward):
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
    def end_episode(self):
        self.rewards.append(self.current_episode_reward)
        self.episode_lengths.append(self.current_episode_length)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    @property
    def avg_reward(self):
        return np.mean(self.rewards) if self.rewards else 0
        
    @property
    def avg_episode_length(self):
        return np.mean(self.episode_lengths) if self.episode_lengths else 0