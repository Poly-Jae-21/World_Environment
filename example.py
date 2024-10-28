import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

env = gym.make("FrozenLake-v1", desc=generate_random_map(size=8))

observation, info = env.reset(seed=42)
print(env.observation_space)