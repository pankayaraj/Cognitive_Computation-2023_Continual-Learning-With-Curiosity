from custom_envs.sumo_rl.environment.env import SumoEnvironment
import argparse
from test_alogrithms.test_other.normalized_env import NormalizedEnv
import gym
import numpy as np
import gym_miniworld

env = gym.make('MiniWorld-CollectHealth-v0')


s = env.reset()
env.render()
print(np.shape(s))
print(env.action_space)
for  i in range(10000):
    a = env.action_space.sample()
    env.render()
    n_s, r, done, _ = env.step(a)
    s = n_s
