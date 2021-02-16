import torch
import numpy as np
import gym
from algorithms.q_learning_cnn import Q_learning
from custom_envs.custom_acrobat import AcrobotEnv
from custom_envs.custom_mountain_car_discrete import MountainCarEnv
import gym_sokoban
from atari_wrapper import make_atari, wrap_deepmind
import gym_miniworld

from model import NN_Paramters
from parameters import Algo_Param, Save_Paths, Load_Paths
# Use the Baseline Atari environment because of Deepmind helper functions
"""
env = gym.make('MiniWorld-CollectHealth-v0')
env2 = gym.make('MiniWorld-CollectHealth-v0')
state = env.reset()
print(np.shape(state))
"""
# Warp the frames, grey scale, stake four frame and scale to smaller ratio
#env = wrap_deepmind(env, frame_stack=True, scale=True)




#env = MountainCarEnv()
#env2 = MountainCarEnv()
env = gym.make('MiniGrid-Empty-8x8-v0')
env2 = gym.make('MiniGrid-Empty-8x8-v0')

print(env.action_space.shape)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[512],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.001)
algo_param = Algo_Param()
algo_param.gamma = 0.99

max_episodes = 200





Q = Q_learning(env, q_nn_param=q_nn_param, algo_param=algo_param)
print(Q.Q)
#env2 = env
#Q.load("q", "target_q")
update_interval = 10
save_interval = 1000
eval_interval = 10000
state = Q.initalize()

for i in range(100000):
    print(i)
    if i%100 == 0:
        print(i)
    Q.update()
    state = Q.step(state)
    if i%update_interval == 0:
        Q.hard_update()
    if i%save_interval == 0:
        print("saving")
        Q.save("q", "target_q")
    if i%eval_interval == 0:
        s = env2.reset()

        i_s = s
        rew = 0
        for j in range(max_episodes):
            a = Q.get_action(s)
            s, r, d, _ = env2.step(a)
            rew += r
            env2.render()
            if j == max_episodes-1:
                d = True
            if d == True:
                break
        print("reward at itr " + str(i) + " = " + str(rew) )
#Q.memory = torch.load("mem")
torch.save(Q.memory, "mem")
