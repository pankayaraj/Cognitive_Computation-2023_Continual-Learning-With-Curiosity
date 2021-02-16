import torch
from algorithms.q_learning import Q_learning
from custom_envs.sumo.custom_sumo_env import SUMOEnv
from custom_envs.custom_acrobat import AcrobotEnv
from model import NN_Paramters
from parameters import Algo_Param
import gym_minigrid
import gym
env = AcrobotEnv()
env2 = AcrobotEnv()
import numpy as np



s = env.reset()

#env = MountainCarEnv()
#env2 = MountainCarEnv()
print("a")
print(env.action_space, env.observation_space.shape)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001,)
algo_param = Algo_Param()
algo_param.gamma = 0.99

max_episodes = 200




Q = Q_learning(env, q_nn_param=q_nn_param, algo_param=algo_param,  tau=0.0005)

#Q.load("q", "target_q")
update_interval = 10
save_interval = 1000
eval_interval = 1000
state = Q.initalize()

for i in range(200000):

    Q.update()
    state = Q.step(state)

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
