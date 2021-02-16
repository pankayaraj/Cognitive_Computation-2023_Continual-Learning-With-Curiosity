import torch
from algorithms.q_learning import Q_learning
from custom_envs.sumo.custom_sumo_env import SUMOEnv
from custom_envs.custom_acrobat import AcrobotEnv
from model import NN_Paramters
from parameters import Algo_Param
import gym_minigrid
import gym
env1 = AcrobotEnv(mass=1.)
env_eval1 = AcrobotEnv(mass=1.)
env2 = AcrobotEnv(mass=3.)
env_eval2 = AcrobotEnv(mass=3.)

env = env1
env_eval = env_eval1
import numpy as np



s = env.reset()

#env = MountainCarEnv()
#env2 = MountainCarEnv()
print("a")
print(env.action_space, env.observation_space.shape)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n


q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)
algo_param = Algo_Param()
algo_param.gamma = 0.99

max_episodes = 200




Q = Q_learning(env, q_nn_param=q_nn_param, algo_param=algo_param, tau = 0.005)

#Q.load("q", "target_q")
update_interval = 10
save_interval = 1000
eval_interval = 1000
state = Q.initalize()


envs = [env1, env2]
env1.reset()
env2.reset()

x = 2000

print("x = " + str(x) )
for i in range(200000):
    if i%1000 == 0:
        print("power = " + str(Q.env.m))
        if i%x == 0:
            Q.env = envs[0]
        else:
            Q.env = envs[1]


    Q.update()
    state = Q.step(state)

    if i%save_interval == 0:
        print("saving")
        #Q.save("q", "target_q")
    if i%eval_interval == 0:

        e = env_eval1
        s = e.reset()

        i_s = s
        rew = 0
        for j in range(max_episodes):
            a = Q.get_action(s)
            s, r, d, _ = e.step(a)
            rew += r
            #e.render()
            if j == max_episodes-1:
                d = True
            if d == True:
                break
        print("reward at itr " + str(i) + " = " + str(rew)  + " mass = " + str(e.m))

        e = env_eval2
        s = e.reset()

        i_s = s
        rew = 0
        for j in range(max_episodes):
            a = Q.get_action(s)
            s, r, d, _ = e.step(a)
            rew += r
            #e.render()
            if j == max_episodes - 1:
                d = True
            if d == True:
                break
        print("reward at itr " + str(i) + " = " + str(rew) + " mass = " + str(e.m))
#Q.memory = torch.load("mem")
#torch.save(Q.memory, "mem")
