import torch
import numpy as np

import gym

from algorithms.SAC import SAC
from parameters import Algo_Param, NN_Paramters, Load_Paths, Save_Paths
from custom_envs.custom_pendulum import PendulumEnv

env = gym.make("Pendulum-v0")
env_eval = gym.make("Pendulum-v0")

#env = gym.make("MountainCarContinuous-v0")
#env_eval = gym.make("MountainCarContinuous-v0")

#env = PendulumEnv()
#env_eval = PendulumEnv()

seed = env.seed()[0]
env_eval.seed(seed)

env.l = 1.
env_eval.l  = 1.

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.003)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.003)

algo_nn_param = Algo_Param(gamma=0.99, alpha=0.2, tau=0.005, target_update_interval=1, automatic_alpha_tuning=True)


A = SAC(env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes=200, memory_capacity=100000
        ,batch_size=256, alpha_lr=0.0003)
#A.load("q1", "q2", "q1", "q2", "policy_target")

save_interval = 1000
eval_interval = 1000

state = A.initalize()

for i in range(100000):

    A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        #A.save("q1", "q2", "q1_target", "q2_target", "policy_target")
        pass
    if i%eval_interval==0:

        e = env_eval


        rew_total = 0
        for _ in range(10):
            rew = 0
            s = e.reset()
            for j in range(A.max_episodes):
                a = A.get_action(s, evaluate=True)
                s, r, d, _ = e.step(a)
                rew += r

                #e.render()

                if d == True:
                    break

            rew_total += rew
        rew_total = rew_total/10
        print("reward at itr " + str(i) + " = " + str(rew_total) + " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )
torch.save(A.replay_buffer, "mem")