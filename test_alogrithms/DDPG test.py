import torch
import numpy as np
import gym

from algorithms.DDPG import DDPG
from parameters import Algo_Param_DDPG, NN_Paramters, Load_Paths, Save_Paths


env = gym.make("Pendulum-v0")
env_eval = gym.make("Pendulum-v0")

#env = gym.make("MountainCarContinuous-v0")
#env_eval = gym.make("MountainCarContinuous-v0")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cpu"), l_r=0.003)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cpu"), l_r=0.003)

algo_nn_param = Algo_Param_DDPG(gamma=0.995, tau=0.005, target_update_interval=1, noise="gaussian", std=1)


A = DDPG(env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes=1000, memory_capacity=100000
        ,batch_size=256, noise ="gaussian")


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
        A.save("q", "q_target", "policy", "policy_target")
    if i%eval_interval==0:

        e = env_eval
        s = e.reset()
        rew = 0
        for j in range(A.max_episodes):
            a = A.get_action(s, evaluate=True)
            s, r, d, _ = e.step(a)
            rew += r
            e.render()
            if d == True:
                break

        print("reward at itr " + str(i) + " = " + str(rew)  )
torch.save(A.replay_buffer, "mem")