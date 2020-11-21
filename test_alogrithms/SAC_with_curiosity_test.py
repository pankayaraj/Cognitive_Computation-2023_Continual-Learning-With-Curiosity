import torch
import numpy as np
import gym

from algorithms.SAC_w_Curiosity import SAC_with_Curiosity
from parameters import Algo_Param, NN_Paramters, Load_Paths, Save_Paths


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
icm_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256, 256],
                          non_linearity=torch.relu, device=torch.device("cpu"), l_r=0.003)
algo_nn_param = Algo_Param(gamma=0.99, alpha=0.2, tau=0.005, target_update_interval=1, automatic_alpha_tuning=True)


A = SAC_with_Curiosity(env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes=1000, memory_capacity=100000
        ,batch_size=256, alpha_lr=0.0003)
length = [1, 100]

save_interval = 1000
eval_interval = 1000

#A.env.l = 10

state = A.initalize()

for i in range(100000):

    A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        A.save("q1", "q2", "q1_target", "q2_target", "policy_target")
    if i%eval_interval==0:

        for k in range(2):
            e = env_eval
            e.l = float(length[k])

            rew_total = 0
            icm_s_total = 0
            icm_a_total = 0
            for _ in range(10):
                rew = 0
                i_s = 0
                i_a = 0
                s = e.reset()
                for j in range(A.max_episodes):

                    a = A.get_action(s, evaluate=True)
                    n_s, r, d, _ = e.step(a)
                    rew += r

                    s_r, a_r = A.get_curiosity_rew(s, a, n_s)
                    i_s += s_r
                    i_a += a_r
                    #e.render()

                    s = n_s

                    if d == True:
                        break

                rew_total += rew
                icm_a_total += i_a
                icm_s_total += i_s
            rew_total = rew_total/10
            print("reward at itr " + str(i) + " = " + str(rew_total) + " at alpha: " + str(A.alpha.cpu().detach().numpy()[0])
                  + "for l " + str(length[k]) )
            print(icm_s_total, icm_a_total)

torch.save(A.replay_buffer, "mem")