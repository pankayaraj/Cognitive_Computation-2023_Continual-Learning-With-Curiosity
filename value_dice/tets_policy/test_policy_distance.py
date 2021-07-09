import torch
import numpy as np
import copy
import gym

from algorithms.test_algo.SAC_task_relevance_test_policy_distance import SAC_TR_P_test
from parameters import Algo_Param, NN_Paramters, Load_Paths, Save_Paths, Log_Ratio_Algo_Param
from util.new_replay_buffers.task_relevance_policy.replay_buffer import Replay_Memory_TR_P
from custom_envs.custom_pendulum import PendulumEnv
from custom_envs.custom_lunar_lander import LunarLanderContinuous
import gym  # open ai gym
from custom_envs.sumo.custom_sumo_env import SUMOEnv

import custom_envs.pybulletgym_custom


from gym.envs.registration import register



env = PendulumEnv()
env_eval = PendulumEnv()
env_eval2 = PendulumEnv()


env.__init__()
env_eval.__init__()

env.l = 1.8
env_eval.l  = 1.8
val = str("1_8")
exp = 1
old_memory = torch.load("old_buffers/Pendulum/l_1_4/mem")


#state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print((state_dim, action_dim))

q_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256,256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)
nu_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256,256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)
policy_nn_param = NN_Paramters(state_dim, action_dim, hidden_layer_dim=[256,256],
                          non_linearity=torch.relu, device=torch.device("cuda"), l_r=0.0001)

algo_nn_param = Algo_Param(gamma=0.99, alpha=0.2, tau=0.005, target_update_interval=1, automatic_alpha_tuning=True)
log_algo_param = Log_Ratio_Algo_Param(gamma=0.99, hard_update_interval=1)

#Pendulum
A = SAC_TR_P_test(env,nu_param=nu_nn_param, log_algo_param=log_algo_param,
        q_nn_param=q_nn_param, policy_nn_param=policy_nn_param, algo_nn_param=algo_nn_param, max_episodes=200, memory_capacity=10000,
                log_ratio_memory_capacity=10000, fifo_frac=0.05,
        batch_size=512, alpha_lr=0.0003, env_type="roboschool", buffer_type="Half_Reservior_FIFO_with_FT")





save_interval = 500
eval_interval = 500

state = A.initalize()



#A.load("q1", "q2", "q1", "q2", "policy_target")
L1, L2, L3, L4 = [], [], [], []
eval = False
#eval = True
for i in range(60000):

    #if i%1000 == 0:
    #    print(i, A.alpha)

    if eval == False:
        A.update()

    if i < A.batch_size:
        state = A.step(state, random=True)
    else:
        state = A.step(state, random=False)
    if i%save_interval==0:
        if eval == False:
            #A.save("q1", "q2", "q1_target", "q2_target", "policy_target")
            pass
    if i%eval_interval==0:
        print("testing")
        e = env_eval

        rew_total = 0
        for _ in range(10):
            rew = 0

            #e.render()
            s = e.reset()
            i_s = s
            for j in range(A.max_episodes):
                a = A.get_action(s, evaluate=True)
                n_s, r, d, _ = e.step(a)
                #e.render("gui")
                #if i%(save_interval*1) == 0:
                #    e.render(mode='rgb_array')
                rew += r

                s = n_s
                if d == True:
                    break


            rew_total += rew
        rew_total = rew_total/10


        print("reward at itr " + str(i) + " = " + str(rew_total) + " length = " + str(e.l) )#+ " at alpha: " + str(A.alpha.cpu().detach().numpy()[0]) )

        l1 = A.get_bhattacharyya_distance(old_memory.sample(len(old_memory)))
        #l2 = A.get_KL(data=A.log_ratio_memory[0].sample(A.log_ratio_memory_capacity), unweighted=True  )[0].item()
        #l3 = A.get_KL(data=last_ten_eps.sample(len(last_ten_eps)), unweighted=False)[0].item()
        #l4 = A.get_KL(data=last_ten_eps.sample(len(last_ten_eps)), unweighted=True)[0].item()

        L1.append(l1)
        #L2.append(l2)
        #L3.append(l3)
        #L4.append(l4)

        print("log ratio = " + str(l1))
        #print("log ratio = " + str(l2))
        #print("log ratio2 = " + str(l3))
        #print("log ratio2 = " + str(l4))



#torch.save(A.replay_buffer, "old_buffers/Pendulum/l_1_0/mem")

torch.save(L1, "distance_values/l_1_0/"+ val + "/L1_" + str(exp) )
#torch.save(L2, "KL_values/l_0_0/"+ val + "/L2_" + str(exp))
#torch.save(L3, "KL_values/l_0_0/"+ val + "/L3_" + str(exp))
#torch.save(L4, "KL_values/l_0_0/"+ val + "/L4_" + str(exp))