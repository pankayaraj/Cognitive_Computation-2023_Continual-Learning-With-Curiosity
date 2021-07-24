import torch
import numpy as np
import copy
import gym

from algorithms.test_algo.SAC_task_relevance_test_log_ratio import SAC_TR_test
from parameters import Algo_Param, NN_Paramters, Load_Paths, Save_Paths, Log_Ratio_Algo_Param
from util.new_replay_buffers.task_relevance.replay_buffer import Replay_Memory_TR
from custom_envs.custom_pendulum import PendulumEnv
from custom_envs.custom_lunar_lander import LunarLanderContinuous
import gym  # open ai gym
from custom_envs.sumo.custom_sumo_env import SUMOEnv

import custom_envs.pybulletgym_custom
from value_dice.war import Wasserstein
from gym.envs.registration import register



env = PendulumEnv()
env_eval = PendulumEnv()
env_eval2 = PendulumEnv()


env.__init__()
env_eval.__init__()

env.l = 1.4
env_eval.l  = 1.4
val = str("1_4")
exp = 1
buff1 = torch.load("old_buffers/Pendulum/l_1_0/mem")
buff2 = torch.load("old_buffers/Pendulum/l_1_4/mem")


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

save_path = Save_Paths()
load_path= Load_Paths()

War = Wasserstein( nu_param=nu_nn_param, algo_param=log_algo_param, deterministic_env=False, averege_next_nu=True,
                                    discrete_policy=False, save_path=save_path.nu_path, load_path=load_path.nu_path)
batch_size = 10
save_interval = 500
eval_interval = 5

for i in range(60000):

    #if i%1000 == 0:
    #    print(i, A.alpha)

    War.train_ratio(buff1.sample(batch_size=batch_size), buff1.sample(batch_size=batch_size), unweighted=True)

    if i%eval_interval==0:
        print("testing")
        e = env_eval



        print("KL at itr " + str(i) + " = " + str(KL.get_KL(buff1.sample(batch_size=batch_size), buff1.sample(batch_size=batch_size), unweighted=True)) )
        print(War.loss1, War.loss2)
        print(War.u_loss1)
        print(War.x)




#torch.save(A.replay_buffer, "old_buffers/Pendulum/l_1_0/mem")

