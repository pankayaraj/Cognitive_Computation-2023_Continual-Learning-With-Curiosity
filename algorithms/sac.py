import numpy as np
import torch

from model import Q_Function_NN, Value_Function_NN, Continuous_Gaussian_Policy
from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths
from util.replay_buffer import Replay_Memory



class SAC():

    def __init__(self, env, q_nn_param, policy_nn_param, algo_nn_param, max_episodes =100, memory_capacity =10000,
                 batch_size=400, save_path = Save_Paths(), load_path= Load_Paths(), action_space = None
                 ):

        self.env = env

        self.q_nn_param = q_nn_param
        self.policy_nn_param = policy_nn_param
        self.algo_nn_param = algo_nn_param


        self.gamma = self.algo_nn_param.gamma
        self.alpha = self.algo_nn_param.alpha
        self.tau   = self.algo_nn_param.tau

        self.target_update_interval = self.algo_nn_param.target_update_interval
        self.automatic_alpha_tuning = self.algo_nn_param.automatic_alpha_tuning

        self.critic = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
        self.critic_target = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
        self.critic_target.load_state_dict(self.critic.state_dict())


        self.policy = Continuous_Gaussian_Policy(policy_nn_param, save_path=save_path.policy_path,
                                                 load_path=load_path.policy_path, action_space=action_space)

        self.critic_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)
        self.policy_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)

        