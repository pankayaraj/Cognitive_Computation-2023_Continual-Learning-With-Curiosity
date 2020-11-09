import numpy as np
import torch

from model import Q_Function_NN, Value_Function_NN, Continuous_Gaussian_Policy
from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths
from util.replay_buffer import Replay_Memory



class SAC():

    def __init__(self, env, q_nn_param, policy_nn_param, algo_param, max_episodes =100, memory_capacity =10000,
                 batch_size=400, save_path = Save_Paths(), load_path= Load_Paths(),
                 ):

        self.env = env

        self.gamma = algo_param.gamma
        self.alpha = algo_param.alpha
        self.tau   = algo_param.tau

        self.q_nn_param = q_nn_param
        self.policy_nn_param = policy_nn_param
        self.algo_nn_param = algo_param