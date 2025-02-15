import torch
import numpy as np

class NN_Paramters():
    def __init__(self, state_dim, action_dim, non_linearity = torch.nn.functional.tanh, weight_initializer = 'xavier', bias_initializer = 'zero',
                 hidden_layer_dim = [128, 128], device= torch.device('cuda'), l_r=0.0001, CNN_layers = [(4, 32, 8, 4), (32, 64, 4, 2), (64, 64 ,3, 1)],
                 flatten_dim = 7*7*64, CNN_initalizer = "kaiming"):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.non_linearity = non_linearity
        self.l_r = l_r

        self.CNN_layers = CNN_layers
        self.flatten_dim = flatten_dim
        self.CNN_initalizer = CNN_initalizer


        self.device = device



class Algo_Param():
    def __init__(self, gamma=0.995, alpha=0.2, tau=0.005, target_update_interval=1,
                automatic_alpha_tuning=False
                ):
        self.gamma = gamma
        self.alpha = alpha
        self.tau   = tau
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning

class Log_Ratio_Algo_Param():
    def __init__(self, gamma=0.9, hard_update_interval=1):
        self.gamma = gamma
        self.hard_update_interval = hard_update_interval

class Algo_Param_DDPG():
    def __init__(self, gamma=0.995, tau=0.005, target_update_interval=1,
                noise = "gaussian", depsilon = 50000, std = 1.0
                ):
        self.gamma = gamma
        self.noise = "gaussian"
        self.tau   = tau
        self.target_update_interval = target_update_interval
        self.depsilon = depsilon
        self.std = std



class Save_Paths():

    def __init__(self, policy_path="policy_temp", q_path="q_temp", target_q_path="target_q_temp",
                 v_path="v_temp", nu_path = "nu_temp", zeta_path="zeta_temp",
                 icm_n_state_path = "icm_n_state", icm_action_path = "icm_action",
                 icm_reward_path = "icm_reward_path"):

        self.policy_path = policy_path
        self.q_path = q_path
        self.target_q_path = target_q_path
        self.v_path = v_path
        self.nu_path = nu_path
        self.zeta_path = zeta_path
        self.icm_n_state_path = icm_n_state_path
        self.icm_action_path  = icm_action_path
        self.icm_reward_path = icm_reward_path

class Load_Paths():

    def __init__(self, policy_path="policy_temp", q_path="q_temp", target_q_path="target_q_temp",
                 v_path="v_temp", nu_path="nu_temp", zeta_path="zeta_temp",
                 icm_n_state_path = "icm_n_state", icm_action_path = "icm_action",
                 icm_reward_path = "icm_reward_path"):

        self.policy_path = policy_path
        self.q_path = q_path
        self.target_q_path = target_q_path
        self.v_path = v_path
        self.nu_path = nu_path
        self.zeta_path = zeta_path
        self.icm_n_state_path = icm_n_state_path
        self.icm_action_path = icm_action_path
        self.icm_reward_path = icm_reward_path