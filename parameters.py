import torch
import numpy as np

class NN_Paramters():
    def __init__(self, state_dim, action_dim, non_linearity = torch.nn.functional.tanh, weight_initializer = 'xavier', bias_initializer = 'zero',
                 hidden_layer_dim = [128, 128], device= torch.device('cuda'), l_r=0.0001):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.non_linearity = non_linearity
        self.l_r = l_r

        self.device = device



class Algo_Param():
    def __init__(self, gamma=0.995, alpha=0.2, tau=0.005):
        self.gamma = gamma
        self.alpha = alpha
        self.tau   = tau




class Save_Paths():

    def __init__(self, policy_path="policy_temp", q_path="q_temp", target_q_path="target_q_temp",
                 v_path="v_temp", nu_path = "nu_temp", zeta_path="zeta_temp"):

        self.policy_path = policy_path
        self.q_path = q_path
        self.target_q_path = target_q_path
        self.v_path = v_path
        self.nu_path = nu_path
        self.zeta_path = zeta_path


class Load_Paths():

    def __init__(self, policy_path="policy_temp", q_path="q_temp", target_q_path="target_q_temp",
                 v_path="v_temp", nu_path="nu_temp", zeta_path="zeta_temp"):
        self.policy_path = policy_path
        self.q_path = q_path
        self.target_q_path = target_q_path
        self.v_path = v_path
        self.nu_path = nu_path
        self.zeta_path = zeta_path
