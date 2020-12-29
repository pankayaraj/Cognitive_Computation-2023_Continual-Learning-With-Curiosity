import torch
import numpy

from model import ICM_Next_State_NN, ICM_Action_NN, Value_Function_NN, Q_Function_NN, ICM_Reward_NN
from parameters import Save_Paths, Load_Paths

class CDESP():

    def __init__(self, env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes =100, memory_capacity =10000,
                 batch_size=400, save_path = Save_Paths(), load_path= Load_Paths()):

        self.env = env

        self.q_nn_param = q_nn_param
        self.icm_nn_param = icm_nn_param
        self.algo_nn_param = algo_nn_param
        self.policy_nn_param = policy_nn_param

        self.max_episodes = max_episodes
        self.steps_done = 0  # total no of steps done
        self.steps_per_eps = 0  # this is to manually enforce max eps length
        self.update_no = 0
        self.batch_size = batch_size

        self.target_update_interval = self.algo_nn_param.target_update_interval

        self.critic = Q_Function_NN(self.q_nn_param, save_path, load_path)

        self.critic_target = Q_Function_NN(self.q_nn_param, save_path, load_path)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_optim = torch.optim.Adam(self.critic.parameters(), self.q_nn_param.l_r)

        self.icm_nxt_state = ICM_Next_State_NN(icm_nn_param, save_path.icm_n_state_path, load_path.icm_n_state_path)
        self.icm_action = ICM_Action_NN(icm_nn_param, save_path.icm_action_path, load_path.icm_action_path)
        self.icm_reward = ICM_Reward_NN(icm_nn_param, save_path.icm_reward_path, load_path.icm_reward_path)

        self.icm_nxt_state_optim = torch.optim.Adam(self.icm_nxt_state.parameters(), self.icm_nn_param.l_r)
        self.icm_action_optim = torch.optim.Adam(self.icm_action.parameters(), self.icm_nn_param.l_r)
        self.icm_reward_optim = torch.optim.Adam(self.icm_reward.parameters(), self.icm_nn_param.l_r)

