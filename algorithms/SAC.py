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

        self.critic_1 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
        self.critic_2 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)

        self.critic_target_1 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
        self.critic_target_2 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.policy = Continuous_Gaussian_Policy(policy_nn_param, save_path=save_path.policy_path,
                                                 load_path=load_path.policy_path, action_space=action_space)

        self.critic_1_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)
        self.critic_2_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)
        self.policy_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)


        self.replay_buffer = Replay_Memory(capacity=memory_capacity)


    def get_action(self, state, evaluate=False):

        action, log_prob, action_mean = self.policy.sample(state, format="torch")

        if evaluate == False:
            return action.cpu().detach().numpy()
        else:
            return action_mean.detach().numpy()



    def update(self, batch_size, update_no):

        batch = self.replay_buffer.sample(batch_size=batch_size)

        state_batch = torch.tensor(batch.state)
        action_batch = torch.tensor(batch.action)
        next_state_batch = torch.tensor(batch.next_state)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1)

        with torch.no_grad():
            next_action_batch, next_log_prob_batch, _ = self.policy.sample(next_state_batch, format="torch")
            q1_next_target = self.critic_target_1.get_value(next_state_batch, next_action_batch, format="torch")
            q2_next_target = self.critic_target_2.get_value(next_state_batch, next_action_batch, format="torch")
            min_q_target = torch.min(q1_next_target, q2_next_target) - self.alpha*next_log_prob_batch
            next_q_value = reward_batch - done_mask_batch*self.gamma*min_q_target

        q1 = self.critic_1.get_value(state_batch, action_batch)
        q2 = self.critic_2.get_value(state_batch, action_batch)

        q1_loss = torch.nn.functional.mse_loss(q1, next_q_value)
        q2_loss = torch.nn.functional.mse_loss(q2, next_q_value)

        self.critic_1_optim.zero_grad()
        q1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        q2_loss.backward()
        self.critic_2_optim.step()

        pi, log_pi, _ = self.policy.sample(state_batch)

        q1_pi = self.critic_1.get_value(state_batch, pi)
        q2_pi = self.critic_2.get_value(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((self.alpha*log_pi) - min_q_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if update_no%self.target_update_interval == 0:
            self.soft_update(self.critic_target_1, self.critic_1, self.tau)
            self.soft_update(self.critic_target_2, self.critic_2, self.tau)

        update_no += 1

        return update_no

    def step(self, state):

        batch_size = 1  #since step is for a single sample
        action, _, _ = self.policy.sample(state, format="numpy")
        action = action[0]

        next_state, reward, done, _ = self.env.step(action)

        if done:
            mask = 0.0
            self.replay_buffer.push(state, action, reward, next_state, mask)
            next_state = self.env.reset()
            return next_state

        if self.time_step == self.max_episodes:
            mask = 1.0
            self.replay_buffer.push(state, action, reward, next_state, mask)
            next_state = self.env.reset()
            return next_state
        mask = 1.0
        self.replay_buffer.push(state, action, reward, next_state, mask)
        return next_state

    def hard_update(self):
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
