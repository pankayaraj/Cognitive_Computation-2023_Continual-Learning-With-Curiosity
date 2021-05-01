import numpy as np
import torch

from model import Q_Function_NN, Value_Function_NN, Continuous_Gaussian_Policy, ICM_Action_NN, ICM_Next_State_NN, \
    ICM_Reward_NN
from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths

from util.old_buffers.gradual_penalized_cur.half_res_w_cur_ft_fifo_gradual import Half_Reservoir_Flow_Through_w_Cur_Gradual
from util.old_buffers.gradual_penalized_cur.custom_hrf import Custom_HRF

class Debug():
    def __init__(self):
        self.icm_next_state_loss = 0
        self.icm_action_loss = 0

        self.f_icm_r = 0
        self.i_icm_r = 0

    def print_all(self):
        print("ICM_LOSS = " + str(self.icm_next_state_loss.item()) + " , " + str(self.icm_action_loss.item())
              + " I_rew = " + str(self.f_icm_r.item())
              + ", " + str(self.i_icm_r.item()))


class SAC_Test():

    def __init__(self, env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param, max_episodes=100,
                 memory_capacity=10000,
                 batch_size=400, save_path=Save_Paths(), load_path=Load_Paths(), action_space=None, alpha_lr=0.0003,
                 debug=Debug(), buffer_type="Reservior", update_curiosity_from_fifo=True, fifo_frac=0.34,
                 no_cur_network=5, change_at = [0, 50000, 350000]):

        self.env = env
        self.device = q_nn_param.device

        self.q_nn_param = q_nn_param
        self.policy_nn_param = policy_nn_param
        self.algo_nn_param = algo_nn_param
        self.icm_nn_param = icm_nn_param

        self.update_curiosity_from_fifo = update_curiosity_from_fifo

        self.alpha_lr = alpha_lr
        self.gamma = self.algo_nn_param.gamma
        self.alpha = self.algo_nn_param.alpha
        self.tau = self.algo_nn_param.tau

        self.max_episodes = max_episodes
        self.steps_done = 0  # total no of steps done
        self.steps_per_eps = 0  # this is to manually enforce max eps length
        self.update_no = 0
        self.batch_size = batch_size

        self.target_update_interval = self.algo_nn_param.target_update_interval
        self.automatic_alpha_tuning = self.algo_nn_param.automatic_alpha_tuning

        self.critic_1 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
        self.critic_2 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)

        self.critic_target_1 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path,
                                             load_path=load_path.q_path)
        self.critic_target_2 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path,
                                             load_path=load_path.q_path)

        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

        self.icm_action = []
        self.icm_next_state = []
        self.icm_reward = []

        self.icm_next_state_optim = []
        self.icm_action_optim = []
        self.icm_reward_optim = []

        self.no = no_cur_network
        for i in range(self.no):
            self.icm_next_state.append(
                ICM_Next_State_NN(icm_nn_param, save_path.icm_n_state_path, load_path.icm_n_state_path))
            self.icm_action.append(ICM_Action_NN(icm_nn_param, save_path.icm_action_path, load_path.icm_action_path))
            self.icm_reward.append(ICM_Reward_NN(icm_nn_param, save_path.icm_action_path, load_path.icm_action_path))

        for i in range(self.no):
            self.icm_next_state_optim.append(
                torch.optim.Adam(self.icm_next_state[i].parameters(), self.icm_nn_param.l_r))
            self.icm_action_optim.append(torch.optim.Adam(self.icm_action[i].parameters(), self.icm_nn_param.l_r))
            self.icm_reward_optim.append(torch.optim.Adam(self.icm_reward[i].parameters(), self.icm_nn_param.l_r))

        self.icm_i_r = [[] for i in range(self.no)]
        self.icm_f_r = [[] for i in range(self.no)]
        self.icm_r = [[] for i in range(self.no)]

        self.policy = Continuous_Gaussian_Policy(policy_nn_param, save_path=save_path.policy_path,
                                                 load_path=load_path.policy_path, action_space=action_space)

        self.critic_1_optim = torch.optim.Adam(self.critic_1.parameters(), self.q_nn_param.l_r)
        self.critic_2_optim = torch.optim.Adam(self.critic_2.parameters(), self.q_nn_param.l_r)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), self.q_nn_param.l_r)

        if self.automatic_alpha_tuning is True:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer_type = buffer_type


        if buffer_type == "Custom":
            self.replay_buffer = Custom_HRF(capacity=memory_capacity, fifo_fac=fifo_frac, change_at = change_at)
        elif buffer_type == "Half_Reservior_FIFO_with_FT":
            self.replay_buffer = Half_Reservoir_Flow_Through_w_Cur_Gradual(capacity=memory_capacity,
                                                                           curisoity_buff_frac=0.34,
                                                                           seperate_cur_buffer=True,
                                                                           fifo_fac=fifo_frac)


        self.debug = debug

    def get_action(self, state, evaluate=False):

        action, log_prob, action_mean = self.policy.sample(state, format="torch")

        if evaluate == False:
            return action.cpu().detach().numpy(), action_mean.cpu().detach().numpy()
        else:
            return action_mean.cpu().detach().numpy()

    def initalize(self):

        # inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.steps_per_eps = 0
        state = self.env.reset()
        for i in range(self.batch_size):
            state = self.step(state)
        return state

    def update(self, batch_size=None, factor=0.7):

        if batch_size == None:
            batch_size = self.batch_size
        if batch_size > len(self.replay_buffer):
            return

        self.update_no += 1

        batch = self.replay_buffer.sample(batch_size=batch_size)

        state_batch = batch.state
        action_batch = batch.action
        action_mean_batch = batch.action_mean
        next_state_batch = batch.next_state
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.q_nn_param.device)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1).to(self.q_nn_param.device)

        # critic update
        with torch.no_grad():
            next_action_batch, next_log_prob_batch, _ = self.policy.sample(next_state_batch, format="torch")
            q1_next_target = self.critic_target_1.get_value(next_state_batch, next_action_batch, format="torch")
            q2_next_target = self.critic_target_2.get_value(next_state_batch, next_action_batch, format="torch")
            min_q_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob_batch
            next_q_value = reward_batch + done_mask_batch * self.gamma * min_q_target

        q1 = self.critic_1.get_value(state_batch, action_batch)
        q2 = self.critic_2.get_value(state_batch, action_batch)

        q1_loss = 0.5 * torch.nn.functional.mse_loss(q1, next_q_value)
        q2_loss = 0.5 * torch.nn.functional.mse_loss(q2, next_q_value)

        self.critic_1_optim.zero_grad()
        q1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        q2_loss.backward()
        self.critic_2_optim.step()

        # decide weather to update the curiosity from the entire buffer or just the fifo buffer
        for i in range(self.no):
            if len(self.replay_buffer.reservior_buffer) == 0:
                fifo_batchs = [self.replay_buffer.fifo_buffer.sample(batch_size)]
            else:
                fifo_batchs = self.replay_buffer.sample_individual(batch_size=batch_size, factor=factor)
            self.update_curiosity(fifo_batchs, index=i)

        # policy update
        pi, log_pi, pi_m = self.policy.sample(state_batch)

        # alpha update
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().detach()

        q1_pi = self.critic_1.get_value(state_batch, pi)
        q2_pi = self.critic_2.get_value(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if self.update_no % self.target_update_interval == 0:
            self.soft_update(self.critic_target_1, self.critic_1, self.tau)
            self.soft_update(self.critic_target_2, self.critic_2, self.tau)



    def update_curiosity(self, batchs, index):
        # icm update
        for ind, batch in enumerate(batchs):

            state_batch = batch.state
            action_batch = batch.action
            action_mean_batch = batch.action_mean
            next_state_batch = batch.next_state
            reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.q_nn_param.device)
            done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1).to(self.q_nn_param.device)

            pred_next_state = self.icm_next_state[index].get_next_state(state_batch, action_batch, format="torch")
            pred_action = self.icm_action[index].get_action(state_batch, next_state_batch, format="torch")
            pred_reward = self.icm_reward[index].get_reward(state_batch, action_batch, format="torch")

            factor =1e-4


            if ind < len(batchs) - 1:
                if ind == 0:

                    #loss = torch.nn.functional.mse_loss(pred_action, torch.FloatTensor(action_batch).to(
                    #    self.icm_nn_param.device))

                    loss = 0.5 * torch.nn.functional.mse_loss(pred_reward, reward_batch).to(self.icm_nn_param.device)

                    #icm_action_loss = -factor * 0.5 * torch.sigmoid_(loss)
                    icm_reward_loss = -factor * 0.5 * torch.sigmoid_(loss)


                else:

                    #loss = torch.nn.functional.mse_loss(pred_action, torch.FloatTensor(action_batch).to(
                    #    self.icm_nn_param.device))

                    loss = 0.5 * torch.nn.functional.mse_loss(pred_reward, reward_batch).to(self.icm_nn_param.device)

                    #icm_action_loss += -factor * 0.5 * torch.sigmoid_(loss)
                    icm_reward_loss += -factor * 0.5 * torch.sigmoid_(loss)
            else:

                if ind == 0:

                    #icm_action_loss = 0.5 * torch.nn.functional.mse_loss(pred_action,
                    #                                                     torch.FloatTensor(action_batch).to(
                    #                                                         self.icm_nn_param.device))

                    icm_reward_loss = 0.5 * torch.nn.functional.mse_loss(pred_reward, reward_batch).to(self.icm_nn_param.device)
                else:

                    #icm_action_loss += 0.5 * torch.nn.functional.mse_loss(pred_action,
                    #                                                      torch.FloatTensor(action_batch).to(
                    #                                                          self.icm_nn_param.device))

                    icm_reward_loss += 0.5 * torch.nn.functional.mse_loss(pred_reward, reward_batch).to(
                        self.icm_nn_param.device)

        #self.icm_next_state_optim[index].zero_grad()
        #icm_next_state_loss.backward()
        #self.icm_next_state_optim[index].step()

        #self.icm_action_optim[index].zero_grad()
        #icm_action_loss.backward()
        #self.icm_action_optim[index].step()

        self.icm_reward_optim[index].zero_grad()
        icm_reward_loss.backward()
        self.icm_reward_optim[index].step()

    def step(self, state, random=False):
        batch_size = 1  # since step is for a single sample

        if random:
            action = self.env.action_space.sample()
            action_mean = action
        else:
            action, action_mean = self.get_action(state, evaluate=False)

        next_state, reward, done, _ = self.env.step(action)
        curiosity = torch.Tensor([0])
        for i in range(self.no):
            p_next_state = self.icm_next_state[i].get_next_state(state, action)
            p_action = self.icm_action[i].get_action(state, next_state)
            p_reward = self.icm_reward[i].get_reward(state, action)


            with torch.no_grad():
                f_icm_r = torch.nn.functional.mse_loss(p_next_state,
                                                       torch.Tensor(next_state).to(
                                                           self.icm_nn_param.device)).cpu().detach().numpy()
                i_icm_r = torch.nn.functional.mse_loss(p_action, torch.Tensor(action).to(
                    self.icm_nn_param.device)).cpu().detach().numpy()

                r_icm_r = (p_reward - reward) ** 2

            self.icm_f_r[i].append(f_icm_r)
            self.icm_i_r[i].append(i_icm_r)
            self.icm_r[i].append(r_icm_r)

            #curiosity[0] += i_icm_r.item() / self.no
            curiosity[0] += r_icm_r.item() / self.no

        self.steps_done += 1
        self.steps_per_eps += 1

        if done:
            mask = 0.0
            self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask)
            next_state = self.env.reset()
            self.steps_per_eps = 0
            return next_state

        if self.steps_per_eps == self.max_episodes:
            mask = 1.0
            self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask)
            next_state = self.env.reset()
            self.steps_per_eps = 0
            return next_state
        mask = 1.0

        self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask)

        return next_state

    def hard_update(self):
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, critic_1_path="critic_1", critic_2_path="critic_2",
             critic_1_target_path="critic_1_target", critic_2_target_path="critic_2_target",
             policy_path="policy_target", icm_state_path="icm_state", icm_action_path="icm_action"):

        self.critic_1.save(critic_1_path)
        self.critic_2.save(critic_2_path)
        self.critic_target_1.save(critic_1_target_path)
        self.critic_target_2.save(critic_2_target_path)
        self.policy.save(policy_path)

        for i in range(self.no):
            self.icm_next_state[i].save(icm_state_path + str(i))
            self.icm_action[i].save(icm_action_path + str(i))

    def load(self, critic_1_path="critic_1", critic_2_path="critic_2",
             critic_1_target_path="critic_1_target", critic_2_target_path="critic_2_target",
             policy_path="policy_target", icm_state_path="icm_state", icm_action_path="icm_action"):

        self.critic_1.load(critic_1_path)
        self.critic_2.load(critic_2_path)
        self.critic_target_1.load(critic_1_target_path)
        self.critic_target_2.load(critic_2_target_path)
        self.policy.load(policy_path)

        for i in range(self.no):
            self.icm_next_state[i].load(icm_state_path + str(i))
            self.icm_action[i].load(icm_action_path + str(i))

    def get_curiosity_rew(self, state, action, next_state):

        p_n_s = self.icm_nxt_state.get_next_state(state, action)
        p_a = self.icm_action.get_action(state, next_state)

        f_icm_r = torch.nn.functional.mse_loss(p_n_s, torch.FloatTensor(next_state))
        i_icm_r = torch.nn.functional.mse_loss(p_a, torch.FloatTensor(action))

        return f_icm_r, i_icm_r