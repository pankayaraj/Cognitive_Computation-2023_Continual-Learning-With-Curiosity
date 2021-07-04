import numpy as np
import torch

from model import Q_Function_NN, Value_Function_NN, Continuous_Gaussian_Policy, ICM_Action_NN, ICM_Next_State_NN, ICM_Reward_NN
from parameters import Algo_Param, NN_Paramters, Save_Paths, Load_Paths

from value_dice.log_ratio import Log_Ratio



#temporary buffers that holds the data while log_ratio is trained
from util.new_replay_buffers.task_relevance.gradual_task_with_relevance.buffer_for_log_ratio_training.reservoir_with_fifo_replay_buffer_flow_through import Half_Reservoir_with_FIFO_Flow_Through_Replay_Buffer_TR_w_Cur as HRF_LR

#replay buffer for main algorithm
from util.new_replay_buffers.task_relevance.gradual_task_with_relevance.half_res_w_cur_ft_fifo_gradual import Half_Reservoir_Flow_Through_w_Cur_Gradual_TR as HRF

class Debug():
    def __init__(self):
        self.icm_next_state_loss = 0
        self.icm_action_loss = 0

        self.f_icm_r = 0
        self.i_icm_r = 0

    def print_all(self):
        print("ICM_LOSS = " + str(self.icm_next_state_loss.item()) + " , " + str(self.icm_action_loss.item())
              + " I_rew = " + str(self.f_icm_r.item())
              + ", " + str(self.i_icm_r.item()) )



class SAC_with_Curiosity_Buffer():

    def __init__(self, env,nu_param, log_algo_param, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param,
                 no_log_ratio_eval_steps=40000, KL_threshold = 1000,
                 max_episodes =100, memory_capacity =10000,
                 batch_size=400, save_path = Save_Paths(), load_path= Load_Paths(), action_space = None, alpha_lr=0.0003,
                 debug=Debug(), buffer_type = "Reservior", update_curiosity_from_fifo = True, fifo_frac=0.34, curisoity_buff_frac = 0.34, no_cur_network=5,
                 reset_cur_on_task_change=True, reset_alpha_on_task_change=True, change_at = [100000, 350000],
                 fow_cur_w = 0.0, inv_cur_w = 1.0, rew_cur_w = 0.0,
                 n_k=600, l_k=8000, m_k=1.5,
                 priority = "uniform", cur_res_frac = 0.5):

        self.env = env
        self.device = q_nn_param.device

        self.priority = priority

        self.q_nn_param = q_nn_param
        self.policy_nn_param = policy_nn_param
        self.algo_nn_param = algo_nn_param
        self.icm_nn_param = icm_nn_param

        self.nu_param = nu_param
        self.log_algo_param = log_algo_param

        self.save_path = save_path
        self.load_path = load_path

        self.update_curiosity_from_fifo = update_curiosity_from_fifo
        self.reset_cur_on_task_change = reset_cur_on_task_change
        self.reset_alpha_on_task_change = reset_alpha_on_task_change

        self.alpha_lr = alpha_lr
        self.gamma = self.algo_nn_param.gamma
        self.alpha = self.algo_nn_param.alpha
        self.tau   = self.algo_nn_param.tau

        self.max_episodes = max_episodes
        self.steps_done = 0   #total no of steps done
        self.update_no = 0
        self.batch_size = batch_size

        self.fifo_frac = fifo_frac
        self.curiosity_frac = curisoity_buff_frac
        self.memory_capacity = memory_capacity
        self.action_sapce = action_space

        # log ratio
        self.steps_per_eps = 0  # this is to manually enforce max eps length and also to use in log ration calculation
        self.initial_state = None
        self.log_ratio = []  #add log ratio networks as needed later on
        self.temporary_reservoir = [] #store data during trainng log ratio phase so we can consolidate later on
        self.no_log_ratio_evaluation_steps = no_log_ratio_eval_steps
        self.KL_threshold = KL_threshold


        self.target_update_interval = self.algo_nn_param.target_update_interval
        self.automatic_alpha_tuning = self.algo_nn_param.automatic_alpha_tuning

        self.critic_1 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
        self.critic_2 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)

        self.critic_target_1 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)
        self.critic_target_2 = Q_Function_NN(nn_params=q_nn_param, save_path=save_path.q_path, load_path=load_path.q_path)

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
            self.icm_next_state.append(ICM_Next_State_NN(icm_nn_param, save_path.icm_n_state_path, load_path.icm_n_state_path))
            self.icm_action.append(ICM_Action_NN(icm_nn_param, save_path.icm_action_path, load_path.icm_action_path))
            self.icm_reward.append(ICM_Reward_NN(icm_nn_param, save_path.icm_action_path, load_path.icm_action_path))

        for i in range(self.no):
            self.icm_next_state_optim.append(torch.optim.Adam(self.icm_next_state[i].parameters(), self.icm_nn_param.l_r))
            self.icm_action_optim.append(torch.optim.Adam(self.icm_action[i].parameters(), self.icm_nn_param.l_r))
            self.icm_reward_optim.append(torch.optim.Adam(self.icm_reward[i].parameters(), self.icm_nn_param.l_r))

        self.icm_i_r = [[] for i in range(self.no)]
        self.icm_f_r = [[] for i in range(self.no)]
        self.icm_r = [[] for i in range(self.no)]

        self.alpha_history = []

        self.fow_cur_w = fow_cur_w
        self.inv_cur_w = inv_cur_w
        self.rew_cur_w = rew_cur_w


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


        if buffer_type == "Half_Reservior_FIFO_with_FT":
            self.replay_buffer = HRF(capacity=memory_capacity, curisoity_buff_frac=curisoity_buff_frac, seperate_cur_buffer=True,
                                                                          fifo_fac=fifo_frac,
                                                                          avg_len_snr=n_k, repetition_threshold=l_k, snr_factor=m_k,
                                                                           priority=priority)
        else:
            print("buffer type not specified")



        self.debug = debug

        self.change_var_at = change_at

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
        self.initial_state = state
        for i in range(self.batch_size):
            state = self.step(state)
        return state

    def get_target_policy(self):
        #this is the current policy the agent should evaluate against given the data
        target_policy = self.temporary_policy
        return target_policy

    def train_log_ratio(self):
        for i in range(self.no_tasks):
            data = self.replay_buffer.reservior_buffer.sample_from_sub_buffer(index=i, batch_size=self.batch_size)
            target_policy = self.get_target_policy()
            self.log_ratio[i].train_ratio(data, target_policy)

    def get_log_ratio(self, data):
        #here data can be off current policy's memory
        log_ratio_values = []
        for i in range(len(self.log_ratio)):
            target_policy = self.get_target_policy()
            log_ratio_values.append(self.log_ratio[i].get_log_state_action_density_ratio(data, target_policy))
        return log_ratio_values

    def get_neg_KL(self, data, unweighted=False):
        # here data can be off current policy's memory
        KL_values = []
        for i in range(len(self.log_ratio)):
            target_policy = self.get_target_policy()
            KL_values.append(self.log_ratio[i].get_KL(data, target_policy, unweighted))
        return KL_values




    def update(self, batch_size=None):

        if self.reset_cur_on_task_change:
            if self.replay_buffer.reservior_buffer.t_c:
                print("init")
                for N in self.icm_action:
                    N.__init__(self.icm_nn_param, self.save_path.icm_n_state_path, self.load_path.icm_n_state_path)
                for N in self.icm_next_state:
                    N.__init__(self.icm_nn_param, self.save_path.icm_n_state_path, self.load_path.icm_n_state_path)
                for N in self.icm_reward:
                    N.__init__(self.icm_nn_param, self.save_path.icm_n_state_path, self.load_path.icm_n_state_path)

        if self.reset_alpha_on_task_change:
            if self.replay_buffer.reservior_buffer.t_c:
            #for i in self.change_var_at:
                #if i == self.replay_buffer.reservior_buffer.time:
                self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.alpha_lr)



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

        #critic update
        with torch.no_grad():
            next_action_batch, next_log_prob_batch, _ = self.policy.sample(next_state_batch, format="torch")
            q1_next_target = self.critic_target_1.get_value(next_state_batch, next_action_batch, format="torch")
            q2_next_target = self.critic_target_2.get_value(next_state_batch, next_action_batch, format="torch")
            min_q_target = torch.min(q1_next_target, q2_next_target) - self.alpha*next_log_prob_batch

            next_q_value = reward_batch + done_mask_batch*self.gamma*min_q_target

        q1 = self.critic_1.get_value(state_batch, action_batch)
        q2 = self.critic_2.get_value(state_batch, action_batch)

        q1_loss = 0.5*torch.nn.functional.mse_loss(q1, next_q_value)
        q2_loss = 0.5*torch.nn.functional.mse_loss(q2, next_q_value)

        self.critic_1_optim.zero_grad()
        q1_loss.backward()
        self.critic_1_optim.step()

        self.critic_2_optim.zero_grad()
        q2_loss.backward()
        self.critic_2_optim.step()

        if self.replay_buffer_type == "FIFO" or self.replay_buffer_type == "Reservior_TR" or self.replay_buffer_type == "Reservior":
            for i in range(self.no):
                cur_batch = self.replay_buffer.sample(self.batch_size)

                self.update_curiosity(cur_batch, index=i)
        else:
            if self.update_curiosity_from_fifo == False:
                for i in range(self.no):
                    cur_batch = self.replay_buffer.sample(self.batch_size)
                    self.update_curiosity(cur_batch, index=i)
            else:
                for i in range(self.no):

                    if self.replay_buffer.seperate_cur_buffer == True:
                        fifo_batch = self.replay_buffer.curiosity_buffer.sample(batch_size=batch_size)
                    else:
                        fifo_batch = self.replay_buffer.fifo_buffer.sample(batch_size=batch_size)
                    self.update_curiosity(fifo_batch, index=i)


        #policy update
        pi, log_pi, pi_m = self.policy.sample(state_batch)

        # alpha update
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().detach()
        """
        state_batch_2 = fifo_batch.state
        pi, log_pi, pi_m = self.policy.sample(state_batch_2)

        # alpha update
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().detach()
        """
        q1_pi = self.critic_1.get_value(state_batch, pi)
        q2_pi = self.critic_2.get_value(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)


        policy_loss = ((self.alpha*log_pi) - min_q_pi).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        #if self.alpha.device == "cuda":
        #    self.alpha.detach()
        if self.automatic_alpha_tuning:
            self.alpha_history.append(self.alpha.item())
        else:
            self.alpha_history.append(self.alpha)


        if self.update_no%self.target_update_interval == 0:

            self.soft_update(self.critic_target_1, self.critic_1, self.tau)
            self.soft_update(self.critic_target_2, self.critic_2, self.tau)

    def update_curiosity(self, batch, index):
        # icm update

        state_batch = batch.state
        action_batch = batch.action
        action_mean_batch = batch.action_mean
        next_state_batch = batch.next_state
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.q_nn_param.device)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1).to(self.q_nn_param.device)

        pred_next_state = self.icm_next_state[index].get_next_state(state_batch, action_batch, format="torch")
        pred_action = self.icm_action[index].get_action(state_batch, next_state_batch, format="torch")
        pred_reward = self.icm_reward[index].get_reward(state_batch, action_batch, format="torch")

        icm_next_state_loss = 0.5 * torch.nn.functional.mse_loss(pred_next_state,
                                                                torch.FloatTensor(next_state_batch).to(
                                                                    self.icm_nn_param.device))

        icm_action_loss = 0.5 * torch.nn.functional.mse_loss(pred_action, torch.FloatTensor(action_batch).to(self.icm_nn_param.device))

        icm_reward_loss = 0.5 * torch.nn.functional.mse_loss(pred_reward, reward_batch).to(self.icm_nn_param.device)


        self.icm_next_state_optim[index].zero_grad()
        icm_next_state_loss.backward()
        self.icm_next_state_optim[index].step()

        self.icm_action_optim[index].zero_grad()
        icm_action_loss.backward()
        self.icm_action_optim[index].step()

        self.icm_reward_optim[index].zero_grad()
        icm_reward_loss.backward()
        self.icm_reward_optim[index].step()

    def step(self, state, random=False):
        batch_size = 1  #since step is for a single sample

        if random:
            action = self.env.action_space.sample()
            action_mean = action
        else:
            action, action_mean = self.get_action(state, evaluate=False)

        next_state, reward, done, _ = self.env.step(action)
        curiosity = 0
        for i in range(self.no):

            p_next_state = self.icm_next_state[i].get_next_state(state, action)
            p_action = self.icm_action[i].get_action(state, next_state)
            p_reward = self.icm_reward[i].get_reward(state, action)


            with torch.no_grad():
                f_icm_r = torch.nn.functional.mse_loss(p_next_state,torch.Tensor(next_state).to(self.icm_nn_param.device)).cpu().detach().numpy()
                i_icm_r = torch.nn.functional.mse_loss(p_action, torch.Tensor(action).to(self.icm_nn_param.device)).cpu().detach().numpy()
                r_icm_r = torch.nn.functional.mse_loss(p_reward, torch.FloatTensor([reward]).to(self.icm_nn_param.device))


            self.icm_f_r[i].append(f_icm_r.item())
            self.icm_i_r[i].append(i_icm_r.item())
            self.icm_r[i].append(r_icm_r.item())

            curiosity += self.fow_cur_w*f_icm_r.item() / self.no
            curiosity += self.inv_cur_w*i_icm_r.item() / self.no
            curiosity += self.rew_cur_w*r_icm_r.item() / self.no



        self.steps_done += 1
        self.steps_per_eps += 1

        if done:
            mask = 0.0
            self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask, self.initial_state, self.steps_per_eps)
            next_state = self.env.reset()
            self.initial_state = next_state
            self.steps_per_eps = 0
            return next_state

        if self.steps_per_eps == self.max_episodes:
            mask = 1.0
            self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask, self.initial_state, self.steps_per_eps)
            next_state = self.env.reset()
            self.initial_state = next_state
            self.steps_per_eps = 0
            return next_state
        mask = 1.0

        self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask, self.initial_state, self.steps_per_eps)

        return next_state

    def hard_update(self):
        self.critic_target_1.load_state_dict(self.critic_1.state_dict())
        self.critic_target_2.load_state_dict(self.critic_2.state_dict())

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def save(self, critic_1_path="critic_1", critic_2_path="critic_2",
             critic_1_target_path = "critic_1_target", critic_2_target_path = "critic_2_target",
             policy_path= "policy_target", icm_state_path = "icm_state", icm_action_path = "icm_action"):

        self.critic_1.save(critic_1_path)
        self.critic_2.save(critic_2_path)
        self.critic_target_1.save(critic_1_target_path)
        self.critic_target_2.save(critic_2_target_path)
        self.policy.save(policy_path)

        for i in range(self.no):
            self.icm_next_state[i].save(icm_state_path + str(i))
            self.icm_action[i].save(icm_action_path + str(i))

    def load(self, critic_1_path="critic_1", critic_2_path="critic_2",
             critic_1_target_path = "critic_1_target", critic_2_target_path = "critic_2_target",
             policy_path= "policy_target", icm_state_path = "icm_state", icm_action_path = "icm_action"):

        self.critic_1.load(critic_1_path)
        self.critic_2.load(critic_2_path)
        self.critic_target_1.load(critic_1_target_path)
        self.critic_target_2.load(critic_2_target_path)
        self.policy.load(policy_path)

        for i in range(self.no):
            self.icm_next_state[i].load(icm_state_path + str(i))
            self.icm_action[i].load(icm_action_path + str(i))

    def get_curiosity_rew(self, state, action, next_state):

        p_n_s = self.icm_nxt_state.get_next_state(state, action )
        p_a = self.icm_action.get_action(state, next_state)

        f_icm_r = torch.nn.functional.mse_loss(p_n_s, torch.FloatTensor(next_state))
        i_icm_r = torch.nn.functional.mse_loss(p_a, torch.FloatTensor(action))

        return f_icm_r, i_icm_r



    def initialize_log_ratio(self):
        self.no_tasks = self.replay_buffer.reservior_buffer.no_tasks
        self.log_ratio = [Log_Ratio(nu_param=self.nu_param, algo_param=self.log_algo_param, deterministic_env=False, averege_next_nu=True,
                                    discrete_policy=False, save_path=self.save_path.nu_path, load_path=self.load_path.nu_path)
                          for _ in range(self.no_tasks)]

        #these just hold the data during training so it can be consolidated later on
        self.temporary_reservoir = HRF_LR(capacity=self.memory_capacity, fifo_fac=self.fifo_frac,
                                          curisoity_buff_frac=self.curiosity_frac, seperate_cur_buffer=True, )

        self.temporary_critic1 = Q_Function_NN(nn_params=self.q_nn_param, save_path=self.save_path.q_path, load_path=self.load_path.q_path)
        self.temporary_critic2 = Q_Function_NN(nn_params=self.q_nn_param, save_path=self.save_path.q_path,load_path=self.load_path.q_path)

        self.temporary_target_critic1 = Q_Function_NN(nn_params=self.q_nn_param, save_path=self.save_path.q_path,load_path=self.load_path.q_path)
        self.temporary_target_critic2 = Q_Function_NN(nn_params=self.q_nn_param, save_path=self.save_path.q_path,load_path=self.load_path.q_path)

        self.temporary_target_critic1.load_state_dict(self.temporary_critic1.state_dict())
        self.temporary_target_critic2.load_state_dict(self.temporary_critic2.state_dict())

        self.temporary_policy = Continuous_Gaussian_Policy(self.policy_nn_param, save_path=self.save_path.policy_path,load_path=self.load_path.policy_path, action_space=self.action_sapce)

        self.temporary_critic1_optim = torch.optim.Adam(self.temporary_critic1.parameters(), self.q_nn_param.l_r)
        self.temporary_critic2_optim = torch.optim.Adam(self.temporary_critic2.parameters(), self.q_nn_param.l_r)
        self.temporary_policy_optim = torch.optim.Adam(self.temporary_policy.parameters(), self.q_nn_param.l_r)

        if self.automatic_alpha_tuning is True:
            self.temporary_target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
            self.temporary_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.temporary_alpha_optim = torch.optim.Adam([self.temporary_log_alpha], lr=self.alpha_lr)

        self.temporary_update_no = 0

    def clear_log_ratio(self):
        del self.log_ratio
        del self.temporary_reservoir

        del self.temporary_critic1
        del self.temporary_critic2
        del self.temporary_target_critic1
        del self.temporary_target_critic2
        del self.temporary_policy
        del self.temporary_critic1_optim
        del self.temporary_critic2_optim
        del self.temporary_policy_optim
        del self.temporary_target_entropy
        del self.temporary_log_alpha
        del self.temporary_alpha_optim

        self.log_ratio = []
        self.temporary_reservoir = []

    def step_for_log_ratio(self, state, random=False):
        batch_size = 1  # since step is for a single sample

        if random:
            action = self.env.action_space.sample()
            action_mean = action
        else:
            action, action_mean = self.get_action(state, evaluate=False)

        next_state, reward, done, _ = self.env.step(action)
        curiosity = 0
        for i in range(self.no):
            p_next_state = self.icm_next_state[i].get_next_state(state, action)
            p_action = self.icm_action[i].get_action(state, next_state)
            p_reward = self.icm_reward[i].get_reward(state, action)

            with torch.no_grad():
                f_icm_r = torch.nn.functional.mse_loss(p_next_state, torch.Tensor(next_state).to(
                    self.icm_nn_param.device)).cpu().detach().numpy()
                i_icm_r = torch.nn.functional.mse_loss(p_action, torch.Tensor(action).to(
                    self.icm_nn_param.device)).cpu().detach().numpy()
                r_icm_r = torch.nn.functional.mse_loss(p_reward,
                                                       torch.FloatTensor([reward]).to(self.icm_nn_param.device))

            self.icm_f_r[i].append(f_icm_r.item())
            self.icm_i_r[i].append(i_icm_r.item())
            self.icm_r[i].append(r_icm_r.item())

            curiosity += self.fow_cur_w * f_icm_r.item() / self.no
            curiosity += self.inv_cur_w * i_icm_r.item() / self.no
            curiosity += self.rew_cur_w * r_icm_r.item() / self.no

        self.steps_done += 1
        self.steps_per_eps += 1

        if done:
            mask = 0.0
            self.temporary_reservoir.push(state, action, action_mean, reward, curiosity, next_state, mask, self.initial_state,
                                    self.steps_per_eps)
            next_state = self.env.reset()
            self.initial_state = next_state
            self.steps_per_eps = 0
            return next_state

        if self.steps_per_eps == self.max_episodes:
            mask = 1.0
            self.temporary_reservoir.push(state, action, action_mean, reward, curiosity, next_state, mask, self.initial_state,
                                    self.steps_per_eps)
            next_state = self.env.reset()
            self.initial_state = next_state
            self.steps_per_eps = 0
            return next_state
        mask = 1.0

        self.temporary_reservoir.push(state, action, action_mean, reward, curiosity, next_state, mask, self.initial_state,
                                self.steps_per_eps)

        return next_state

    def update_temporary_for_log_ratio(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        if batch_size > len(self.temporary_reservoir):
            return

        batch = self.temporary_reservoir.sample(batch_size=batch_size)
        self.temporary_update_no += 1



        state_batch = batch.state
        action_batch = batch.action
        action_mean_batch = batch.action_mean
        next_state_batch = batch.next_state
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.q_nn_param.device)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1).to(self.q_nn_param.device)

        # critic update
        with torch.no_grad():
            next_action_batch, next_log_prob_batch, _ = self.temporary_policy.sample(next_state_batch, format="torch")
            q1_next_target = self.temporary_target_critic1.get_value(next_state_batch, next_action_batch, format="torch")
            q2_next_target = self.temporary_target_critic2.get_value(next_state_batch, next_action_batch, format="torch")
            min_q_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_prob_batch

            next_q_value = reward_batch + done_mask_batch * self.gamma * min_q_target

        q1 = self.temporary_critic1.get_value(state_batch, action_batch)
        q2 = self.temporary_critic2.get_value(state_batch, action_batch)

        q1_loss = 0.5 * torch.nn.functional.mse_loss(q1, next_q_value)
        q2_loss = 0.5 * torch.nn.functional.mse_loss(q2, next_q_value)

        self.temporary_critic1_optim.zero_grad()
        q1_loss.backward()
        self.critic_1_optim.step()

        self.temporary_critic2_optim.zero_grad()
        q2_loss.backward()
        self.critic_2_optim.step()

        # policy update
        pi, log_pi, pi_m = self.temporary_policy.sample(state_batch)

        # alpha update
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.temporary_log_alpha * (log_pi + self.temporary_target_entropy).detach()).mean()
            self.temporary_alpha_optim.zero_grad()
            alpha_loss.backward()
            self.temporary_critic2_optim.step()

            self.temporary_alpha = self.temporary_log_alpha.exp().detach()

        q1_pi = self.temporary_critic1.get_value(state_batch, pi)
        q2_pi = self.temporary_critic2.get_value(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)

        policy_loss = ((self.temporary_alpha * log_pi) - min_q_pi).mean()

        self.temporary_policy_optim.zero_grad()
        policy_loss.backward()
        self.temporary_policy_optim.step()


        if self.temporary_update_no % self.target_update_interval == 0:
            self.soft_update(self.temporary_target_critic1, self.temporary_critic1, self.tau)
            self.soft_update(self.temporary_target_critic2, self.temporary_critic2, self.tau)
            

    def start_log_ratio_evaluation(self):
        self.initialize_log_ratio()
        partition = False
        minimum_KL = None


        for i in range(self.no_log_ratio_evaluation_steps):
            state = self.initalize()
            self.step_for_log_ratio(state=state, random=False)



            self.update_temporary_for_log_ratio() # do the training of temporary policy
            self.train_log_ratio() #do the traning for log_ratio using that policy










        if minimum_KL <  self.KL_threshold:
            partition = False
        else:
            partition = True


        if partition:
            self.replay_buffer.new_partition(self.temporary_reservoir)
        else:
            self.replay_buffer.no_new_partion(self.temporary_reservoir)
        self.clear_log_ratio()