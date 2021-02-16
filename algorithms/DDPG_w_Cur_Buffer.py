import torch
import numpy as np

from models.DDPG_model import Q_Function_NN, DDPG_Policy, ICM_Reward_NN, ICM_Action_NN, ICM_Next_State_NN
from models.DDPG_sumo_model import Q_Function_sumo_NN, DDPG_Policy_sumo, ICM_Next_State_sumo_NN, ICM_Action_sumo_NN, ICM_Reward_sumo_NN
from parameters import Save_Paths, Load_Paths

from util.replay_buffer import Replay_Memory
from util.new_replay_buffers.gradual.half_res_w_cur_ft_fifo_gradual import Half_Reservoir_Flow_Through_w_Cur_Gradual


from models.noise import OrnsteinUhlenbeckProcess, GaussianWhiteNoiseProcess


class DDPG_with_Curiosity_Buffer():

    def __init__(self, env, q_nn_param, policy_nn_param, icm_nn_param, algo_nn_param
                 , max_episodes=100, memory_capacity=10000,
                 batch_size=400, save_path=Save_Paths(), load_path=Load_Paths(), noise_type ="Ornstein",
                 ou_theta = 0.15, ou_sigma = 0.3,  ou_mu = 0.0, sigma_min = None, anneal_epsilon = False,
                 env_type= "sumo", change_at = [100000, 350000],
                 buffer_type= "FIFO", fifo_frac=0.34, update_curiosity_from_fifo = True, no_cur_network=5,
                 reset_cur_on_task_change=True, reset_alpha_on_task_change=True,

                 ):

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.env_type = env_type

        self.save_path = save_path
        self.load_path = load_path

        self.q_nn_param = q_nn_param
        self.policy_nn_param = policy_nn_param
        self.algo_nn_param = algo_nn_param
        self.icm_nn_param = icm_nn_param

        self.update_curiosity_from_fifo = update_curiosity_from_fifo
        self.reset_cur_on_task_change = reset_cur_on_task_change
        self.reset_alpha_on_task_change = reset_alpha_on_task_change

        self.gamma = algo_nn_param.gamma
        self.epsilon = 1.0
        self.depsilon = 1/self.algo_nn_param.depsilon
        self.anneal_epsilon = anneal_epsilon

        self.tau = algo_nn_param.tau

        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_mu = ou_mu

        self.max_episodes = max_episodes
        self.steps_done = 0
        self.update_no = 0
        self.batch_size = batch_size
        self.target_update_interval = self.algo_nn_param.target_update_interval



        if self.env_type == "sumo":


            self.critic = Q_Function_sumo_NN(q_nn_param, save_path, load_path)
            self.critic_target = Q_Function_sumo_NN(q_nn_param, save_path, load_path)
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.policy = DDPG_Policy_sumo(policy_nn_param, save_path, load_path)
            self.policy_target = DDPG_Policy_sumo(policy_nn_param, save_path, load_path)
            self.policy_target.load_state_dict(self.policy.state_dict())
        else:

            self.critic = Q_Function_NN(q_nn_param, save_path, load_path)
            self.critic_target = Q_Function_NN(q_nn_param, save_path, load_path)
            self.critic_target.load_state_dict(self.critic.state_dict())

            self.policy = DDPG_Policy(policy_nn_param, save_path, load_path)
            self.policy_target = DDPG_Policy(policy_nn_param, save_path, load_path)
            self.policy_target.load_state_dict(self.policy.state_dict())

        self.crtic_optim = torch.optim.Adam(self.critic.parameters(), self.q_nn_param.l_r)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), self.policy_nn_param.l_r)


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

        self.replay_buffer_type = buffer_type


        if buffer_type == "FIFO":
            self.replay_buffer = Replay_Memory(capacity=memory_capacity)
        elif buffer_type == "Half_Reservior_FIFO_with_FT":
            self.replay_buffer = Half_Reservoir_Flow_Through_w_Cur_Gradual(capacity=memory_capacity,
                                                                           curisoity_buff_frac=0.34,
                                                                           seperate_cur_buffer=True,
                                                                           fifo_fac=fifo_frac)


        self.device = q_nn_param.device
        self.noise_type = noise_type
        if self.noise_type == "gaussian":
            self.noise = GaussianWhiteNoiseProcess(mu = ou_mu, sigma=ou_sigma, sigma_min=sigma_min,size=self.action_dim)

        elif self.noise_type == "Ornstein":
            self.noise = OrnsteinUhlenbeckProcess(size=self.action_dim, theta=self.ou_theta, mu=self.ou_mu, sigma=self.ou_sigma)

        self.change_var_at = change_at

    def get_action(self, state, evaluate=False ):
        #returns action mean if evaluate
        #else both action, action mean are returned

        if evaluate == True:
            return self.policy.sample(state, format="numpy")
        else:
            if self.noise_type == "gaussian":
                noise = self.noise.sample()
            elif self.noise_type == "Ornstein" :
                noise = self.noise.sample()

            action_mean = self.policy.sample(state, format="numpy")

            action = action_mean + max(self.epsilon,0)*noise
            action = np.clip(action, -1, 1)

            if self.anneal_epsilon == True:
                self.epsilon -= self.depsilon
            return action, action_mean

    def initalize(self):
        # inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        state = self.env.reset()
        for i in range(self.batch_size):
            state = self.step(state)
        return state

    def step(self, state, random=False):
        batch_ize = 1
        self.steps_done += 1

        if random == True:
            action = self.env.action_space.sample()
            action_mean = self.env.action_space.sample()
        else:
            action, action_mean = self.get_action(state, evaluate=False)

        next_state, reward, done, _ = self.env.step(action)
        self.steps_done += 1

        curiosity = 0
        for i in range(self.no):
            # p_next_state = self.icm_next_state[i].get_next_state(state, action)
            p_action = self.icm_action[i].get_action(state, next_state)
            p_reward = self.icm_reward[i].get_reward(state, action)

            with torch.no_grad():
                # f_icm_r = torch.nn.functional.mse_loss(p_next_state,torch.Tensor(next_state).to(self.icm_nn_param.device)).cpu().detach().numpy()
                i_icm_r = torch.nn.functional.mse_loss(p_action, torch.Tensor(action).to(
                    self.icm_nn_param.device)).cpu().detach().numpy()

                r_icm_r = torch.nn.functional.mse_loss(p_reward,
                                                       torch.FloatTensor([reward]).to(self.icm_nn_param.device))

            # self.icm_f_r[i].append(f_icm_r.item())
            self.icm_i_r[i].append(i_icm_r.item())
            self.icm_r[i].append(r_icm_r.item())

            # half_cheetah
            curiosity += 0.0 * i_icm_r.item() / self.no
            curiosity += 1.0 * r_icm_r.item() / self.no

        if done:
            mask = 0.0
            self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask)
            next_state = self.env.reset()
            return next_state

        if self.steps_done == self.max_episodes:
            #mask = 0.0
            mask = 0.0
            self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask)
            next_state = self.env.reset()
            return next_state
        mask = 1.0

        self.replay_buffer.push(state, action, action_mean, reward, curiosity, next_state, mask)
        return next_state

    def update(self, batch_size=None):

        if batch_size == None:
            batch_size = self.batch_size
        if batch_size > len(self.replay_buffer):
            return

        self.update_no += 1

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
                self.epsilon = 1.0
                self.noise.reset_annealing()

        batch = self.replay_buffer.sample(batch_size)

        state_batch = batch.state
        action_batch = batch.action
        next_state_batch = batch.next_state
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.q_nn_param.device)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1).to(self.q_nn_param.device)

        with torch.no_grad():
            next_action_batch = self.policy_target.sample(next_state_batch, format="torch")
            next_q_value = self.critic_target.get_value(next_state_batch, next_action_batch, format="torch")

            q_target = reward_batch + self.gamma*done_mask_batch*next_q_value

        q = self.critic.get_value(state_batch, action_batch)

        q_loss = torch.nn.functional.mse_loss(q, q_target)


        policy_loss = -self.critic.get_value(state_batch,
                                             self.policy.sample(state_batch, format="torch"),
                                             format = "torch"
                                             ).mean()




        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        self.crtic_optim.zero_grad()
        q_loss.backward()
        self.crtic_optim.step()


        if self.update_no%self.target_update_interval == 0:

            self.soft_update(target=self.critic_target, source=self.critic, tau=self.tau)
            self.soft_update(target=self.policy_target, source=self.policy, tau=self.tau)

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

        #icm_next_state_loss = 0.5 * torch.nn.functional.mse_loss(pred_next_state,
        #                                                         torch.FloatTensor(next_state_batch).to(
        #                                                             self.icm_nn_param.device))

        icm_action_loss = 0.5 * torch.nn.functional.mse_loss(pred_action, torch.FloatTensor(action_batch).to(self.icm_nn_param.device))

        icm_reward_loss = 0.5 * torch.nn.functional.mse_loss(pred_reward, reward_batch).to(self.icm_nn_param.device)


        #self.icm_next_state_optim[index].zero_grad()
        #icm_next_state_loss.backward()
        #self.icm_next_state_optim[index].step()

        self.icm_action_optim[index].zero_grad()
        icm_action_loss.backward()
        self.icm_action_optim[index].step()

        self.icm_reward_optim[index].zero_grad()
        icm_reward_loss.backward()
        self.icm_reward_optim[index].step()


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def save(self, critic_path="critic",
            critic_target_path="critic_target",
            policy_path="policy", policy_target_path = "policy_target",
             icm_state_path = "icm_state", icm_action_path = "icm_action"):

        self.critic.save(critic_path)
        self.critic_target.save(critic_target_path)
        self.policy.save(policy_path)
        self.policy_target.save(policy_target_path)

        for i in range(self.no):
            self.icm_next_state[i].save(icm_state_path + str(i))
            self.icm_action[i].save(icm_action_path + str(i))

    def load(self, critic_path="critic",
            critic_target_path="critic_target",
            policy_path="policy", policy_target_path = "policy_target",
             icm_state_path = "icm_state", icm_action_path = "icm_action"):

        self.critic.load(critic_path)
        self.critic_target.load(critic_target_path)
        self.policy.load(policy_path)
        self.policy_target.load(policy_target_path)

        for i in range(self.no):
            self.icm_next_state[i].load(icm_state_path + str(i))
            self.icm_action[i].load(icm_action_path + str(i))

