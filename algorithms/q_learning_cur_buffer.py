import torch
from model import Discrete_Q_Function_NN, ICM_Next_State_NN, ICM_Action_NN ,ICM_Reward_NN
from parameters import NN_Paramters, Algo_Param, Save_Paths, Load_Paths

import numpy as np
from algorithms.epsilon_greedy import epsilon_greedy

from util.new_replay_buffers.replay_buff_cur import Replay_Memory_Cur
#from reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
from util.new_replay_buffers.gradual.half_res_w_cur_ft_fifo_gradual import Half_Reservoir_Flow_Through_w_Cur_Gradual



class Q_learning_w_cur_buf():

    def __init__(self, env, q_nn_param,  icm_nn_param, algo_param, max_episodes =100, memory_capacity =50000,
                 batch_size=32, save_path = Save_Paths(), load_path= Load_Paths(), buffer_type = "FIFO",
                 update_curiosity_from_fifo=True, fifo_frac=0.34, no_cur_network=1,
                 reset_cur_on_task_change=True, reset_alpha_on_task_change=True,  tau=0.005
                 ):

        self.state_dim = q_nn_param.state_dim
        self.action_dim = q_nn_param.action_dim
        self.q_nn_param = q_nn_param
        self.algo_param = algo_param
        self.icm_nn_param = icm_nn_param
        self.max_episodes = max_episodes

        self.replay_buffer_type = buffer_type


        self.update_curiosity_from_fifo = update_curiosity_from_fifo
        self.reset_cur_on_task_change = reset_cur_on_task_change
        self.reset_alpha_on_task_change = reset_alpha_on_task_change

        self.inital_state = None
        self.time_step = 0
        self.target_update_interval = self.algo_param.target_update_interval
        self.tau = tau

        self.save_path = Save_Paths()
        self.load_path = Load_Paths()

        self.inital_state = None
        self.time_step = 0

        self.Q = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                        save_path= self.save_path.q_path, load_path=self.load_path.q_path)

        self.Target_Q = Discrete_Q_Function_NN(nn_params=q_nn_param,
                                        save_path= self.save_path.q_path, load_path=self.load_path.q_path)
        self.Target_Q.load_state_dict(self.Q.state_dict())



        #self.loss_function = torch.nn.functional.smooth_l1_loss
        self.loss_function = torch.nn.functional.mse_loss
        self.Q_optim = torch.optim.Adam(self.Q.parameters(), self.q_nn_param.l_r)

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


        #if buffer_type == "Reservior":
        #    self.replay_buffer = Reservoir_with_Cur_Replay_Memory(capacity=memory_capacity)
        if buffer_type == "FIFO":
            self.replay_buffer = Replay_Memory_Cur(capacity=memory_capacity)
        elif buffer_type == "Half_Reservior_FIFO_with_FT":
            self.replay_buffer = Half_Reservoir_Flow_Through_w_Cur_Gradual(capacity=memory_capacity, curisoity_buff_frac=0.34, seperate_cur_buffer=True,
                                                                          fifo_fac=fifo_frac)


        self.memory_capacity = memory_capacity
        self.batch_size = batch_size
        self.env = env


        self.update_no = 0

    def save(self, critic_path="critic",
             critic_target_path="critic_target",
             icm_state_path="icm_state", icm_action_path="icm_action"):

        self.Q.save(critic_path)
        self.Target_Q.save(critic_target_path)


        for i in range(self.no):
            self.icm_next_state[i].save(icm_state_path + str(i))
            self.icm_action[i].save(icm_action_path + str(i))

    def load(self, critic_path="critic",
             critic_target_path="critic_target",
             icm_state_path="icm_state", icm_action_path="icm_action"):

        self.Q.load(critic_path)
        self.Target_Q.load(critic_target_path)

        for i in range(self.no):
            self.icm_next_state[i].load(icm_state_path + str(i))
            self.icm_action[i].load(icm_action_path + str(i))


    def step(self, state, random=False):

        #since step is done on the basis of single states and not as a batch
        batch_size = 1

        q_values = self.Target_Q.get_value(state, format="numpy")
        action, self.steps_done, self.epsilon = epsilon_greedy(q_values, self.steps_done, self.epsilon, self.action_dim)

        next_state, reward, done, _ = self.env.step(action)


        #converting the action for buffer as one hot vector
        sample_hot_vec = np.array([0.0 for i in range(self.q_nn_param.action_dim)])
        sample_hot_vec[action] = 1




        action = sample_hot_vec

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

            self.icm_i_r[i].append(i_icm_r.item())
            self.icm_r[i].append(r_icm_r.item())

            curiosity += 1.0 * i_icm_r.item() / self.no
            curiosity += 1.0 * r_icm_r.item() / self.no


        self.time_step += 1

        if done:
            next_state = None
            self.replay_buffer.push(state, action, None, reward, curiosity, next_state, self.time_step)
            state = self.env.reset()
            self.inital_state = state
            self.time_step = 0
            return state

        if self.time_step == self.max_episodes:
            self.replay_buffer.push(state, action, None, reward, curiosity, next_state, self.time_step)
            state = self.env.reset()
            self.inital_state = state
            self.time_step = 0
            return state

        self.replay_buffer.push(state, action, None, reward, curiosity, next_state, self.time_step)
        return next_state

    def get_action(self, state, evaluate=True):
        q_values = self.Q.get_value(state, format="numpy")
        action_scaler = np.argmax(q_values)
        return action_scaler

    def update(self):

        self.update_no += 1

        batch_size = self.batch_size
        if len(self.replay_buffer) < batch_size:
            return


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
                print("current epsilon = " + str(self.epsilon))
                self.epsilon = 1.0



        batch = self.replay_buffer.sample(batch_size)
        state = batch.state
        action = torch.Tensor(batch.action).to(self.q_nn_param.device)
        action_scaler = action.max(1)[1].unsqueeze(1).to(
            self.q_nn_param.device)  # to make them as indices in the gather function
        reward = torch.Tensor(batch.reward).to(self.q_nn_param.device)
        next_state = batch.next_state



        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.q_nn_param.device, dtype=torch.bool).to(self.q_nn_param.device)

        non_final_next_states = torch.Tensor([s for s in next_state if s is not None]).to(self.q_nn_param.device)



        #get only the q value relevant to the actions
        state_action_values = self.Q.get_value(state).gather(1, action_scaler)


        with torch.no_grad():
            next_state_action_values = torch.zeros(batch_size, device=self.q_nn_param.device).to(self.q_nn_param.device)
            next_state_action_values[non_final_mask] = self.Target_Q.get_value(non_final_next_states).max(1)[0]
            #now there will be a zero if it is the final state and q*(n_s,n_a) is its not None


        expected_state_action_values = (self.algo_param.gamma*next_state_action_values).unsqueeze(1) + reward.unsqueeze(1)


        loss = self.loss_function( state_action_values, expected_state_action_values)

        self.Q_optim.zero_grad()
        loss.backward()
        self.Q_optim.step()


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

        if self.update_no%self.target_update_interval == 0:
            self.soft_update(target=self.Target_Q, source=self.Q, tau=self.tau)


    def update_curiosity(self, batch, index):
        # icm update

        state_batch = batch.state
        action_batch = batch.action
        action_mean_batch = batch.action_mean
        next_state_batch = batch.next_state
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.q_nn_param.device)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1).to(self.q_nn_param.device)




        non_final_next_states = torch.Tensor([s for s in next_state_batch if s is not None]).to(self.q_nn_param.device)

        non_final_states = torch.Tensor([state_batch[s_i] for s_i in range(np.shape(next_state_batch)[0])
                                         if batch.next_state[s_i] is not None]).to(self.q_nn_param.device)
        non_final_action = torch.Tensor([action_batch[s_i] for s_i in range(np.shape(next_state_batch)[0])
                                         if batch.next_state[s_i] is not None]).to(self.q_nn_param.device)
        non_final_reward = torch.Tensor([reward_batch[s_i] for s_i in range(np.shape(next_state_batch)[0])
                                         if batch.next_state[s_i] is not None]).to(self.q_nn_param.device).unsqueeze(1)


        pred_next_state = self.icm_next_state[index].get_next_state(non_final_states, non_final_action, format="torch")
        pred_action = self.icm_action[index].get_action(non_final_states, non_final_next_states, format="torch")
        pred_reward = self.icm_reward[index].get_reward(non_final_states, non_final_action, format="torch")

        #icm_next_state_loss = 0.5 * torch.nn.functional.mse_loss(pred_next_state,
        #                                                         non_final_next_states)

        icm_action_loss = 0.5 * torch.nn.functional.mse_loss(pred_action, non_final_action).to(self.icm_nn_param.device)

        icm_reward_loss = 0.5 * torch.nn.functional.mse_loss(pred_reward, non_final_reward).to(self.icm_nn_param.device)


        #self.icm_next_state_optim[index].zero_grad()
        #icm_next_state_loss.backward()
        #self.icm_next_state_optim[index].step()

        self.icm_action_optim[index].zero_grad()
        icm_action_loss.backward()
        self.icm_action_optim[index].step()

        self.icm_reward_optim[index].zero_grad()
        icm_reward_loss.backward()
        self.icm_reward_optim[index].step()


    def hard_update(self):
        self.Target_Q.load_state_dict(self.Q.state_dict())

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def initalize(self):

        #inital_phase train after this by continuing with step and train at single iteration and hard update at update interval
        self.steps_done = 0
        self.epsilon = 0.9
        state = self.env.reset()
        self.inital_state = state
        for i in range(self.batch_size):
            state = self.step(state)
        return state






