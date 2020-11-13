import torch
import numpy as np

from model import Q_Function_NN, DDPG_Policy
from parameters import NN_Paramters, Algo_Param_DDPG, Save_Paths, Load_Paths
from util.replay_buffer import Replay_Memory



class DDPG():

    def __init__(self, env, q_nn_param, policy_nn_param, algo_nn_param
                 , max_episodes=100, memory_capacity=10000,
                 batch_size=400, save_path=Save_Paths(), load_path=Load_Paths(), noise ="gaussian"
                 ):

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.q_nn_param = q_nn_param
        self.policy_nn_param = policy_nn_param
        self.algo_nn_param = algo_nn_param

        self.gamma = algo_nn_param.gamma
        self.epsilon = 1.0
        self.depsilon = 1/self.algo_nn_param.depsilon
        self.tau = algo_nn_param.tau

        self.max_episodes = max_episodes
        self.steps_done = 0
        self.update_no = 0
        self.batch_size = batch_size
        self.target_update_interval = self.algo_nn_param.target_update_interval


        self.critic = Q_Function_NN(q_nn_param, save_path, load_path)
        self.critic_target = Q_Function_NN(q_nn_param, save_path, load_path)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.policy = DDPG_Policy(policy_nn_param, save_path, load_path)
        self.policy_target = DDPG_Policy(policy_nn_param, save_path, load_path)
        self.policy_target.load_state_dict(self.policy.state_dict())

        self.crtic_optim = torch.optim.Adam(self.critic.parameters(), self.q_nn_param.l_r)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), self.policy_nn_param.l_r)

        self.replay_buffer = Replay_Memory(capacity=memory_capacity)

        self.device = q_nn_param.device

        if noise == "gaussian":
            mean = torch.zeros(size = (self.action_dim,)).to(self.policy_nn_param.device)
            sigma = torch.ones(size=(self.action_dim,)).to(self.policy_nn_param.device) * float(self.algo_nn_param.std)
            self.noise = torch.distributions.Normal(loc=mean, scale=sigma)

    def get_action(self, state, evaluate=False ):

        if evaluate == True:
            return self.policy.sample(state, format="numpy")
        else:
            noise = self.noise.sample().cpu().detach().numpy()
            action_mean = self.policy.sample(state, format="numpy")
            action = action_mean + self.epsilon*noise
            action = np.clip(action, -1, 1)
            self.epsilon += self.depsilon
            return action

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
        else:
            action = self.get_action(state, evaluate=False)

        next_state, reward, done, _ = self.env.step(action)
        self.steps_done += 1
        if done:
            mask = 0.0
            self.replay_buffer.push(state, action, reward, next_state, mask)
            next_state = self.env.reset()
            return next_state

        if self.steps_done == self.max_episodes:
            mask = 1.0
            self.replay_buffer.push(state, action, reward, next_state, mask)
            next_state = self.env.reset()
            return next_state
        mask = 1.0

        self.replay_buffer.push(state, action, reward, next_state, mask)
        return next_state

    def update(self, batch_size=None):

        if batch_size == None:
            batch_size = self.batch_size
        if batch_size > len(self.replay_buffer):
            return

        self.update_no += 1


        batch = self.replay_buffer.sample(batch_size)

        state_batch = batch.state
        action_batch = batch.action
        next_state_batch = batch.next_state
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1)
        done_mask_batch = torch.FloatTensor(batch.done_mask).unsqueeze(1)

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

        if self.update_no%self.target_update_interval:
            self.soft_update(target=self.critic_target, source=self.critic)
            self.soft_update(target=self.policy_target, source=self.policy)


    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


    def save(self, critic_path="critic",
            critic_target_path="critic_target",
            policy_path="policy", policy_target_path = "policy_target"):

        self.critic.save(critic_path)
        self.critic_target.save(critic_target_path)
        self.policy.save(policy_path)
        self.policy_target.save(policy_target_path)

    def load(self, critic_path="critic",
            critic_target_path="critic_target",
            policy_path="policy", policy_target_path = "policy_target"):

        self.critic.load(critic_path)
        self.critic_target.load(critic_target_path)
        self.policy.load(policy_path)
        self.policy_target.load(policy_target_path)
