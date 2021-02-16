import torch
import torch.nn as nn
import torch.nn.functional as F

from util.weight_initalizer import weight_initialize, bias_initialize

import numpy as np
from models.base import NN_Paramters, BaseNN

#for soft actor critic
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class Continuous_Gaussian_Policy_Sumo(BaseNN):
    # adapted from https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
    def __init__(self, nn_params, save_path, load_path, action_space=None):

        super(Continuous_Gaussian_Policy_Sumo, self).__init__(save_path=save_path, load_path=load_path)

        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        self.batch_size = None
        # Hidden layers

        layer_input_dim = self.nn_params.state_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim

        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.mean = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.mean, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.log_std = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.log_std, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

        # action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.).to(self.nn_params.device)
            self.action_bias = torch.tensor(0.).to(self.nn_params.device)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.).to(self.nn_params.device)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.).to(self.nn_params.device)


    def forward(self, state):

        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        self.batch_size = state.size()[0]

        if len(state.shape)  == 3:
            state = torch.flatten(state)
        else:

            state = torch.flatten(state, start_dim=1)

        inp = state
        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)

        mean = self.mean(inp)
        log_std = self.log_std(inp)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def sample(self, state, format="torch"):

        mean , log_std = self.forward(state=state)
        std = log_std.exp()

        gaussian = torch.distributions.Normal(loc=mean, scale=std)

        #sample for reparametrization trick
        x_t = gaussian.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = gaussian.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)


        if len(log_prob.shape) != 1:
            log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias



        if format == "torch":
            return action, log_prob, mean
        else:
            return action.cpu().detach().numpy(), log_prob.cpu().detach().numpy(), mean.cpu().detach().numpy()


    def to(self, device):
        super().to(device)
        self.nn_params.device= device


class Q_Function_sumo_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path):

        super(Q_Function_sumo_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        # Hidden layers
        layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        #Final Layer
        self.Q_value = nn.Linear(layer_input_dim, 1)
        self.weight_init(self.Q_value, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, action):

        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(action) != torch.Tensor:
            action = torch.Tensor(action).to(self.nn_params.device)

        if len(state.shape)  == 3:
            state = torch.flatten(state)
        else:
            state = torch.flatten(state, start_dim=1)

        inp = torch.cat((state, action), dim= 1)

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        Q_s_a = self.Q_value(inp)

        return Q_s_a

    def get_value(self, state, action, format="torch"):

        if format == "torch":
            return self.forward(state, action)
        elif format == "numpy":
            return  self.forward(state, action).cpu().detach().numpy()

    def to(self, device):
        super().to(device)
        self.nn_params.device= device


class ICM_Next_State_sumo_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(ICM_Next_State_sumo_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.next_state = nn.Linear(layer_input_dim, self.nn_params.state_dim)
        self.weight_init(self.next_state, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, action):
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(action) != torch.Tensor:
            action = torch.Tensor(action).to(self.nn_params.device)

        if len(state.shape) == 3:
            state = torch.flatten(state)
        else:
            state = torch.flatten(state, start_dim=1)


        if len(state.size()) == 1:
            inp = torch.cat((state, action), dim=0)
        else:
            inp = torch.cat((state, action), dim=1)



        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        next_state_pred = self.next_state(inp)

        return next_state_pred

    def get_next_state(self, state, action, format="torch"):
        next_state = self.forward(state, action)

        if format == "torch":
            return next_state
        else:
            return next_state.cpu().detach().numpy()


    def to(self, device):
        super().to(device)
        self.nn_params.device = device

class ICM_Action_sumo_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(ICM_Action_sumo_NN, self).__init__(save_path=save_path, load_path=load_path)

        print("cur_init")
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        layer_input_dim = self.nn_params.state_dim + self.nn_params.state_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.action = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.action, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, next_state):
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(next_state) != torch.Tensor:
            next_state = torch.Tensor(next_state).to(self.nn_params.device)


        if len(state.shape)  == 3:
            state = torch.flatten(state)
            next_state = torch.flatten(next_state)
        else:
            state = torch.flatten(state, start_dim=1)
            next_state = torch.flatten(next_state, start_dim=1)



        if len(state.size()) == 1:
            inp = torch.cat((state, next_state), dim=0)
        else:
            inp = torch.cat((state, next_state), dim=1)

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        action_pred = self.action(inp)

        return action_pred

    def get_action(self, state, next_state, format="torch"):
        action =  self.forward(state, next_state)

        if format == "torch":
            return action
        else:
            return action.cpu().detach().numpy()

    def to(self, device):
        super().to(device)
        self.nn_params.device = device


class ICM_Reward_sumo_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(ICM_Reward_sumo_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.reward = nn.Linear(layer_input_dim, 1)
        self.weight_init(self.reward, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, action):
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(action) != torch.Tensor:
            action = torch.Tensor(action).to(self.nn_params.device)

        if len(state.shape) == 3:
            state = torch.flatten(state)
        else:
            state = torch.flatten(state, start_dim=1)

        if len(state.size()) == 1:
            inp = torch.cat((state, action), dim=0)
        else:
            inp = torch.cat((state, action), dim=1)

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        pred_reward = self.reward(inp)

        return pred_reward

    def get_reward(self, state, action, format="torch"):
        reward =  self.forward(state, action)

        if format == "torch":
            return reward
        else:
            return reward.cpu().detach().numpy()

    def to(self, device):
        super().to(device)
        self.nn_params.device = device