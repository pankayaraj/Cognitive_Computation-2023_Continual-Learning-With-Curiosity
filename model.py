import torch
import torch.nn as nn
import torch.nn.functional as F

from util.weight_initalizer import weight_initialize, bias_initialize

import numpy as np
'''
Contains

Policies that samples action and the corresponding action's probabilty and log probabilty
Value functions and action value functions
Nu and Zeta values
'''

class NN_Paramters():
    def __init__(self, state_dim, action_dim, non_linearity = F.tanh, weight_initializer = 'xavier', bias_initializer = 'zero',
                 hidden_layer_dim = [128, 128], device= torch.device('cuda'), l_r=0.0001):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.non_linearity = non_linearity
        self.l_r = l_r

        self.device = device


class BaseNN(nn.Module):

    '''
    Base Neural Network function to inherit from
    save_path       : default path for saving neural network weights
    load_path       : default path for loading neural network weights
    '''

    def __init__(self, save_path, load_path):
        super(BaseNN, self).__init__()

        self.save_path = save_path
        self.load_path = load_path

    def weight_init(self, layer, w_initalizer, b_initalizer):
        #initalize weight
        weight_initialize(layer, w_initalizer)
        bias_initialize(layer, b_initalizer)

    def save(self, path=None):
        #save state dict
        if path is None:
            path = self.save_path
        torch.save(self.state_dict(), path)

    def load(self, path=None):
        #load state dict
        if path is None:
            path = self.load_path
        self.load_state_dict(torch.load(path))


class DiscretePolicyNN(BaseNN):

    '''
    nn_prams        : a class with neural network's paramters
    save_path       : default path for saving neural network weights
    load_path       : default path for loading neural network weights

    All the fuctions that interface with outside take tensors as numpy array and returns torch tensor by default
    Can be changed via format="numpy"
    '''
    def __init__(self, nn_params, save_path, load_path):
        super(DiscretePolicyNN, self).__init__(save_path=save_path, load_path=load_path)

        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        self.batch_size = None
        #Hidden layers
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

        self.to(self.nn_params.device)


    def forward(self, state):

        state = torch.Tensor(state).to(self.nn_params.device)
        self.batch_size = state.size()[0]
        inp = state
        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        action_prob = F.softmax(self.mean(inp), dim=1)
        return action_prob


    def forward_temp(self, state):
        # this function is for testing purposes only
        state = torch.Tensor(state).to(self.nn_params.device)
        self.batch_size = state.size()[0]
        inp = state
        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        return self.mean(inp)

    def sample(self, state, format="torch"):
        '''
        Returns : Sample of the action(one_hot_vector) as a numpy array for the whole batch
        '''
        action_prob = self.forward(state)
        dist = torch.distributions.categorical.Categorical(action_prob)

        sample = dist.sample()
        sample_hot_vec = torch.tensor([[0.0 for i in range(self.nn_params.action_dim)]
                                       for j in range(self.batch_size)]).to(self.nn_params.device)
        for i in range(self.batch_size):
            sample_hot_vec[i][sample[i]] = 1

        if format == "torch":
            return sample_hot_vec
        elif format == "numpy":
            return sample_hot_vec.cpu().detach().numpy()

    def get_probability(self, state, action_no, format="torch"):
        action_prob = self.forward(state)

        if format == "torch":
            return torch.reshape(action_prob[:, action_no], shape=(self.batch_size, 1))
        elif format == "numpy":
            return action_prob[:, action_no].cpu().detach().numpy()

    def get_probabilities(self, state, format="torch"):
        if format == "torch":
            return self.forward(state)
        elif format == "numpy":
            return self.forward(state).cpu().detach().numpy()

    def get_log_probability(self, state, action_no, format="torch"):

        if format == "torch":
            return torch.log( 1e-8 + self.get_probability(state, action_no, format = "torch")).to(self.nn_params.device)
        elif format == "numpy":
            return np.log(1e-8 + self.get_probability(state, action_no, format="numpy"))


    def to(self, device):
        super().to(device)
        self.nn_params.device = device

class Q_Function_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path):

        super(Q_Function_NN, self).__init__(save_path=save_path, load_path=load_path)
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

        state = torch.Tensor(state).to(self.nn_params.device)
        action = torch.Tensor(action).to(self.nn_params.device)

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

class Value_Function_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path):

        super(Value_Function_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        # Hidden layers
        layer_input_dim = self.nn_params.state_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.value = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.value, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state):

        inp = torch.Tensor(state).to(self.nn_params.device)


        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        V_s = self.value(inp)

        return V_s

    def get_value(self, state, format="torch"):

        if format == "torch":
            return self.forward(state)
        elif format == "numpy":
            return  self.forward(state).cpu().detach().numpy()

    def to(self, device):
        super().to(device)
        self.nn_params.device = device


class Nu_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(Nu_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity
        self.state_action = state_action
        # Hidden layers
        if state_action:
            layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim
        else:
            layer_input_dim = self.nn_params.state_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.nu = nn.Linear(layer_input_dim, 1)
        self.weight_init(self.nu, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, action):

        state = torch.Tensor(state).to(self.nn_params.device)

        if self.state_action:

            action = torch.Tensor(action).to(self.nn_params.device)
            inp = torch.cat((state, action), dim=1)
        else:
            inp = state

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        NU = self.nu(inp)

        return NU
        #NU = torch.clamp(self.nu(inp), 70, -70)

class Zeta_NN(BaseNN):
    """
        state_action : weather to estimate for state action or just state.
    """

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(Zeta_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity
        self.state_action = state_action

        if state_action:
            layer_input_dim = self.nn_params.state_dim + self.nn_params.action_dim
        else:
            layer_input_dim = self.nn_params.state_dim

        hidden_layer_dim = self.nn_params.hidden_layer_dim

        # Hidden layers
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        # Final Layer
        self.zeta = nn.Linear(layer_input_dim, 1)
        self.weight_init(self.zeta, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state, action):
        """ Here the input can either be the state or a concatanation of state and action"""
        state = torch.Tensor(state).to(self.nn_params.device)
        if self.state_action:
            action = torch.Tensor(action).to(self.nn_params.device)
            inp = torch.cat((state, action), dim=1)
        else:
            inp =state

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        Zeta = self.zeta(inp)

        return Zeta


class Discrete_Q_Function_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path):

        super(Discrete_Q_Function_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        # Hidden layers
        layer_input_dim = self.nn_params.state_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim
        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer)
            self.layers.append(l)
            layer_input_dim = dim

        #Final Layer
        self.Q_value = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.Q_value, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state):

        state = torch.Tensor(state).to(self.nn_params.device)
        inp = state

        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)
        Q_s_a = self.Q_value(inp)

        return Q_s_a

    def get_value(self, state, format="torch"):

        if format == "torch":
            return self.forward(state)
        elif format == "numpy":
            return self.forward(state).cpu().detach().numpy()