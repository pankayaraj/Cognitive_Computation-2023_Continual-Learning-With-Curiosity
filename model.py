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

#for soft actor critic
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class NN_Paramters():
    def __init__(self, state_dim, action_dim, non_linearity = F.tanh, weight_initializer = 'xavier', bias_initializer = 'zero',
                 hidden_layer_dim = [128, 128], device= torch.device('cuda'), l_r=0.0001, CNN_layers = [(3, 32, 8, 4), (32, 64, 4, 2), (64, 64 ,3, 1)],
                 flatten_dim =1536 ,  CNN_initalizer = "kaiming"):
                #[(4, 32, 8, 4), (32, 64, 4, 2), (64, 64 ,3, 1)]
                #7*7*64
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_layer_dim = hidden_layer_dim
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.non_linearity = non_linearity
        self.l_r = l_r


        self.CNN_layers = CNN_layers
        self.flatten_dim = flatten_dim
        self.CNN_initalizer = CNN_initalizer

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

    def weight_init(self, layer, w_initalizer, b_initalizer, non_lin=None):
        #initalize weight
        weight_initialize(layer, w_initalizer, non_lin=non_lin)
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

class Continuous_Gaussian_Policy(BaseNN):
    # adapted from https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/model.py
    def __init__(self, nn_params, save_path, load_path, action_space=None):

        super(Continuous_Gaussian_Policy, self).__init__(save_path=save_path, load_path=load_path)

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

class DDPG_Policy(BaseNN):
    def __init__(self, nn_params, save_path, load_path):
        super(DDPG_Policy, self).__init__(save_path=save_path, load_path=load_path,
                                          )

        self.layers = nn.ModuleList([])
        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        self.noise = "gaussian"


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
        self.action = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.action, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state):

        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        self.batch_size = state.size()[0]
        inp = state
        for i, layer in enumerate(self.layers):
            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)

        action = self.action(inp)
        action = torch.tanh(action)
        return action

    def sample(self, state, format="torch"):
        action = self.forward(state)

        if format == "torch":
            return action
        elif format=="numpy":
            return action.cpu().detach().numpy()

    def to(self, device):
        super().to(device)
        self.nn_params.device= device


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

        if type(state) != torch.Tensor:
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

        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)
        if type(action) != torch.Tensor:
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

        if type(state) != torch.Tensor:
            inp = torch.Tensor(state).to(self.nn_params.device)
        else:
            inp = state

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
        if type(state) != torch.Tensor:
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


class Discrete_Q_Function_CNN_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path):

        super(Discrete_Q_Function_CNN_NN, self).__init__(save_path=save_path, load_path=load_path)
        self.cnn_layers = nn.ModuleList([])
        self.flatten_layers = nn.ModuleList([])
        self.layers = nn.ModuleList([])


        self.nn_params = nn_params
        self.non_lin = self.nn_params.non_linearity

        # Hidden layers
        layer_input_dim = self.nn_params.state_dim
        hidden_layer_dim = self.nn_params.hidden_layer_dim

        #CNN layers
        cnn_layers = self.nn_params.CNN_layers
        flatten_dim = self.nn_params.flatten_dim
        cnn_initializer = self.nn_params.CNN_initalizer

        for i, spec in enumerate(cnn_layers):
            l = nn.Conv2d(spec[0], spec[1], kernel_size=spec[2], stride=spec[3])
            self.weight_init(l, cnn_initializer, cnn_initializer, non_lin=self.non_lin)
            self.cnn_layers.append(l)

        l = nn.Flatten(start_dim=1)
        self.flatten_layers.append(l)
        layer_input_dim = flatten_dim


        for i, dim in enumerate(hidden_layer_dim):
            l = nn.Linear(layer_input_dim, dim)
            self.weight_init(l, self.nn_params.weight_initializer, self.nn_params.bias_initializer, non_lin=None)
            self.layers.append(l)
            layer_input_dim = dim

        #Final Layer
        self.Q_value = nn.Linear(layer_input_dim, self.nn_params.action_dim)
        self.weight_init(self.Q_value, self.nn_params.weight_initializer, self.nn_params.bias_initializer)

        self.to(self.nn_params.device)

    def forward(self, state):
        if type(state) != torch.Tensor:
            state = torch.Tensor(state).to(self.nn_params.device)

        if len(state.shape) == 3:
            #inp = torch.reshape(state, (1,4,84,84))
            inp = torch.reshape(state, (1, 3, 80, 60))
        else:

            batch_size = state.shape[0]
            #new_dim = (batch_size, 4, 84, 84)
            new_dim = (batch_size, 3, 80, 60)

            inp = state.reshape(shape=new_dim)


        for i, layer in enumerate(self.cnn_layers):

            if self.non_lin != None:
                inp = self.non_lin(layer(inp))
            else:
                inp = layer(inp)

        for i, layer in enumerate(self.flatten_layers):
            inp = layer(inp)


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

class ICM_Next_State_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(ICM_Next_State_NN, self).__init__(save_path=save_path, load_path=load_path)
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

class ICM_Action_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(ICM_Action_NN, self).__init__(save_path=save_path, load_path=load_path)

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


class ICM_Reward_NN(BaseNN):

    def __init__(self, nn_params, save_path, load_path, state_action=True):
        super(ICM_Reward_NN, self).__init__(save_path=save_path, load_path=load_path)
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