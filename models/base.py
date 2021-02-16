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
                 hidden_layer_dim = [128, 128], device= torch.device('cuda'), l_r=0.0001, CNN_layers = [(4, 32, 8, 4), (32, 64, 4, 2), (64, 64 ,3, 1)],
                 flatten_dim = 7*7*64,  CNN_initalizer = "kaiming"):

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