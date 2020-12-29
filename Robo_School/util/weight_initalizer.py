import torch


def weight_initialize(layer, name):
    if name == 'xavier':
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
    if name == 'zero':
        torch.nn.init.constant_(layer.weight, 0)


def bias_initialize(layer, name):
    if name == 'zero':
        torch.nn.init.constant_(layer.bias, 0)


