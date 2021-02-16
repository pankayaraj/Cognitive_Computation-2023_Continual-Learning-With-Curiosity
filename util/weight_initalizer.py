import torch


def weight_initialize(layer, name, non_lin = None):
    if name == 'xavier':
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(layer.weight, gain=1)
    if name == "kaiming":
        if non_lin == None:
            print("need non linearity for kaiming")
        else:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')

    if name == 'zero':
        torch.nn.init.constant_(layer.weight, 0)


def bias_initialize(layer, name):
    if name == 'zero':
        torch.nn.init.constant_(layer.bias, 0.0)
    if name == "kaiming":
        torch.nn.init.constant_(layer.bias, 0.0)
