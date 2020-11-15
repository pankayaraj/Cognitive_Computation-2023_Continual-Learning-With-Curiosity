import torch
import matplotlib
from pathlib import Path

load_dir = "main/results/results_gravity__s_i_1000"




rewards = torch.load(load_dir)
print(rewards)