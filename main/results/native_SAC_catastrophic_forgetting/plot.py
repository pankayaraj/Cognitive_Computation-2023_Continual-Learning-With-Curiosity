import torch
import matplotlib.pyplot as plt
from pathlib import Path


load_dir = "results_length__s_i_1000_5"


rewards = torch.load(load_dir)

print(len(rewards[0]))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for i in range(len(rewards)):
    plt.plot(rewards[i], linewidth=3)
plt.show()