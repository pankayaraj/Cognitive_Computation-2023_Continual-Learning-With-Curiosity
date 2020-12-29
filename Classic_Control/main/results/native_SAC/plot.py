import torch
import matplotlib.pyplot as plt
from pathlib import Path

load_dir = "results_length__s_i_1000"
changing_variable_name = "length"
changeing_variable = [1, 2, 10, 20]
legend = [str(i) for i in changeing_variable]


rewards = torch.load(load_dir)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for i in range(len(changeing_variable)):
    plt.plot(rewards[i], linewidth=3)

plt.legend(legend)
plt.xlabel('No of steps X1000')
plt.ylabel("Reward")
plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changeing_variable[0]))
name = "graph/" + changing_variable_name
plt.savefig(name)
plt.close(fig)