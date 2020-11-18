import torch
import matplotlib.pyplot as plt
from pathlib import Path

length_interval = 10000
l_interval_rate = 5
l_linear_rate = 0.0005
update_on_interval = True
no_steps = 100000

changing_variable = [1.0  for i in range(int(no_steps / length_interval))]
if update_on_interval:
    for i in range(1, int(no_steps / length_interval)):
        changing_variable[i] = float(changing_variable[i-1] + l_interval_rate)
else:
    for i in range(1, int(no_steps / length_interval)):
        changing_variable[i] = changing_variable[i-1] + l_linear_rate*update_on_interval

load_dir = "results_length__s_i_1000"
changing_variable_name = "length"

legend = [str(i) for i in changing_variable]


rewards = torch.load(load_dir)

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for i in range(len(changing_variable)):
    plt.plot(rewards[i], linewidth=3)

plt.legend(legend)
plt.xlabel('No of steps X1000')
plt.ylabel("Reward")
plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[0]))
name = "graph/" + changing_variable_name
plt.savefig(name)
plt.close(fig)