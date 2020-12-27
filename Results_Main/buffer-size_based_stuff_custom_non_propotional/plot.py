import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

length_interval = 30000
l_interval_rate = 0.4
l_linear_rate = 65e-7
update_on_interval = False
no_steps = 200000

dir_name = "buff_size_2000/linear_False_m_s_2000__restart_alpha_False_Buffer_HRF_TR_Cur"


changing_variable = [1.0  for i in range(int(no_steps / length_interval))]
if update_on_interval:
    for i in range(1, int(no_steps / length_interval)):
        changing_variable[i] = float(changing_variable[i-1] + l_interval_rate)
else:
    for i in range(1, int(no_steps / length_interval)):

        changing_variable[i] = changing_variable[i-1] + l_linear_rate*length_interval

changing_variable = [1.0, 1.2, 1.4, 1.6]

load_dir_1 = "results_length__s_i_1000_1"
load_dir_2 = "results_length__s_i_1000_2"
load_dir_3 = "results_length__s_i_1000_3"
load_dir_4 = "results_length__s_i_1000_4"
load_dir_5 = "results_length__s_i_1000_5"

changing_variable_name = "length"


legend = [str(i) for i in changing_variable]

r1 = torch.load(dir_name + "/" + load_dir_1)
r2 = torch.load(dir_name + "/" + load_dir_2)
r3 = torch.load(dir_name + "/" + load_dir_3)
r4 = torch.load(dir_name + "/" + load_dir_4)
r5 = torch.load(dir_name + "/" + load_dir_5)

rewards = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

rew_total = [[] for j in range(len(changing_variable))]
for j in range(len(changing_variable)):
    rew_total[j].append(r1[j])
    rew_total[j].append(r2[j])
    rew_total[j].append(r3[j])
    rew_total[j].append(r4[j])
    rew_total[j].append(r5[j])

    for i in range(len(r1[0])):
        rewards[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards[j][i] = rewards[j][i]/5

rew_total = np.array(rew_total)
rew_std = np.std(rew_total, axis=1)
x = [i for i in range(no_steps//1000)]

fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for i in range(len(changing_variable)):
    plt.plot(x, rewards[i], linewidth=3)
    plt.fill_between(x, rewards[i] + rew_std[i], rewards[i] - rew_std[i], alpha = 0.3)

plt.legend(legend)
plt.xlabel('No of steps X1000')
plt.ylabel("Reward")
plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[0]))
name = dir_name + "/" + changing_variable_name
plt.savefig(name)
plt.close(fig)