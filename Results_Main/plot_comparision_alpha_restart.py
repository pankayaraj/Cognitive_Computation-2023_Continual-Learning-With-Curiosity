import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

length_interval = 30000
l_interval_rate = 0.4
l_linear_rate = 65e-7
update_on_interval = False
no_steps = 90000

dir_name_r_t = "curiosity_False_linear_True_m_s_30000__restart_alpha_True"
dir_name_r_f = "curiosity_False_linear_True_m_s_30000__restart_alpha_False"

changing_variable = [1.0  for i in range(int(no_steps / length_interval))]
if update_on_interval:
    for i in range(1, int(no_steps / length_interval)):
        changing_variable[i] = float(changing_variable[i-1] + l_interval_rate)
else:
    for i in range(1, int(no_steps / length_interval)):

        changing_variable[i] = changing_variable[i-1] + l_linear_rate*length_interval
load_dir_1 = "results_length__s_i_1000_1"
load_dir_2 = "results_length__s_i_1000_2"
load_dir_3 = "results_length__s_i_1000_3"
load_dir_4 = "results_length__s_i_1000_4"

changing_variable_name = "reinitalized_alpha_comparision"


legend = ["Reinitalized alpha", "Non Reinitalized alpha"]


r1 = torch.load(dir_name_r_t + "/" + load_dir_1)
r2 = torch.load(dir_name_r_t + "/" + load_dir_2)
r3 = torch.load(dir_name_r_t + "/" + load_dir_3)
#r4 = torch.load(dir_name + "/" + load_dir_4)
rewards_r_t = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_t[j][i] = r1[j][i] + r2[j][i] + r3[j][i] #+ r4[j][i]
        rewards_r_t[j][i] = rewards_r_t[j][i]/4

r1 = torch.load(dir_name_r_f + "/" + load_dir_1)
r2 = torch.load(dir_name_r_f + "/" + load_dir_2)
r3 = torch.load(dir_name_r_f + "/" + load_dir_3)
#r4 = torch.load(dir_name + "/" + load_dir_4)
rewards_r_f = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_f[j][i] = r1[j][i] + r2[j][i] + r3[j][i] #+ r4[j][i]
        rewards_r_f[j][i] = rewards_r_f[j][i]/4

rewards_r_t_avg = np.sum(rewards_r_t, axis=0)/len(rewards_r_t)
rewards_r_f_avg = np.sum(rewards_r_f, axis=0)/len(rewards_r_f)


fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(rewards_r_t_avg, linewidth=3)
plt.plot(rewards_r_f_avg, linewidth=3)
plt.legend(legend)
plt.xlabel('No of steps X1000')
plt.ylabel("Reward")
plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[0]) )
name = "reinitalizing_alpha" + "/" + changing_variable_name
plt.savefig(name)
plt.close(fig)

for i in range(len(changing_variable)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.plot(rewards_r_t[i], linewidth=3)
    plt.plot(rewards_r_f[i], linewidth=3)
    plt.legend(legend)
    plt.xlabel('No of steps X1000')
    plt.ylabel("Reward")
    plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[0]))
    name = "reinitalizing_alpha" + "/" + "time_stamp_" + str(changing_variable[i]) + ".png"
    plt.savefig(name)
    plt.close(fig)
