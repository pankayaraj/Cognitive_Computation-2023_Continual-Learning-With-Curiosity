import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

length_interval = 30000
l_interval_rate = 0.4
l_linear_rate = 65e-7
update_on_interval = False
no_steps = 90000

dir_name_r_1 = "curiosity_False_linear_True_m_s_10000__restart_alpha_False"
dir_name_r_2 = "curiosity_False_linear_True_m_s_30000__restart_alpha_False"
dir_name_r_3 = "curiosity_False_linear_True_m_s_90000__restart_alpha_False"

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

changing_variable_name = "buffer size"


legend = ["buff size = 10000", "buff size = 30000", "buff size = 90000" ]

r1 = torch.load(dir_name_r_1 + "/" + load_dir_1)
r2 = torch.load(dir_name_r_1 + "/" + load_dir_2)
r3 = torch.load(dir_name_r_1 + "/" + load_dir_3)
#r4 = torch.load(dir_name + "/" + load_dir_4)
rewards_r_1 = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_1[j][i] = r1[j][i] + r2[j][i] + r3[j][i] #+ r4[j][i]
        rewards_r_1[j][i] = rewards_r_1[j][i]/4

r1 = torch.load(dir_name_r_2 + "/" + load_dir_1)
r2 = torch.load(dir_name_r_2 + "/" + load_dir_2)
r3 = torch.load(dir_name_r_2 + "/" + load_dir_3)
#r4 = torch.load(dir_name + "/" + load_dir_4)
rewards_r_2 = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_2[j][i] = r1[j][i] + r2[j][i] + r3[j][i] #+ r4[j][i]
        rewards_r_2[j][i] = rewards_r_2[j][i]/3

rewards_r_3 = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

r1 = torch.load(dir_name_r_3 + "/" + load_dir_1)
r2 = torch.load(dir_name_r_3 + "/" + load_dir_2)
r3 = torch.load(dir_name_r_3 + "/" + load_dir_3)

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_3[j][i] = r1[j][i] + r2[j][i] + r3[j][i] #+ r4[j][i]
        rewards_r_3[j][i] = rewards_r_3[j][i]/3

rewards_r_1_avg = np.sum(rewards_r_1, axis=0)/len(rewards_r_1)
rewards_r_2_avg = np.sum(rewards_r_2, axis=0)/len(rewards_r_2)
rewards_r_3_avg = np.sum(rewards_r_3, axis=0)/len(rewards_r_3)



fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(rewards_r_1_avg, linewidth=3)
plt.plot(rewards_r_2_avg, linewidth=3)
plt.plot(rewards_r_3_avg, linewidth=3)
plt.legend(legend)
plt.xlabel('No of steps X1000')
plt.ylabel("Reward")
plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[0]) )
name = "diff_buffer_size" + "/" + changing_variable_name
plt.savefig(name)
plt.close(fig)

for i in range(len(changing_variable)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.plot(rewards_r_1[i], linewidth=3)
    plt.plot(rewards_r_2[i], linewidth=3)
    plt.plot(rewards_r_3[i], linewidth=3)
    plt.legend(legend)
    plt.xlabel('No of steps X1000')
    plt.ylabel("Reward")
    plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[0]))
    name = "diff_buffer_size" + "/" + "time_stamp_" + str(changing_variable[i]) + ".png"
    plt.savefig(name)
    plt.close(fig)
