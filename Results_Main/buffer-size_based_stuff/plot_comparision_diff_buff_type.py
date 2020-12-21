import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

length_interval = 30000
l_interval_rate = 0.4
l_linear_rate = 65e-7
update_on_interval = False
no_steps = 120000

dir_name_r_fifo = "buff_size_2000/linear_False_m_s_2000__restart_alpha_False_Buffer_FIFO"
dir_name_r_hrf = "buff_size_2000/linear_False_m_s_2000__restart_alpha_False_Buffer_HRF"
dir_name_r_res = "buff_size_2000/linear_False_m_s_2000__restart_alpha_False_Buffer_Res"


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
load_dir_5 = "results_length__s_i_1000_5"
changing_variable_name = "diff_buffer_type_comparision"


legend = [ "FIFO", "Reservior", "Reservior with FIFO", ]


r1 = torch.load(dir_name_r_hrf + "/" + load_dir_1)
r2 = torch.load(dir_name_r_hrf + "/" + load_dir_2)
r3 = torch.load(dir_name_r_hrf + "/" + load_dir_3)
r4 = torch.load(dir_name_r_hrf + "/" + load_dir_4)
r5 = torch.load(dir_name_r_hrf + "/" + load_dir_5)

rewards_r_hrf = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_hrf[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_hrf[j][i] = rewards_r_hrf[j][i]/5

r1 = torch.load(dir_name_r_fifo + "/" + load_dir_1)
r2 = torch.load(dir_name_r_fifo + "/" + load_dir_2)
r3 = torch.load(dir_name_r_fifo + "/" + load_dir_3)
r4 = torch.load(dir_name_r_fifo + "/" + load_dir_4)
r5 = torch.load(dir_name_r_fifo + "/" + load_dir_5)

rewards_r_fifo = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_fifo[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_fifo[j][i] = rewards_r_fifo[j][i]/5

r1 = torch.load(dir_name_r_res+ "/" + load_dir_1)
r2 = torch.load(dir_name_r_res + "/" + load_dir_2)
r3 = torch.load(dir_name_r_res + "/" + load_dir_3)
r4 = torch.load(dir_name_r_res + "/" + load_dir_4)
r5 = torch.load(dir_name_r_res + "/" + load_dir_5)

rewards_r_res = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]

for j in range(int(no_steps/length_interval)):
    for i in range(len(r1[0])):
        rewards_r_res[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_res[j][i] = rewards_r_res[j][i]/5


rewards_r_fifo_avg = np.sum(rewards_r_fifo, axis=0)/len(rewards_r_fifo)
rewards_r_res_avg = np.sum(rewards_r_res, axis=0)/len(rewards_r_hrf)
rewards_r_hrf_avg = np.sum(rewards_r_hrf, axis=0)/len(rewards_r_hrf)



fig, ax = plt.subplots(1, 1, figsize=(10, 10))
plt.plot(rewards_r_fifo_avg, linewidth=3)
plt.plot(rewards_r_res_avg, linewidth=3)
plt.plot(rewards_r_hrf_avg, linewidth=3)
plt.legend(legend)
plt.xlabel('No of steps X1000')
plt.ylabel("Reward")
plt.title("For SAC trained at " + str(changing_variable_name) + " average " )
name = "buff_size_2000/diff_buffer_type" + "/" + changing_variable_name
plt.savefig(name)
plt.close(fig)

for i in range(len(changing_variable)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.plot(rewards_r_fifo[i], linewidth=3)
    plt.plot(rewards_r_res[i], linewidth=3)
    plt.plot(rewards_r_hrf[i], linewidth=3)
    plt.legend(legend)
    plt.xlabel('No of steps X1000')
    plt.ylabel("Reward")
    plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[i]))
    name = "buff_size_2000/diff_buffer_type" + "/" + "diff_buffer_type_" + str(changing_variable[i]) + ".png"
    plt.savefig(name)
    plt.close(fig)