import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path


no_steps = 400000
n_step = int(no_steps/1000)
m_s = "50k"

dir_name_r_fifo = "buff_size_" + m_s +"/fifo"
dir_name_r_hrf = "buff_size_" + m_s + "/hrf_ft"
#dir_name_r_res = "buff_size_"+ str(m_s) + "/linear_False_m_s_"+ str(m_s) + "__restart_alpha_False_Buffer_Res"
#dir_name_r_res_cur = "buff_size_"+ str(m_s) + "/linear_False_m_s_"+ str(m_s) + "__restart_alpha_False_Buffer_Res_Cur"
dir_name_r_hrf_cur =  "buff_size_" + m_s + "/hrf_snr_ft"

changing_variable = [0.75, 1.75, 2.75, 3.75]
changing_variable_at = [0, 100, 150, 350]

load_dir_1 = "results_length__s_i_1000_1"
load_dir_2 = "results_length__s_i_1000_2"
load_dir_3 = "results_length__s_i_1000_3"
load_dir_4 = "results_length__s_i_1000_4"
load_dir_5 = "results_length__s_i_1000_5"

changing_variable_name = "diff_buffer_type_comparision"

legend = ["FIFO", "HRF Flow Through", "Curious HRF Flow Through"]


#HRF
r1 = torch.load(dir_name_r_hrf + "/" + load_dir_1)
r2 = torch.load(dir_name_r_hrf + "/" + load_dir_2)
r3 = torch.load(dir_name_r_hrf + "/" + load_dir_3)
r4 = torch.load(dir_name_r_hrf + "/" + load_dir_4)
r5 = torch.load(dir_name_r_hrf + "/" + load_dir_5)

for i in range(len(changing_variable)):
    r1[i] = r1[i][0:n_step]
    r2[i] = r2[i][0:n_step]
    r3[i] = r3[i][0:n_step]
    r4[i] = r4[i][0:n_step]
    r5[i] = r5[i][0:n_step]

r1 = r1[0:len(changing_variable)]
r2 = r2[0:len(changing_variable)]
r3 = r3[0:len(changing_variable)]
r4 = r4[0:len(changing_variable)]
r5 = r5[0:len(changing_variable)]

rewards_r_hrf = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]
rew_hrf_total = [[] for j in range(len(changing_variable))]
rew_hrf_ind_avg = []

for j in range(len(changing_variable)):
    rew_hrf_total[j].append(r1[j])
    rew_hrf_total[j].append(r2[j])
    rew_hrf_total[j].append(r3[j])
    rew_hrf_total[j].append(r4[j])
    rew_hrf_total[j].append(r5[j])
    for i in range(len(r1[0])):
        rewards_r_hrf[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_hrf[j][i] = rewards_r_hrf[j][i]/5

rew_hrf_std = np.std(rew_hrf_total, axis=1)

rew_hrf_ind_total = np.array([r1, r2, r3, r4, r5])
rew_hrf_ind_avg = np.sum(rew_hrf_ind_total, axis=1)/len(rewards_r_hrf)
rew_hrf_ind_std = np.std(rew_hrf_ind_avg, axis=0)

#fifo

r1 = torch.load(dir_name_r_fifo + "/" + load_dir_1)
r2 = torch.load(dir_name_r_fifo + "/" + load_dir_2)
r3 = torch.load(dir_name_r_fifo + "/" + load_dir_3)
r4 = torch.load(dir_name_r_fifo + "/" + load_dir_4)
r5 = torch.load(dir_name_r_fifo + "/" + load_dir_5)

for i in range(len(changing_variable)):
    r1[i] = r1[i][0:n_step]
    r2[i] = r2[i][0:n_step]
    r3[i] = r3[i][0:n_step]
    r4[i] = r4[i][0:n_step]
    r5[i] = r5[i][0:n_step]

r1 = r1[0:len(changing_variable)]
r2 = r2[0:len(changing_variable)]
r3 = r3[0:len(changing_variable)]
r4 = r4[0:len(changing_variable)]
r5 = r5[0:len(changing_variable)]

rewards_r_fifo = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]
rew_fifo_total = [[] for j in range(len(changing_variable))]
rew_fifo_ind_avg = []


for j in range(len(changing_variable)):
    rew_fifo_total[j].append(r1[j])
    rew_fifo_total[j].append(r2[j])
    rew_fifo_total[j].append(r3[j])
    rew_fifo_total[j].append(r4[j])
    rew_fifo_total[j].append(r5[j])

    for i in range(len(r1[0])):
        rewards_r_fifo[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_fifo[j][i] = rewards_r_fifo[j][i]/5
rew_fifo_std = np.std(rew_fifo_total, axis=1)

rew_fifo_ind_total = np.array([r1, r2, r3, r4, r5])
rew_fifo_ind_avg = np.sum(rew_fifo_ind_total, axis=1)/len(rewards_r_fifo)
rew_fifo_ind_std = np.std(rew_fifo_ind_avg, axis=0)

"""
#res

r1 = torch.load(dir_name_r_res+ "/" + load_dir_1)
r2 = torch.load(dir_name_r_res + "/" + load_dir_2)
r3 = torch.load(dir_name_r_res + "/" + load_dir_3)
r4 = torch.load(dir_name_r_res + "/" + load_dir_4)
r5 = torch.load(dir_name_r_res + "/" + load_dir_5)

for i in range(len(changing_variable)):
    r1[i] = r1[i][0:n_step]
    r2[i] = r2[i][0:n_step]
    r3[i] = r3[i][0:n_step]
    r4[i] = r4[i][0:n_step]
    r5[i] = r5[i][0:n_step]

r1 = r1[0:len(changing_variable)]
r2 = r2[0:len(changing_variable)]
r3 = r3[0:len(changing_variable)]
r4 = r4[0:len(changing_variable)]
r5 = r5[0:len(changing_variable)]

rewards_r_res = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]
rew_res_total = [[] for j in range(len(changing_variable))]
rew_res_ind_avg = []

for j in range(len(changing_variable)):
    rew_res_total[j].append(r1[j])
    rew_res_total[j].append(r2[j])
    rew_res_total[j].append(r3[j])
    rew_res_total[j].append(r4[j])
    rew_res_total[j].append(r5[j])

    for i in range(len(r1[0])):
        rewards_r_res[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_res[j][i] = rewards_r_res[j][i]/5

rew_res_std = np.std(rew_res_total, axis=1)

rew_res_ind_total = np.array([r1, r2, r3, r4, r5])
rew_res_ind_avg = np.sum(rew_res_ind_total, axis=1)/len(rewards_r_res)
rew_res_ind_std = np.std(rew_res_ind_avg, axis=0)


#res cur

r1 = torch.load(dir_name_r_res_cur+ "/" + load_dir_1)
r2 = torch.load(dir_name_r_res_cur + "/" + load_dir_2)
r3 = torch.load(dir_name_r_res_cur + "/" + load_dir_3)
r4 = torch.load(dir_name_r_res_cur + "/" + load_dir_4)
r5 = torch.load(dir_name_r_res_cur + "/" + load_dir_5)

for i in range(len(changing_variable)):
    r1[i] = r1[i][0:n_step]
    r2[i] = r2[i][0:n_step]
    r3[i] = r3[i][0:n_step]
    r4[i] = r4[i][0:n_step]
    r5[i] = r5[i][0:n_step]

r1 = r1[0:len(changing_variable)]
r2 = r2[0:len(changing_variable)]
r3 = r3[0:len(changing_variable)]
r4 = r4[0:len(changing_variable)]
r5 = r5[0:len(changing_variable)]

rewards_r_res_cur = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]
rew_res_cur_total = [[] for j in range(len(changing_variable))]
rew_res_cur_ind_avg = []

for j in range(len(changing_variable)):
    rew_res_cur_total[j].append(r1[j])
    rew_res_cur_total[j].append(r2[j])
    rew_res_cur_total[j].append(r3[j])
    rew_res_cur_total[j].append(r4[j])
    rew_res_cur_total[j].append(r5[j])

    for i in range(len(r1[0])):
        rewards_r_res_cur[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_res_cur[j][i] = rewards_r_res_cur[j][i]/5

rew_res_cur_std = np.std(rew_res_cur_total, axis=1)

rew_res_cur_ind_total = np.array([r1, r2, r3, r4, r5])
rew_res_cur_ind_avg = np.sum(rew_res_cur_ind_total, axis=1)/len(rewards_r_res_cur)
rew_res_cur_ind_std = np.std(rew_res_cur_ind_avg, axis=0)
"""

#res fifo cur

r1 = torch.load(dir_name_r_hrf_cur+ "/" + load_dir_1)
r2 = torch.load(dir_name_r_hrf_cur + "/" + load_dir_2)
r3 = torch.load(dir_name_r_hrf_cur + "/" + load_dir_3)
r4 = torch.load(dir_name_r_hrf_cur + "/" + load_dir_4)
r5 = torch.load(dir_name_r_hrf_cur + "/" + load_dir_5)

for i in range(len(changing_variable)):
    r1[i] = r1[i][0:n_step]
    r2[i] = r2[i][0:n_step]
    r3[i] = r3[i][0:n_step]
    r4[i] = r4[i][0:n_step]
    r5[i] = r5[i][0:n_step]

r1 = r1[0:len(changing_variable)]
r2 = r2[0:len(changing_variable)]
r3 = r3[0:len(changing_variable)]
r4 = r4[0:len(changing_variable)]
r5 = r5[0:len(changing_variable)]

rewards_r_hrf_cur = [[0. for j in range(len(r1[0]))] for i in range(len(r1))]
rew_hrf_cur_total = [[] for j in range(len(changing_variable))]
rew_hrf_cur_ind_avg = []

for j in range(len(changing_variable)):
    rew_hrf_cur_total[j].append(r1[j])
    rew_hrf_cur_total[j].append(r2[j])
    rew_hrf_cur_total[j].append(r3[j])
    rew_hrf_cur_total[j].append(r4[j])
    rew_hrf_cur_total[j].append(r5[j])

    for i in range(len(r1[0])):
        rewards_r_hrf_cur[j][i] = r1[j][i] + r2[j][i] + r3[j][i] + r4[j][i] + r5[j][i]
        rewards_r_hrf_cur[j][i] = rewards_r_hrf_cur[j][i]/5

rew_hrf_cur_std = np.std(rew_hrf_cur_total, axis=1)

rew_hrf_cur_ind_total = np.array([r1, r2, r3, r4, r5])
rew_hrf_cur_ind_avg = np.sum(rew_hrf_cur_ind_total, axis=1)/len(rewards_r_hrf_cur)
rew_hrf_cur_ind_std = np.std(rew_hrf_cur_ind_avg, axis=0)


rewards_r_fifo_avg = np.sum(rewards_r_fifo, axis=0)/len(rewards_r_fifo)
#rewards_r_res_avg = np.sum(rewards_r_res, axis=0)/len(rewards_r_res)
rewards_r_hrf_avg = np.sum(rewards_r_hrf, axis=0)/len(rewards_r_hrf)
#rewards_r_res_cur_avg = np.sum(rewards_r_res_cur, axis=0)/len(rewards_r_res_cur)
rewards_r_hrf_cur_avg = np.sum(rewards_r_hrf_cur, axis=0)/len(rewards_r_hrf_cur)

x = [i for i in range(no_steps//1000)]

fig, ax = plt.subplots(1, 1, figsize=(20, 10))
plt.plot(rewards_r_fifo_avg, linewidth=5)
plt.fill_between(x, rewards_r_fifo_avg + rew_fifo_ind_std, rewards_r_fifo_avg - rew_fifo_ind_std, alpha = 0.1)


#plt.plot(rewards_r_res_avg, linewidth=5)
#plt.fill_between(x, rewards_r_res_avg + rew_res_ind_std, rewards_r_res_avg - rew_res_ind_std, alpha = 0.1)

plt.plot(rewards_r_hrf_avg, linewidth=5)
plt.fill_between(x, rewards_r_hrf_avg + rew_hrf_ind_std, rewards_r_hrf_avg - rew_hrf_ind_std, alpha = 0.1)

#plt.plot(rewards_r_res_cur_avg, linewidth=5)
#plt.fill_between(x, rewards_r_res_cur_avg + rew_res_cur_ind_std, rewards_r_res_cur_avg - rew_res_cur_ind_std, alpha = 0.1)


plt.plot(rewards_r_hrf_cur_avg, linewidth=5)
plt.fill_between(x, rewards_r_hrf_cur_avg + rew_hrf_cur_ind_std, rewards_r_hrf_cur_avg - rew_hrf_cur_ind_std, alpha = 0.1)


plt.legend(legend,  prop={'size':30})
plt.xlabel('No of steps X1000')
plt.ylabel("Reward")
plt.title("For SAC trained at " + str(changing_variable_name) + " average " )
name = "buff_size_" + m_s + "/diff_buffer_type" + "/" + changing_variable_name
plt.savefig(name)
plt.close(fig)

for i in range(len(changing_variable)):
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    plt.plot(rewards_r_fifo[i], linewidth=5)
    plt.fill_between(x, rewards_r_fifo[i] + rew_fifo_std[i], rewards_r_fifo[i] - rew_fifo_std[i], alpha=0.1)


    #plt.plot(rewards_r_res[i], linewidth=5)
    #plt.fill_between(x, rewards_r_res[i] + rew_res_std[i], rewards_r_res[i] - rew_res_std[i], alpha=0.1)


    plt.plot(rewards_r_hrf[i], linewidth=5)
    plt.fill_between(x, rewards_r_hrf[i] + rew_hrf_std[i], rewards_r_hrf[i] - rew_hrf_std[i], alpha=0.1)


    #plt.plot(rewards_r_res_cur[i], linewidth=5)
    #plt.fill_between(x, rewards_r_res_cur[i] + rew_res_cur_std[i], rewards_r_res_cur[i] - rew_res_cur_std[i], alpha=0.1)

    plt.plot(rewards_r_hrf_cur[i], linewidth=5)
    plt.fill_between(x, rewards_r_hrf_cur[i] + rew_hrf_cur_std[i], rewards_r_hrf_cur[i] - rew_hrf_cur_std[i], alpha=0.1)


    plt.axvline(changing_variable_at[i], color="black", linewidth=3)

    plt.legend(legend, prop={'size':30})
    plt.xlabel('No of steps X1000')
    plt.ylabel("Reward")
    plt.title("For SAC trained at " + str(changing_variable_name) + " of " + str(changing_variable[i]))
    name = "buff_size_" + m_s + "/diff_buffer_type" + "/" + "diff_buffer_type_" + str(changing_variable[i]) + ".png"
    plt.savefig(name)
    plt.close(fig)