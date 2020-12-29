import torch
import heapq
import matplotlib.pyplot as plt
import numpy as np

mem_size = 1000
cur = True
prop = False
custom = True

change_var_at = [30000, 60000, 120000, 200000]
dir = "custom_non_proportional_2/"
M1 = []
M2 = []

for i in range(1,5):
    M1.append(torch.load(dir + "replay_m_cur_hrf_tr_ft/" + "replay_memn_c_t" + str(i)))
    M2.append(torch.load(dir + "replay_m_cur_hrf/" + "replay_memn_c_t" + str(i)))
ratio1 = []
ratio2 = []


for j in range(len(M1)):

    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    a = M1[j].reservior_buffer.storage

    #a = M.storage


    for i in range(len(a)):
        if a[i][1] >= 0 and a[i][1] < 30000:
            x1 += 1
        elif a[i][1] >= 30000 and a[i][1] < 60000:
            x2 += 1
        elif a[i][1] >= 60000 and a[i][1] < 120000:
            x3 += 1
        elif a[i][1] >= 120000 and a[i][1] < 200000:
            x4 += 1

    size = len(a)
    ratio1.append([x1/size, x2/size, x3/size, x4/size])

    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0

    a = M2[j].reservior_buffer.storage

    # a = M.storage

    for i in range(len(a)):
        if a[i][1] >= 0 and a[i][1] < 30000:
            x1 += 1
        elif a[i][1] >= 30000 and a[i][1] < 60000:
            x2 += 1
        elif a[i][1] >= 60000 and a[i][1] < 120000:
            x3 += 1
        elif a[i][1] >= 120000 and a[i][1] < 200000:
            x4 += 1

    size = len(a)
    ratio2.append([x1 / size, x2 / size, x3 / size, x4 / size])

labels = ["l = 1.0", "l = 1.2", "l=1.4", "l=1.6"]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

for i in range(len(ratio1)):
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, ratio1[i], width, label='HRF with curisoity')
    rects2 = ax.bar(x + width/2, ratio2[i], width, label="HRF")

    ax.set_ylabel('Precentage of data')
    ax.set_xlabel('l value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Buffer at time = " + str(change_var_at[i]))
    plt.savefig(dir + "ratio_comparision_at_" + str(change_var_at[i]))
