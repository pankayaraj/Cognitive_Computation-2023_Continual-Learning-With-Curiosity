import torch
import heapq
import matplotlib.pyplot as plt
import numpy as np


cur = True
prop = False
custom = True

mul = 1000
m_s = "50k"

change_var_at = [0, 100, 150, 350, 400]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]


dir = ""
M1 = []
M2 = []

for i in range(1,5):
    M1.append(torch.load(dir + "hrf_ft_" + m_s + "/" + "replay_mem" + str(i)))
    M2.append(torch.load(dir + "hrf_snr_ft_" + m_s + "/" + "replay_mem" + str(i)))

ratio1 = []
ratio2 = []


for j in range(len(M1)):

    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    x5 = 0

    a = M1[j].reservior_buffer.storage
    #a = M1[j].storage


    for i in range(len(a)):
        
        if a[i][1] >= change_var_at[0] and a[i][1] < change_var_at[1]:
            x1 += 1
        elif a[i][1] >= change_var_at[1] and a[i][1] < change_var_at[2]:
            x2 += 1
        elif a[i][1] >= change_var_at[2] and a[i][1] < change_var_at[3]:
            x3 += 1
        elif a[i][1] >= change_var_at[3] and a[i][1] < change_var_at[4] :
            x4 += 1
        elif a[i][1] >= change_var_at[4]:
            x5 += 1
    print("M1")
    print(x1, x2, x3, x4, x5)
    size = len(a)
    ratio1.append([x1/size, x2/size, x3/size, x4/size, x5/size])

    x1 = 0
    x2 = 0
    x3 = 0
    x4 = 0
    x5 = 0



    a = M2[j].reservior_buffer.storage
    #a = M2[j].storage

    for i in range(len(a)):

        if a[i][1] >= change_var_at[0] and a[i][1] < change_var_at[1]:
            x1 += 1
        elif a[i][1] >= change_var_at[1] and a[i][1] < change_var_at[2]:
            x2 += 1
        elif a[i][1] >= change_var_at[2] and a[i][1] < change_var_at[3]:
            x3 += 1
        elif a[i][1] >= change_var_at[3] and a[i][1] < change_var_at[4]:
            x4 += 1
        elif a[i][1] >= change_var_at[4]:
            x5 += 1
    size = len(a)
    ratio2.append([x1 / size, x2 / size, x3 / size, x4 / size, x5/size])

    print("M2")
    print(x1, x2, x3, x4, x5)


labels = ["p = 0.75", "p = 1.75", "p = 2.75", "p = 3.75",  "p = 4.75"]

labels = labels[0:4]
for i in range(len(M1)):
    ratio1[i] = ratio1[i][0:4]
    ratio2[i] = ratio2[i][0:4]

print(len(ratio1), len(ratio2), len(labels))

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

for i in range(len(ratio1)):

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, ratio1[i], width, label='HRF with flow through')
    rects2 = ax.bar(x + width/2, ratio2[i], width, label="HRF with curisoity and flow through")

    ax.set_ylabel('Precentage of data')
    ax.set_xlabel('l value')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_title("Buffer at time = " + str(change_var_at[i]))
    plt.savefig(dir + "ratio_comparision_at_" + str(change_var_at[i]))
