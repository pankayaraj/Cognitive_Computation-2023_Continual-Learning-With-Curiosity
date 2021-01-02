import random
import matplotlib.pyplot as plt
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import numpy as np
s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]

M = Reservoir_with_Cur_Replay_Memory(50000)

SNR_bool = torch.load("bool")
avg_len_snr = 60

measure = 1.0
last_bool = 0
measure_increase_threshold = 40000

reduce = 10e-5

t = 0
a = 0
z = []
y = []
for i in range(len(a_c)):
    t += 1
    c = a_c[i]

    if i < avg_len_snr:
        pushed =M.push(0,0,0,0,c,0,0)

    else:
        if SNR_bool[i-avg_len_snr] == 1.0:
            last_bool = 0
        else:
            last_bool += 1


        if last_bool >= measure_increase_threshold:
            measure = 1.0
        else:
            if measure > 0:
                measure -= reduce

        if measure > 0.5:
            pushed = M.push(0,0,0,0,c,0,0)
        else:
            next(M.tiebreaker)
            pushed = False

    if pushed:
        z.append(1)
    else:
        z.append(0)
    #y.append(avg_t)
    y.append(measure)


x1 = 0
x2 = 0
x3 = 0
x4 = 0
a = M.storage
for i in range(len(a)):
    if a[i][1] >= change_var_at[0] and a[i][1] < change_var_at[1]:
        x1 += 1
    elif a[i][1] >= change_var_at[1] and a[i][1] < change_var_at[2]:
        x2 += 1
    elif a[i][1] >= change_var_at[2] and a[i][1] < change_var_at[3]:
        x3 += 1
    elif a[i][1] >= change_var_at[3]:
        x4 += 1


plt.plot(z)
plt.plot(y)
plt.legend(["bool", "measure"])
plt.show()


print(x1, x2, x3, x4)
size = len(a)
data = [x1/size, x2/size, x3/size, x4/size]
print(data)
print(len(M))

