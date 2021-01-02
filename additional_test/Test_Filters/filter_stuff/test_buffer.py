import random
import matplotlib.pyplot as plt
import torch
from util.new_replay_buffers.reservior_buffer_w_cur_n_SNR import Reservoir_with_Cur_SNR_Replay_Memory
import numpy as np
s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]

M = Reservoir_with_Cur_SNR_Replay_Memory(50000)

for c in a_c:

    M.push(0, 0, 0, 0, c, 0, 0)


plt.plot(M.PUSH)
plt.plot(M.MEASURE)
plt.show()


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

print(x1, x2, x3, x4)
size = len(a)
data = [x1/size, x2/size, x3/size, x4/size]
print(data)
print(len(M))

