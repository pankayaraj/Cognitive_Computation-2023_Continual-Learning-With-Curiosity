import random
import matplotlib.pyplot as plt
import torch
from util.new_replay_buffers.reservior_buffer_w_cur_n_SNR import Reservoir_with_Cur_SNR_Replay_Memory
from util.new_replay_buffers.reservior_buffer_w_cur_n_SNR_with_FT_FIFO import Half_Reservoir_Cur_n_SNR_FIFO_Flow_Through_Replay_Buffer
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory

import numpy as np

"""
s_c = torch.load("forward_curiosity")[0:400000]
a_c = torch.load("inverse_curiosity")[0:400000]
#r_c = torch.load("reward_curiosity")[0:400000]
"""


s_c = torch.load("forward_curiosity1")
a_c = torch.load("inverse_curiosity1")
#r_c =""" torch.load("reward_curiosity")
""""""



a =[]
for j in range(len(a_c[0])):
    c = 0
    c_ = []

    for i in range(len(a_c)):
        c += a_c[i][j]

        c_.append(a_c[i][j])
    std = np.std(c_)
    a.append(c/len(a_c))
print(len(a_c))

a_c = a
"""
plt.plot(a_c)
plt.plot(a)
plt.show()
"""
mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]
#M = Reservoir_with_Cur_Replay_Memory(capacity=33000)
M = Half_Reservoir_Cur_n_SNR_FIFO_Flow_Through_Replay_Buffer(50000, fifo_fac=0.34)
t = 0

#a_c = r_c
for c in a_c:

    if t == 100000 or t == 150000 or t==350000:


        x1 = 0
        x2 = 0
        x3 = 0
        x4 = 0
        a = M.reservior_buffer.storage
        for i in range(len(a)):
            if a[i][1] >= change_var_at[0] and a[i][1] < change_var_at[1]:
                x1 += 1
            elif a[i][1] >= change_var_at[1] and a[i][1] < change_var_at[2]:
                x2 += 1
            elif a[i][1] >= change_var_at[2] and a[i][1] < change_var_at[3]:
                x3 += 1
            elif a[i][1] >= change_var_at[3]:
                x4 += 1
        print(x1,x2,x3,x4)
        
        size = len(a)
        data = [x1 / size, x2 / size, x3 / size, x4 / size]
        print(data)
        #print(len(M))

    t += 1
    M.push(0, 0, 0, 0, c, 0, 0)




#plt.plot(a_c)
#plt.plot(M.reservior_buffer.SNR)
plt.plot(M.reservior_buffer.MEAN)
#plt.plot(M.reservior_buffer.MEASURE)
plt.plot(M.reservior_buffer.BOOL)
plt.show()


x1 = 0
x2 = 0
x3 = 0
x4 = 0
a = M.reservior_buffer.storage
for i in range(len(a)):
    if a[i][1] >= change_var_at[0] and a[i][1] < change_var_at[1]:
        x1 += 1
    elif a[i][1] >= change_var_at[1] and a[i][1] < change_var_at[2]:
        x2 += 1
    elif a[i][1] >= change_var_at[2] and a[i][1] < change_var_at[3]:
        x3 += 1
    elif a[i][1] >= change_var_at[3]:
        x4 += 1
print(size)
print(x1, x2, x3, x4)
size = len(a)
data = [x1/size, x2/size, x3/size, x4/size]
print(data)
print(len(M))

