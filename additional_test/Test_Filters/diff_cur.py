import matplotlib.pyplot as plt
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import numpy as np
s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]



new_a_c = []
n_c = []
average = 0
avg_len = 50
avg = []
t = 0



for i in range(len(a_c)):
    t += 1
    if t < avg_len:
        avg.append(a_c[i])
        average += a_c[i]/avg_len

    else:
        avg.append(a_c[i])
        temp = avg.pop(0)

        average += a_c[i] / avg_len
        average -= temp / avg_len

        new_a_c.append(average)

#plt.plot(new_a_c)
#plt.show()


for i in range(avg_len, len(a_c)):
    n_c.append(a_c[i]/new_a_c[i-avg_len])

plt.plot(n_c)
plt.show()