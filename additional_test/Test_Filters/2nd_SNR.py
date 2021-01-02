import scipy
from scipy.signal.signaltools import wiener
import matplotlib.pyplot as plt
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import numpy as np
#s_c = torch.load("forward_curiosity")
#a_c = torch.load("inverse_curiosity")
a_c = torch.load("bool_cur")
bool = torch.load("bool")

plt.plot(bool[0:50000])
plt.show()
mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]

avg_len = 5000
mean = []
std = []
SNR = []

delta = 1e-5
bool = []
bool_cur = []
for i in range(len(a_c)-avg_len):
    mean.append(np.mean(a_c[i:i+avg_len]))
    std.append(np.std(a_c[i:i+avg_len]))
    SNR.append(mean[-1]/std[-1]+delta)

    if SNR[-1] < 3*mean[-1]:
        bool.append(1)
        bool_cur.append(mean[-1])
    else:
        bool.append(0)
        bool_cur.append(0)

#plt.plot(SNR)
plt.plot(mean)
plt.plot(std)
plt.legend(["SNR", "mean"])
plt.show()
plt.close()

plt.plot(bool)
#plt.savefig("bool")
plt.show()
plt.close()

plt.plot(bool_cur)
#plt.savefig("bool_cur")
plt.show()
plt.close()

#torch.save(bool_cur, "bool_cur")
#torch.save(bool, "bool")
