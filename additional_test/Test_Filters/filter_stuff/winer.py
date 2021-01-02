import scipy
from scipy.signal.signaltools import wiener
import matplotlib.pyplot as plt
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import numpy as np
s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]


#n_a_c = wiener(a_c, mysize=None, noise=20)
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w



def signaltonoise(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)

    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)
n_a_c= signaltonoise(a_c)
#n_a_c = moving_average(a_c, 50)
plt.plot(n_a_c)
plt.show()