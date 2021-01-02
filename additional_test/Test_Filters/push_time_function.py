import numpy as np
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import random
import matplotlib.pyplot as plt

s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

#plt.plot(a_c)
#plt.show()

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]


def push_fun(x, slope, shift):
    return 1 / (1 + np.exp(-slope * x + shift))

def time_fun(x, slope, shift):
    return 1 / (1 + np.exp(slope * x - shift))

t_slope, t_shift = 0.001, 10
p_slope, p_shift = 0.0005, 10
shift = 10
slope = 0.001

x = [i for i in range(50000)]
y = [push_fun(i, p_slope, p_shift ) for i in x]
z = [time_fun(i, t_slope, t_shift) for i in x]

plt.plot(x,y)
plt.plot(x,z)
plt.show()

class push_trace():

    def __init__(self, lam, slope=3.0, shift=5.0):

        self.E = 0
        self.lam = lam
        self.slope = slope
        self.shift = shift

    def get_trace(self):
        return push_fun(self.E, slope=self.slope, shift=self.shift)

    def not_pushed(self):
        self.E = self.lam*self.E

    def is_pushed(self):
        self.E += 1

class time_trace():
    def __init__(self, lam, slope=3.0, shift=5.0):

        self.E = 0
        self.lam = lam
        self.slope = slope
        self.shift = shift

    def get_trace(self):
        return time_fun(self.E, slope=self.slope, shift=self.shift)

    def not_pushed(self):
        self.E = self.lam*self.E

    def is_pushed(self):
        self.E += 1
P = push_trace(lam=0.99, slope=p_slope, shift=p_shift)
T = time_trace(lam=0.99, slope=t_slope, shift=t_shift)
M = Reservoir_with_Cur_Replay_Memory(50000)

bool = []

for c in a_c:
    t_t = T.get_trace()
    p_t = P.get_trace()


    if p_t < t_t:
        bool.append(1)
        pushed = M.push(0,0,0,0,c,0,0)
        if pushed:
            P.is_pushed()
            T.is_pushed()
        else:
            T.not_pushed()
            P.not_pushed()
    else:
        pushed = M.push(0, 0, 0, 0, c, 0, 0)
        bool.append(0)
        T.not_pushed()
        P.not_pushed()

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]

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


#plt.plot(bool)
#plt.plot(s_c)
#plt.show()

print(x1, x2, x3, x4)
size = len(a)
data = [x1/size, x2/size, x3/size, x4/size]
print(data)
print(len(M))