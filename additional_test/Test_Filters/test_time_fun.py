import numpy as np
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import random
import matplotlib.pyplot as plt

s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]


def time_fun(x, slope, shift):
    return 1/(1+np.exp((slope*x - shift)))


shift = 10
slope = 0.0001
z = []
x = [i for i in range(50000)]
y = [time_fun(i, slope, shift ) for i in x]
#plt.plot(x,y)
#plt.show()

class eligibility_trace():

    def __init__(self, lambda_v, r, slope=3, shif=5):

        self.E = 0
        self.lambda_v = lambda_v
        self.r = r
        self.slope = slope
        self.shift = shif

    def get_trace(self):
        return time_fun(self.E, slope=self.slope, shift=self.shift)

    def general_iterate(self):
        self.E = self.r*self.lambda_v*self.E

    def is_pushed(self):
        self.E += 1


E = eligibility_trace(lambda_v=0.99995, r=1, shif=shift, slope=slope)
M = Reservoir_with_Cur_Replay_Memory(50000)
t = 0
for i in a_c:
    r = random.uniform(0,1)

    if r > 0.0:
        ran = random.uniform(0,1)
        E.general_iterate()
        z.append(E.E)

        print(E.E)
        if ran < E.get_trace():
            pusehd = M.push(0, 0, 0, 0, i, 0, 0)

            if pusehd:
                t += 1

                E.is_pushed()


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


plt.plot( z)
plt.show()

print(x1, x2, x3, x4)
size = len(a)
data = [x1/size, x2/size, x3/size, x4/size]
print(data)
print(len(M))