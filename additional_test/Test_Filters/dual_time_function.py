import numpy as np
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import random

s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

delta = 1e-15
change_variable_at = [0, 100000, 150000, 350000]

def time_fun(x, slope, shift):
    return 1/(1+np.exp((slope*x - shift)))

class eligibility_trace():

    def __init__(self, lambda_v, r, slope=3, shif=5):

        self.E = 0
        self.lambda_v = lambda_v
        self.r = r
        self.slope = slope
        self.shift = shif

    def get_trace(self):
        return time_fun(self.E, slope=self.slope, shift=self.shift)

    def get_trace_unscaled(self):
        return self.E

    def general_iterate(self):
        self.E = self.r*self.lambda_v*self.E

    def increment(self):
        self.E += 1


t_trace = eligibility_trace(lambda_v=0.99, r=1)
p_trace = eligibility_trace(lambda_v=0.99, r=1)

M = Reservoir_with_Cur_Replay_Memory(50000)

t = 0
for i in a_c:
    if t == 20000:
        break
    t +=1
    thres = (t_trace.get_trace() + delta)/(p_trace.get_trace() + delta)
    thres = time_fun(thres, 0.001, 5)
    print(t, t_trace.get_trace(), p_trace.get_trace(), thres)

    if thres <  p_trace.get_trace() :
        pusehd = M.push(0, 0, 0, 0, i, 0, 0)
        if pusehd:
            p_trace.increment()
        else:
            t_trace.increment()
    else:
        print("a")
        t_trace.increment()




x1 = 0
x2 = 0
x3 = 0
x4 = 0
a = M.storage

for i in range(len(a)):
    if a[i][1] >= change_variable_at[0] and a[i][1] < change_variable_at[1]:
        x1 += 1
    elif a[i][1] >= change_variable_at[1] and a[i][1] < change_variable_at[2]:
        x2 += 1
    elif a[i][1] >= change_variable_at[2] and a[i][1] < change_variable_at[3]:
        x3 += 1
    elif a[i][1] >= change_variable_at[3] :
        x4 += 1


print(x1, x2, x3, x4)
size = len(a)

data = [x1/size, x2/size, x3/size, x4/size]
print(data)