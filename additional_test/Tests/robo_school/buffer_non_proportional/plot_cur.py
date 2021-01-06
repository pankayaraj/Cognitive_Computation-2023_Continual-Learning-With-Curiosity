import torch
import matplotlib.pyplot as plt
import numpy as np

s_c = torch.load("forward_curiosity4")
a_c = torch.load("inverse_curiosity4")
#r_c =""" torch.load("reward_curiosity")
""""""
#a_c = s_c
print(len(a_c))

a =[]
for j in range(len(a_c[0])):
    c = 0
    c_ = []

    for i in range(len(a_c)):

        c += a_c[i][j]

        c_.append(a_c[i][j])
    std = np.std(c_)
    a.append(c/len(a_c))

"""
fig = plt.figure()
plt.plot(s_c)
plt.savefig("forward_curiosity_linear_false")
plt.close(fig)
"""

fig = plt.figure()
plt.plot(a)
plt.savefig("inverse_curiosity_linear_false")
plt.close(fig)
"""
fig = plt.figure()
plt.plot(a_c)
plt.savefig("inverse_curiosity_linear_false")
plt.close(fig)

"""
"""
fig = plt.figure()
plt.plot(r_c)
plt.savefig("reward_curiosity_linear_false")
plt.close(fig)
"""