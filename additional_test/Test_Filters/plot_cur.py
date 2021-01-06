import torch
import matplotlib.pyplot as plt


s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")
#r_c = torch.load("reward_curiosity")

c = []
for i in range(len(a_c)):
    c.append(a_c[i] + 0.5*s_c[i])

plt.plot(c)
plt.show()

fig = plt.figure()
plt.plot(s_c)
plt.savefig("forward_curiosity_linear_false")
plt.close(fig)

fig = plt.figure()
plt.plot(a_c)
plt.savefig("inverse_curiosity_linear_false")
plt.close(fig)