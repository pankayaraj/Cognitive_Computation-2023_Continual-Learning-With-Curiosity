import torch
import matplotlib.pyplot as plt


s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

fig = plt.figure()
plt.plot(s_c)
plt.savefig("forward_curiosity_linear_false")
plt.close(fig)

fig = plt.figure()
plt.plot(a_c)
plt.savefig("inverse_curiosity_linear_false")
plt.close(fig)