import torch
import matplotlib.pyplot as plt

s_c = torch.load("s_c")
a_c = torch.load("a_c")

plt.plot(a_c)
plt.show()