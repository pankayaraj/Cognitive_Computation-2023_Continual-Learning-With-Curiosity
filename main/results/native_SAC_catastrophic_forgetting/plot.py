import torch
import matplotlib.pyplot as plt
from pathlib import Path


load_dir = "results_length__s_i_2000_1"
a_c = torch.load("inverse_curiosity4")
r_c = torch.load("reward_curiosity4")

plt.plot(a_c[0])
plt.show()
a_c_1 = torch.load("inverse_curiosity4")
a_c_2 = torch.load("reward_curiosity4")
a_c = [[]]
for i in range(len(a_c_1[0])):
    a_c[0].append(a_c_1[0][i] + a_c_2[0][i]*1)
print(len(a_c))
plt.plot(a_c[0])
plt.show()
plt.plot(r_c[0])
plt.show()

rewards = torch.load(load_dir)

print(len(rewards[0]))
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

for i in range(len(rewards)):
    plt.plot(rewards[i], linewidth=3)
plt.show()