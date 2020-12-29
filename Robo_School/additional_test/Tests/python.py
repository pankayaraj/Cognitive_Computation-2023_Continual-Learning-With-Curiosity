import torch
import heapq
import matplotlib.pyplot as plt
import numpy as np

mem_size = 1000
cur = True
prop = False
custom = True
if prop:
    p = "proportional"
elif  custom:
    p = "custom_non_proportional_2"
else:
    p = "non_proportional"

if cur == True:
    M = torch.load(p + "/replay_m_cur_res_" + str(mem_size) + "/replay_mem")
else:

    M = torch.load(p + "/replay_m_res_" + str(mem_size) + "/replay_mem")
#M.reservior_buffer.storage
M = torch.load("replay_memn_c_t2")

x1 = 0
x2 = 0
x3 = 0
x4 = 0
a = M.reservior_buffer.storage

#a = M.storage


for i in range(len(a)):
    if a[i][1] >= 0 and a[i][1] < 30000:
        x1 += 1
    elif a[i][1] >= 30000 and a[i][1] < 60000:
        x2 += 1
    elif a[i][1] >= 60000 and a[i][1] < 120000:
        x3 += 1
    elif a[i][1] >= 120000 and a[i][1] < 200000:
        x4 += 1

"""        
for i in range(len(a)):
    if a[i][2][-1] >= 0 and a[i][2][-1] < 30000:
        x1 += 1
    elif a[i][2][-1] >= 30000 and a[i][2][-1] < 60000:
        x2 += 1
    elif a[i][2][-1] >= 60000 and a[i][2][-1] < 120000:
        x3 += 1
    elif a[i][2][-1] >= 120000 and a[i][2][-1] < 200000:
        x4 += 1
"""
"""
    elif a[i][1] >= 60000 and a[i][1] < 90000:
        x3 += 1
"""

print(x1, x2, x3, x4)
size = len(a)
data = [x1/size, x2/size, x3/size, x4/size]
legend = ["l = 1.0", "l = 1.2", "l = 1.4", "l = 1.6"]
#data = [x1/size, x2/size, x4/size]
#legend = ["l = 1.0", "l = 1.2", "l = 1.4",]
i = 0
while True:
    if i == len(data):
        break
    if data[i] == 0:
        data.pop(i)
        legend.pop(i)
    else:
        i += 1

print(data, legend)

fig, ax = plt.subplots(figsize=(8, 3), subplot_kw=dict(aspect="equal"))
wedges, texts = ax.pie(data, wedgeprops=dict(width=0.5), startangle=-40)

bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
kw = dict(arrowprops=dict(arrowstyle="-"),
          bbox=bbox_props, zorder=0, va="center")

for i, p in enumerate(wedges):
    ang = (p.theta2 - p.theta1)/2. + p.theta1
    y = np.sin(np.deg2rad(ang))
    x = np.cos(np.deg2rad(ang))
    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
    connectionstyle = "angle,angleA=0,angleB={}".format(ang)
    kw["arrowprops"].update({"connectionstyle": connectionstyle})
    ax.annotate(legend[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                horizontalalignment=horizontalalignment, **kw)

ax.set_title("Data Propotionality")

if cur == True:
    plt.savefig("Data_in_reservior_" + "with_cur_m_" + str(mem_size) )
else:
    plt.savefig("Data_in_reservior_" +  str(mem_size))

plt.close()