import random
import matplotlib.pyplot as plt
import torch
from util.reservoir_w_cur_replay_buffer import Reservoir_with_Cur_Replay_Memory
import numpy as np
s_c = torch.load("forward_curiosity")
a_c = torch.load("inverse_curiosity")

mul = 1000
change_var_at = [0, 100, 150, 350]
change_var_at = [change_var_at[i]*mul for i in range(len(change_var_at))]

M = Reservoir_with_Cur_Replay_Memory(50000)



measure = 1.0
measure_delta =  58e-6

switch = False
temp_t = 0

cur_frame = 1000

time_frame = 400
t_count = 0

time = []
cur = []

last_cur = 0
last_push = 0

avg_cur = 0
avg_t = 0
t = 0
z = []
y = []

a = 0
for c in s_c:
    t += 1

    if cur_frame > t:

        pushed = M.push(0, 0, 0, 0, c, 0, 0)
        avg_cur += c / cur_frame
        cur.append(c)
        if time_frame > t:
            if pushed:
                avg_t += 1.0 / time_frame
                time.append(1.0)
            else:
                if pushed:
                    avg_t += 1.0 / time_frame
                    time.append(1.0)
                else:
                    avg_t += 0.0 / time_frame
                    time.append(0.0)
    elif 165*avg_cur < c:


        temp_t = 0
        measure = 1.0

        pushed = M.push(0, 0, 0, 0, c, 0, 0)
        avg_cur += c / cur_frame
        avg_cur -= cur.pop(0)/cur_frame
        cur.append(c)

        if pushed:
            avg_t += 1.0 / time_frame
            avg_t -= time.pop(0)/time_frame
            time.append(1.0)
        else:
            avg_t += 0.0 / time_frame
            avg_t -= time.pop(0) / time_frame
            time.append(0.0)

    elif temp_t < time_frame:

        measure -= measure_delta
        temp_t += 1

        pushed = M.push(0, 0, 0, 0, c, 0, 0)
        avg_cur += c / cur_frame
        avg_cur -= cur.pop(0) / cur_frame
        cur.append(c)

        if pushed:
            avg_t += 1.0 / time_frame
            avg_t -= time.pop(0)/time_frame
            time.append(1.0)
        else:
            avg_t += 0.0 / time_frame
            avg_t -= time.pop(0) / time_frame
            time.append(0.0)

    elif measure > 0.6:

        if avg_t > 0.8:
            measure -= measure_delta
            pushed = M.push(0, 0, 0, 0, c, 0, 0)

            avg_cur += c / cur_frame
            avg_cur -= cur.pop(0) / cur_frame
            cur.append(c)

            if pushed:
                avg_t += 1.0 / time_frame
                avg_t -= time.pop(0)/time_frame
                time.append(1.0)
            else:
                avg_t += 0.0 / time_frame
                avg_t -= time.pop(0) / time_frame
                time.append(0.0)

        else:
            measure = 1.0
            pushed = M.not_pushed()
    else:
        pushed = M.not_pushed()


    if pushed:
        z.append(1)
    else:
        z.append(0)
    #y.append(avg_t)
    y.append(measure)


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


plt.plot(y)
plt.show()

print(x1, x2, x3, x4)
size = len(a)
data = [x1/size, x2/size, x3/size, x4/size]
print(data)
print(len(M))

