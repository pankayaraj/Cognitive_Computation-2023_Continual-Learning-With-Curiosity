import numpy as np
import torch
from custom_envs.custom_pendulum import PendulumEnv


e = PendulumEnv()
s = e.reset()
for i in range(100):

    a = e.action_space.sample()
    e.set_gravity(1.)
    n_s1, r1, d1, _ = e.step(a)
    e.set_gravity(1.2)
    n_s, r, d, _ = e.step(a)
    e.set_gravity(1.4)
    n_s2, r2, d2, _ = e.step(a)
    print(np.sum(n_s-n_s1), np.sum(n_s2-n_s1), n_s-n_s1, n_s2-n_s1)
    #print(n_s, n_s2, r, r2, s)

    s = n_s1