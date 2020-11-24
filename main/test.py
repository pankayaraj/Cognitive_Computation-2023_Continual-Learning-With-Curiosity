import numpy
import torch



from custom_envs.custom_pendulum import PendulumEnv
e = PendulumEnv()
s = e.reset()
for i in range(100):
    a = e.action_space.sample()
    e.set_gravity(10.0)
    n_s, r, d, _ = e.step(a)
    e.set_gravity(100.0)
    n_s2, r2, d2, _ = e.step(a)

    print(n_s, n_s2, r, r2, s)

    s = n_s