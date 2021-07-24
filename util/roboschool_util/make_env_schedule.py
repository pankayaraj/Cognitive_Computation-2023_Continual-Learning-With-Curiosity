from math import sin

import matplotlib.pyplot as plt


class linear_schedule_generator():
    def __init__(self,min_limit, max_limit, no_iteration):
        self.max_l = max_limit
        self.min_l = min_limit
        self.no_iteration = no_iteration
        self.current = min_limit

    def param_generator(self):
        for i in range(self.no_iteration):
            self.current = self.current + (self.max_l-self.min_l) /self.no_iteration
            yield self.current

    def get_param(self,iter):
        self.current = self.min_l + iter*(self.max_l - self.min_l) / self.no_iteration
        return self.current

#here no_iteration_used_for_pendulum was 400000 instead of 150000
class sine_schedule_generator():
    def __init__(self,min_limit, max_limit, factor, no_iteration):
        self.max_l = max_limit
        self.min_l = min_limit
        self.factor = factor
        self.no_iteration = no_iteration

    def param_generator(self):
        for i in range(self.no_iteration):
            yield (sin(self.factor*i)+1)*(self.max_l-self.min_l)/2 + self.min_l

    def get_param(self,iter):
        return (sin(self.factor*iter)+1)*(self.max_l-self.min_l)/2 + self.min_l
from random import randrange, randint


class spurcious_fluxuation_generator():
    def __init__(self,min_limit, max_limit,
                 #flux_points=[(32275, 0), (56602, 0), (81694, 2), (105977, 2), (118155, 0), (141158, 2)],
                 flux_points=[(32275, 0), (56602, 2), (81694, 0), (105977, 0), (118155, 2), (141158, 2)],
                 #flux_points = [[86067, 0], [150939, 2], [217851, 0], [282606, 0], [315080, 2], [376422, 2]],
                 #interval_length=12500, no_iteration=400000):
                  interval_length=5000, no_iteration=150000):

                 #interval_length=2000, no_iteration=150000):

        self.max_l = max_limit
        self.min_l = min_limit
        self.flux_points = flux_points
        self.no_iteration = no_iteration


        self.interval_length = interval_length
        self.interval_length_main = interval_length

        self.i = 0
        self.param = [ self.min_l, (self.min_l+self.max_l)/2,  self.max_l,]
        #self.param = [self.min_l,  self.max_l, (self.min_l + self.max_l) / 2,]
        #self.param = [ (self.min_l + self.max_l) / 2, self.min_l, self.max_l]

        self.current_ind = 1
        self.count = 0

    def param_generator(self):
        for iter in range(self.no_iteration):

            if self.flux_points[self.i][0] == iter:
                if self.i == 0:
                    self.interval_length = 10000
                else:
                    self.interval_length = self.interval_length_main

                self.count = 1
                self.current_ind = self.flux_points[self.i][1]
                if self.i != len(self.flux_points)-1:
                    self.i += 1

            if self.count != 0 and self.count < self.interval_length:
                self.count += 1
            elif self.count == self.interval_length:
                self.count = 0
                self.current_ind = 1

            yield  self.param[self.current_ind]

    def get_param(self, iter):
        if self.flux_points[self.i][0] == iter:

            if self.i == 0:
                self.interval_length = 2*self.interval_length_main
            else:
                self.interval_length = self.interval_length_main

            self.count = 1
            self.current_ind = self.flux_points[self.i][1]
            if self.i != len(self.flux_points) - 1:
                self.i += 1
        if self.count != 0 and self.count < self.interval_length:
            self.count += 1
        elif self.count == self.interval_length:
            self.count = 0
            self.current_ind = 1
        return self.param[self.current_ind]



"""
x = []
g = linear_schedule_generator(1,1.8,150000)
for i in g.param_generator():
    x.append(i)

plt.plot(x)
plt.ylabel("Length", size=20)
plt.xlabel("Time", size = 20)
plt.savefig("Pendulum_linear")

"""

"""

x = []
#g = sine_schedule_generator(1,1.8,0.00003,400000)
g = sine_schedule_generator(1,1.8,0.00004,150000)
j = 0
for i in g.param_generator():
    j += 1
    x.append(i)
    #if j == 140000:
    #    break

plt.plot(x)
plt.tick_params(axis='both', which='major', labelsize=15)
plt.ylabel("Length", size=20)
plt.xlabel("Time", size=20)
plt.show()
plt.savefig("Pendulum_sine_flux")
"""

"""
a = [ randrange(0, 150000) for i in range(6) ]
a.sort()
print(a)
b = []
for i in a:
    b.append((i, randint(0,1)))
print(b)
"""
"""

x = []
g = spurcious_fluxuation_generator(min_limit=1.0, max_limit=1.8)
for i in g.param_generator():
    x.append(i)

plt.plot(x)
plt.ylabel("Length", size=20)
plt.xlabel("Time", size=20)
plt.axvline(141158 + 5000, color="green")
plt.axvline(105977+5000, color="green")
plt.axvline(81694, color="green")
plt.savefig("Pendulum_spur_flux_with_lines")
"""

"""
c = [20000, 120000, 150000]
v = [1.0, 1.4, 1.8]
x = [0 for i in range(150000)]
for i in range(len(x)):
    if i <= c[0]:
        x[i] = v[0]
    elif i > c[0] and i <= c[1]:
        x[i] = v[1]
    else:
        x[i] = v[2]


plt.plot(x)
plt.ylabel("Length", size=20)
plt.xlabel("Time", size=20)
plt.savefig("Pendulum_schedule")
"""

hopper_erratic = [[86067, 0], [150939, 2], [217851, 0], [282606, 0], [315080, 2], [376422, 2]]

"""
from math import ceil

c = [[32275, 0], [56602, 2], [81694, 0], [105977, 0], [118155, 2], [141158, 2]]
for i in range(len(c)):
    c[i][0] = ceil(40*c[i][0]/15)

print(c)
"""