import matplotlib.pyplot as plt
import numpy as np

def time_fun(x, slope, shift):
    return 1/(1+np.exp((slope*x - shift)))


shift = 10
slope = 0.001
z = []
x = [i for i in range(50000)]
y = [time_fun(i, slope, shift ) for i in x]
plt.plot(x,y)
plt.show()