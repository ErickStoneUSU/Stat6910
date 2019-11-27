import numpy as np
import matplotlib.pylab as plt

np.random.seed(0)
n = 200
K = 2
e = np.array([0.7, 0.3])
w = np.array([-2, 1])
b = np.array([0.5, 0.5])
v = np.array([0.2, 0.1])
x = np.zeros([n])
y = np.zeros([n])
for i in range(0, n):
    x[i] = np.random.rand(1)
    if np.random.rand(1) < e[0]:
        y[i] = w[0] * x[i] + b[0] + np.random.rand(1) * np.sqrt(v[0])
    else:
        y[i] = w[1] * x[i] + b[1] + np.random.rand(1) * np.sqrt(v[1])
plt.plot(x, y, 'bo')
t = np.linspace(0, 1, num=100)
plt.plot(t, w[0]*t+b[0], 'k')
plt.plot(t, w[1]*t+b[1], 'k')
plt.show()
# TODO implement init variables
# TODO implement log-likelihood
# TODO implement plots

prev = 0.0
curr = 1.0
while abs(prev - curr) > 0.0:
    # E Step
    print('a')
    # M Step
    print('b')