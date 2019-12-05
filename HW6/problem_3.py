import numpy as np
import matplotlib.pylab as plt
from scipy.stats import norm

np.random.seed(0)
n = 200
K = 2
e = np.array([0.5, 0.5])  #mixing weight
w = np.array([1, -1]) #slopes of line
b = np.array([0.0, 0.0]) #offsets of line
v = np.array([0.5, 0.5]) #variance
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

# TODO double check v is sample variance of y data
# TODO implement log-likelihood
# TODO implement plots

prev = 0.0
curr = 1.0
itt = 0
ll = []
sig = [0.5, 0.5]
mu = [np.var(y), np.var(y)]

while abs(curr - prev) > 0.0001:
    prev = curr

    # E Step - this is where we update the weights and biases
    b[0] = np.sum(w[0] * x) / (K * n)
    b[1] = np.sum(w[1] * x) / (K * n)

    w[0] = np.sum(norm(mu[0], sig[0]).pdf(x)) * e[0]
    w[1] = np.sum(norm(mu[1], sig[1]).pdf(x)) * e[1]

    # M Step - this is where we update the hypothesis
    e = w / n
    mu = np.sum(w.dot(x)) / np.sum(w)
    sig = np.sum(w.dot((x-mu)**2))

    itt += 1
    curr = np.log(sum(w))
    ll.append(abs(curr))

print('iterations: ' + str(itt))
# plot ll to itt
print('W: ' + str(w))
print('b: ' + str(b))
# plot showing data, true lines, estimated lines

