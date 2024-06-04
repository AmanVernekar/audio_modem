from ldpc_jossy.py import ldpc
import numpy as np
import matplotlib.pyplot as plt

k = 648
n = 2*k
# llr = 0.01
c = ldpc.code('802.16', '1/2', 54)

u = np.random.randint(0, 2, k)
x = c.encode(u)
x = 2 * x - 1
y = x + np.random.normal(0, 0.05, (n))

z = np.zeros(n)
llrs = np.linspace(0.1, 7, 100)
errors = np.zeros(len(llrs))

for p, llr in enumerate(llrs):
    z[y>=0] = llr
    z[y<0] = -llr

    app,it = c.decode(z)
    app[app>=0] = 1
    app[app<0] = -1
    errors[p] = np.count_nonzero(app[:k]-x[:k])*100/k

plt.plot(llrs, errors)
plt.show()
