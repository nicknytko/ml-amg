import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.interpolate as sint

domain_x = (0,1)
domain_y = (0,1)

n = 250
S = np.random.uniform(0.0, 1.0, size=(2, n))

theta = np.random.uniform(0, 2*np.pi)
xs = np.random.uniform(0.1, 10)
ys = np.random.uniform(0.1, 10)
R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
D = np.diag([xs,ys])
b = np.random.uniform(-10, 10, size=(2,))

D_S = R.T@(D@R@S+b.reshape((2,1)))
Sd = (np.cos(D_S[0]) ** 2 + np.cos(D_S[1]) ** 2)*1.5 + 0.2

plt.figure()
plt.scatter(S[0], S[1], c=10.**Sd, norm=matplotlib.colors.LogNorm())
plt.colorbar()

N = 50
x = np.linspace(domain_x[0], domain_x[1], N)
y = np.linspace(domain_y[0], domain_y[1], N)
xx, yy = np.meshgrid(x, y)

interpolator = sint.NearestNDInterpolator(S.T, Sd)
dd = interpolator(xx, yy)

plt.figure()
plt.pcolormesh(xx, yy, 10.**dd, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.show()
