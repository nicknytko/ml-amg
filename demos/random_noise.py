import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.interpolate as sint

domain_x = (0,1)
domain_y = (0,1)

n = 250
S = np.random.uniform(0.0, 1.0, size=(n, 2))
Sd = 10 ** np.random.uniform(0., 4., size=n)

plt.figure()
plt.scatter(S[:,0], S[:,1], c=Sd, norm=matplotlib.colors.LogNorm())
plt.colorbar()

N = 50
x = np.linspace(domain_x[0], domain_x[1], N)
y = np.linspace(domain_y[0], domain_y[1], N)
xx, yy = np.meshgrid(x, y)

interpolator = sint.NearestNDInterpolator(S, np.log10(Sd))
dd = 10 ** interpolator(xx, yy)

plt.figure()
plt.pcolormesh(xx, yy, dd, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.show()
