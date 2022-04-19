import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors
import scipy.spatial as spat

domain_x = (0,1)
domain_y = (0,1)

Ns = np.random.randint(2,4) # number of seedpts

# Random seeds, diffusion coeff
S = np.column_stack((
    np.random.uniform(domain_x[0], domain_x[1], size=Ns),
    np.random.uniform(domain_y[0], domain_y[1], size=Ns)
))

while True:
    D = 10 ** np.random.uniform(-4, 4, size=Ns)
    if np.ptp(D) > 1e3:
        # Force us to have a wide range of diff coeff
        break

# ez Voronoi visualization -- make a bunch of points and assign them to nearest seed
N = 200
x = np.linspace(domain_x[0], domain_x[1], N)
y = np.linspace(domain_y[0], domain_y[1], N)
xx, yy = np.meshgrid(x, y)

# Form Xr as (numpts, numseeds, 2) tensor
#  Subtract Xr-S, then contract norm over mode 2: (numpts, numseeds, 2) -> (numpts, numseeds)
#  take argmin over mode 1:  (numpts, numseeds) -> (numpts,)
#  to assign each point to nearest seed.

X = np.column_stack((xx.flatten(), yy.flatten())).reshape((1, -1, 2))
Xr = np.transpose(np.repeat(X, Ns, axis=0), axes=(1,0,2))
assignment = np.argmin(la.norm(Xr-S, axis=2), axis=1)
dd = D[assignment].reshape(xx.shape)

plt.pcolormesh(xx, yy, dd, norm=matplotlib.colors.LogNorm())
plt.colorbar()
plt.plot(S[:,0], S[:,1], 'o')
plt.show()
