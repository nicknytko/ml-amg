import numpy as np
import numpy.linalg as la
import matplotlib
import matplotlib.pyplot as plt
import pickle
from scipy.interpolate import interp1d

with open('lcurve.pkl', 'rb') as f:
    lcurve = pickle.load(f)

alphas = np.array(sorted(list(lcurve.keys())))
e_loss = np.array([lcurve[a]['E_loss'] for a in alphas])
c = np.array([lcurve[a]['C'] for a in alphas])
c_nrms = np.array([la.norm(lcurve[a]['C'].flatten(), ord=1) for a in alphas])

not_same = []

for i in range(len(e_loss)):
    if e_loss[i] > 1.:
        e_loss[i] = 1.

not_same = np.array(not_same)

for i in range(len(alphas)):
    print(f'${alphas[i]:.3e}$ & ${e_loss[i]:.3e}$ & ${int(c_nrms[i])}$ & ${np.sum(1.0 - c[i]) / len(c[i].flatten()):.3f}$ \\\\')

# plt.figure(figsize=(8,6))
# arst = np.arange(len(e_loss))
# plt.plot(e_loss[arst], c_nrms[arst], 'o-', label='raw data')
# plt.ylabel(r'$\|\| c \|\|_1$')
# plt.xlabel(r'$\rho(E)$')
# plt.title('"L-Curve" of discrete training loss')

def get_marker(c):
    N = 50 # number of verts

    r = np.linspace(0, np.pi * 2, N)
    xy_2 = np.array([np.cos(r), np.sin(r)])
    xy_inf = xy_2 / la.norm(xy_2, ord=np.inf, axis=0)
    xy = c * xy_inf + (1.0 - c) * xy_2

    return xy.T

def display_grid_2d(G):
    n, m = G.shape
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, n)

    xx, yy = np.meshgrid(x, y)

    C = np.where(G >= 0.5)
    F = np.where(G < 0.5)

    plt.clf()

    for xi in range(m):
        for yi in range(n):
            c = G[xi, yi]
            verts = get_marker(c)
            path = matplotlib.path.Path(verts, closed=True, readonly=True)
            if c >= 0.5:
                col = (1., 0., 0.)
            else:
                col = (0., 0., 1.)
            plt.plot(xx[xi, yi], yy[xi, yi], ms=15, marker=path, markerfacecolor="None", markeredgecolor=col, markeredgewidth=2)

for i in range(len(alphas)):
    if int(c_nrms[i]) != 0 and int(c_nrms[i]) != 225:
        plt.figure(figsize=(7,7))
        display_grid_2d(c[i])
        plt.title(f'C/F partitioning for alpha={alphas[i]:.3e}')
        plt.savefig(f'cf_{i}.pdf')
#plt.show()

# x = np.logspace(np.log10(np.min(e_loss)), np.log10(np.max(e_loss)), 200)
# f = interp1d(e_loss[arst], c_nrms[arst], 'cubic')
# plt.loglog(x, f(x), '-', label='cubic poly')

#plt.legend()
# plt.xscale('symlog')
# plt.yscale('symlog')
# plt.savefig('l.svg')
# plt.show()
