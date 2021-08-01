import numpy as np
import pyamg
import torch
import torch.optim as optim
import os
import sys
import traceback
import matplotlib.pyplot as plt
import matplotlib.path
import random

sys.path.append(os.path.dirname(os.getcwd()))
from ns.model.interpolation import *

grid_dim = (15,15)

epsilon = 3
theta_deg = 45

S = pyamg.gallery.diffusion_stencil_2d(epsilon=epsilon, theta=(theta_deg * (np.pi / 180)))
A = pyamg.gallery.stencil_grid(S, grid=grid_dim, format='csr')

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
            plt.plot(xx[xi, yi], yy[xi, yi], ms=15, marker=path, markerfacecolor="None", markeredgecolor=(c, 0., 1.0-c), markeredgewidth=2)

G_RS = pyamg.classical.CR(A).reshape(grid_dim)

plt.figure(figsize=(7,7))
plt.ion()
display_grid_2d(G_RS)
plt.title('RS AMG C/F for Diffusion Problem')
plt.show(block=True)
