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

np.set_printoptions(linewidth=300)

#cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = 'cpu'
print(f'Training on device "{cuda}"')

N_s = 50
grid_dim = (15,15)

samples = []
for i in range(N_s):
    epsilon = random.uniform(1,5)
    theta_deg = random.uniform(0,180)
    if i == 0:
        epsilon = 3.0
        theta_deg = 45
        print('epsilon', epsilon, 'theta', theta_deg)
    S = pyamg.gallery.diffusion_stencil_2d(epsilon=epsilon, theta=(theta_deg * (np.pi / 180)))
    A = pyamg.gallery.stencil_grid(S, grid=grid_dim, format='csr')

    A_t = torch.tensor(A.todense()).float()
    G = mat_to_graph(A)
    R = R_jacobi(A)

    sample = {
        'A': A,
        'A_t': A_t.to(cuda),
        'G': G.to(cuda),
        'R': R.to(cuda),
        'c_rs': pyamg.classical.RS(A)
    }
    samples.append(sample)

GNN = ContinuousInterpolationFullNetwork(cuda).to(cuda)

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

# Full training loop

print('Training full net')

optimizer = optim.Adam(GNN.parameters(), lr=0.01)
num_epochs = 20

batch_size = 10

for i in range(num_epochs):
    ordering = np.arange(N_s)
    np.random.shuffle(ordering)

    for k in range(0, N_s, batch_size):
        optimizer.zero_grad()
        loss = 0
        for j_i in range(k, min(N_s, k + batch_size)):
            j = ordering[j_i]
            sample = samples[j]
            P, Phat, C = GNN(sample['G'])
            loss += EC_loss(sample['A_t'], Phat, C, sample['R'])
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        P, Phat, C = GNN(samples[0]['G'])
        C_n = C.cpu().numpy().diagonal().reshape(grid_dim)
        print((C_n >= 0.5).astype(np.int64))
        display_grid_2d(C_n)
        plt.title(f'C/F Predictions for Diffusion Problem (training epoch {i+1})')
        plt.pause(0.1)
        plt.savefig(f'frames/prediction_{i:04d}.png', dpi=200)

    print(f'{i}/{num_epochs}', loss.item() / N_s)

plt.figure(figsize=(8,8))
plt.ion()
plt.show(block=False)
with torch.no_grad():
    P, Phat, C = GNN(samples[0]['G'])
    C_n = C.cpu().numpy().diagonal().reshape(grid_dim)
    print((C_n >= 0.5).astype(np.int64))
    display_grid_2d(C_n)
    plt.title(f'C/F Predictions for Diffusion Problem (training epoch {i+1})')
    plt.pause(0.1)
    plt.savefig(f'frames/prediction_{i:04d}.png', dpi=200)

plt.show(block=True)
