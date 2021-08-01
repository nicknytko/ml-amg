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
import nevergrad as ng
import collections

sys.path.append(os.path.dirname(os.getcwd()))
from ns.model.interpolation import *

np.set_printoptions(linewidth=300)

#cuda = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cuda = 'cpu'
print(f'Training on device "{cuda}"')

N_s = 100
grid_dim = (15,15)

def R_jacobi_sp(A, omega=0.666, nu=5):
    D = np.array(A.diagonal())
    Dinv = 1.0 / D
    J_ep = sp.eye(A.shape[0]) - omega * sp.spdiags(Dinv, [0], m=A.shape[0], n=A.shape[1]) @ A

    J_epn = sp.eye(A.shape[0])
    for i in range(nu):
        J_epn = J_epn @ J_ep

    return J_epn

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
    R = R_jacobi_sp(A)

    sample = {
        'A': A,
        'A_t': A_t.to(cuda),
        'G': G.to(cuda),
        'R': R,
        'c_rs': pyamg.classical.RS(A)
    }
    samples.append(sample)

GNN = ContinuousInterpolationFullNetwork(cuda).to(cuda)
#GNN.CF.load_state_dict(torch.load('gnn_model'))

def torch_to_ng_param(d):
    ng_dict = ng.p.Dict()
    for k, v in d.items():
        ng_dict[k] = ng.p.Array(init=v.numpy())
        #ng_dict[k] = ng.p.Array(shape=v.shape)
    return ng_dict

ngp = torch_to_ng_param(GNN.CF.state_dict())

def ng_to_torch_state(d):
    od = collections.OrderedDict()
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            od[k] = torch.tensor(v)
        else:
            od[k] = torch.Tensor(v.value)
    return od

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

iteration = 0
def plot_predictions():
    global iteration
    c = GNN.CF(samples[0]['G']).cpu().detach().numpy()
    C_n = c.reshape(grid_dim)
    print(iteration, (C_n >= 0.5).astype(np.int64))
    display_grid_2d(C_n)
    plt.title(f'C/F Predictions for Diffusion Problem')
    plt.pause(0.1)
    plt.savefig(f'frames/prediction_{iteration:04d}.png', dpi=200)
    iteration += 1

# Training CF Net

def E_loss_discrete_nograd(A, c, R):
    if la.norm(c,ord=1) == 0:
        return 30

    P = pyamg.classical.direct_interpolation(A, A, c)
    I = sp.eye(A.shape[0])
    G = I - P @ spla.spsolve(P.T @ A @ P, P.T @ A)
    E = R @ G @ R
    rho = np.real(spla.eigs(E, k=1, return_eigenvectors=False)).item()
    return rho**2 + (la.norm(c, ord=1) * 0.1)

def training_loop(x):
    state = ng_to_torch_state(x)
    GNN.CF.load_state_dict(state)

    loss = 0
    sample_idx = np.arange(N_s)
    np.random.shuffle(sample_idx)

    for j in sample_idx[:N_s//2]:
        sample = samples[j]
        c = GNN.CF(sample['G']).cpu().detach().numpy()
        c = (c >= 0.5).astype(np.int32)
        loss += E_loss_discrete_nograd(sample['A'], c, sample['R'])
    print(loss)
    plot_predictions()

    return loss

print('Training CF Net')
plt.figure(figsize=(8,8))
plt.ion()
plt.show(block=False)
display_grid_2d(np.zeros(grid_dim))

optimizer = ng.optimizers.NGOpt4(parametrization=ngp, budget=1500)
weights = optimizer.minimize(training_loop)
torch.save(GNN.CF.state_dict(), 'gnn_model')

print('done')
plt.show(block=True)
