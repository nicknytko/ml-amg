import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle
import bz2
import sys
import scipy.sparse as sp
import scipy.cluster.vq as vq
sys.path.append('../')

import ns.model.data
import ns.lib.multigrid
import ns.lib.graph
import ns.optimize.spsa
import ns.ga.parga
import ns.parallel.pool
import pyamg
import torch
import abc
import ns.lib.aggplot

G = ns.model.data.Grid.structured_2d_poisson_dirichlet(16, 16)

A = G.A
n = A.shape[0]

class NumpyModule(abc.ABC):
    @abc.abstractmethod
    def get_weights(self):
        return np.zeros(self.get_weights_len())

    @abc.abstractmethod
    def get_weights_len(self):
        return 0

    @abc.abstractmethod
    def set_weights(self, w):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

def ReLU(x):
    return np.maximum(x, 0)

def ELU(x, alpha=1.0):
    if len(x.shape) > 1:
        x = np.squeeze(x)
    z = x.copy()
    xneg = x<=0
    z[xneg] = alpha * (np.exp(x[xneg]) - 1)
    return z

class MatConv(NumpyModule):
    def __init__(self, in_dim, out_dim, K=3):
        self.weights = np.random.randn(K, in_dim, out_dim)
        self.bias = np.random.randn(out_dim)
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, A, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, axis=1)

        out = np.zeros((A.shape[0], self.out_dim))
        for i in range(self.K):
            x = A @ x
            out += x@self.weights[i]
        out += self.bias
        return out

    def get_weights(self):
        w = np.zeros(self.get_weights_len())
        w[:-self.out_dim] = self.weights.reshape((-1,))
        w[-self.out_dim:] = self.bias
        return w

    def get_weights_len(self):
        return self.K * self.in_dim * self.out_dim + self.out_dim

    def set_weights(self, w):
        self.weights[:] = w[:-self.out_dim].reshape((self.K, self.in_dim, self.out_dim))
        self.bias[:] = w[-self.out_dim:]

class MatConvSequential(NumpyModule):
    def __init__(self, in_dim, hid_dim, out_dim, K=3):
        self.conv1 = MatConv(in_dim, hid_dim, K)
        self.conv2 = MatConv(hid_dim, hid_dim, K)
        self.conv3 = MatConv(hid_dim, out_dim, K)

    def forward(self, A, x):
        x = ReLU(self.conv1(A, x))
        x = ReLU(self.conv2(A, x))
        x = self.conv3(A, x)
        return x

    def get_weights_len(self):
        wl1 = self.conv1.get_weights_len()
        wl2 = self.conv2.get_weights_len()
        wl3 = self.conv3.get_weights_len()
        return wl1 + wl2 + wl3

    def get_weights(self):
        w = np.zeros(self.get_weights_len())
        wl1 = self.conv1.get_weights_len()
        wl2 = self.conv2.get_weights_len()

        w[:wl1] = self.conv1.get_weights()
        w[wl1:wl1+wl2] = self.conv2.get_weights()
        w[wl1+wl2:] = self.conv3.get_weights()

        return w

    def set_weights(self, w):
        wl1 = self.conv1.get_weights_len()
        wl2 = self.conv2.get_weights_len()

        self.conv1.set_weights(w[:wl1])
        self.conv2.set_weights(w[wl1:wl1+wl2])
        self.conv3.set_weights(w[wl1+wl2:])

class MatConvAggNet(NumpyModule):
    def __init__(self, dim=64):
        self.PNet = MatConvSequential(1, dim, 1, K=3)

    def get_weights_len(self):
        return self.PNet.get_weights_len()

    def get_weights(self):
        return self.PNet.get_weights()

    def set_weights(self, w):
        self.PNet.set_weights(w)

    def forward(self, A, seed, alpha):
        x = ReLU(self.PNet(A, seed)).reshape((-1,))
        n = A.shape[0]
        k = int(np.ceil(n * alpha))

        top_k = np.argsort(x)[::-1][:k]
        distance, nearest_center = pyamg.graph.bellman_ford(abs(A), top_k)
        un, cluster = np.unique(nearest_center, return_inverse=True)
        AggOp = sp.csr_matrix((np.ones(n), (np.arange(n), cluster)), shape=(n, k))

        return x, top_k, AggOp, ns.lib.multigrid.smoothed_aggregation_jacobi(A, AggOp)


model = MatConvAggNet()

def loss(it, w, idx):
    model.set_weights(w)
    #x = np.random.RandomState(it).uniform(0, 1, n)
    x = np.ones(n)/n
    b = np.zeros(n)
    node_sc, top_k, AggOp, P = model(A, x, 0.1)
    x = np.random.RandomState(0).randn(n)
    x /= la.norm(x, 2)
    return 1./ns.lib.multigrid.amg_2_v(A, P, b, x, pre_smoothing_steps=2, post_smoothing_steps=2, error_tol=1e-8)[1]


with ns.parallel.pool.WorkerPool(4) as pool:
    plt = ns.lib.aggplot.ThreadedPlot()

    initial_pop = np.tile(model.get_weights(), (200, 1))
    mut_prob = 0.5
    mut_perturb = 0.05
    ga = ns.ga.parga.ParallelGA(initial_population=initial_pop,
                                worker_pool=pool,
                                fitness_func=loss,
                                crossover_probability=0.0,
                                mutation_probability=mut_prob,
                                mutation_min_perturb=-mut_perturb,
                                mutation_max_perturb=mut_perturb,
                                steady_state_top_use=1./2.,
                                steady_state_bottom_discard=1./2.,)
    i=0
    while True:
        ga.iteration()
        if i % 10 == 0:
            sol = ga.best_solution()
            print(i, 1./sol[1])
            w = sol[0]
            model.set_weights(w)
            node_sc, top_k, AggOp, P = model(A, np.ones(n)/n, 0.1)
            #print(node_sc)
            plt.clear()
            plt.plot_agg_2d(G, aggregation=AggOp, cluster_centers=top_k, node_values=node_sc)
            plt.set_title(f'Iter {i}, conv {1./sol[1]:0.4f}')
        i += 1
