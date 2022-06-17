import numpy as np
import sys

# Test of the genetic algorithm
# Training a neural network to fit a noisy quadratic regression

import torch

sys.path.append('../')
import ns.ga.parga
import ns.ga.torch
import ns.parallel.pool

# Random datapoints to fit
np.random.seed(0)
N = 200
x = np.linspace(-1, 1, N)
y = x ** 2 + np.random.randn(N)*0.01

# Model we're using
H = 20
model = torch.nn.Sequential(
    torch.nn.Linear(1,H), torch.nn.Softplus(),
    torch.nn.Linear(H,H), torch.nn.Softplus(),
    torch.nn.Linear(H,H), torch.nn.Softplus(),
    torch.nn.Linear(H,1)
)

# Optimizer settings
N_workers = 4
N_population = 1000
mut_perturb = 1e-1
mut_prob = 0.1
stop_loss = 1e-3

def fitness(generation, weights, idx):
    '''
    Fitness function for the optimizer.  This computes the overall loss
    when given a set of network weights.

    Parameters
    ----------
    generation : integer
      The generation number we are on
    weights : numpy.ndarray
      Vector containing flattened weights for this individual
    idx : integer
      Index that identifies this individual in the population

    Returns
    -------
    fitness : float
      How "fit" this individual is.  Higher is better.
    '''

    # Load population weights into the model
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weights))
    model.eval()

    # Evaluate fitness (which is inverse to loss -- higher is better)
    y_eval = model(torch.Tensor(x).reshape((-1, 1))).detach().numpy().flatten()
    return 1. / (np.sum((y - y_eval)**2) / N) # MSE


with ns.parallel.pool.WorkerPool(N_workers) as pool:
    import matplotlib.pyplot as plt

    # Create our population of *N_population* individuals that we will use
    population = ns.ga.torch.TorchGA(model, N_population)

    # Optimizer setup
    ga = ns.ga.parga.ParallelGA(worker_pool=pool,
                                initial_population=population.population_weights,
                                model_folds=population.folds,
                                fitness_func=fitness,
                                mutation_probability=mut_prob,
                                mutation_min_perturb=-mut_perturb,
                                mutation_max_perturb=mut_perturb,
                                steady_state_top_use=1./2.,
                                steady_state_bottom_discard=1./2)

    # Optimizer loop
    print(ga.num_generation, 1./ga.best_solution()[1])
    loss_vals = [1./ga.best_solution()[1]]
    while True:
        ga.iteration()

        loss = 1. / ga.best_solution()[1]
        loss_vals.append(loss)
        print(ga.num_generation, loss)
        if loss < stop_loss:
             break

    # Evaluate and show results
    weight = ga.best_solution()[0]
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weight))
    model.eval()
    y_eval = model(torch.Tensor(x).reshape((-1, 1))).detach().numpy().flatten()

    res = y - y_eval
    var = y - np.average(y)
    det = 1 - (res@res)/(var@var)

    plt.figure()
    plt.plot(x, y, label='Noisy data')
    plt.plot(x, y_eval, 'o-', markersize=1.5, label=f'NN fit, $r^2$={det:.4f}')
    plt.title('Quadratic Regression')
    plt.legend()

    plt.figure()
    plt.semilogy(loss_vals)
    plt.title('Loss History')
    plt.ylabel('MSE')
    plt.xlabel('Iteration')
    plt.grid()
    plt.show(block=True)
