import numpy as np
import sys

# Test of the genetic algorithm
# Finding a regression for a noisy quadratic function

import torch

sys.path.append('../')
import ns.ga.parga
import ns.ga.torch

np.random.seed(0)
N = 200
batch_size = 32
#batch_size = N
x = np.linspace(-1, 1, N)
y = x ** 2 + np.random.randn(N)*0.01

N_population = 100

H = 20
model = torch.nn.Sequential(
    torch.nn.Linear(1,H), torch.nn.Softplus(),
    torch.nn.Linear(H,H), torch.nn.Softplus(),
    torch.nn.Linear(H,H), torch.nn.Softplus(),
    torch.nn.Linear(H,1)
)

print(model.state_dict().keys())
population = ns.ga.torch.TorchGA(model, N_population)

def fitness(gen, weight, idx):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weight))
    model.eval()

    r = np.random.RandomState(gen)
    batch_idx = r.choice(N, size=batch_size, replace=False)

    y_eval = model(torch.Tensor(x[batch_idx]).reshape((-1, 1))).detach().numpy().flatten()
    return 1. / (np.sum((y[batch_idx] - y_eval)**2)/batch_size)


def overall_fitness(weight):
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weight))
    model.eval()

    y_eval = model(torch.Tensor(x).reshape((-1, 1))).detach().numpy().flatten()
    return 1. / (np.sum((y - y_eval)**2)/N)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    num_workers = 4
    perturb = 1e-2
    ga = ns.ga.parga.ParallelGA(num_workers=num_workers,
                                initial_population=population.population_weights,
                                model_folds=population.folds,
                                fitness_func=fitness,
                                mutation_probability=0.5,
                                mutation_min_perturb=-perturb,
                                mutation_max_perturb=perturb,
                                steady_state_top_use=3./4.,
                                steady_state_bottom_discard=1./4)
    ga.start_workers()
    print(ga.num_generation, ga.best_solution())
    while True:
        ga.stochastic_iteration()
        print(ga.num_generation, ga.best_solution()[1], overall_fitness(ga.best_solution()[0]))
        coeffs = ga.best_solution()[0]
        if ga.best_solution()[1] > 200:
             break
    ga.finish_workers()

    weight = ga.best_solution()[0]
    model.load_state_dict(ns.ga.torch.model_weights_as_dict(model, weight))
    model.eval()
    y_eval = model(torch.Tensor(x).reshape((-1, 1))).detach().numpy().flatten()

    res = y - y_eval
    var = y - np.average(y)
    det = 1 - (res@res)/(var@var)

    plt.plot(x, y, label='Noisy data')
    plt.plot(x, y_eval, 'o-', markersize=1.5, label=f'NN fit, $r^2$={det:.4f}')
    plt.legend()
    plt.show(block=True)
