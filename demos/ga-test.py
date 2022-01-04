import numpy as np
import matplotlib.pyplot as plt
import sys

# Test of the genetic algorithm
# Finding a regression for a noisy quadratic function

sys.path.append('../')
import ns.ga.parga as parga

np.random.seed(0)
N = 300
x = np.linspace(-1, 1, N)
y = x ** 2 + np.random.randn(N)*0.01

N_population = 100
N_coeffs = 3

def fitness(weight, idx):
    y_eval = np.polyval(weight, x)
    return 1. / np.sum((y - y_eval)**2)

if __name__ == '__main__':
    num_workers = 4
    ga = parga.ParallelGA(num_workers=num_workers,
                          initial_population=np.random.randn(N_population, N_coeffs),
                          fitness_func=fitness,
                          mutation_probability=0.5,
                          mutation_min_perturb=-5.,
                          mutation_max_perturb=5.)
    ga.start_workers()
    print(ga.num_generation, ga.best_solution())
    while True:
        ga.iteration()
        print(ga.num_generation, ga.best_solution())
        coeffs = ga.best_solution()[0]
        if ga.best_solution()[1] > 10:
            break
    ga.finish_workers()

    res = y - np.polyval(coeffs, x)
    var = y - np.average(y)
    det = 1 - (res@res)/(var@var)

    plt.plot(x, y, label='Noisy data')
    plt.plot(x, np.polyval(coeffs, x), 'o-', markersize=1.5, label=f'Quadratic fit, $r^2$={det:.4f}')
    plt.legend()
    plt.show(block=True)
