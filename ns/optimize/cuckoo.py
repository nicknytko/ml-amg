import numpy as np
import ns.optimize.base_optimizer
import scipy.stats


class CuckooSearchOptimizer(ns.optimize.base_optimizer.BaseEvolutionaryOptimizer):
    def __init__(self, x,
                 population_size=20,
                 fitness_func=None,
                 fitness_args=None,
                 pa=0.4,
                 alpha=1e-4,
                 lmbda=3.0):
        super().__init__(x, population_size, fitness_func, fitness_args)
        self.pa = pa
        self.levy = scipy.stats.levy(lmbda)
        self.alpha = alpha

    def step(self):
        self.compute_fitness()

        ## 2) perturbation
        signs = np.random.choice([-1., 1.], size=self.population.shape)
        lengths = self.levy.rvs(self.population.shape)
        perturb = signs * length * self.alpha

        x_i_prime = self.population + perturb
        x_i_prime_fit = self._compute_fitness_on_population(self.generation, x_prime)
        x_prime = self.population.copy()

        ## 3) selection
        # x_prime will consist of the best of either individual from both populations
        for i in range(self.population):
            if x_i_prime_fit[i] < self.population_fitness[i]:
                x_prime[i] = x_i_prime[i]

        ## 4) recombination
        self.population[:] = x_prime.copy()
        pa_prob = np.random.uniform(0., 1., self.population.shape)
        to_recombine = np.where(pa_prob >= self.pa)[0]

        xl = np.random.choice(self.num_individuals, size=to_recombine.shape[0], replace=False)
        xm = np.random.choice(self.num_individuals, size=to_recombine.shape[0], replace=False)

        for i, individual in enumerate(to_recombine):
            if len(individual) > 0:
                self.population[i, individual] += np.random.uniform(0, 1., len(individual)) * \
                    (x_prime[xl[i], individual] - x_prime[xm[i], individual])
                self.population_computed_fitness[i] = False

        self.compute_fitness()
        self.generation += 1
