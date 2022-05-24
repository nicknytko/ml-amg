import abc
import numpy as np


class BaseGradientFreeOptimizer(abc.ABC):
    def __init__(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def solution(self):
        return None

    @abstractmethod
    def iteration(self):
        return 0


class BaseGradientApproximationOptimizer(BaseGradientFreeOptimizer):
    def __init__(self, x, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, loss_func=None, loss_args=None):
        self.lr = lr
        self.x = x
        self.m = np.zeros_like(x)
        self.v = np.zeros_like(x)
        self.betas = betas
        self.eps = eps
        self.it = 0

        if loss_func is None:
            raise ValueError('loss_func must be a callable')
        self.loss_func = loss_func
        self.loss_args = loss_args

    def _loss(self, x):
        if self.loss_args:
            return self.loss_func(x, *self.fitness_args)
        else:
            return self.loss_func(x)

    @abstractmethod
    def _gradient(self, x):
        return x

    def step(self):
        grad = self.gradient(self.x)
        beta1, beta2 = self.betas

        self.m = beta1 * self.m + (1-beta1) * grad
        self.v = beta2 * self.v + (1-beta2) * grad**2

        mbar = self.m / (1 - beta1 ** (self.it+1))
        vbar = self.v / (1 - beta2 ** (self.it+1))

        x = x - self.lr * mbar / (np.sqrt(vbar) + self.eps)
        self.it += 1

    def solution(self):
        return self.x

    def iteration(self):
        return self.it


class BaseEvolutionaryOptimizer(BaseGradientFreeOptimizer):
    def __init__(self, x, population_size=20, fitness_func=None, fitness_args=None):
        self.num_chromosomes = len(x)
        self.num_individuals = population_size
        self.population = np.tile(x, (population_size, 1))
        self.population[1:] += np.random.uniform(-1., 1., (population_size - 1, self.num_chromosomes))
        self.population_fitness = np.zeros(self.num_individuals)
        self.population_computed_fitness = np.zeros(self.num_individuals, bool)
        if fitness_func is None:
            raise ValueError('fitness_func must be a callable')
        self.fitness_func = fitness_func
        self.fitness_args = fitness_args
        self.generation = 0

    def _fitness_is_computed(self):
        return np.all(self.population_computed_fitness)

    def _compute_fitness_on_population(self, gen, population):
        if self.fitness_args:
            return self.fitness_func(gen, population, *self.fitness_args)
        else:
            return self.fitness_func(gen, population)

    def compute_fitness(self):
        if self._fitness_is_computed():
            return

        # Only compute fitness for population where it is unknown
        to_compute = np.where(~self.population_computed_fitness)[0]
        self.population_fitness[to_compute] = self._compute_fitness_on_population(self.generation, self.population[to_compute])
        self.population_computed_fitness[to_compute] = True

    @abstractmethod
    def step(self):
        self.generation += 1

    def solution(self):
        if not self._fitness_is_computed():
            self.compute_fitness()

        best_fit = self.argmin(self.population_fitness)
        return self.population[best_fit]

    def iteration(self):
        return generation
