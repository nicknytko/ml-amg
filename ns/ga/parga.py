import multiprocessing
import numpy as np
from worker import *


class ParallelGA:
    def __init__(self, **kwargs):
        self.population = np.copy(kwargs.get('initial_population'))
        self.population_size = self.population.shape[0]
        self.population_fitness = np.zeros(self.population_size)
        self.population_computed_fitness = np.zeros(self.population_size, bool)

        self.fitness_func = kwargs.get('fitness')
        self.crossover_probability = kwargs.get('crossover_probability', 0.5)
        self.mutation_probability = kwargs.get('mutation_probability', 0.3)

        self.mutation_min_perturb = kwargs.get('mutation_min_perturb', -1.)
        self.mutation_max_perturb = kwargs.get('mutation_max_perturb',  1.)

        self.num_generation = 0

        self.num_workers = kwargs.get('num_workers', 2)
        self.workers = WorkerQueue(self.num_workers)


    def compute_fitness(self):
        if np.all(self.population_computed_fitness):
            return

        # Only compute fitness for population where it is unknown
        to_compute = np.where(~self.population_computed_fitness)[0]

        # Divvy up the population for the workers using a cyclic mapping.
        # The mapping we use doesn't really matter here, but I'm lazy and it's
        # really easy to slice a cycling mapping in Python.
        for i, worker in enumerate(self.workers):
            local_indices = to_compute[i::self.num_workers]
            local_population = self.population[local_indices]
            if len(local_indices) != 0:
                worker.send_command(WorkerCommand.create(WorkerCommand.FITNESS,
                                                         population=local_population,
                                                         indices=local_indices,
                                                         fitness_func=self.fitness_func))
            else:
                worker.send_command(WorkerCommand.create(WorkerCommand.NOOP))

        # Now, assemble data we get back from the workers
        data = self.workers.receive_all()
        for datum in data:
            if datum['command'] == WorkerCommand.NOOP:
                continue
            local_indices = datum['indices']
            local_fitness = datum['fitness']
            self.population_fitness[local_indices] = local_fitness
            self.population_computed_fitness[local_indices] = True


    def crossover(self):
        N_per_worker = int(np.ceil(self.population_size / self.num_workers / 2)) * 2
        chromosomes = self.population.shape[1]
        N = N_per_worker * self.num_workers

        for worker in self.workers:
            worker.send_command(WorkerCommand.create(WorkerCommand.CROSSOVER,
                                                     population=self.population,
                                                     fitness=self.population_fitness,
                                                     crossover_probability=self.crossover_probability,
                                                     num_to_create=N_per_worker))

        self.population = np.zeros((N, chromosomes))
        self.population_fitness = np.zeros(N)
        self.population_computed_fitness = np.zeros(N, bool)

        # We will compute more pairs than needed, then discard a random subset
        indices_to_use = np.random.choice(np.arange(0, N), size=self.population_size, replace=False)

        data = self.workers.receive_all()
        for i, datum in enumerate(data):
            self.population[i * N_per_worker:(i+1) * N_per_worker] = datum['pairs']
            self.population_computed_fitness[i * N_per_worker:(i+1) * N_per_worker] = datum['pairs_computed_fitness']
            self.population_fitness[i * N_per_worker:(i+1) * N_per_worker] = datum['pairs_fitness']

        # Discard extras
        self.population = self.population[indices_to_use]
        self.population_computed_fitness = self.population_computed_fitness[indices_to_use]
        self.population_fitness = self.population_fitness[indices_to_use]


    def mutation(self):
        for i, worker in enumerate(self.workers):
            local_indices = np.arange(0, self.population_size)[i::self.num_workers]
            local_population = self.population[local_indices]
            if len(local_indices) != 0:
                worker.send_command(WorkerCommand.create(WorkerCommand.MUTATION,
                                                         population=local_population,
                                                         indices=local_indices,
                                                         mutation_probability=self.mutation_probability,
                                                         mutation_perturb=(self.mutation_min_perturb, self.mutation_max_perturb)))
            else:
                worker.send_command(WorkerCommand.create(WorkerCommand.NOOP))

        # Now, assemble data we get back from the workers
        data = self.workers.receive_all()
        for datum in data:
            if datum['command'] == WorkerCommand.NOOP:
                continue
            local_indices = datum['indices']
            local_population = datum['population']
            self.population[local_indices] = local_population
            self.population_computed_fitness[local_indices] = False


    def iteration(self):
        self.num_generation += 1
        self.compute_fitness()
        self.crossover()
        self.mutation()
        self.compute_fitness()


    def best_solution(self):
        self.compute_fitness()
        idx = np.argmax(self.population_fitness)
        return self.population[idx], self.population_fitness[idx], idx


    def start_workers(self):
        self.workers.start()


    def finish_workers(self):
        self.workers.finish()
