import os
import enum
import numpy as np
import numpy.linalg as la


class WorkerCommand(enum.Enum):
    EXIT = 0
    NOOP = 1
    STARTED = 2
    FITNESS = 3
    CROSSOVER = 4
    MUTATION = 5


    def create(enum, **kwargs):
        kwargs['command'] = enum
        return kwargs


class Worker:
    def __init__(self, ctx):
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(self.child_pipe,))
        self.process.start()
        self.started = False


    def start(self):
        if self.started:
            return

        cmd = self.parent_pipe.recv()
        if cmd['command'] != WorkerCommand.STARTED:
            raise RuntimeError('Did not receive reply from worker!')
        self.started = True


    def finish(self):
        self.send_command(WorkerCommand.create(WorkerCommand.EXIT))


    def send_command(self, cmd):
        self.parent_pipe.send(cmd)


    def receive(self):
        return self.parent_pipe.recv()


class WorkerQueue:
    def __init__(self, num_workers):
        self.mp_ctx = multiprocessing.get_context('spawn')
        self.num_workers = num_workers
        self.workers = []
        for i in range(num_workers):
            self.workers.append(Worker(self.mp_ctx))
        self.started = False


    def __getitem__(self, key):
        return self.workers[key]


    def __len__(self):
        return self.num_workers


    def start(self):
        if self.started:
            return

        for worker in self.workers:
            worker.start()

        self.started = True


    def receive_all(self):
        data = []
        for worker in self.workers:
            data.append(worker.receive())
        return data


    def finish(self):
        if not self.started:
            raise RuntimeError('Cannot close WorkerQueue: not started')

        for worker in self.workers:
            worker.finish()


def worker_fitness(pipe, cmd):
    population = cmd['population']
    indices = cmd['indices']
    fitness_func = cmd['fitness_func']
    computed_fitness = np.zeros_like(indices, dtype=np.float64)

    for i, idx in enumerate(indices):
        computed_fitness[i] = fitness_func(population[i], idx)

    pipe.send(WorkerCommand.create(WorkerCommand.FITNESS,
                                   indices=indices,
                                   fitness=computed_fitness))


def worker_crossover(pipe, cmd):
    population = cmd['population']
    fitness = cmd['fitness']
    fitness = fitness / la.norm(fitness, 1)
    crossover_probability = cmd['crossover_probability']
    N = cmd['num_to_create']
    n_pairs = N // 2

    pairs = np.zeros((n_pairs * 2, population.shape[1]))
    pairs_computed_fitness = np.zeros(n_pairs * 2, dtype=bool)
    pairs_fitness = np.zeros(n_pairs * 2)

    np.random.seed(os.getpid())

    for i in range(0, n_pairs, 2):
        # draw two parents
        parent_one_idx, parent_two_idx = np.random.choice(np.arange(population.shape[0]), size=2, replace=False, p=fitness)

        # now, with probability p do some sort of crossover, otherwise use the parents directly
        p = np.random.rand()
        if p <= crossover_probability:
            crossover_pt = np.random.randint(0, population.shape[1])

            # single point crossover
            pairs[i, :crossover_pt] = population[parent_one_idx, :crossover_pt]
            pairs[i, crossover_pt:] = population[parent_two_idx, crossover_pt:]

            pairs[i+1, crossover_pt:] = population[parent_one_idx, crossover_pt:]
            pairs[i+1, :crossover_pt] = population[parent_two_idx, :crossover_pt]
        else:
            pairs[i] = population[parent_one_idx]
            pairs_fitness[i] = fitness[parent_one_idx]
            pairs_computed_fitness[i] = True

            pairs[i+1] = population[parent_two_idx]
            pairs_fitness[i+1] = fitness[parent_two_idx]
            pairs_computed_fitness[i+1] = True

    pipe.send(WorkerCommand.create(WorkerCommand.CROSSOVER,
                                   pairs=pairs,
                                   pairs_computed_fitness=pairs_computed_fitness,
                                   pairs_fitness=pairs_fitness))


def worker_mutation(pipe, cmd):
    population = cmd['population']
    indices = cmd['indices']
    mutated = np.zeros(population.shape[0], bool)
    mutation_probability = cmd['mutation_probability']
    perturb_min, perturb_max = cmd['mutation_perturb']
    C = population.shape[1]

    np.random.seed(os.getpid())

    for i in range(population.shape[0]):
        p = np.random.rand()
        if p <= mutation_probability:
            population[i] += (np.random.rand(C) * (perturb_max - perturb_min)) + perturb_min
            mutated[i] = True

    # Only send back parts of the population we have mutated
    indices = indices[mutated]
    population = population[mutated]
    pipe.send(WorkerCommand.create(WorkerCommand.FITNESS,
                                   indices=indices,
                                   population=population))


def worker_process(pipe):
    # There may be some startup cost involved in creating the process
    # (loading datasets, etc), so send a messgae when we've started
    pipe.send(WorkerCommand.create(WorkerCommand.STARTED))

    while True:
        # Poll endlessly until we are told to exit

        worker_cmd = pipe.recv()
        cmd_enum = worker_cmd['command']

        if cmd_enum == WorkerCommand.EXIT:
            return
        elif cmd_enum == WorkerCommand.NOOP:
            pipe.send(WorkerCommand.create(WorkerCommand.NOOP))
        elif cmd_enum == WorkerCommand.FITNESS:
            worker_fitness(pipe, worker_cmd)
        elif cmd_enum == WorkerCommand.CROSSOVER:
            worker_crossover(pipe, worker_cmd)
        elif cmd_enum == WorkerCommand.MUTATION:
            worker_mutation(pipe, worker_cmd)
        else:
            raise RuntimeError(f'Unknown command {cmd_enum}')
