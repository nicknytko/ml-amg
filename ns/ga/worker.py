import os
import enum
import numpy as np
import numpy.linalg as la
import multiprocessing


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
    '''
    Representation of a child worker process.
    Has routines for sending and receiving data to the worker.
    '''

    def __init__(self, ctx):
        '''
        Creates and starts a worker process.  Note that the process is transparently
        started in the constructor, though start() should be called before anything
        else is run.
        '''
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()
        self.process = ctx.Process(target=worker_process, args=(self.child_pipe,))
        self.process.start()
        self.started = False


    def start(self):
        '''
        Waits for the worker process to start and get to a ready state.
        This should be called before any data is sent/received to the worker.
        '''

        if self.started:
            return

        cmd = self.parent_pipe.recv()
        if cmd['command'] != WorkerCommand.STARTED:
            raise RuntimeError('Did not receive reply from worker!')
        self.started = True


    def finish(self):
        '''
        Closes the worker process
        '''
        self.send_command(WorkerCommand.create(WorkerCommand.EXIT))


    def send_command(self, cmd):
        '''
        Send a WorkerCommand to the worker process
        '''
        self.parent_pipe.send(cmd)


    def receive(self):
        '''
        Receive data from the worker process.  This will block until something is received.
        '''
        return self.parent_pipe.recv()


class WorkerQueue:
    def __init__(self, num_workers):
        '''
        Creates and transparently starts a queue of workers.
        '''

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
        '''
        Wait for all workers to start.  This should be called before
        any data is sent or received to a worker.
        '''
        if self.started:
            return

        for worker in self.workers:
            worker.start()

        self.started = True


    def receive_all(self):
        '''
        Gathers data from all worker processes.

        Returns
        -------
        data - python list
          List of data, such that data[i] is what was received from worker i.
        '''

        data = []
        for worker in self.workers:
            data.append(worker.receive())
        return data


    def finish(self):
        '''
        Closes all workers in the queue, and frees associated resources.
        '''

        if not self.started:
            raise RuntimeError('Cannot close WorkerQueue: not started')

        for worker in self.workers:
            worker.finish()


def worker_fitness(random, pipe, cmd):
    population = cmd['population']
    indices = cmd['indices']
    fitness_func = cmd['fitness_func']
    generation = cmd['generation']
    computed_fitness = np.zeros_like(indices, dtype=np.float64)

    for i, idx in enumerate(indices):
        computed_fitness[i] = fitness_func(generation, population[i], idx)

    pipe.send(WorkerCommand.create(WorkerCommand.FITNESS,
                                   indices=indices,
                                   fitness=computed_fitness))


def worker_crossover(random, pipe, cmd):
    population = cmd['population']
    fitness = cmd['fitness']

    # Probability for picking individuals
    if cmd['selection_uniform_probability']:
        probability = np.ones(len(fitness)) / len(fitness)
    else:
        probability = fitness / la.norm(fitness, 1)

    # Crossover probability
    crossover_probability = cmd['crossover_probability']
    N = cmd['num_to_create']
    n_pairs = N // 2

    # How many of the top population to use
    population_to_use = np.argsort(fitness)[::-1]
    if cmd['top_population_to_use'] != -1:
        population_to_use = population_to_use[:cmd['top_population_to_use']]
        probability = probability[population_to_use]
        probability /= la.norm(probability, 1)

    pairs = np.zeros((n_pairs * 2, population.shape[1]))
    pairs_computed_fitness = np.zeros(n_pairs * 2, dtype=bool)
    pairs_fitness = np.zeros(n_pairs * 2)

    folds = cmd['folds']

    for i in range(0, n_pairs, 2):
        # draw two parents
        parent_one_idx, parent_two_idx = random.choice(population_to_use, size=2, replace=False, p=probability)

        # now, with probability p do some sort of crossover, otherwise use the parents directly
        p = random.rand()
        if p <= crossover_probability:
            if folds is None:
                crossover_pt = random.randint(0, population.shape[1])

                # single point crossover
                pairs[i, :crossover_pt] = population[parent_one_idx, :crossover_pt]
                pairs[i, crossover_pt:] = population[parent_two_idx, crossover_pt:]

                pairs[i+1, crossover_pt:] = population[parent_one_idx, crossover_pt:]
                pairs[i+1, :crossover_pt] = population[parent_two_idx, :crossover_pt]
            else:
                # Fold-based crossover
                for fold in folds:
                    a, b = random.choice([i, i+1], size=2, replace=False)
                    for rng in fold.ranges:
                        pairs[a, rng.low:rng.high] = population[parent_one_idx, rng.low:rng.high]
                        pairs[b, rng.low:rng.high] = population[parent_two_idx, rng.low:rng.high]
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


def worker_mutation(random, pipe, cmd):
    population = cmd['population']
    indices = cmd['indices']
    mutated = np.zeros(population.shape[0], bool)
    mutation_probability = cmd['mutation_probability']
    perturb_min, perturb_max = cmd['mutation_perturb']
    C = population.shape[1]

    # Have at least one mutation.  Numpy doesn't like when we send back an empty array.
    while not np.any(mutated):
        for i in range(population.shape[0]):
            p = random.rand()
            if p <= mutation_probability:
                population[i] += (random.rand(C) * (perturb_max - perturb_min)) + perturb_min
                mutated[i] = True

    # Only send back parts of the population we have mutated
    indices = indices[mutated]
    population = population[mutated]

    pipe.send(WorkerCommand.create(WorkerCommand.FITNESS,
                                   indices=indices,
                                   population=population))


def worker_process(pipe):
    '''
    Main entrypoint for the worker process.  This will poll for commands
    to run from the main process, until it is told to terminate.
    '''

    # There may be some startup cost involved in creating the process
    # (loading datasets, etc), so send a messgae when we've started
    pipe.send(WorkerCommand.create(WorkerCommand.STARTED))
    random = np.random.RandomState(seed=os.getpid())

    while True:
        # Poll endlessly until we are told to exit

        worker_cmd = pipe.recv()
        cmd_enum = worker_cmd['command']

        if cmd_enum == WorkerCommand.EXIT:
            return
        elif cmd_enum == WorkerCommand.NOOP:
            pipe.send(WorkerCommand.create(WorkerCommand.NOOP))
        elif cmd_enum == WorkerCommand.FITNESS:
            worker_fitness(random, pipe, worker_cmd)
        elif cmd_enum == WorkerCommand.CROSSOVER:
            worker_crossover(random, pipe, worker_cmd)
        elif cmd_enum == WorkerCommand.MUTATION:
            worker_mutation(random, pipe, worker_cmd)
        else:
            raise RuntimeError(f'Unknown command {cmd_enum}')
