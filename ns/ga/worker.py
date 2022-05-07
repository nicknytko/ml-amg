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
    MAP = 6


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


class DummySingleProcessPipeEnd:
    def __init__(self, pipe, recv_buf, send_buf):
        self.pipe = pipe
        self.recv_buf = recv_buf
        self.send_buf = send_buf

    def recv(self):
        if len(self.recv_buf) == 0:
            raise RuntimeError('Cannot receive from empty pipe when running in single-threaded mode.')
        return self.recv_buf.pop(0)

    def send(self, data):
        self.send_buf.append(data)


class DummySingleProcessPipe:
    def __init__(self):
        self.buf_a = []
        self.buf_b = []

    def __new__(cls):
        pipe = object.__new__(cls)
        pipe.__init__()

        return (DummySingleProcessPipeEnd(pipe, pipe.buf_a, pipe.buf_b),
                DummySingleProcessPipeEnd(pipe, pipe.buf_b, pipe.buf_a))


class DummySingleProcessWorker(Worker):
    def __init__(self, ctx):
        self.started = False
        self.parent_pipe, self.child_pipe = DummySingleProcessPipe()

    def start(self):
        self.started = True

    def finish(self):
        pass

    def send_command(self, cmd):
        self.parent_pipe.send(cmd)

    def receive(self):
        self.parent_pipe.send(WorkerCommand.create(WorkerCommand.EXIT))
        worker_process(self.child_pipe) # run worker code
        self.parent_pipe.recv() # started command
        return self.parent_pipe.recv() # return output


class WorkerQueue:
    def __init__(self, num_workers):
        '''
        Creates and transparently starts a queue of workers.
        '''

        self.mp_ctx = multiprocessing.get_context('spawn')
        self.num_workers = num_workers
        self.workers = []
        for i in range(num_workers):
            if i == 0:
                self.workers.append(DummySingleProcessWorker(self.mp_ctx))
            else:
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
    folds = cmd['folds']
    C = population.shape[1]

    mut_rand = np.random.RandomState(os.getpid() ^ random.randint(os.getpid()))

    # Have at least one mutation.  Numpy doesn't like when we send back an empty array.
    while not np.any(mutated):
        for i in range(population.shape[0]):
            if folds is None:
                # Non-folded mutation.  Mutate random subsets of the weights.
                to_mutate = mut_rand.choice([False, True], size=C, replace=True, p=[1-mutation_probability, mutation_probability])
                if np.any(to_mutate):
                    #mutations = (mut_rand.rand(np.sum(to_mutate)) * (perturb_max - perturb_min)) + perturb_min
                    mutations = mut_rand.normal(scale=min(abs(perturb_max), abs(perturb_min)), size=np.sum(to_mutate))
                    population[i, to_mutate] += mutations
                    mutated[i] = True
            else:
                # Folded mutation.  Randomly mutate an entire fold.
                to_mutate = mut_rand.choice([False, True], size=len(folds), replace=True, p=[1-mutation_probability, mutation_probability])
                for j, fold in enumerate(folds):
                    if not to_mutate[j]:
                        continue

                    for rng in fold.ranges:
                        low, high = rng.low, rng.high
                        mutations = mut_rand.normal(scale=min(abs(perturb_max), abs(perturb_min)), size=high-low)
                        population[i, low:high] += mutations
                        mutated[i] = True

    # Only send back parts of the population we have mutated
    indices = indices[mutated]
    population = population[mutated]

    pipe.send(WorkerCommand.create(WorkerCommand.FITNESS,
                                   indices=indices,
                                   population=population))


def worker_map(random, pipe, cmd):
    iterable = cmd['iterable']
    f = cmd['function']
    output = [f(i, *cmd['args']) for i in iterable]
    pipe.send(WorkerCommand.create(WorkerCommand.MAP, output=output, worker_idx=cmd['worker_idx']))


def worker_process(pipe):
    '''
    Main entrypoint for the worker process.  This will poll for commands
    to run from the main process, until it is told to terminate.
    '''

    # There may be some startup cost involved in creating the process
    # (loading datasets, etc), so send a message when we've started
    pipe.send(WorkerCommand.create(WorkerCommand.STARTED))

    if multiprocessing.current_process().name == 'MainProcess':
        random = np.random.RandomState()
    else:
        random = np.random.RandomState(seed=os.getpid())

    cmds = {
        WorkerCommand.NOOP: lambda r,p,c: p.send(WorkerCommand.create(WorkerCommand.NOOP)),
        WorkerCommand.FITNESS: worker_fitness,
        WorkerCommand.CROSSOVER: worker_crossover,
        WorkerCommand.MUTATION: worker_mutation,
        WorkerCommand.MAP: worker_map
    }

    while True:
        # Poll endlessly until we are told to exit

        worker_cmd = pipe.recv()
        cmd_enum = worker_cmd['command']

        if cmd_enum == WorkerCommand.EXIT:
            return
        elif cmd_enum in cmds:
            cmds[cmd_enum](random, pipe, worker_cmd)
        else:
            raise RuntimeError(f'Unknown command {cmd_enum}')
