import os
from ns.parallel.worker import BaseWorker

import mpi4py
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
import mpi4py.MPI


def mpi_local_rank():
    '''
    If running in MPI, finds the local rank of this process.
    This is a unique identifier per *node*.

    Returns
    -------
    rank : integer
      Local rank of this process
    '''

    if 'SLURM_LOCALID' in os.environ:
        # If we're running w/ slurm, then we'll have a local ID assigned
        return int(os.environ['SLURM_LOCALID'])
    else:
        # TODO: we can probably get this with a local communicator
        raise RuntimeError('Cannot get local rank')


def mpi_rank():
    return mpi4py.MPI.COMM_WORLD.Get_rank()


def mpi_enabled():
    '''
    Checks if MPI is enabled.  Currently this is assumed only
    if there are multiple MPI processes detected.

    Returns
    -------
    enabled : boolean
      Are we running with MPI?
    '''

    return (mpi4py.MPI.COMM_WORLD.Get_size() > 1)


def mpi_initialize():
    mpi4py.MPI.Init()


def mpi_finalize():
    mpi4py.MPI.Finalize()


def mpi_get_num_workers():
    return mpi4py.MPI.COMM_WORLD.Get_size()


class MPIPipeEnd:
    def __init__(self, pipe, this_idx, other_idx):
        self.pipe = pipe
        self.this_idx = this_idx
        self.other_idx = other_idx

    def recv(self):
        return mpi4py.MPI.COMM_WORLD.recv(source=self.other_idx, tag=0)

    def send(self, data):
        return mpi4py.MPI.COMM_WORLD.send(data, dest=self.other_idx, tag=0)


class MPIPipe:
    def __init__(self, a_idx, b_idx):
        self.a_idx = a_idx
        self.b_idx = b_idx

    def __new__(cls, a_idx, b_idx):
        pipe = object.__new__(cls)
        pipe.__init__(a_idx, b_idx)

        return (MPIPipeEnd(pipe, a_idx, b_idx),
                MPIPipeEnd(pipe, b_idx, a_idx))


class MPIWorker(BaseWorker):
    '''
    A child worker process that communicates to this process via MPI
    '''

    def __init__(self, idx):
        self.started = False
        self.parent_pipe, self.child_pipe = MPIPipe(0, idx)
