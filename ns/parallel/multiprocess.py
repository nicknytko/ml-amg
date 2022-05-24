import os
from ns.parallel.worker import BaseWorker, worker_process
import multiprocessing
import sys


def mp_is_subprocess():
    '''
    Checks if the current process is a child subprocess or the main process.
    This is achieved in a somewhat hacky way by checking stack frames.

    Returns
    -------
    is_subprocess : boolean
      Is this process a subprocess?
    '''

    frame = sys._getframe()
    while frame.f_back is not None:
        if frame.f_globals['__name__'] == '__mp_main__':
            return True
        frame = frame.f_back
    return False


class MultiprocessWorker(BaseWorker):
    '''
    A child worker process that runs in a separate process using Python multiprocessing.
    '''

    def __init__(self):
        '''
        Creates and starts a worker process.  Note that the process is transparently
        started in the constructor, though start() should be called before anything
        else is run.
        '''
        self.mp_ctx = multiprocessing.get_context('spawn')
        self.parent_pipe, self.child_pipe = multiprocessing.Pipe()
        self.process = self.mp_ctx.Process(target=worker_process, args=(self.child_pipe,))
        self.process.start()
        self.started = False
