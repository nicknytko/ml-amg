import os
from ns.parallel.worker import BaseWorker, WorkerCommand, worker_process


class SingleProcessPipeEnd:
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


class SingleProcessPipe:
    def __init__(self):
        self.buf_a = []
        self.buf_b = []

    def __new__(cls):
        pipe = object.__new__(cls)
        pipe.__init__()

        return (SingleProcessPipeEnd(pipe, pipe.buf_a, pipe.buf_b),
                SingleProcessPipeEnd(pipe, pipe.buf_b, pipe.buf_a))


class SingleProcessWorker(BaseWorker):
    '''
    A worker process that runs in the same process/thread as the caller.
    This queues up commands and runs them all when receive() is called.
    '''

    def __init__(self):
        self.started = False
        self.parent_pipe, self.child_pipe = SingleProcessPipe()

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
