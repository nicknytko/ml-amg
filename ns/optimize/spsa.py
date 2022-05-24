import numpy as np
import ns.optimize.base_optimizer


class SPSAOptimizer(BaseGradientApproximationOptimizer):
    '''
    An approximation to the gradient of a loss function using
    Simultaneous perturbation stochastic approximations.
    All components of the search space are simultaneously perturbed
    by a consistent value.
    '''

    def __init__(self, x, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, loss_func=None, loss_args=None, c=1e-4):
        super().__init__(x, lr, betas, eps, loss_func, loss_args)
        self.c = c

    def _gradient(self, x):
        delta_n = np.random.choice([1., -1.], size=len(self.x), replace=True)
        f = self._loss(x + delta_n * self.c)
        b = self._loss(x - delta_n * self.c)
        return (f-b) / (2*c*delta_n)
