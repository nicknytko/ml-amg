from firedrake import *
from firedrake.petsc import PETSc
from firedrake.assemble import allocate_matrix, assemble
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import functools
import matplotlib.pyplot as plt
import scipy.io as sio
from ns.model.ali_interp import InterpolationNetwork
from ns.lib.greedy import greedy_coarsening
import torch
import traceback
import numpy as np
import numpy.linalg as la
import time

start_time = 0
def init_timer():
    global start_time
    start_time = (time.time_ns() // 1000000)

def timeit(s):
    global start_time
    print('TIME', (time.time_ns() // 1000000) - start_time, s)


class MLAMG(PCBase):
    _prefix = 'mlamg_'

    def initialize(self, pc):
        try:
            self._initialize(pc)
        except Exception as e:
            traceback.print_exc()
            raise e

    def _initialize(self, pc):
        _, P = pc.getOperators()

        if pc.getType() != 'python':
            raise ValueError('Expecting PC type python')
        opc = pc
        appctx = self.get_appctx(pc)
        fcp = appctx.get('form_compiler_parameters')

        V = get_function_space(pc.getDM())
        if len(V) == 1:
            V = FunctionSpace(V.mesh(), V.ufl_element())
        else:
            V = MixedFunctionSpace([V_ for V_ in V])
        test = TestFunction(V)
        trial = TrialFunction(V)

        if P.type == 'python':
            context = P.getPythonContext()
            # It only makes sense to precondition/invert a diagonal
            # block in general.  That's all we're going to allow.
            if not context.on_diag:
                raise ValueError('Only makes sense to invert diagonal block')

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()
        self.amg_rtol = opts.getScalar(f'{options_prefix}amg_rtol', 1e-8)
        self.coarsening_theta = opts.getScalar(f'{options_prefix}greedy_theta', 0.56)
        self.jacobi_weight = opts.getScalar(f'{options_prefix}jacobi_weight', 2./3.)
        self.pnet_model_fname = opts.getString(f'{options_prefix}pnet_model')

        mat_type = PETSc.Options().getString(options_prefix + 'mat_type', 'aij')

        (a, bcs) = self.form(pc, test, trial)

        self.P = allocate_matrix(a, bcs=bcs,
                                 form_compiler_parameters=fcp,
                                 mat_type=mat_type,
                                 options_prefix=options_prefix)
        self._assemble_P = functools.partial(assemble,
                                             a,
                                             tensor=self.P,
                                             bcs=bcs,
                                             form_compiler_parameters=fcp,
                                             mat_type=mat_type)

        self.update(pc)

    def _createAmgSolver(self, pc):
        _, P = pc.getOperators()

        init_timer()
        timeit('initialization of preconditioner')

        # Transfer nullspace over
        Pmat = self.P.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)
        Pmat.setNearNullSpace(P.getNearNullSpace())

        timeit('transferred nullspace')

        row, col, val = Pmat.getValuesCSR()
        self.A = sp.csr_matrix((val, col, row))
        self.Dinv = sp.diags(1.0 / self.A.diagonal()) * self.jacobi_weight
        self.P_net = InterpolationNetwork()
        try:
            state_dict = torch.load(self.pnet_model_fname)
            self.P_net.model.load_state_dict(state_dict)
            self.P_net.eval()
        except Exception as e:
            raise RuntimeError(f'Could not load PNet model "{self.pnet_model_fname}": {e}')

        timeit('created CSR matrix and loaded P net')

        # Create coarsening and interpolation
        _, F, C = greedy_coarsening(self.A, self.coarsening_theta)

        timeit('greedy coarsening')

        self.P_amg = self.P_net.forward_mat(self.A, C, F)
        self.A_H = self.P_amg.T @ self.A @ self.P_amg
        self.A_H_lu = spla.splu(self.A_H, permc_spec='COLAMD')

        timeit('generated interpolation matrix')

    def update(self, pc):
        try:
            self._assemble_P()
            self._createAmgSolver(pc)
        except Exception as e:
            traceback.print_exc()
            raise e

    def form(self, pc, test, trial):
        _, P = pc.getOperators()
        if P.getType() == 'python':
            context = P.getPythonContext()
            return (context.a, context.row_bcs)
        else:
            context = dmhooks.get_appctx(pc.getDM())
            return (context.Jp or context.J, context._problem.bcs)

    def jacobi(self, b, x, nu=2):
        for i in range(nu):
            x += self.Dinv @ (b - self.A @ x)
        return x

    def amg_2_v(self, P, b, x,
            pre_smoothing_steps=1,
            post_smoothing_steps=1,
            max_iter=500):
        '''
        Two-level AMG solver.

        Parameters
        ----------
        A : array, matrix, sparse matrix
            n x n linear system to solve
        P : array, matrix, sparse matrix
            n_F x n_C interpolation matrix
        b : array
            rhs for linear system, should be an array of shape (n,)
        x : array
            initial guess for solution, should be an array of shape (n,)
        jacobi_weight : float
            value of omega for Jacobi relaxation scheme
        res_tol : None, float
            if set, will stop iteration when absolute solution residual is below this value
        error_tol : None, float
            if set, will stop iteration when absolute solution norm is below this value
        max_iter : int
            maximum number of iterations before algorithm is stopped

        Returns
        -------
        (x, conv_factor, err, num_iterations)
        x : array
            approximate solution to the system
        conv_factor : float
            convergence factor, approximate to how much error is "dissipated" at each iteration
        err : array
            history of residuals or errors, depending on which tolerance is set
        num_iterations : int
            number of iterations completed until convergence
        '''

        b_H = P.T @ b

        for i in range(max_iter):
            x = self.jacobi(b, x, nu=pre_smoothing_steps)
            x += P @ self.A_H_lu.solve(P.T @ (b - self.A @ x))
            x = self.jacobi(b, x, nu=post_smoothing_steps)

            if la.norm(b - self.A@x, 2) <= self.amg_rtol:
                break

        return x

    def apply(self, pc, X, Y):
        try:
            self._apply(pc, X, Y)
        except Exception as e:
            traceback.print_exc()
            raise e

    def _apply(self, pc, X, Y):
        b = X.array_r

        x = np.random.normal(size=self.A.shape[1])
        x = self.amg_2_v(self.P_amg, b, x)

        Y.setArray(x)

    def applyTranspose(self, pc, X, Y):
        print('PyAMG applyTranspose!')
        pass

    def view(self, pc, viewer=None):
        super(PyAMG, self).view(pc, viewer)
        viewer.printfASCII('PyAMG Solver:\n')
        viewer.printfASCII(f' amg solver: {str(self.Amg)}\n')
        viewer.printfASCII(f' amg rtol: {self.amg_rtol}\n')
