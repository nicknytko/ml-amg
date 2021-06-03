from firedrake import *
from firedrake.petsc import PETSc
from firedrake.assemble import allocate_matrix, assemble
import pyamg
import scipy.sparse as sp
import functools
import matplotlib.pyplot as plt

class PyAMG(PCBase):
    _prefix = "pyamg_"

    def initialize(self, pc):
        from firedrake.assemble import allocate_matrix, assemble
        _, P = pc.getOperators()

        if pc.getType() != "python":
            print('Expecting PC type python')
            raise ValueError("Expecting PC type python")
        opc = pc
        appctx = self.get_appctx(pc)
        fcp = appctx.get("form_compiler_parameters")

        V = get_function_space(pc.getDM())
        if len(V) == 1:
            V = FunctionSpace(V.mesh(), V.ufl_element())
        else:
            V = MixedFunctionSpace([V_ for V_ in V])
        test = TestFunction(V)
        trial = TrialFunction(V)

        if P.type == "python":
            context = P.getPythonContext()
            # It only makes sense to preconditioner/invert a diagonal
            # block in general.  That's all we're going to allow.
            if not context.on_diag:
                print('Only makes sense to invert diagonal block')
                raise ValueError("Only makes sense to invert diagonal block")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix

        mat_type = PETSc.Options().getString(options_prefix + "mat_type", "aij")

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

        self._assemble_P()
        self._createAmgSolver(pc)

    def _createAmgSolver(self, pc):
        _, P = pc.getOperators()
        
        # Transfer nullspace over
        Pmat = self.P.petscmat
        Pmat.setNullSpace(P.getNullSpace())
        tnullsp = P.getTransposeNullSpace()
        if tnullsp.handle != 0:
            Pmat.setTransposeNullSpace(tnullsp)
        Pmat.setNearNullSpace(P.getNearNullSpace())

        # Create PyAMG solver from CSR matrix
        row, col, val = Pmat.getValuesCSR()
        self.Pcsr = sp.csr_matrix((val, col, row))
        self.Amg = pyamg.ruge_stuben_solver(self.Pcsr)        
        
    def update(self, pc):
        self._assemble_P()
        self._createAmgSolver(pc)

    def form(self, pc, test, trial):
        _, P = pc.getOperators()
        if P.getType() == "python":
            context = P.getPythonContext()
            return (context.a, context.row_bcs)
        else:
            context = dmhooks.get_appctx(pc.getDM())
            return (context.Jp or context.J, context._problem.bcs)

    def apply(self, pc, X, Y):
        y = self.Amg.solve(X.array_r, tol=1e-6) # todo: make the tolerance a petsc option
        Y.setArray(y)

    def applyTranspose(self, pc, X, Y):
        pass
