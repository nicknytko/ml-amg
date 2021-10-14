from firedrake import *
from firedrake.petsc import PETSc
from firedrake.assemble import allocate_matrix, assemble
import pyamg
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import functools
import matplotlib.pyplot as plt
import scipy.io as sio
import traceback


class PyAMG(PCBase):
    _prefix = "pyamg_"
    _dump_mats = False

    def initialize(self, pc):
        try:
            self._initialize(pc)
        except Exception as e:
            traceback.print_exc()
            raise e

    def _initialize(self, pc):
        from firedrake.assemble import allocate_matrix, assemble
        _, P = pc.getOperators()

        if pc.getType() != "python":
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
                raise ValueError("Only makes sense to invert diagonal block")

        prefix = pc.getOptionsPrefix()
        options_prefix = prefix + self._prefix
        opts = PETSc.Options()
        self.amg_rtol = opts.getScalar(f'{options_prefix}amg_rtol', 1e-8)
        self.amg_max_levels = opts.getInt(f'{options_prefix}amg_max_levels', 10)
        self.amg_precon_gmres = opts.getBool(f'{options_prefix}amg_precondition_with_gmres', True)

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

        self.update(pc)

    _matidx = 0
    def dump_mat(self, pc, csr):
        prefix = pc.getOptionsPrefix() + self._prefix
        sio.savemat(f'../out_matrices/{prefix}mat_{self._matidx:04}.mat', {'A': csr})
        self._matidx += 1

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
        #self.Amg = pyamg.classical.ruge_stuben_solver(self.Pcsr, strength=('classical', {'theta': 0.5}), max_levels=self.amg_max_levels)
        self.Amg = pyamg.aggregation.smoothed_aggregation_solver(self.Pcsr, max_levels=self.amg_max_levels)
        if PyAMG._dump_mats:
            self.dump_mat(pc, self.Pcsr)

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
        try:
            self._apply(pc, X, Y)
        except Exception as e:
            traceback.print_exc()
            raise e

    def _apply(self, pc, X, Y):
        y = self.Amg.solve(X.array_r, tol=self.amg_rtol, accel=('gmres' if self.amg_precon_gmres else None))
        Y.setArray(y)

    def applyTranspose(self, pc, X, Y):
        print('PyAMG applyTranspose!')
        pass

    def view(self, pc, viewer=None):
        super(PyAMG, self).view(pc, viewer)
        viewer.printfASCII( 'PyAMG Solver:\n')
        viewer.printfASCII(f' amg solver: {str(self.Amg)}\n')
        viewer.printfASCII(f' amg rtol: {self.amg_rtol}\n')
