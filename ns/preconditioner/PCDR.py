from firedrake import *
from firedrake.petsc import PETSc
from firedrake.assemble import allocate_matrix, assemble
import scipy.sparse as sp
import functools
import matplotlib.pyplot as plt
import numpy as np
import ns.lib.petsc

class PCDR(PCBase):
    # Loosely based on the PCDR preconditioner from FEniCS
    # https://fenapack.readthedocs.io/en/2019.1.0/math.html#math-background-pcdr-extension
    # Follows the X_{BRM2} form of
    # S^{-1} \approx R_p^{-1} + A_p^{-1} K_p M_p^{-1} (FEniCS has an identity matrix in there, not sure why...)
    # where R_p \approx dt B D_M^{-1} B^T,
    # B^T is the pressure gradient operator,
    # D_M = diag(M_u) = diagonal of the velocity mass matrix
    # K_p is the pressure convection matrix
    # M_p is the pressure mass matrix
    # A_p is the pressure laplacian
    # dt is the timestep

    # This reuses a lot of code from the existing Firedrake PCD preconditioner
    
    _prefix = 'pcdr_'
    needs_python_pmat = True

    def initialize(self, pc):
        try:
            self._initialize(pc)
        except Exception as e:
            print(e)
    
    def _initialize(self, pc):
        _, P = pc.getOperators()
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

        # Grab relevant options
        prefix = pc.getOptionsPrefix() + self._prefix

        opts = PETSc.Options()
        mat_type = opts.getString(f'{prefix}_mat_type', "aij")
        Re = appctx.get('Re', 1.0)
        dt = appctx.get('dt', 1.0)
        velid = appctx.get('velocity_space', 0)
        self.Re = Re
        self.dt = dt
        self.velid = velid
        
        # Assemble operators
        context = P.getPythonContext()
        test, trial = context.a.arguments()
        if test.function_space() != trial.function_space():
            raise ValueError("Pressure space test and trial space differ")
        
        Q = test.function_space()
        
        p = TrialFunction(Q)
        q = TestFunction(Q)

        mass = p*q*dx
        # Regularisation to avoid having to think about nullspaces.
        stiffness = inner(grad(p), grad(q)) * dx + Constant(1e-6) * mass

        default = parameters["default_matrix_type"]
        Mp_mat_type = opts.getString(prefix+"Mp_mat_type", default)
        Kp_mat_type = opts.getString(prefix+"Kp_mat_type", default)
        Mu_mat_type = opts.getString(prefix+"Mu_mat_type", default)
        self.Fp_mat_type = opts.getString(prefix+"Fp_mat_type", "matfree")


        Mp = assemble(mass, form_compiler_parameters=context.fc_params,
                      mat_type=Mp_mat_type,
                      options_prefix=prefix + "Mp_")
        Kp = assemble(stiffness, form_compiler_parameters=context.fc_params,
                      mat_type=Kp_mat_type,
                      options_prefix=prefix + "Kp_")

        Mksp = PETSc.KSP().create(comm=pc.comm)
        Mksp.incrementTabLevel(1, parent=pc)
        Mksp.setOptionsPrefix(prefix + "Mp_")
        Mksp.setOperators(Mp.petscmat)
        Mksp.setUp()
        Mksp.setFromOptions()
        self.Mksp = Mksp

        Kksp = PETSc.KSP().create(comm=pc.comm)
        Kksp.incrementTabLevel(1, parent=pc)
        Kksp.setOptionsPrefix(prefix + "Kp_")
        Kksp.setOperators(Kp.petscmat)
        Kksp.setUp()
        Kksp.setFromOptions()
        self.Kksp = Kksp

        # Get current state
        state = appctx['state']
        u0 = split(state)[velid]
        fp = 1.0/Re * inner(grad(p), grad(q))*dx + inner(u0, grad(p))*q*dx

        self.Fp = allocate_matrix(fp,
                                  form_compiler_parameters=fcp,
                                  mat_type=self.Fp_mat_type,
                                  options_prefix=prefix + "Fp_")

        self._assemble_Fp = functools.partial(assemble,
                                              fp,
                                              tensor=self.Fp,
                                              form_compiler_parameters=fcp,
                                              mat_type=self.Fp_mat_type)
        self._assemble_Fp()
        Fpmat = self.Fp.petscmat

        # Create reaction matrix
        P2 = appctx['velocity_p_space']
        u = TrialFunction(P2)
        v = TestFunction(P2)
        velocity_bcs = appctx['velocity_bcs']
        mu = inner(u, v) * dx

        Mu = assemble(mu, form_compiler_parameters=context.fc_params,
                      mat_type=Mu_mat_type,
                      bcs=velocity_bcs,
                      options_prefix=prefix + "Mu_")

        Mumat = Mu.petscmat
        Dvec = Mumat.createVecLeft()
        Mumat.getDiagonal(Dvec)
        Dry = np.array(Dvec.array_r)
        Dvec.reciprocal()
        Dvec.sqrtabs()

        # Get this in the "tall skinny" matrix form.
        # i.e., the (0,1) block of the original matrix
        b = - p * div(v) * dx
        B = assemble(b)
        Bdsqrt = B.petscmat
        Bdsqrt.diagonalScale(L=Dvec)

        # Rp = B D^{-1} B^T
        #    = (sqrt(D^{-1}) B)^T (sqrt(D^{-1}) B)
        self.Rp = Bdsqrt.transposeMatMult(Bdsqrt) * dt

        Rksp = PETSc.KSP().create(comm=pc.comm)
        Rksp.incrementTabLevel(1, parent=pc)
        Rksp.setOptionsPrefix(prefix + "Rp_")
        Rksp.setOperators(self.Rp)
        Rksp.setUp()
        Rksp.setFromOptions()
        self.Rksp = Rksp
            
        # Create scratch vectors to hold intermediate computations
        self.workspace = [Fpmat.createVecLeft() for i in range(4)]
        
    def update(self, pc):
        try:
            self._assemble_Fp()
        except Exception as e:
            print(e)

    def apply(self, pc, X, Y):
        try:
            self._apply(pc,X,Y)
        except Exception as e:
            print(e)
        
    def _apply(self, pc, X, Y):
        MinvX, FMinvX, Pcd, R = self.workspace

        self.Mksp.solve(X, MinvX)
        self.Fp.petscmat.mult(MinvX, FMinvX)
        self.Kksp.solve(FMinvX, Pcd)

        self.Rksp.solve(X, R)

        pcdr = Pcd + R
        pcdr.copy(Y)
        Y.scale(-1.0)

    def applyTranspose(self, pc, X, Y):
        print('applyTranspose() called! This is not implemented!!')
        pass
