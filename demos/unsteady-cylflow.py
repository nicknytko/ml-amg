from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import sys
import os
import time

sys.path.append(os.path.dirname(os.getcwd()))
import ns
import ns.preconditioner.PyAMG

save_frames = True

M = Mesh('../mesh/wide_cyl_tiny.msh')
#M = Mesh('../mesh/wide_cyl_small.msh')
x, y = SpatialCoordinate(M)
print('Vertices in Mesh', M.num_vertices())
print('Edges in mesh', M.num_edges())

# Set up function spaces
P2 = VectorFunctionSpace(M, "CG", 2) # Velocity
P1 = FunctionSpace(M, "CG", 1) # Pressure
TaylorHood = P2 * P1

up = Function(TaylorHood)
u, p = split(up)

up0 = Function(TaylorHood)
u0, p0 = split(up0)

v, q = TestFunctions(TaylorHood)

# Constants
# "True" reynolds number
Re_t = 500.0
U_r = 2 # Inlet velocity
L_r = 4 # Inlet width
# Re_{eff} = Re / (U_r L_r)
# "effective" reynolds number.  Since I'm lazy, just take above
# and divide by U_rL_r so we just need to take inverse in formulation
Re = Re_t / (U_r * L_r)
print('Re', Re_t, '"Effective" Re', Re)
Re_inv = Constant(1.0/Re)
dt = 0.005
dt_inv = Constant(1.0/dt)

# Use euler backward for time discretization
F = (
    dt_inv * inner(u - u0, v) * dx +
    Re_inv * inner(grad(u), grad(v)) * dx +
    inner(dot(grad(u), u), v) * dx -
    p * div(v) * dx +
    div(u) * q * dx
)

# Boundary conditions
inlet_vel = U_r
bcs_U = [
    DirichletBC(TaylorHood.sub(0), Constant((5, 0)), [6]), # Inlet velocity = (1,0)
    DirichletBC(TaylorHood.sub(0), Constant((0, 0)), [8])  # No-slip walls
]
nullspace = MixedVectorSpaceBasis(
    TaylorHood, [TaylorHood.sub(0), VectorSpaceBasis(constant=True)])

appctx = {
    'Re': Re,
    'dt': dt,
    'velocity_space': 0,
    'velocity_p_space': P2,
    'velocity_functions': [u, v],
    'velocity_bcs': bcs_U
}
solver_params_gmres_only = {
    "ksp_type": "gmres"
}
solver_params = {
    "ksp_type": "fgmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_gmres_restart": 100,
    # Debug output, uncomment if you want to see SNES and linear solve residuals
    # "ksp_monitor_true_residual": None,
    "ksp_monitor": None,
    "snes_monitor": None,
    #"ksp_view": None,

    # Use PETSC's fieldsplit preconditioner to "invert" the matrix blocks
    "mat_type": "matfree",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "pc_type": "fieldsplit",

    # Momentum block
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "python",

#    "fieldsplit_0_assembled_pc_python_type": "ns.preconditioner.MLAMG",
    "fieldsplit_0_assembled_mlamg_amg_rtol": 1e-8,
    "fieldsplit_0_assembled_mlamg_pnet_model": "../models/model_epoch_9.pth",

    "fieldsplit_0_assembled_pc_python_type": "ns.preconditioner.MLAMG",
    "fieldsplit_0_assembled_pyamg_amg_rtol": 1e-8,
    "fieldpslit_0_assembled_pyamg_amg_max_levels": 2,
    "fieldsplit_0_assembled_pyamg_amg_precondition_with_gmres": False,

    # Schur complement
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "ns.preconditioner.PCDR",  # Our nice PCD-R preconditioner :)

    # Pressure laplacian
    "fieldsplit_1_pcdr_Kp_pc_type": "lu",
    "fieldsplit_1_pcdr_Kp_ksp_type": "preonly",

    # reaction
    "fieldsplit_1_pcdr_Rp_pc_type": "lu",
    "fieldsplit_1_pcdr_Rp_ksp_type": "preonly",

    # Pressure mass
    "fieldsplit_1_pcdr_Mp_pc_type": "lu",
    "fieldsplit_1_pcdr_Mp_ksp_type": "preonly",

    # Pressure convection-diffusion matrix
    "fieldsplit_1_pcdr_Fp_mat_type": "matfree",
}

up0.assign(up)
u,p = up.split()

T = 5
n_t = int(T / dt)
#n_t = 1
vmin = 0
vmax = 8

start_time = time.time()
dump_mats_modulus = 20

for i in range(1, n_t+1):
    ns.preconditioner.PyAMG._dump_mats = (i % dump_mats_modulus == 0)
    solve(F == 0, up, bcs=bcs_U, solver_parameters=solver_params, appctx=appctx)
    up0.assign(up)
    print(f'Timestep {i:03} -- elapsed {time.time()-start_time:.2f}s')

    plt.clf()
    u,p = up.split()
    tripcolor(u, cmap='plasma', axes=plt.gca(), vmin=vmin, vmax=vmax)
    plt.title(f't={dt*i:.3f}')
    plt.gca().set_aspect('equal')
    plt.pause(0.01)

    if save_frames:
        plt.savefig(f'../frames/ns_{i:03d}.png', dpi=200)
