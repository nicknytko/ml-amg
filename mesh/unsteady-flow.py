from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import sys
import os
import time

save_frames = False
if save_frames:
    if not os.path.exists('frames'):
        os.makedirs('frames')

M = Mesh('wide_cyl_small.msh')
x, y = SpatialCoordinate(M)
print('Vertices in Mesh', M.num_vertices())
print('Edges in mesh', M.num_edges())

# Set up function spaces
P2 = VectorFunctionSpace(M, "CG", 2) # Velocity
P1 = FunctionSpace(M, "CG", 1) # Pressure
TH = P2 * P1

up = Function(TH)
u, p = split(up)

up0 = Function(TH)
u0, p0 = split(up0)

v, q = TestFunctions(TH)

# Constants
U_r = 5 # Inlet velocity
Re = 62 # This isn't necessarily the true Reynolds number because we dont have unit inlet velocity or inlet length. /shrug
Re_inv = Constant(1.0/Re)
dt = 0.01
dt_inv = Constant(1.0/dt)

# Use implicit euler for time discretization
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
    DirichletBC(TH.sub(0), Constant((U_r, 0)), [6]), # Inlet velocity = (U_r,0)
    DirichletBC(TH.sub(0), Constant((0, 0)), [8])  # No-slip walls
]

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
    # "ksp_view": None,
    # "ksp_monitor": None,
    # "snes_monitor": None,

    # Use PETSC's fieldsplit preconditioner to "invert" the matrix blocks
    "mat_type": "matfree",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "pc_type": "fieldsplit",

    # Momentum block
    "fieldsplit_0_pc_type": "python",
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_python_type": "firedrake.AssembledPC",
    "fieldsplit_0_assembled_pc_type": "lu",

    # Schur complement
    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_ksp_rtol": 1e-4,
}

up0.assign(up)
u,p = up.split()

T = 10
n_t = int(T / dt)
vmin = 0
vmax = 8

start_time = time.time()

for i in range(1, n_t+1):
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
        plt.savefig(f'/frames/ns_{i:03d}.png', dpi=200)
