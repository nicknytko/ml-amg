from firedrake import *
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))
import ns

save_frames = False

M = Mesh('../mesh/wide_cyl_small.msh') # Using low-res mesh for testing
# M = Mesh('../mesh/wide_cyl.msh')
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

appctx = {
    'Re': Re,
    'dt': dt,
    'velocity_space': 0,
    'velocity_p_space': P2,
    'velocity_functions': [u, v],
    'velocity_bcs': bcs_U
}
solver_params = {
    "ksp_type": "gmres",
    "ksp_gmres_modifiedgramschmidt": None,
    "ksp_monitor_true_residual": None,
    "snes_monitor": None,
    "ksp_view": None,

    "mat_type": "matfree",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "lower",
    "pc_type": "fieldsplit",

    "fieldsplit_0_ksp_type": "gmres",
    "fieldsplit_0_pc_type": "python",
    #"fieldsplit_0_pc_python_type": "ns.preconditioner.PyAMG",
    "fieldsplit_0_pc_type": "jacobi",
    
    "fieldsplit_1_ksp_type": "gmres",
    "fieldsplit_1_ksp_rtol": 1e-8,
    # "fieldsplit_1_pc_type": "jacobi", # testing...
    "fieldsplit_1_pc_type": "python",
    "fieldsplit_1_pc_python_type": "ns.preconditioner.PCDR",
    "fieldsplit_1_pcdr_Kp_pc_type": "lu",
    "fieldsplit_1_pcdr_Kp_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_pcdr_Rp_pc_type": "lu",
    "fieldsplit_1_pcdr_Rp_pc_factor_mat_solver_type": "mumps",
    "fieldsplit_1_pcdr_Mp_pc_type": "lu",
    "fieldsplit_1_pcdr_Mp_pc_factor_mat_solver_type": "mumps",
}

up0.assign(up)
u,p = up.split()

T = 5
n_t = int(T / dt)
vmin = 0
vmax = 8

for i in range(1, n_t+1):
    solve(F == 0, up, bcs=bcs_U, solver_parameters=solver_params, appctx=appctx)
    up0.assign(up)
    print('timestep', i)
    
    plt.clf()
    u,p = up.split()
    tripcolor(u, cmap='plasma', axes=plt.gca(), vmin=vmin, vmax=vmax)
    plt.title(f't={dt*i:.3f}')
    plt.gca().set_aspect('equal')
    plt.pause(0.01)
    
    if save_frames:
        plt.savefig(f'frames/ns_{i:03d}.png', dpi=200)

