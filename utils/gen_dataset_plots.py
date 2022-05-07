import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Computer Modern Roman"]})

if len(sys.argv) != 3:
    print(f'usage: {sys.argv[0]} [input pickle] [output prefix]')

with open(sys.argv[1], 'rb') as f:
    ds = pickle.load(f)


baseline = ds['baseline']; baseline_avg = np.average(baseline)
lloyd = ds['lloyd']; lloyd_avg = np.average(lloyd)
ml = ds['ml']; ml_avg = np.average(ml)
ds_size = ds['sizes']
extras = ds['extras']

print(f'Smallest problem: {np.min(ds_size)};  largest problem: {np.max(ds_size)}')

all_conv = np.concatenate((baseline, lloyd, ml))
min_conv = np.min(all_conv)
max_conv = np.max(all_conv)

prefix = sys.argv[2]

figsize = (6,6)

plt.figure(figsize=figsize)
plt.scatter(baseline, ml, s=np.array(ds_size)**0.6, alpha=0.7, label='Convergence')
plt.plot([min_conv, max_conv], [min_conv, max_conv], 'tab:orange', label='Diagonal')
plt.plot([min_conv, max_conv], [ml_avg, ml_avg], 'tab:green', label='ML Average')
plt.plot([baseline_avg, baseline_avg], [min_conv, max_conv], 'tab:red', label='Lloyd Average')
plt.xlim((min_conv, max_conv))
plt.ylim((min_conv, max_conv))
plt.xlabel('Baseline Convergence')
plt.ylabel('ML Convergence')
plt.title('ML vs Baseline')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig(f'{prefix}_baseline_ml_convergence.pdf')


plt.figure(figsize=figsize)
plt.scatter(lloyd, ml, s=np.array(ds_size)**0.6, alpha=0.7, label='Convergence')
plt.plot([min_conv, max_conv], [min_conv, max_conv], 'tab:orange', label='Diagonal')
plt.plot([min_conv, max_conv], [ml_avg, ml_avg], 'tab:green', label='ML Average')
plt.plot([lloyd_avg, lloyd_avg], [min_conv, max_conv], 'tab:red', label='Lloyd Average')
plt.xlim((min_conv, max_conv))
plt.ylim((min_conv, max_conv))
plt.xlabel('Lloyd Convergence')
plt.ylabel('ML Convergence')
plt.title('ML vs Lloyd')
plt.axis('equal')
plt.grid()
plt.legend()
plt.savefig(f'{prefix}_lloyd_ml_convergence.pdf')


plt.figure(figsize=figsize)
plt.title(f'Convergence of ML and Lloyd vs Problem Size')
plt.semilogx(ds_size, lloyd, 'o', label='Lloyd Convergence')
plt.semilogx(ds_size, ml, 'o', label='ML Convergence')
# plt.semilogx(ds_size, baseline, 'o', label='Baseline Convergence')
plt.xlabel('Problem Size (DOF)')
plt.ylabel('Convergence Rate')
plt.grid()
plt.legend()
plt.savefig(f'{prefix}_convergence_per_size.pdf')


plt.figure(figsize=figsize)
plt.title(f'Relative performance of ML to Lloyd')
plt.semilogx(ds_size, ml / lloyd, 'o', label='Relative Performance')
plt.xlabel('Problem Size (DOF)')
plt.ylabel('Ratio of ML to Lloyd convergence')
plt.grid()
plt.legend()
plt.savefig(f'{prefix}_rel_perf.pdf')


if 'theta' in extras[0] and 'epsilon' in extras[0]:
    epsilons = np.array(list(map(lambda x: x['epsilon'], extras)))
    thetas = np.array(list(map(lambda x: x['epsilon'], extras)))

    plt.figure(figsize=figsize)
    plt.title(f'Convergence of Lloyd and ML vs Magnitude of Anisotropy')
    plt.semilogx(epsilons, baseline, 'o', label='Baseline Convergence')
    plt.semilogx(epsilons, ml, 'o', label='ML Convergence')
    plt.xlabel('Epsilon')
    plt.ylabel('Convergence Rate')
    plt.grid()
    plt.legend()
    plt.savefig(f'{prefix}_convergence_per_epsilon.pdf')

    plt.figure(figsize=figsize)
    plt.title(f'Convergence of Lloyd and ML vs Rotation of Anisotropy')
    plt.plot(thetas, baseline, 'o', label='Lloyd Convergence')
    plt.plot(thetas, ml, 'o', label='ML Convergence')
    plt.xlabel('Theta')
    plt.ylabel('Convergence Rate')
    plt.grid()
    plt.legend()
    plt.savefig(f'{prefix}_convergence_per_theta.pdf')

if 'theta_y' in extras[0] and 'theta_z' in extras[0]:
    theta_z = np.array(list(map(lambda x: x['theta_z'], extras)))
    theta_y = np.array(list(map(lambda x: x['theta_y'], extras)))

    eps_x = np.array(list(map(lambda x: x['eps_x'], extras)))
    eps_y = np.array(list(map(lambda x: x['eps_y'], extras)))
    eps_z = np.array(list(map(lambda x: x['eps_z'], extras)))

    eigs = np.column_stack((eps_x, eps_y, eps_z))
    cond = np.max(eigs, axis=1) / np.min(eigs, axis=1)

    plt.figure(figsize=figsize)
    plt.title(f'Convergence of Lloyd and ML vs Conditioning of Diffusion Tensor')
    plt.semilogx(cond, baseline, 'o', label='Lloyd Convergence')
    plt.semilogx(cond, ml, 'o', label='ML Convergence')
    plt.xlabel(r'$\left\|{\bf D}\right\|_2\left\|{\bf D}^{-1}\right\|_2$')
    plt.ylabel('Convergence Rate')
    plt.grid()
    plt.legend()
    plt.savefig(f'{prefix}_convergence_per_cond.pdf')
