import argparse
import numpy as np
import scipy.io as sio
import sys
import time
import pyamg
import multiprocessing
import os
import os.path as path
import itertools
from tqdm import tqdm

sys.path.append(os.path.dirname(os.getcwd()))
import ns.lib.helpers as helpers
from ns.lib.multigrid import *

parser = argparse.ArgumentParser(description='Randomly generates 2D C/F split grids.  Takes various reference grids and generates random permutations by flipping points at various probabilities')
parser.add_argument('--iterations', type=int, default=5, help='Number of permutations for each probability', required=False)
parser.add_argument('--mgtrials', type=int, default=20, help='Number of multigrid trials to run for each problem', required=False)
parser.add_argument('--maxthreads', type=int, default=16, help='Maximum number of worker processes to use when generating perturbations', required=False)
parser.add_argument('--randseed', type=int, default=None, help='Random seed to use when perturbing grids', required=False)

args = vars(parser.parse_args())
I = args['iterations']
maxthreads = max(1, args['maxthreads'])

out_dir = path.join('..', 'out_grids')
mat_dir = path.join('..', 'out_matrices')
mat_names = list(filter(lambda x: x.endswith('.mat'), os.listdir(mat_dir)))
num_mats = len(mat_names)

# grid generation
coarsenings = [ref_all_fine, ref_all_coarse, ref_amg(theta=0.50), ref_coarsen_by_bfs(2), ref_coarsen_by_bfs(3), ref_coarsen_by_bfs(4), ref_coarsen_by_bfs(5)]
p_trials = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75])
trial_seeds = None


def run_trial(indices):
    midx, cidx, pidx = indices
    grids = []
    convs = []
    A = sio.loadmat(path.join(mat_dir, mat_names[midx]))['A'].tocsr()
    N = A.shape[0]
    np.random.seed(trial_seeds[midx])

    coarsening = coarsenings[cidx]
    name, G = coarsening(A)
    p = p_trials[pidx]
    for i in range(I):
        def permute_grid():
            # Automatically try again for degenerate cases
            while True:
                perm_G = G.copy()
                for j in range(N):
                    if np.random.rand(1) < p:
                        perm_G[j] = not perm_G[j]
                if not np.all(perm_G == False):
                    P = create_interp(A, perm_G)
                    break

            return perm_G, P

        # Create the randomly permuted "perm_C"
        perm_G, P = permute_grid()
        conv = mgavg(P, A, N=args['mgtrials'], omega=0.66)
        grids.append((perm_G * 2) - 1)
        convs.append(conv)

    return grids, convs, mat_names[midx]


if __name__ == '__main__':
    grids = []
    convs = []
    matnames = []

    t_start = time.time()

    # Generate a random seed using the "master seed" that gets distributed to each runner
    np.random.seed(args['randseed'])
    int_max = np.iinfo(np.int32).max
    trial_seeds = (np.random.random_sample(num_mats)*int_max).astype(np.int32)

    midx = range(num_mats)
    cidx = range(len(coarsenings))
    pidx = range(len(p_trials))
    indices = itertools.product(midx,cidx,pidx)
    num_indices = len(midx) * len(cidx) * len(pidx)

    threads = min(maxthreads, num_indices)
    pool = multiprocessing.Pool(processes=threads)
    outs_iter = pool.imap_unordered(run_trial, indices)

    completed = 0
    for out in tqdm(outs_iter, total=num_indices):
        g, c, m = out
        grids += g
        convs += c
        matnames += [m] * len(g)

    pool.close()

    outputs = [
        {
            'var': grids,
            'fname': path.join(out_dir, 'output-splittings.pkl.bz2')
        },
        {
            'var': convs,
            'fname': path.join(out_dir, 'output-conv.pkl.bz2')
        },
        {
            'var': matnames,
            'fname': path.join(out_dir, 'output-matnames.pkl.bz2')
        }
    ]

    for o in outputs:
        var = o['var']
        fname = o['fname']
        try:
            existing = helpers.pickle_load_bz2(fname)
            existing = existing + var
            helpers.pickle_save_bz2(fname, existing)
        except Exception:
            existing = var
            helpers.pickle_save_bz2(fname, existing)

    t_end = time.time()
    print(f'finished in {int(t_end-t_start)} seconds')
