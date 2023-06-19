import numpy as np
import matplotlib.pyplot as plt

from numba import njit, jit, prange

import fun


L = 1620
nn_mat = fun.return_nn(L)
beta = (1.)/2.26918531421

generate_initial_conf = False

if generate_initial_conf:
    lattice = np.random.choice([-1,1], size = L**2).reshape(L,L)
    lattice = lattice.astype(np.int8)

    lattice = fun.wolff_wrapper(lattice, nn_mat, beta, Nsteps = int(1e6))
    np.save('data/IsingCritConf.npy', lattice)

else:
    lattice = np.load('data/IsingCritConf.npy')

NBatch = 1000
nconf = 32

for batch in range(NBatch):
    print('Batch: ', batch)
    if batch == 0:
        equilibrium_conf = fun.generate_configurations(lattice, nn_mat, beta, nconf = nconf, Nsteps = 50)
    else:
        equilibrium_conf = fun.generate_configurations(equilibrium_conf[-1], nn_mat, beta, nconf = nconf, Nsteps = 50)

    np.save('data/all_confs/IsingCritEquilibriumConf_batch' + str(batch), equilibrium_conf)