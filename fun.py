import numpy as np
import matplotlib.pyplot as plt

from numba import njit, jit, prange
from scipy.signal import convolve2d

@njit
def return_nn(L):
    """
    Creates the nearest neighbors matrix, a LxLx4x2 array so each
    site of the LxL sublattice is a 4x2 array that contains the
    indexes of the nns.

    Parameters
    ----------
    L : int
        Linear size of the lattice.

    Returns
    -------
    nnlist : LxLx4x2 array
        Array containing the indexes of the nearest neighbors of each
        site of the LxL sublattice.
    """
    L = np.int64(L)
    nnlist = np.zeros((L, L, np.int64(4), np.int64(2)), dtype = np.int64)
    
    moves_list = [np.array([-1,0]), np.array([0,+1]),
                  np.array([+1,0]), np.array([0,-1])]
    
    for site, _ in np.ndenumerate(np.zeros((L,L))):
        nn = np.zeros((4,2))
        for idx, move in enumerate(moves_list):
            nn[idx] = (np.array(site) + move)%L
        nnlist[site] = nn
    
    return nnlist

@njit
def wolff_flip(lattice, nn_mat, p, L):
    """
    Build and flip a cluster using the Wolff algorithm.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration to be updated.
    nn_mat : LxLx4x2 array
        Array containing the indexes of the nearest neighbors of each
        site of the LxL sublattice.
    p : float
        Probability of adding a spin to the cluster.
    L : int
        Linear size of the lattice.

    Returns
    -------
    lattice : LxL array
        Updated Ising configuration.
    """
    start_x = np.random.randint(0, L-1)
    start_y = np.random.randint(0, L-1)
    
    cluster = np.ones((L, L), dtype = np.int8)
    cluster[start_x][start_y] = -1
    to_visit = [[start_x, start_y]]
    
    while len(to_visit) > 0:
        i = to_visit[0][0]
        j = to_visit[0][1]
                
        for nn in nn_mat[i][j]:
            nn_spin = lattice[nn[0]][nn[1]]
            if (cluster[nn[0]][nn[1]] == 1) and (nn_spin == lattice[i][j]) and (np.random.rand() < p):
                to_visit.append([nn[0], nn[1]])
                cluster[nn[0]][nn[1]] = -1
        to_visit = to_visit[1:]
            
    return lattice*cluster

@njit
def wolff_wrapper(lattice, nn_mat, beta, Nsteps = 10000, printidx = True):
    """
    Generate a new Ising configuration using the Wolff algorithm.

    Parameters
    ----------
    lattice : LxL array
        Initial Ising configuration.
    nn_mat : LxLx4x2 array
        Array containing the indexes of the nearest neighbors of each
        site of the LxL sublattice.
    beta : float
        Inverse temperature.
    Nsteps : int, optional
        Number of steps of the Wolff algorithm. The default is 10000.
    printidx : bool, optional
        If True, print the step number. The default is True.

    Returns
    -------
    lattice : LxL array
        Sampled Ising configuration.
    """
    L = lattice.shape[0]
    p = 1-np.exp(-2*beta)

    for i in range(Nsteps):
        if i % 1000 == 0:
            if printidx:
                print(i)
        lattice = wolff_flip(lattice, nn_mat, p, L)

    return lattice

@njit(parallel = True)
def generate_configurations(lattice_stationary, nn_mat, beta, nconf = 100, Nsteps = 100, printidx = False):
    """
    Sample Ising configurations using the Wolff algorithm.
    
    Parameters
    ----------
    lattice_stationary : LxL array  
        Initial Ising configuration.
    nn_mat : LxLx4x2 array
        Array containing the indexes of the nearest neighbors of each
        site of the LxL sublattice.
    beta : float
        Inverse temperature.
    nconf : int, optional
        Number of configurations to be sampled. The default is 100.
    Nsteps : int, optional
        Number of steps of the Wolff algorithm. The default is 100.
    printidx : bool, optional
        If True, print the step number. The default is False.

    Returns
    -------
    data : nconfxLxL array
        Array containing the sampled Ising configurations.
    """
    L = lattice_stationary.shape[0]
    data = np.empty((nconf, L, L), dtype = np.int8)

    for idx in prange(nconf):
        data[idx] = wolff_wrapper(lattice_stationary.copy(), nn_mat, beta, Nsteps = Nsteps, printidx = printidx).astype(np.int8)
    return data


def majority_rule(cg_mat, block_size):
    """
    Apply the majority rule to a coarse-grained matrix.

    Parameters
    ----------
    cg_mat : LxL array
        Coarse-grained matrix.
    block_size : int
        Size of the blocks.
    
    Returns
    -------
    cg_mat : LxL array
        Renormalized matrix.
    """

    if block_size % 2 != 0:
        cg_mat[cg_mat < 0] = -1
        cg_mat[cg_mat > 0] = +1
    else:
        mask = (cg_mat == 0)
        cg_mat[cg_mat < 0] = -1
        cg_mat[cg_mat > 0] = +1
        cg_mat[mask] = np.random.choice([-1,1], size = np.where(mask == True)[0].size)
    return cg_mat

def block_transform(mat, block_size):
    """
    Renormalizes a matrix by applying the majority rule to blocks of size
    block_size x block_size.

    Parameters
    ----------
    mat : LxL array
        Matrix to be renormalized.
    block_size : int
        Size of the blocks.

    Returns
    -------
    mat : LxL array
        Renormalized matrix.
    """
    shape = tuple([i//block_size for i in mat.shape])
    
    sh = shape[0],mat.shape[0]//shape[0],shape[1],mat.shape[1]//shape[1]
    cutoff = (mat.shape[0] - (mat.shape[0] % block_size),
              mat.shape[1] - (mat.shape[1] % block_size))
    
    mat = mat[:cutoff[0], :cutoff[1]].reshape(sh).sum(-1).sum(1)
    
    return majority_rule(mat, block_size)

def sum_four(lattice):
    """
    Sum of the 2x2 plaquette in the lattice.
    This is an even coupling.
    
    Parameters
    ----------
    lattice : LxL array 
        Ising configuration.

    Returns
    -------
    b : LxL array
        Sum of the 2x2 plaquette in the lattice.
    """
    b = np.pad(lattice, 1, 'wrap')[:-1,:-1]
    b = b[1:,:]*b[:-1,:]
    b = b[:,1:]*b[:,:-1]

    return np.sum(b)

def sum_six(lattice):
    """
    Sum of the 2x3 plaquette in the lattice.
    This is an even coupling.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration.
    
    Returns
    -------
    b : LxL array
        Sum of the 2x3 plaquette in the lattice.
    """
    b = np.pad(lattice, 2, 'wrap')[:-2,:-3]
    b = b[:-2,:]*b[1:-1,:]*b[2:,:]
    b = b[:,1:]*b[:,:-1]

    return np.sum(b)

def sum_six_ver(lattice):
    """
    Sum of the 3x2 plaquette in the lattice.
    This is an even coupling.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration.

    Returns
    -------
    b : LxL array
        Sum of the 3x2 plaquette in the lattice.
    """
    b = np.pad(lattice, 2, 'wrap')[:-3,:-2]
    b = b[:,:-2]*b[:,1:-1]*b[:,2:]
    b = b[1:,:]*b[:-1,:]
    return np.sum(b)

def sum_eight(lattice):
    """
    Sum of the 2x4 plaquette in the lattice.
    This is an even coupling.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration.

    Returns
    -------
    b : LxL array
        Sum of the 2x4 plaquette in the lattice.
    """
    b = np.pad(lattice, 3, 'wrap')[:-5,:-3]
    b = b[1:,:]*b[:-1,:]
    b = b[:,:-3]*b[:,1:-2]*b[:,2:-1]*b[:,3:]
    return np.sum(b)

def sum_three_hor(lattice):
    """
    Sum of the 3 spin interaction along the horizontal direction
    in the lattice.
    This is an odd coupling.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration.

    Returns
    -------
    a : LxL array
        Sum of the 3 spin interaction along the horizontal direction
    """
    a = np.pad(lattice, 1, 'wrap')[:-2,:]
    a = a[:,:-2]*a[:,1:-1]*a[:, 2:]
    return np.sum(a)

def sum_three_ver(lattice):
    """
    Sum of the 3 spin interaction along the vertical direction
    in the lattice.
    This is an odd coupling.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration.

    Returns
    -------
    a : LxL array
        Sum of the 3 spin interaction along the vertical direction
    """

    a = np.pad(lattice, 1, 'wrap')[:,:-2]
    a = a[:-2,:]*a[1:-1,:]*a[2:,:]
    return np.sum(a)

def sum_five_hor(lattice):
    """
    Sum of the 5 spin interaction along the horizontal direction
    in the lattice.
    This is an odd coupling.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration.

    Returns
    -------
    a : LxL array
        Sum of the 5 spin interaction along the horizontal direction
    """

    a = np.pad(lattice, 2, 'wrap')[:-4,:]
    a = a[:,:-4]*a[:,1:-3]*a[:,2:-2]*a[:,3:-1]*a[:,4:]
    return np.sum(a)

def sum_five_ver(lattice):
    """
    Sum of the 5 spin interaction along the vertical direction
    in the lattice.
    This is an odd coupling.

    Parameters
    ----------
    lattice : LxL array
        Ising configuration.

    Returns
    -------
    a : LxL array
        Sum of the 5 spin interaction along the vertical direction
    """

    a = np.pad(lattice, 2, 'wrap')[:,:-4]
    a = a[:-4,:]*a[1:-3,:]*a[2:-2,:]*a[3:-1,:]*a[4:,:]
    return np.sum(a)

def cov(x,y):
    """
    Returns the covariance between x and y.

    Parameters
    ----------
    x : array
        First array.
    y : array
        Second array.

    Returns
    -------
    float
        Covariance between x and y.
    """
    return np.sum((x - x.mean())*(y-y.mean()))/(x.shape[0])

def find_spin_products(data):
    """
    Find all spin products given a sequence of Ising configurations.
    The spin products are the following:
        - 1 spin interaction (odd)
        - 3 spin horizontal interaction (odd)
        - 3 spin vertical interaction (odd)
        - 5 spin horizontal interaction (odd)
        - 5 spin vertical interaction (odd)
        - nearest neighbors interaction (even)
        - next nearest neighbors interaction (even)
        - next-next nearest neighbors interaction,
          along the horizontal/vertical directions (even)
        - next-next nearest neighbors interaction, with
          pad 1 (even)
        - next-next nearest neighbors interaction, along
          the diagonal directions (even)
        - 2x2 plaquette (even)
        - 2x3 plaquette (even)
        - 2x4 plaquette (even)

    Parameters
    ----------
    data : NxLxL array
        Sequence of Ising configurations.

    Returns
    -------
    spin_products : Nx13 array
        Sequence of spin products.
    """
    spin_products = np.zeros((data.shape[0], 13), dtype = np.int64)
    
    kernel_nn = np.array([[0,1,0],
                          [1,0,1],
                          [0,1,0]])
    
    kernel_nnn = np.array([[1,0,1],
                           [0,0,0],
                           [1,0,1]])
        
    kernel_3n = np.array([[0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 1, 0, 0]])
        
    kernel_4n = np.array([[0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0]])
    
    kernel_5n = np.array([[1, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0],
                          [1, 0, 0, 0, 1]])

    
    for idx, conf in enumerate(data):
        spin_products[idx][0] = np.sum(conf)
        spin_products[idx][1] = sum_three_hor(conf)
        spin_products[idx][2] = sum_three_ver(conf)
        spin_products[idx][3] = sum_five_hor(conf)
        spin_products[idx][4] = sum_five_ver(conf)
        spin_products[idx][5] = np.sum(conf*convolve2d(conf, kernel_nn, mode = 'same', boundary = 'wrap'))/2
        spin_products[idx][6] = np.sum(conf*convolve2d(conf, kernel_nnn, mode = 'same', boundary = 'wrap'))/2
        spin_products[idx][7] = np.sum(conf*convolve2d(conf, kernel_3n, mode = 'same', boundary = 'wrap'))/2
        spin_products[idx][8] = np.sum(conf*convolve2d(conf, kernel_4n, mode = 'same', boundary = 'wrap'))/2
        spin_products[idx][9] = np.sum(conf*convolve2d(conf, kernel_5n, mode = 'same', boundary = 'wrap'))/2
        spin_products[idx][10] = sum_four(conf)
        spin_products[idx][11] = sum_six(conf)
        spin_products[idx][12] = sum_eight(conf)
        
    return spin_products

def get_even_odd_eigvals(CG_T_mat):
    """
    Finds the eigenvalues of the linearized RG transformation
    for the even and odd couplings.

    Parameters
    ----------
    CG_T_mat : Nx13x13 array
        Linearized RG transformation.

    Returns
    -------
    CG_EigEven : Nx8 array
        Eigenvalues of the even couplings.
    CG_EigOdd : Nx5 array
        Eigenvalues of the odd couplings.
    """
    CG_EigEven = []
    CG_EigOdd = []

    for i, T in enumerate(CG_T_mat):
        eigvals, eigvecs = np.linalg.eig(T)

        ye = []
        for idx in range(8):
            ye.append(np.linalg.eigvals(T[5:6+idx, 5:6+idx])[0])

        CG_EigEven.append(ye)

        yo = []
        for idx in range(5):
            yo.append(np.linalg.eigvals(T[:1+idx, :1+idx])[0])
        CG_EigOdd.append(yo)

    CG_EigEven = np.array(CG_EigEven)
    CG_EigOdd = np.array(CG_EigOdd)

    return CG_EigEven, CG_EigOdd