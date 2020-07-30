import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from multiprocessing import Pool
from functools import partial


class EulerBernoulli:
    """ Computes the displacement of an Euler-Bernoulli beam using a
        second order finite difference scheme.  The beam is assumed
        to be in a cantilever configuration, where the left boundary
        is fixed (u=0, du/dx=0) and the right boundary is  free (d2u/dx2=0, d3u/dx3=0).

        This class can be used in two ways:
        1) Setting the stiffness in the constructor, in which case the ModPiece
           will have a single input: the load.  In this case, the stiffness
           matrix is precomputed and can be accessed
        2) Setting the stiffness during the call to Evaluate, in which case
           there will be two inputs: [load, stiffness]

    """

    def __init__(self, numNodes, length, radius, constMod=np.array([])):
        self.numNodes = numNodes

        self.dx = length/(numNodes-1)

        # Moment of inertia assuming cylindrical beam
        self.I = np.pi/4.0*radius**4

        self.K = self.BuildK(constMod)



    def BuildK(self, modulus):
        # Number of nodes in the finite difference stencil
        n = self.numNodes

        # Create stiffness matrix
        K = np.zeros((n, n))

        # Build stiffness matrix (center)
        for i in range(2, n-2):
            K[i,i+2] = modulus[i]
            K[i,i+1] = modulus[i+1] - 6.0*modulus[i] + modulus[i-1]
            K[i,i]   = -2.0*modulus[i+1] + 10.0*modulus[i] - 2.0*modulus[i-1]
            K[i,i-1] = modulus[i+1] - 6.0*modulus[i] + modulus[i-1]
            K[i,i-2] = modulus[i]

        # Set row i == 1
        K[1,3] = modulus[1]
        K[1,2] = modulus[2] - 6.0*modulus[1] + modulus[0]
        K[1,1] = -2.0*modulus[2] + 11.0*modulus[1] - 2.0*modulus[0]

        # Set row i == n - 2
        K[n-2,n-1] = modulus[n-1] - 4.0*modulus[n-2] + modulus[n-3]
        K[n-2,n-2] = -2.0*modulus[n-1] + 9.0*modulus[n-2] - 2.0*modulus[n-3]
        K[n-2,n-3] = modulus[n-1] - 6.0*modulus[n-2] + modulus[n-3]
        K[n-2,n-4] = modulus[n-2]

        # Set row i == n - 1 (last row)
        K[n-1,n-1] = 2.0*modulus[n-1]
        K[n-1,n-2] = -4.0*modulus[n-1]
        K[n-1,n-3] = 2.0*modulus[n-1]

        # Apply dirichlet BC (w=0 at x=0)
        K[0,:] = 0.0; K[:,0] = 0.0
        K[0,0] = 1

        K /= self.dx**4

        sparseK = sparse.csr_matrix(K, shape=(n,n))
        return sparseK

"""
The stiffness coefficient.

x is the parameters of shape (d, ) for d >= 1
"""
def stiffness(x, t):
    d = len(x)

    x = np.abs(x) # Enforce that the diffusivity is positive.

    # Smoothing function.
    I = lambda y,a: 1.0/(1.0 + np.exp(-(y - a)/0.005))

    E = np.ones(len(t))

    if d == 1:
        # Multiply by the parameter squared to make it nonlinear.
        return E * x**2

    E *= x[0]

    for i in range(1, d):
        E = (1 - I(t, i/d))*E + I(t, i/d)*x[i]
    return E

"""
Set up the forward model. t is the spatial variable and x is the parameters.
"""
def fun(x, d_out, level):

    numPts = level # Number of finite difference points.

    E = lambda t : stiffness(x, t)
    t = np.linspace(0, 1, numPts)

    beam = EulerBernoulli(numPts, length=1, radius=0.1, constMod = E(t))
    load = np.ones(numPts)

    u = linalg.spsolve(beam.K, load)

    # Return the displacement at the end point.
    xx = np.linspace(0, 1, numPts)
    uu = np.interp(np.linspace(0,1,d_out), xx, u)

    return uu

"""
Evaluate the forward model many times.

Takes in (N,dim_in) and outputs (N, dim_out).
"""
def parameter_to_observation_map(x, d_out, lvl, nproc):
    num_proc = nproc
    N = len(x)

    if num_proc == 1:
        evals = np.zeros((N, d_out))
        for i in range(N):
            evals[i] = fun(x[i], d_out, level = lvl)
    else:
        model = partial(fun, d_out = d_out, level = lvl) # Needed since there are 2 arguments.
        with Pool(num_proc) as p:
            evals = p.map(model, x.tolist())
        evals = np.asarray(evals)   # Convert to numpy array.
    return evals
