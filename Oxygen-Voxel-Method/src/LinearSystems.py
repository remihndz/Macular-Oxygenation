import numpy as np
from utils import ControlVolumes, PositivePart, Grid
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from graph_tool.generation import lattice
from graph_tool.spectral import incidence
from graph_tool.util import find_edge
from itertools import chain 
import psutil


def ConvectionMatrices(CV):
    C1 = incidence(CV._vessels).T
    Din = sp.dia_matrix(((CV._vessels.get_in_degrees(list(CV._vessels.iter_vertices()))==0).astype(int), 0),
                        shape=(CV._vessels.num_vertices(), CV._vessels.num_vertices()))
    Dout = sp.dia_matrix(((CV._vessels.get_out_degrees(CV._vessels.get_vertices())==0).astype(int), 0),
                        shape=(CV._vessels.num_vertices(), CV._vessels.num_vertices()))    
    
    v = CV._vessels.ep['q'].a / (np.pi*(CV._vessels.ep['radius'].a**2)) # Flow velocity m/s
    l = CV._vessels.ep['length'].a # Vessel length m
    vol = np.pi*(CV._vessels.ep['radius'].a**2) * l # Volume of the cylinder m^3

    L = sp.dia_matrix((vol/(0.5*l), 0), shape=2*[CV._vessels.num_edges()]) # From the discretization of the divergence we get 1./l
                                                                           # To convert concentrations to moles, we multiply by volume of cylinder, assuming the cylinder is full of solute (blood)
    V = sp.dia_matrix((v, 0), shape=2*[CV._vessels.num_edges()]) # Flow velocity
    # Min  = Din @ C1.T @ Q @ (PositivePart(-C1))
    Min = Din
    M = C1.T@(PositivePart(L@V)@PositivePart(C1) - PositivePart(-L@V)@PositivePart(-C1))
    # MOut = Dout @ C1.T @ Q @ PositivePart(C1)
    # M -= MOut
    return M.T.tocsr(), Min.T.tocsr(), Din.tocsr()

def MakeCSRBinaryMatrix(data, nrows:int, ncols:int):
    '''
    Quickly assemble a CSR matrix where the non-zero elements for each row are
    given by their column number.

    ### Parameters
    - data: list[list[int]]
        The non-zero elements of the matrix. 
        Each list correspond to a row and contains the column number of the nnz entry.
    - nrows, ncols: int
        The number of rows and columns of the matrix.
    '''
    # indices = np.concatenate(data)
    indices = np.fromiter(chain.from_iterable(data), dtype=np.int32)
    # row_indices = np.repeat(np.arange(len(data)), [len(row) for row in data])
    values = np.ones_like(indices)
    indptr = np.concatenate([[0], np.cumsum(np.array([len(row) for row in data]))])    
    return sp.csr_matrix((values, indices, indptr), shape=(nrows,ncols))

def MakeCouplingMatrices(CV:ControlVolumes, h_v:float):
    """
    Creates the coupling matrices C2v and C2t. 
    For a grid with nc 'endothelial connections' between the nv control volumes
    and the nt voxels:
        -C2v is a nc-by-nv binary matrix that 
            maps the control volumes to connections 
            with the voxels.
        -C2t is a nc-by-nt matrix that maps
            the voxels to their connections with
            the control volumes.
        -S is a nc-by-nc diagonal matrix that scales the exchange with the surface 
            area of the control volume and the number of connections to that
            control volumne.
    
    ### Parameters
    - CV: ControlVolumes
        The control volume class. Contains the vessel graph and grid information.
    - h_v: float
        The vessel wall thickness. Controls the discretization of the reference
        cylinder during labelling.

    ### Returns
    - scipy sparse matrices C2v, C2t and S.
    """
    C2v, C2t, N = CV.LabelMesh(h_v)
    S = MakeSurfaceScaling(CV, N)
    assert S.shape[0]==C2v.shape[0], "Dimension mismatch between S and C2v. {}!={}".format(S.shape, C2v.shape)
    return C2v, C2t, S

def MakeSurfaceScaling(CV, N):
    C1 = incidence(CV._vessels)
    S = 2*np.pi*CV._vessels.ep['radius'].a*(CV._vessels.ep['length'].a) 
    S = 0.5*abs(C1)@S # Scale the exchange rates by the surface area of the control volume (each CV is a combination of half the adjacent vessels to a node)    
    nc = sum(N)
    S = np.multiply(1/np.where(N>0, N, np.inf), S)
    S = sp.dia_matrix(([si for si,Ni in zip(S,N) for _ in range(Ni)], 0),
                      shape=(nc,nc))
    return S

def MakeDiffusionReaction(grid:Grid, Gamma_t:float, k_t:float):
    """
    Make the discretized version of -nabla^2 + k_t from a finite volume approximation on cartesian grid.

    ### Parameters:
    - grid: Grid
        The cartesian grid used for the discretization.
    - Gamma_t: float
        The diffusion coefficient of O2 in tissue [m^2/s]
    - k_t: float
        The consumption rate of O2 per volume of tissue [m^-3.s^-1] 
    """
    grid_shape = tuple(grid.shape)
    N = len(grid_shape)
    faces = [
        (slice(None),) * i + (k,) + (slice(None),) * (N-i-1)
        for i in range(N) for k in [0,-1]
        ] # Slices of a 3D cube that return the faces 

    def _D3(x):
        X = x.reshape(grid_shape)
        y = np.zeros_like(X)
        for face in faces:
            y[face] = X[face]
        return y.flatten()

    def _I3minusD3(v):
        y = v.reshape(grid_shape)
        for face in faces:
            y[face] = 0
        return y.flatten()
    
    def _L(x):
        X = x.reshape(grid_shape + (-1,))
        Y = -k_t*np.prod(grid.h)*X # Reaction
        for i in range(N):
            coeff = Gamma_t*grid.h[i]/np.prod(np.delete(grid.h, i))
            Y -= 2*coeff * X
            Y += coeff * np.roll(X, 1, axis=i)
            Y += coeff * np.roll(X, -1, axis=i)
        
        return Y.reshape(-1, X.shape[-1])
    
    L  = spl.LinearOperator(shape=(grid.size, grid.size), matvec=_L)
    I3 = spl.LinearOperator(shape=L.shape, matvec=lambda v: v, matmat=lambda V:V, rmatvec=lambda v:v)
    D3 = spl.LinearOperator(shape=L.shape, matvec=_D3)
    I3minusD3 = spl.LinearOperator(shape=L.shape, matvec=_I3minusD3)
    B = I3minusD3@(-Gamma_t * L) + D3 
    return B, D3, I3minusD3

def MakeAll(
        CV:ControlVolumes, # Contains vascular graph and tissue grid
        C2v, C2t, S, # Coupling matrices
        cvIn:float, # Oxygen concentration at vascular inlets
        ctBC:float, # Oxygen concentration on cube boundaries
        Gamma_v:float, h_v:float, # Vessel wall permeability and thickness 
        Gamma_t:float, k_t:float, # Tissue diffusion and consumption rates     
        ): 
    M, Min, Din = ConvectionMatrices(CV)
    nc, nt, nv = C2v.shape[0], CV.grid.size, CV._vessels.num_vertices()

    # M, Min = spl.aslinearoperator(M), spl.aslinearoperator(Min)
    # C2v, C2t, S = spl.aslinearoperator(C2v), spl.aslinearoperator(C2t), spl.aslinearoperator(S)
    
    # Convection + free O2 in plasma at the interface with tissue
    # sp.eye(nv)-Din cancels exchange for the inlet (BC)
    A = M + Min # Convection 
    A += Gamma_v/h_v * (sp.eye(nv)-Din) @ C2v.T@S@C2v  # Sinks
    print("A, done. CPU:", psutil.virtual_memory())
    del M
    
    C2v, C2t, S = spl.aslinearoperator(C2v), spl.aslinearoperator(C2t), spl.aslinearoperator(S)
    
    E = -Gamma_v/h_v * (sp.eye(nv)-Din) @ C2v.T@S@C2t # Tissue oxygen at interface with vessel 
    print("E, done. CPU:", psutil.virtual_memory())
    
    B, D3 = MakeDiffusionReaction(CV.grid, Gamma_t, k_t)
    print("Got diffusion. CPU:", psutil.virtual_memory())
    B -= Gamma_v/h_v * C2t.T@S@C2t # Oxygen sources
    print("B, done. CPU:", psutil.virtual_memory())
    G = Gamma_v/h_v * C2t.T@S@C2v  # Plasma free oxygen concentration
    print("G, done. CPU,", psutil.virtual_memory())
    del C2t, C2v, S
    cvBar = Min@(cvIn*np.ones(nv))
    ctBar = D3 @ (np.ones(nt) * ctBC)

    return [[A,E], [G,B]], [cvBar, ctBar]