import numpy as np
from utils import ControlVolumes, PositivePart, Grid
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from graph_tool.generation import lattice
from graph_tool.spectral import incidence
from graph_tool.util import find_edge
from itertools import chain 

def ConvectionMatrices(CV):
    C1 = incidence(CV._vessels).T
    Din = sp.dia_matrix(((CV._vessels.get_in_degrees(list(CV._vessels.iter_vertices()))==0).astype(int), 0),
                        shape=(CV._vessels.num_vertices(), CV._vessels.num_vertices()))
    Dout = sp.dia_matrix(((CV._vessels.get_out_degrees(CV._vessels.get_vertices())==0).astype(int), 0),
                        shape=(CV._vessels.num_vertices(), CV._vessels.num_vertices()))    
    q = CV._vessels.ep['q']
    l = CV._vessels.ep['length']
    L = sp.dia_matrix((1.0/l.a, 0), shape=2*[CV._vessels.num_edges()])
    Q = sp.dia_matrix((q.a * (l.a/2), 0), shape=2*[CV._vessels.num_edges()])
    Min  = Din @ C1.T @ Q @ (PositivePart(-C1))
    MOut = Dout @ C1.T @ Q @ PositivePart(C1)
    M = C1.T@(PositivePart(L@Q)@PositivePart(C1) - PositivePart(-L@Q)@PositivePart(-C1))
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
    couplingGraph = CV.LabelMesh(h_v)
    C1 = incidence(CV._vessels)
    S = np.pi*CV._vessels.ep['radius'].a*(CV._vessels.ep['length'].a/2.0) 
    S = abs(C1)@S # Scale the exchange rates by the surface area of the control volume
    
    # The list of endtohelial edges between voxels and control volumes
    endo = find_edge(couplingGraph, couplingGraph.ep['eType'],1)
    nc, nt, nv = len(endo), CV.grid.size, CV._vessels.num_vertices() 
    
    ## Initialize the matrices
    C2v = sp.dok_matrix((nc,nv))
    C2t = sp.dok_matrix((nc,nt))
    N   = np.zeros(nv, dtype=int) # Number of connections to each control volumes. 
                        # Should be the same as C2v.sum(1)
    for i, e in enumerate(endo):
        u,v = int(e.source()), int(e.target()) # u: a control volume, v: a voxel
        C2v[i, u] = 1 
        C2t[i, v] = 1 
        N[u]+=1
    C2v, C2t = C2v.tocsr(), C2t.tocsr()
    S = np.multiply(1/N, S)
    S = sp.dia_matrix(([si for si,Ni in zip(S,N) for _ in range(Ni)], 0),
                      shape=(nc,nc))
    return C2v, C2t, S

def MakeDiffusionReaction(grid:Grid, Gamma_t:float, k_t:float):
    lat = lattice(grid.shape)
    lat.set_directed(True)
    C3 = incidence(lat)
    D3 = sp.dia_matrix(([d!=6 for d in lat.get_total_degrees(lat.get_vertices())], 0)
                       , shape=(grid.size, grid.size))
    I3 = sp.eye(grid.size)
    
    B = (I3-D3)@(-Gamma_t * (grid.h[0]) * C3@C3.T + k_t * np.prod(grid.h) * I3) + D3
    return B, D3

def MakeAll(
        CV:ControlVolumes, # Contains vascular graph and tissue grid
        C2v, C2t, S, # Coupling matrices
        cvIn:float, # Oxygen concentration at vascular inlets
        ctBC:float, # Oxygen concentration on cube boundaries
        Gamma_v:float, h_v:float, # Vessel wall permeability and thickness 
        Gamma_t:float, k_t:float, # Tissue diffusion and consumption     
        ): 
    M, Min, Din = ConvectionMatrices(CV)
    nc, nt, nv = C2v.shape[0], CV.grid.size, CV._vessels.num_vertices() 

    # Convection + free O2 in plasma at the interface with tissue
    # sp.eye(nv)-Din cancels exchange for the inlet (BC)
    A = M + Min # Convection 
    A += Gamma_v/h_v * C2v.T@S@C2v @ (sp.eye(nv)-Din) # Sinks
    E = -Gamma_v/h_v * (sp.eye(nv)-Din) @ C2v.T@S@C2t # Tissue oxygen at interface with vessel 
    
    B, D3 = MakeDiffusionReaction(CV.grid, Gamma_t, k_t)
    B -= Gamma_v/h_v * C2t.T@S@C2t # Oxygen sources
    G = Gamma_v/h_v * C2t.T@S@C2v  # Plasma free oxygen concentration

    cvBar = Din@Min@(cvIn*np.ones(nv))
    ctBar = D3 @ (np.ones(nt) * ctBC)

    return [[A,E], [G,B]], [cvBar, ctBar]