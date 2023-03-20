from Mesh import RectilinearGrid
from VascularNetwork import DAG
from Tissue import TissueNew
from astropy import units as u
import matplotlib.pyplot as plt
import scipy.sparse as sp 
import scipy.sparse.linalg as spl
import numpy as np
import time


G = DAG({'length':'mm', 'time':'s'})
G.CreateGraph('sim_19.cco')
bb = G.BoundingBox()
maxRad = max([e[-1] for e in G.edges.data('radius')])
origin = u.Quantity(bb[0], 'mm')
dimensions = u.Quantity(abs(bb[1]-bb[0]), 'mm')
dimensions += 2*maxRad * u.mm
origin -=  maxRad * u.mm
T = TissueNew(G, origin=origin.value, dimensions=dimensions.value, nCells=[900,800,66])
# G.Repartition(T)
#G.LabelMesh(T, endotheliumThickness=0.01)
#T.ToVTK('TestNewLabelling.vtk')
G.SetLinearSystem({'pressure':50}, {'pressure':25})
# plt.spy(G.Flow_matrix)
# plt.show()
f,p,dp = G.SolveFlow()
T._MakeConvection(1e-4)
T._MakeReactionDiffusion(1e-4, 1e-6)


#### Full system solving
A = sp.bmat([[T.M-T.A, T.E], [T.G, T.B]], format='csr', dtype=float)
n = T.nNodes + T.nVol
b = np.zeros(T.nNodes + T.nVol)
b[T.vessels.inletNodes]=50.0
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            if self.niter%100==0:
                print('iter %3i\trk = %s' % (self.niter, str(rk)))

t = time.time()
# P = spl.splu(A)
# M = spl.LinearOperator((n,n), P.solve)
# M = sp.spdiags(1./A.diagonal(), 0, n,n)
# x,code = spl.gmres(A,b,M=M, callback=gmres_counter(), restart=20, maxiter=10000,tol=1e-5)
# if code!=0:
#     print("gmres did not converge in", time.time()-t, "s.")

#### Schur complement solving
A = T.M - T.A
def make_bc(bt, bv):
    return bt - T.G.dot(spl.spsolve_triangular(A,bv))
def Sc_func(x):
    y = -T.G.dot(spl.spsolve_triangular(A,T.E.dot(x)))
    y += T.B.dot(x)
    return y

Sc = spl.LinearOperator((T.nVol,T.nVol), Sc_func)
bc = make_bc(b[T.nNodes:], b[:T.nNodes])

t = time.time()
x,code = spl.gmres(Sc, bc, restart=10, callback=gmres_counter(),maxiter=1500)
print("Time to solve without preconditioner:", time.time()-t, "s.")

# T.ToVTK('SchurCPU.vtk', x)

## Schurr complement on gpu
import cupyx.scipy.sparse.linalg as cpssl
import cupy as cp
import cupyx.scipy.sparse as cpss

del A
del Sc

b_gpu = cp.array(b)
# del b
# del M
A = cpss.csr_matrix(T.M-T.A)
Am = cpssl.LinearOperator(A.shape, matvec=lambda x:cpssl.spsolve_triangular(A,x))
G = cpss.csr_matrix(T.G)
E = cpss.csr_matrix(T.E)
B = cpss.csr_matrix(T.B)

def Sc_func_gpu(x):
    y = G.dot(Am.matvec(E.dot(x)))
    return B.dot(x) - y
def make_bc_gpu(bt, bv):
    return bt - G.dot(Am.dot(bv))

bc_gpu = make_bc_gpu(b_gpu[T.nNodes:], b_gpu[:T.nNodes])
Sc_gpu = cpssl.LinearOperator((T.nVol, T.nVol), Sc_func_gpu)

t = time.time()
x, code = cpssl.gmres(Sc_gpu, bc_gpu, restart=10, callback=gmres_counter())
print("Time to solve without preconditioner, on GPU:", time.time()-t, "s.")

T.ToVTK('SchurGPU.vtk', x)
