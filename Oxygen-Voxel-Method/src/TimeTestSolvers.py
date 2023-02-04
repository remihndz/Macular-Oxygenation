import importlib                                                                                       
import Tissue    
importlib.reload(Tissue)                                     
import VascularNetwork                                                                  
importlib.reload(VascularNetwork)       
import numpy as np                                                                    
import scipy.sparse as sp          
import scipy.sparse.linalg as spl
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from os import devnull
from time import time

from petsc4py import PETSc

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

w = 1e-3 # 1 micron                                                                                                 
U = 2400*1e-5 # Permeability in mm^2/                                                  
kt = 4.5*1e-5                                                                                         
Rg = 62.36*1e6 # Ideal gas constant, in mm3.mmHg.K-1.mol-1                                          
Tb = 309.25    # Blood temperature, 36.1*C in Kelvin      
c0 = 50/Rg/Tb*1e9
print(f"{c0=}")

def LinearSystem(spacing):
    tissue = Tissue.Tissue(ccoFile='1Vessel.cco', w=w, spacing=3*[spacing])
    tissue.MakeReactionDiffusion(1800, kt, method=1)#, saveIn='ReactionDiffusion.npz')                         
    tissue.MakeMassTransfer(U)#, saveIn='MassTransfer.npz')                                                 
    tissue.MakeConvection()#saveIn='Convection.npz')                                                           
    tissue.MakeRHS(c0)#, saveIn='rhs.npz')
    M, b = tissue.A, tissue.rhs.toarray()[:,0]
    return M,b


def RunTests(spacings=[1e-2,5e-3, 2.5e-3, 1e-3]):
    for spacing in spacings:
        with suppress_stdout_stderr():
            M,b = LinearSystem(spacing)
        print(f"{M.shape[0]} cells.")

        print("Solving with spsolve...")
        t = time()
        x = spl.spsolve(M,b)
        del x
        print("spl.spsolve(M,b): ", time()-t, 's')

        print("Solving with lsmr...")
        t = time()
        x = spl.lsmr(M,b, atol=1e-10)
        del x
        print("spl.lsmr(M,b): ", time()-t, 's')

        print("Solving with ilu and gmres...")
        t = time()
        M = M.tocsc()
        ilu = spl.spilu(M)
        M_x = lambda x: ilu.solve(x)
        n,m = M.shape
        P = spl.LinearOperator((n,n), M_x)
        x, info = spl.gmres(M, b, M=P)
        print("spl.gmres(M, b, M=ilu): ", time()-t, 's')
        del x
        del M_x
        del ilu
        del P
        
        M = M.tocsr()
        print("Solving with PETSc")
        t = time()
        petsc_mat = PETSc.Mat().createAIJ(size=M.shape,
                                          csr=(M.indptr,
                                               M.indices,
                                               M.data))
        del M
        ksp = PETSc.KSP().create()
        ksp.setType('pgmres')
        pc = ksp.getPC()
        pc.setType('ilu')
        ksp.setFromOptions()
        ksp.setOperators(petsc_mat)
        x = PETSc.Vec().create()
        b = PETSc.Vec().create()
        x.setSizes(m, None)
        x.setUp()
        b.setSizes(m, None)
        b.setUp()
        b.setValues([0], [c0], addv=PETSc.InsertMode.ADD_VALUES)
        ksp.solve(b, x)
        print("ksp.solve(b,x): ", time()-t, "s")
        del ksp
        del petsc_mat
        del x
        del b
        
    return

RunTests(spacings=[1e-3])
