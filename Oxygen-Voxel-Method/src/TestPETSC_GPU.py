import petsc4py
petsc4py.init()

from petsc4py import PETSc
import numpy as np
import scipy.sparse as sp

# Create a square sparse matrix A in CSR format
A = sp.load_npz('Overall.npz')
row, col, data = A.indptr, A.indices, A.data
n = A.shape[0]
A = PETSc.Mat().createAIJWithArrays(size=(n, n), csr=(row, col, data))

# Create vectors u and b
u = PETSc.Vec().createSeq(n)
b = PETSc.Vec().createSeq(n)

# Set the right-hand side vector b to random values
b.setArray(np.array([50 if n==0 else 0 for n in range(n)]))

# Create a GPU context and transfer data to the GPU
with PETSc.Log.Stage('Data transfer to GPU'):
    # gpu_ctx = PETSc.GPU().create()
    A.assemble()
    # A.setGPUContext(gpu_ctx)
    A.assemblyBegin(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    A.assemblyEnd(PETSc.Mat.AssemblyType.FINAL_ASSEMBLY)
    u.assemble()
    # u.setGPUContext(gpu_ctx)
    b.assemble()
    # b.setGPUContext(gpu_ctx)

# Create a Krylov subspace solver with GPU acceleration
def monitor(ksp, its, rnorm):
    if its%100 == 0:
        print('%5d      %20.15g / %2.2g'%(its,rnorm, ksp.getTolerances()[0]))

with PETSc.Log.Stage('KSP setup'):
    ksp = PETSc.KSP().create()
    ksp.setType('gmres')
    ksp.setOperators(A)
    ksp.setFromOptions()
    ksp.setTolerances(rtol=1e-5, atol=1e-10)
    ksp.setMonitor(monitor)

# Solve the linear system
with PETSc.Log.Stage('Solve'):
    ksp.solve(b, u)

# Get the number of iterations and the final residual norm
its = ksp.getIterationNumber()
res = ksp.getResidualNorm()

# Free the GPU context and PETSc objects
#gpu_ctx.destroy()
A.destroy()
u.destroy()
b.destroy()
ksp.destroy()
