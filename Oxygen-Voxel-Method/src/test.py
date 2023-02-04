import importlib

import Tissue
importlib.reload(Tissue)

import VascularNetwork
importlib.reload(VascularNetwork)

import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
from vtk.util import numpy_support
import vtk 
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from math import ceil
from tqdm import tqdm
from sparse_dot_mkl import sparse_qr_solve_mkl                                                                        


def fastspy(A, cmap='RdBu'):
    plt.scatter(A.row, A.col, edgecolors='none',c=A.data,
                norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()

w = 1e-3 # 1 micron
U = 2400*1e-5 # Permeability in mm^2/
kt = 4.5*1e-5
Rg = 62.36*1e6 # Ideal gas constant, in mm3.mmHg.K-1.mol-1 
Tb = 309.25    # Blood temperature, 36.1*C in Kelvin
c0 = 50/Rg/Tb*1e9
print(f"{c0=}")

#vessels = VascularNetwork.VascularNetwork('1Vessel.cco', spacing = [w*2, w*2, w*2])
#tissue = Tissue.Tissue(Vessels = vessels)
tissue = Tissue.Tissue(ccoFile='sim_19.cco', spacing=3*[5e-2]) # Large-ish network
#tissue = Tissue.Tissue(ccoFile='1Vessel.cco', w=w, spacing=3*[5e-3])# dimensions=[0.5,0.5,0.5]) # One vessel
#tissue = Tissue.Tissue(ccoFile='Patient1.cco', w=w)

tissue.Vessels.SetLinearSystem(inletBC={'pressure':50},
                               outletBC={'pressure':25})
tissue.Vessels.SolveFlow()

tissue.VesselsToVTK('Vessels.vtp')
tissue.MeshToVTK('LabelledMesh.vtk')

flow, radius, dp, mu, length = tissue.Vessels.GetVesselData(['flow', 'radius', 'dp',
                                                             'viscosity', 'length'],
                                                    returnAList=True)

# inletPressure = dp[tissue.Vessels.inletNodes]/133.3224
# outletPressure = dp[tissue.Vessels.outletNodes]/133.3224

# print(f'{tissue.nPoints=} {tissue.nSeg=}')

# NegativePressure = [tissue.Vessels.outletNodes[i] for i in range(outletPressure.size) if (inletPressure[0]-outletPressure[i]<0)]
# NegativePressure = [tissue.nPoints, tissue.nPoints+tissue.nSeg-1]
# print(NegativePressure)
# print(f'{tissue.Vessels.Flow_rhs[NegativePressure]=}')
# print(f'{tissue.Vessels.Flow_matrix[(tissue.nPoints-1+np.array(NegativePressure)).tolist(),:]}')
# fig, ax = plt.subplots(1,2)
# ax = ax.ravel()
# ax[0].plot(inletPressure-outletPressure)
# ax[1].plot(dp/133.3224) #, flow*8*length*mu/(np.pi*(radius**4)))
# plt.show()

tissue.MakeReactionDiffusion(1800, kt, method=1, )#='ReactionDiffusion.npz')
tissue.MakeMassTransfer(U, )#='MassTransfer.npz')
tissue.MakeConvection()#='Convection.npz')
tissue.MakeRHS(c0, )#='rhs.npz')

# tissue.rhs = np.zeros(tissue.A.shape[1])
# tissue.rhs[:tissue.nPoints] = 50
# I = np.ones(tissue.A.shape[1])
# I[:tissue.nPoints] = 0
# I = sp.diags([I], [0], format='csr')
# tissue.A = I.dot(tissue.A) + sp.eye(I.shape[0]) - I
# plt.spy(tissue.A)
# plt.show()
sp.save_npz("Overall.npz", tissue.A.tocoo())
xb, xt = tissue.Solve(checkForEmptyRows=True)
x = xt.reshape(tissue.nx, order='F')

#M = tissue.A[tissue.nPoints:, tissue.nPoints:].tolil()
# plt.spy(M)
# plt.show()
nx,ny,nz = tissue.nx
dx,dy,dz = tissue.dx

#I = sp.eye(M.shape[0])
#b = np.zeros(M.shape[0])
#x = np.zeros(M.shape[0])
# x = np.zeros((nx,ny,nz))
# i0, i1 = int(nx/3), int(2*nx/3)
# j0, j1 = int(ny/2-ny/10), int(ny/2+ny/10)
# k0, k1 = int(nz/2-nz/10), int(nz/2+nz/10)
# x[i0:i1, j0:j1, k0:k1] = 1.50

#sourceCells = []
#c0 = 2.0741691785719655e-09
#for i in tqdm(range(M.shape[0])):
#    if M[i,i] == 0:
#        M[i,i] = 1.0
#        b[i]   = c0
#        x[i] = c0
#        sourceCells.append(i)

#x = spl.spsolve(M.tocsr(), b).reshape((nx,ny,nz))
#print("preconditionning...")
#M = M.tocsc()
#ilu = spl.spilu(M)                               
#M_x = lambda x: ilu.solve(x)                                          
#P = spl.LinearOperator(M.shape, Mx)
#print("Done. Solving the system...")
#x = spl.gmres(M, b, M=P) 
#print("Done.")

dt = 0.1
fig = plt.figure()

# K = I - dt*M
# for k in tqdm(range(50)):
#     x = spl.spsolve(K, x)
#     print(x.min(), x.max(), x.mean())

#plt.pcolormesh(x[:,:,int(nz/2)])
def WriteVTK(x, fileName):
    f = open(fileName,'w') # change your vtk file name
    f.write('# vtk DataFile Version 2.0\n')
    f.write('test\n')
    f.write('ASCII\n')
    f.write('DATASET STRUCTURED_POINTS\n')
    f.write(f'DIMENSIONS {x.shape[0]} {x.shape[1]} {x.shape[2]}\n') # change your dimension
    f.write(f'SPACING {dx} {dy} {dz}\n')
    #f.write('ORIGIN 0.0 0.0 0.0\n')
    f.write(f'ORIGIN {tissue.Mesh.origin[0]} {tissue.Mesh.origin[1]} {tissue.Mesh.origin[0]}\n')
    f.write(f'POINT_DATA {nx*ny*nz}\n') # change the number of point data
    f.write('SCALARS volume_scalars float 1\n')
    f.write('LOOKUP_TABLE default\n')
    f.write("\n".join(str(xi) for xi in x.ravel(order='F')))
    f.close()

WriteVTK(x, "xt.vtk")
plt.plot(xb, label="Blood concentration")
plt.show()

plt.contourf(x[:,:,int(nz/2)])
plt.colorbar()
plt.show()
print(x.min(), x.max(), x.mean())

# tissue.MakeConvection(saveIn='Convection.npz')
# tissue.MakeRHS(50, saveIn='rhs.npz')

# print("Mesh labels:", tissue.Mesh.LabelsDistribution, 'Total number of cells', tissue.Mesh.nCellsTotal)
    
# plt.spy(tissue.A)
# plt.axhline(tissue.nPoints, c='black')
# plt.axvline(tissue.nPoints, c='black')
# plt.title(f'1D-3D coupling matrix\nwith {tissue.nVol} cells in the grid')
# plt.savefig('OverallSystem.jpg', bbox_inches='tight')
# plt.show()

# xVessels, xTissue = tissue.Solve(checkForEmptyRows = False)
# print(xVessels)
# # plt.plot(xVessels)
# # plt.show()
# # xTissue = xTissue.reshape(tissue.Mesh.nCells)
# np.savetxt('OxygenVessels.dat', xVessels)
# np.savetxt('OxygenTissue.dat', xTissue.reshape( (xTissue.shape[0], -1)) )
# np.save('OxygenTissue.npy', xTissue)

# vtk_data = numpy_support.numpy_to_vtk(num_array=xTissue.ravel(), array_type=vtk.VTK_FLOAT)
# vtk_data.SetName("O2")

# reader = vtk.vtkStructuredPointsReader()
# reader.SetFileName('LabelledMesh.vtk')
# reader.Update()
# data = reader.GetOutput()
# data.GetCellData().AddArray(vtk_data)

# writer = vtk.vtkStructuredPointsWriter()
# writer.SetFileName("LabelledMesh.vtk")
# writer.SetInputData(data)
# writer.Write()

