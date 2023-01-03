from Tissue import Tissue
from VascularNetwork import VascularNetwork
import matplotlib.pyplot as plt
import matplotlib as m
import numpy as np
from vtk.util import numpy_support
import vtk 
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from math import ceil

w = 5e-3 # 1 micron
U = 2400*1e-5 # Permeability in mm^2/s

vessels = VascularNetwork('1Vessel.cco', spacing = [w*2, w*2, w*2])
tissue = Tissue(Vessels = vessels)
#tissue = Tissue(ccoFile='sim_19.cco', w=w) # Large-ish network
#tissue = Tissue(ccoFile='1Vessel.cco', w=w) # One vessel
#tissue = Tissue(ccoFile='Patient1.cco', w=w)

# tissue.Vessels.SetLinearSystem(inletBC={'pressure':50},
#                                outletBC={'pressure':25})
# tissue.Vessels.SolveFlow()

tissue.VesselsToVTK('Vessels.vtp')
tissue.MeshToVTK('LabelledMesh.vtk')

flow, radius, dp, mu, length = tissue.Vessels.GetVesselData(['flow', 'radius', 'dp',
                                                             'viscosity', 'length'],
                                                    returnAList=True)

# inletPressure = dp[tissue.Vessels.inletNodes]/133.3224
# outletPressure = dp[tissue.Vessels.outletNodes]/133.3224

print(f'{tissue.nPoints=} {tissue.nSeg=}')

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

#tissue.MakeMassTransfer(U, saveIn='MassTransfer.npz')
tissue.MakeReactionDiffusion(1e6, 0, saveIn='ReactionDiffusion.npz')
# plt.spy(tissue.A)
# plt.show()

M = tissue.A[tissue.nPoints:, tissue.nPoints:]
# plt.spy(M)
# plt.show()
nx,ny,nz = tissue.nx

I = sp.eye(M.shape[0])
b = np.zeros(M.shape[0])
x = np.zeros(M.shape[0])
x[ceil(nx/2):ceil(3*nx/2)] = 1.50
sourceCells = []
for i in range(M.shape[0]):
    if M[i,i] == 0:
        M[i,i] = 1.0
        b[i]   = 1.50
        x[i] = 1.50
        sourceCells.append(i)

dt = 0.1
fig = plt.figure()

idx0 = tissue.Vessels.mesh.ToFlatIndexFrom3D(tissue.Vessels.mesh.PointToCell((-0.08, -0.08, 0.0)))
idx1 = tissue.Vessels.mesh.ToFlatIndexFrom3D(tissue.Vessels.mesh.PointToCell((0.18, 0.08, 0.0)))
idx0 = 0
idx1 = 5*nx

xim = x[idx0:idx1].reshape((tissue.nx[0], -1))
plt.imshow(xim)
plt.colorbar()
plt.show()
K = I + dt*M
for k in range(50):
    x = spl.spsolve(K, x)
    print(x.min(), x.max(), x.mean())

xim = x[idx0:idx1].reshape((tissue.nx[0], -1))
plt.imshow(xim)
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

