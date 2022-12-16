from Tissue import Tissue
from VascularNetwork import VascularNetwork
import matplotlib.pylab as plt
import numpy as np


w = 1e-1 # 1 micron
U = 2400*1e-6 # Permeability in mm^2/s

#tissue = Tissue(ccoFile='sim_19.cco', w=w) # Large-ish network
#tissue = Tissue(ccoFile='1Vessel.cco', w=w) # One vessel
tissue = Tissue(ccoFile='Patient1.cco', w=w)

tissue.Vessels.SetLinearSystem(inletBC={'pressure':50},
                               outletBC={'pressure':25})
tissue.Vessels.SolveFlow()

tissue.VesselsToVTK('Vessels.vtp')
tissue.MeshToVTK('LabelledMesh.vtk')

flow, radius, dp, mu, length = tissue.Vessels.GetVesselData(['flow', 'radius', 'dp',
                                                             'viscosity', 'length'],
                                                    returnAList=True)

inletPressure = dp[tissue.Vessels.inletNodes]/133.3224
outletPressure = dp[tissue.Vessels.outletNodes]/133.3224

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

tissue.MakeMassTransfer(U)
tissue.MakeReactionDiffusion(1, 1)
tissue.MakeConvection()
tissue.MakeRHS(50)
tissue.A.eliminate_zeros()
plt.spy(tissue.A)
plt.axhline(tissue.nPoints, c='black')
plt.axvline(tissue.nPoints, c='black')
plt.title(f'1D-3D coupling matrix\nwith {tissue.nVol} cells in the grid')
plt.savefig('OverallSystem.jpg', bbox_inches='tight')
plt.show()

x = tissue.Solve()
