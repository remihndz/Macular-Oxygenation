from Tissue import Tissue
import matplotlib.pylab as plt
import numpy as np

w = 1e-1 # 1 micron
U = 2400*1e-6 # Permeability in mm^2/s

#tissue = Tissue(ccoFile='sim_19.cco', w=w)
tissue = Tissue(ccoFile='1Vessel.cco', w=w)
tissue.VesselsToVTK('Vessels.vtp')
tissue.MeshToVTK('LabelledMesh.vtk')



tissue.MakeMassTransfer(U)
plt.spy(tissue.Mc)
plt.title('Mass transfer matrix: coefficients distribution.')
plt.savefig('MassTransferMatrix.jpg')
plt.show()

tissue.MakeReactionDiffusion(1, 1)
plt.spy(tissue.Md)
plt.title('Reaction-diffusion matrix: coefficients distribution.')
plt.savefig('ReactionDiffusionMatrix.jpg')
plt.show()

tissue.MakeConvection()
plt.spy(tissue.Mc)
plt.title('Convection matrix: coefficients distribution')
plt.savefig('ConvectionMatrix.jpg')
plt.show()

