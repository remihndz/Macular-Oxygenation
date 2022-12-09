from VascularNetwork import VascularNetwork
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

w = 1e-3 # 1 micron
V = VascularNetwork('1Vessel.cco', spacing = [1e-1, w, w])#, origin=[-0.5, -0.1, -0.1], dimensions = [2,0.2,0.2])
V.VesselsToVTK('Vessels.vtp')
V.LabelMesh(w)
V.MeshToVTK('LabelledMesh.vtk')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x_, y_, z_ = np.arange(arr.shape[0]), np.arange(arr.shape[1]), np.arange(arr.shape[2])
# X, Y, Z = np.meshgrid(x_, y_, z_, indexing='xy')
# print(X.shape, Y.shape, Z.shape)

# scat = ax.scatter(X,Y, Z, c=arr.flatten(), alpha=0.2, s=0.2)
# fig.colorbar(scat, shrink=0.5, aspect=0.5)
# plt.show()


labels = V.mesh.ToNumpy()
vessels = labels[labels==1].size
endothelium = labels[labels==2].size
tissue = labels.size - vessels - endothelium
print(f'{vessels=} {endothelium=} {tissue=}')

# np.savetxt('Labels.txt', labels.reshape(labels.shape[0],-1).astype(int), fmt='%1d')

## Check for each vessel if the midpoint has been labelled as vessel
# for n1,n2, data in V.G.edges(data=True):
#     p1, p2 = V.G.nodes[n1]['position'], V.G.nodes[n2]['position']
#     r = data['radius']
#     v = p2-p1
#     v = v/np.linalg.norm(v)
#     P = np.outer(v,v)
#     O = np.identity(3)-P
#     cellId  = tuple(V.mesh.PointToCell((p1+p2)/2.0).tolist())
#     newLabel = V._LabelCellWithCylinder(O, p1, p2, r, cellId, w)
#     print(f'New label for {cellId} is {newLabel}')
    
