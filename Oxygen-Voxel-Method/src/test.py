from VascularNetwork import VascularNetwork
import matplotlib.pyplot as plt

w = 1e-3 # 1 micron
V = VascularNetwork('sim_19.cco')
V.LabelMesh(w)
V.ToVTK('test.vtk')

labels = V.mesh.labels
vessels = labels[labels==1].size
endothelium = labels[labels==2].size
tissue = labels.size - vessels - endothelium
print(f'{vessels=} {endothelium=} {tissue=}')
