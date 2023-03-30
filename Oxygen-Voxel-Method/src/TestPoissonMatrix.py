import importlib

import Tissue
importlib.reload(Tissue)

import VascularNetwork
importlib.reload(VascularNetwork)

import Mesh
importlib.reload(Mesh)

import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
from vtk.util import numpy_support
import vtk 
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from math import ceil
from tqdm import tqdm


w = 10e-3
U = 0
mesh = Mesh.UniformGrid(dimensions=[1.0,1.0,1.0], nCells=[50,50,50])
D = 10

M = mesh.MakePoissonWithNeumannBC_Kronecker(D)
b = np.zeros(mesh.nCells)
dx,dy,dz = mesh.spacing
b[int(0.45/dx):int(0.55/dx),
  int(0.45/dy):int(0.55/dy),
  int(0.45/dz):int(0.55/dz)] = -1.0
R = sp.eye(M.shape[0])*10

dx,dy,dz = mesh.spacing
def WriteVTK(x, fileName):
    f = open(fileName,'w') # change your vtk file name
    f.write('# vtk DataFile Version 2.0\n')
    f.write('test\n')
    f.write('ASCII\n')
    f.write('DATASET STRUCTURED_POINTS\n')
    f.write(f'DIMENSIONS {x.shape[0]} {x.shape[1]} {x.shape[2]}\n') # change your dimension
    f.write(f'SPACING {dx} {dy} {dz}\n')
    f.write('ORIGIN 0 0 0\n')
    f.write(f'POINT_DATA {x.size}\n') # change the number of point data
    f.write('SCALARS volume_scalars float 1\n')
    f.write('LOOKUP_TABLE default\n')
    f.write("\n".join(str(xi) for xi in x.ravel(order='F')))
    f.close()

print("Solving...")
#x, info = spl.gmres(-M-R, b.ravel(order='F'))
#print(info)
x = spl.spsolve(M-R, b.ravel(order='F'))

WriteVTK(x.reshape(mesh.nCells, order='F'), "SolutionFromMyCode.vtk")
WriteVTK(b, "RHSFromMyCode.vtk")
