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
             
from astropy import units as u
from astropy.constants import R # Ideal gas constant

def fastspy(A, cmap='RdBu'):
    plt.scatter(A.row, A.col, edgecolors='none',c=A.data,
                norm=matplotlib.colors.LogNorm())
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.show()


unitsL = u.Unit('mm')
unitsT = u.Unit('ms')
units = {'length':unitsL, 'time':unitsT}

#R = R.to((unitsL**3) * u.torr / u.K / u.mol) # Gas constant
alpha = u.Quantity(1.27e-15, u.umol/(u.um**3)/u.torr)
f_CRA = 49.34 * u.uL/u.min
w = 1*u.um # 1 micron
U = 50*2400 * u.um/u.s # Permeability in um^2/s
D = 1800 * u.um*u.um/u.s # Diffusion in um^2/s
c0 = 50 * u.torr
kt = 4.5 / u.min * 1e-5
Tb = 309.25 * u.K    # Blood temperature, 36.1*C in Kelvin
spacing = u.Quantity([5, 5, 10], 'micron').to(unitsL)*5
#dimensions = u.Quantity([30, 30, 10], 'mm')
dimensions = u.Quantity([300,500,400],'micron').to(unitsL)
origin  = u.Quantity([-100, -200, -200], 'micron').to(unitsL)


w = w.to(unitsL, copy=False)
f_CRA = f_CRA.to(unitsL**3 / unitsT, copy=False)
U = U.to(unitsL / unitsT, copy=False)
D = D.to(unitsL**2 / unitsT, copy=False)
kt = kt.to(unitsT**(-1), copy=False)

# c0 = c0/R/Tb
# c0 = c0.to(u.nmol / (unitsL**3), copy=False)
Kw = 1.115e-12 * u.umol/(u.um * u.s * u.torr)
U = (Kw/alpha).to((unitsL**2)/unitsT)*1e2 
D = (2.4e6 * u.um*u.um/u.s).to((unitsL**2)/unitsT)/1e3
c0 = (c0*alpha).to(u.nmol/(unitsL**3), copy=False)

print(f"""Simulation coefficients:
\tInlet concentration: {c0=}
\tWall thickness: {w=}
\tDiffusion: {D=}
\tWall permeability: {U=}
\tTissue consumption: {kt=}
\tInlet blood flow (may not be used): {f_CRA=}
""")


#vessels = VascularNetwork.VascularNetwork('1Vessel.cco', spacing = [w*2, w*2, w*2])
#tissue = Tissue.Tissue(Vessels = vessels)
#tissue = Tissue.Tissue(ccoFile='sim_19.cco', w=w, spacing=spacing, units=units) # Large-ish network
#tissue = Tissue.Tissue(ccoFile='1Vessel.cco', units=units, w=w, spacing=spacing, dimensions=dimensions.to_value(unitsL), origin=origin.to_value(unitsL)) # One vessel
tissue = Tissue.Tissue(ccoFile='Patient1.cco', w=w, spacing=spacing, units=units, dimensions=u.Quantity([4.5, 4.0, 0.235], 'mm').to_value(unitsL), origin=u.Quantity([-2.26,-1.95,-0.235], 'mm').to_value(unitsL))

# tissue.Vessels.SetLinearSystem(inletBC={'flow':f_CRA.value},
#                                outletBC={'pressure':25})
# tissue.Vessels.SolveFlow()

#tissue.MeshToVTK('LabelledMesh.vtk')

tissue.MakeReactionDiffusion(D.value, kt.value, method=1)#='ReactionDiffusion.npz')
tissue.MakeMassTransfer(U.value )#='MassTransfer.npz')
tissue.MakeConvection(inletBC={'pressure':50},
                      outletBC={'pressure':25}, saveIn='Convection.npz')
# tissue.MakeConvection(inletBC={'flow':f_CRA.to_value(unitsL**3/unitsT)},
#                       outletBC={'pressure':10})
tissue.MakeRHS(c0.value)#='rhs.npz')

sp.save_npz("Overall.npz", tissue.A)
sp.save_npz("rhs.npz", tissue.rhs)
xb, xt = tissue.Solve(preconditioner='none', maxIter=200, checkForEmptyRows=False)
xb /= alpha.to_value(u.nmol/(unitsL**3)/u.torr) 
xt /= alpha.to_value(u.nmol/(unitsL**3)/u.torr)

unitsConcentration = u.nmol/(unitsL**3)
# xb = (u.Quantity(xb, unitsConcentration)*R*Tb).to_value('torr')
# xt = (u.Quantity(xt, unitsConcentration)*R*Tb).to_value('torr')
tissue.ToVTK('PO2.vtk', xt)
x = xt.reshape(tissue.nx, order='F')

flow, radius, dp, mu, length, po2 = tissue.Vessels.GetVesselData(['flow', 'radius', 'dp',
                                                             'viscosity', 'length', 'PO2'],
                                                            returnAList=True)
fig, ax = plt.subplots(3)
ax[0].scatter(radius, flow)
ax[0].set(xlabel=f'Radius [{unitsL}]', ylabel=f'Flow [{unitsL**3/unitsT}]', yscale='log')

ax[1].scatter(radius, dp)
ax[1].set(xlabel=f'Radius [{unitsL}]', ylabel=f'Pressure drop [mmHg]', yscale='log')

ax[2].scatter(radius, po2)
ax[2].set(xlabel=f'Radius [{unitsL}]', ylabel=f'PO2 [mmHg]', yscale='log')

plt.show()


tissue.VesselsToVTK('Vessels.vtp')
plt.plot(xb, label="Blood concentration")
plt.show()

nx,ny,nz = tissue.Mesh.nCells
plt.contourf(x[:,:,int(nz/2)])
plt.colorbar()
plt.show()
print(x.min(), x.max(), x.mean(), xb.max(), xb.min())


