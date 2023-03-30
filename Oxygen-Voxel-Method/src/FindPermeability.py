import importlib
import pprint
import Tissue
importlib.reload(Tissue)

import VascularNetwork
importlib.reload(VascularNetwork)

import matplotlib.pyplot as plt
import matplotlib 
import numpy as np
import scipy.sparse as sp

from astropy import units as u

unitsL = u.Unit('mm')
unitsT = u.Unit('ms')
units = {'length':unitsL, 'time':unitsT}

f_CRA = 49.34 * u.uL/u.min
w = 1*u.um # 1 micron
D = 1800 * u.um*u.um/u.s # Diffusion in um^2/s
c0 = 50 * u.torr
kt = 4.5 / u.min 
Tb = 309.25 * u.K    # Blood temperature, 36.1*C in Kelvin
spacing = u.Quantity([10, 10, 10], 'micron').to_value(unitsL)/5
#dimensions = u.Quantity([30, 30, 10], 'mm')
dimensions = u.Quantity([200,400,400],'micron').to_value(unitsL)
origin  = u.Quantity([-100, -200, -200], 'micron').to_value(unitsL)

alpha = u.Quantity(1.27e-15, u.umol/(u.um**3)/u.torr)
c0 = 50 *u.torr
Kw = 1.115e-12 * u.umol/(u.um * u.s * u.torr)
U = (Kw/alpha).to((unitsL**2)/unitsT)
D = (2.4e6 * u.um*u.um/u.s).to((unitsL**2)/unitsT)/1e6
c0 = (c0*alpha).to(u.nmol/(unitsL**3), copy=False)

tiss = Tissue.Tissue(ccoFile='1Vessel.cco', w=w, spacing=spacing, units=units)#, dimensions=u.Quantity([4.5, 4.0, 0.235], 'mm').to_value(unitsL), origin=u.Quantity([-2.26,-1.95,-0.235], 'mm').to_value(unitsL))

tiss.MakeReactionDiffusion(D.value, kt.value, method=1)#='ReactionDiffusion.npz')
tiss.MakeConvection(inletBC={'flow':f_CRA.to_value(unitsL**3/unitsT)},
                      outletBC={'pressure':25})#='Convection.npz')
tiss.MakeRHS(c0.value)

sp.save_npz("A_NoMassTransfer.npz", tiss.A)

results = {}
U = U.value#/1e5
for k,i in enumerate(range(20)):
    U = U*10.0
    tiss.A = sp.load_npz("A_NoMassTransfer.npz")
    tiss.MakeMassTransfer(U)

    xb, xt = tiss.Solve(checkForEmptyRows=False)
    xb /= alpha.to_value(u.nmol/(unitsL**3)/u.torr) 
    xt /= alpha.to_value(u.nmol/(unitsL**3)/u.torr)

    results[U] = f"{xt.mean()=} {xt.min()=} {xb.mean()=} {xb.min()}"

    tiss.ToVTK(f'U_sensitivity_{k}.vtk', xt)
    tiss.VesselsToVTK(f'U_sensitivity_{k}.vtp')

print('\n'.join([f'U={key}: {value}' for key, value in results.items()]))
pprint.pp(results)
