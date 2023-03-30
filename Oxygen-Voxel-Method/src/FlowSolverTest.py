import VascularNetwork as vn; V = vn.VascularNetwork("sim_19.cco", units='microns');
V.SetLinearSystem(inletBC={'flow':10}, outletBC={'flow':5})
f,p,dp = V.SolveFlow()
print(f)
print(p)
