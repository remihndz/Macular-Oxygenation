# Retinal PO2
This folder contains codes on mathematical models of oxygen distribution within the retina. At the moment, the retina is represented as a 2D slice along the depth (from vitreous to choroid).
The model is taken from *Verticchio Vercellin AC, Harris A, Chiaravalli G, Sacco R, Siesky B, Ciulla T, Guidoboni G. Physics-based modeling of Age-related Macular Degeneration-A theoretical approach to quantify retinal and choroidal contributions to macular oxygenation. Math Biosci. 2021 Sep;339:108650. doi: 10.1016/j.mbs.2021.108650. Epub 2021 Jun 29. PMID: 34197878.*

To run a simulation, you need FreeFem++ (I used version 3.47 but other versions should work just fine). First step is to build the mesh and auxiliary functions (oxygen consumption and supply rates) for this mesh (use 'MeshWithChoroid.edp' or 'MeshWithoutChoroid.edp'). 
```
FreeFem++ ./Pre-processing/MeshWithChoroid.edp
```
Then run the corresponding solver ('PO2_NoChoroid.edp' or 'PO2_WithChoroid.edp'):
```
FreeFem++ PO2_WithChoroid.edp 
```
Results are stored by default in ./Results/ and oxygen profiles through the thickness of the retina can be drawn by running
```
gnuplot src/PlotProfiles.gp
```
Most parameters relevant to the simulations are in text files in ./Params and experiments can be run by creating new parameter files. 