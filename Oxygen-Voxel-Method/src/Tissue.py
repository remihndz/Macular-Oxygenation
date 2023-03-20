from VascularNetwork import VascularNetwork, DAG
# from Mesh import UniformGrid
import networkx
import scipy.sparse as sp
import scipy.sparse.linalg
from NDSparseMatrix import SparseRowIndexer
import numpy as np
from typing import Dict, Optional, Union
from astropy import units as u 
# from sparse_dot_mkl import sparse_qr_solve_mkl
#from petsc4py import PETSc
#from mpi4py.MPI import COMM_WORLD as mpi_comm
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from Mesh import UniformGrid

class TissueNew(UniformGrid):
    """A class for the generation of equations of 1D-3D oxygen perfusion of the tissue.

    Attributes:
    -----------
    Vessels : VascularNetwork
        The vascular network embedded within the tissue.
    endotheliumThickness : int
        The thickness of the endothelium. Make sure units match Vessels.
    labels : NDSparseArray
        Access the mesh's labels.
    v : float
        The volume of a mesh's cell.
    nVol : int
        The number of cells in the mesh.
    nPoints : int
        The number of vascular nodes in Vessels.
    nSeg : int
        The number of vascular segments in Vessels.
    A : scipy.sparse.sparse_matrix
    rhs : scipy.sparse.sparse_matrix
        The right hand side of the equation. Includes boundary conditions.
    CellToSegment : scipy.sparse.sparse_matrix
        The connectivity matrix between cells and their associated vascular segments.
    
    Methods:
    --------
    ImportVessels(ccoFileName)
        Reads the vascular network to be used from a .cco file.
    MakeMassTransfer()
        Add the vessel-endothelium mass transfer terms to the system.
    MakeReactionDiffusion(D, kt)
        Add reaction-diffusion terms to the overall system.
    MakeConvection(**kwargs)
        Add O2 convection in the vessels to the system.
    MakeRHS(**kwargs)
        Assembles the RHS according to boundary conditions in **kwargs.
    """

    def __init__(self, vessels : Union[DAG, str], dimensions=..., origin=..., nCells=..., spacing=None, 
                 units : Dict[str,str]={'length':'mm', 'time':'s'}, w : float=1*u.um):
        self.units = units
        self.w = w.to_value(self.unitsL)
        if isinstance(vessels, str):
            self.vessels = DAG(units)
            self.vessels.CreateGraph(vessels)
        elif isinstance(vessels, DAG):
            self.vessels = vessels
            self.vessels.units = units # This converts the values stored in the graph (length, radius, flow, positions...)
        else:
            raise ValueError("'vessels' must be a .CCO file or an instance of DAG class.")    
        
        bb = self.vessels.BoundingBox()
        maxRad = max([e[-1] for e in self.vessels.edges.data('radius')])
        orig = bb[0]-maxRad
        dims = abs(bb[1]-bb[0])+2.0*maxRad

        super().__init__(np.maximum(dimensions, dims), np.minimum(origin, orig)
                         , nCells, spacing, units=self.unitsL)
        
        self.C2v, self.C4, self.I4 = self.vessels.LabelMesh(self, endotheliumThickness = self.w)
        self.C2t = np.zeros(self.nVol)
        for cell, label in self.labels.elements.items():
            if label==2:
                flatId = self.ToFlatIndexFrom3D(cell)
                self.C2t[flatId] = 1.0
        self.C2t = sp.diags([self.C2t], [0]) 

    @property
    def w(self) -> float:
        return self._w
    @w.setter
    def w(self, newW):
        self._w = u.Quantity(newW,self.unitsL).value
    @property
    def nNodes(self)->int:
        return self.vessels.nNodes    
    @property
    def units(self) -> Dict[str,str]:
        return self._units
    @property
    def unitsL(self) -> str:
        return self.units['length']
    @property
    def unitsT(self) -> str:
        return self.units['time']
    @units.setter
    def units(self, newUnits : Dict[str, str]):
        unitsL = u.Unit(newUnits.get('length', 'mm'))
        unitsT = u.Unit(newUnits.get('time', 's'))

        self._units = {'length':unitsL, 'time':unitsT}
        self._lengthConversionFactor   = u.cm.to(unitsL)
        self._TorrConversionFactor     = u.torr.to(u.g / unitsL / (unitsT**2))
        self._cPConversionFactor       = u.cP.to(u.g / unitsL / unitsT)

    def _ComputeFlow(self, inletBC : Dict[str, float]={'pressure':100},
                        outletBC : Dict[str, float]={'pressure':26}):
        self.vessels.SetLinearSystem(inletBC, outletBC)
        self.vessels.SolveFlow()

    def _MakeConvection(self, U : float,
                    inletBC_Flow : Dict[str, float]={'pressure':50},
                    outletBC_Flow : Dict[str, float]={'pressure':26},
                    concentrationBC : Dict[str, float]={'inlet':50},
                    saveIn : Optional[str] = None):
        """Assembles the convection matrix for intravascular transport of oxygen.

        The convection operator -div(f.c_v) is discretized using an
        upwind scheme. At vascular node i with links to j and k, the
        discretization reads:
        f_{i,j}(c_i-c_j)/l_{i,j} + f_{i,k}(c_i-c_k)/l_{i,k}.
        The method solves for blood flow in the vascular network.
        Boundary conditions for the haemodynamics simulations
        can be specified as dictionnaries {'pressure' or 'flow':value}.

        Parameters
        ----------
        U : float
            Transmembrane permeability (length^2/time).
        inletBC_Flow : Dict[str, float]
            Inlet boundary condition, either 'pressure' or 'flow'.
        outletBC_Flow : Dict[str, float]
            Outlet boundary condition, either 'pressure' or 'flow'.
        concentrationBC : Dict[str, float]
            Either inlet or outlet boundary condition for vascular
            oxygen concentration.
        saveIn : str
            Optional, saves the matrix in npz format.
        """
        
        print("Assembling the convection matrix")
        self._ComputeFlow(inletBC_Flow, outletBC_Flow)

        bc,c0 = next(iter(concentrationBC.items()))
        if bc=='outlet':
            raise ValueError('Advection with Dirichlet condition on the venous end of the network has not been implemented yet.')

        self.M = sp.lil_matrix((self.nNodes, self.nNodes))
        inlets = self.vessels.inletNodes
        for n1 in self.vessels.nodes():
            if n1 in inlets:
                self.M[n1,n1] = 1.0
                
            for n2 in self.vessels.successors(n1):
                v, l = self.vessels.GetVelocity((n1,n2)), self.vessels[n1][n2]['length']
                self.M[n2,n2] = -v/l
                self.M[n2,n1] = v/l
            
        permeability = U/self.w # U = K_w/alpha with alpha solubility of O2 
        Gamma = np.array([1 if self.vessels.predecessors(n)
                          else 0 for n in self.vessels.nodes()])*permeability
        S     = np.array([np.pi*self.vessels[next(self.vessels.predecessors(n))][n]['length']*self.vessels[next(self.vessels.predecessors(n))][n]['radius']
                          if self.vessels.pred[n] else 0.0
                          for n in self.vessels.nodes()])
        Gamma = sp.diags([Gamma],[0]) # Permeability
        S     = sp.diags([S],[0])     # Vessel wall surface area (for each node) -> has to be broadcasted to the endothelial cells
        # Add the mass exchange terms
        
        self.E = -(Gamma.dot(self.C2v.T)).dot(self.C2t)
        #E = -Gamma.dot(self.C2v.T.dot(S).dot(self.C2t)) # Block (0,1) -> tissue (endothlial) part
        self.A = (Gamma.dot(self.C2v.T)).dot(self.C2v)
        #A += Gamma.dot(self.C2v.T.dot(S).dot(self.C2v)) # Block (0,0) -> vessel part
        del Gamma
        del S
        return 

    def _MakeReactionDiffusion(self, D : float, kt : float,
                               saveIn : Optional[str] = None, method=2):
        """Assembles the Reaction-Diffusion matrix for tissue cells.

        Parameters
        ----------
        D : float
            Oxygen diffusion coefficient in tissue.
        kt : float
            Oxygen consumption rates in tissue.
        saveIn : str
            Optional, saves the matrix in the file in .npz format.

        TODO: code the possibility for non-uniform grid.
        """
        print("Assembling the reaction-diffusion matrix", end='....')
        print(f"\n\tReaction rate {kt=}")
        print(f"\tDiffusion rate {D=}")
        nx, ny,nz  = self.nCells
        dx,dy,dz = self.spacing
        v = np.prod(self.spacing)

        # N.B.: this coefficients correspond to a discretization of the Laplacian operator -div(D*grad(f))

        coeffs = [D/dz/dz, D/dy/dy, D/dx/dx, # Lower diagonals
                  0,        # Main diagonal, coeff is set below
                  D/dz/dz, D/dy/dy, D/dx/dx] # Upper diagonals
        coeffs[3] = -sum(coeffs)
        offsets = [-nx*ny, -nx, -1,  # Lower diagonals
                   0,                                     # Main diagonal
                   nx*ny, nx, 1]     # Upper diagonals
    
        self.B = self.MakePoissonWithNeumannBC(D)
        # Adjust equations to account for internal vascular voxels (no diffusion nor reaction)
        # and endothelial cells (no reaction)
        DeleteCells = sp.eye(self.nVol, format='lil')
        R = sp.eye(self.nVol, format='lil') # The reaction term
        offsets.pop(3)
        coeffs.pop(3)
        D3 = np.zeros(self.nVol)
        for index3D, label in self.labels.elements.items():
            flatId = self.ToFlatIndexFrom3D(index3D)
            R[flatId, flatId] = 0.0                               
            if label==1:
                DeleteCells[flatId, flatId] = 0.0

                # Delete diffusion from vascular cells
                for m,n, neighbour, flux in zip((2,1,0,2,1,0), (-1,-1,-1,1,1,1), 
                                                (flatId+offset for offset in offsets), coeffs):
                    if index3D[m] + n < 0 or index3D[m] + n >= self.nCells[m]:
                        continue
                    self.B[neighbour, flatId] = 0.0
                    self.B[neighbour, neighbour] += flux

            else: #Label==2
                D3[flatId] = 1.0
                    
        self.B = -DeleteCells.dot(self.B) + R*kt*v - self.C2v.dot(self.E) + self.I4
        self.G = sp.diags(D3).dot(self.E.T) + self.C4

        del DeleteCells
        del D3
        del R

    def ToVTK(self, VTKFileName : str, X : np.ndarray):
        """
        Saves the oxygen array X (1D) to VTKFileName in
        VTK_STRUCTURED_POINTS format.
        """

        with open(VTKFileName, 'w') as f:
            print("Writing solution to", VTKFileName)

            f.write("# vtk DataFile Version 3.0\n")
            f.write("A mesh for the computation of oxygen perfusion.\n")
            # f.write("BINARY\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {self.nCells[0]+1} {self.nCells[1]+1} {self.nCells[2]+1}\n")
            f.write(f"ORIGIN {self.origin[0]} {self.origin[1]} {self.origin[2]}\n")
            f.write(f"SPACING {self.spacing[0]} {self.spacing[1]} {self.spacing[2]}\n")
            
            # Writing the data
            f.write(f"CELL_DATA {self.nVol}\n")
            f.write(f"SCALARS labels int 1\n")
            f.write(f"LOOKUP_TABLE default")
            f.write("\n")
            f.write("\n".join(tqdm(self.labels, desc=f"Writing labels to {VTKFileName}", total=self.nVol)))

            f.write(f"SCALARS PO2 float 1\n")
            f.write(f"LOOKUP_TABLE default")
            f.write("\n")
            f.write("\n".join(tqdm((str(xi) for xi in X), desc=f"Writing PO2 to {VTKFileName}", total=self.nVol)))

        return  
    
class TissueWithChoroid(TissueNew):
    def __init__(self, vessels: Union[DAG, str], dimensions=..., origin=..., nCells=..., spacing=None, units: Dict[str, str] = { 'length': 'mm','time': 's' }, w: float = 1 * u.um):
        super().__init__(vessels, dimensions, origin, nCells, spacing, units, w)
    
        """Assembles the Reaction-Diffusion matrix for tissue cells.

        Parameters
        ----------
        D : float
            Oxygen diffusion coefficient in tissue.
        kt : float
            Oxygen consumption rates in tissue.
        saveIn : str
            Optional, saves the matrix in the file in .npz format.

        TODO: code the possibility for non-uniform grid.
        """
        print("Assembling the reaction-diffusion matrix", end='....')
        print(f"\n\tReaction rate {kt=}")
        print(f"\tDiffusion rate {D=}")
        nx, ny,nz  = self.nCells
        dx,dy,dz = self.spacing
        v = np.prod(self.spacing)

        # N.B.: this coefficients correspond to a discretization of the Laplacian operator -div(D*grad(f))

        coeffs = [D/dz/dz, D/dy/dy, D/dx/dx, # Lower diagonals
                  0,        # Main diagonal, coeff is set below
                  D/dz/dz, D/dy/dy, D/dx/dx] # Upper diagonals
        coeffs[3] = -sum(coeffs)
        offsets = [-nx*ny, -nx, -1,  # Lower diagonals
                   0,                                     # Main diagonal
                   nx*ny, nx, 1]     # Upper diagonals
    
        self.B = self.MakePoissonWithNeumannBC(D)
        # Adjust equations to account for internal vascular voxels (no diffusion nor reaction)
        # and endothelial cells (no reaction)
        DeleteCells = sp.eye(self.nVol, format='lil')
        R = sp.eye(self.nVol, format='lil') # The reaction term
        offsets.pop(3)
        coeffs.pop(3)
        D3 = np.zeros(self.nVol)
        for index3D, label in self.labels.elements.items():
            flatId = self.ToFlatIndexFrom3D(index3D)
            R[flatId, flatId] = 0.0                               
            if label==1:
                DeleteCells[flatId, flatId] = 0.0

                # Delete diffusion from vascular cells
                for m,n, neighbour, flux in zip((2,1,0,2,1,0), (-1,-1,-1,1,1,1), 
                                                (flatId+offset for offset in offsets), coeffs):
                    if index3D[m] + n < 0 or index3D[m] + n >= self.nCells[m]:
                        continue
                    self.B[neighbour, flatId] = 0.0
                    self.B[neighbour, neighbour] += flux

            else: #Label==2
                D3[flatId] = 1.0
                    
        self.B = -DeleteCells.dot(self.B) + R*kt*v - self.C2v.dot(self.E) + self.I4
        self.G = sp.diags(D3).dot(self.E.T) + self.C4

        del DeleteCells
        del D3
        del R

class Tissue(object):
    """A class for the generation of equations of 1D-3D oxygen perfusion of the tissue.

    Attributes:
    -----------
    Vessels : VascularNetwork
        The vascular network embedded within the tissue.
    endotheliumThickness : int
        The thickness of the endothelium. Make sure units match Vessels.
    labels : NDSparseArray
        Access the mesh's labels.
    v : float
        The volume of a mesh's cell.
    nVol : int
        The number of cells in the mesh.
    nPoints : int
        The number of vascular nodes in Vessels.
    nSeg : int
        The number of vascular segments in Vessels.
    A : scipy.sparse.sparse_matrix
    rhs : scipy.sparse.sparse_matrix
        The right hand side of the equation. Includes boundary conditions.
    CellToSegment : scipy.sparse.sparse_matrix
        The connectivity matrix between cells and their associated vascular segments.
    
    Methods:
    --------
    ImportVessels(ccoFileName)
        Reads the vascular network to be used from a .cco file.
    MakeMassTransfer()
        Add the vessel-endothelium mass transfer terms to the system.
    MakeReactionDiffusion(D, kt)
        Add reaction-diffusion terms to the overall system.
    MakeConvection(**kwargs)
        Add O2 convection in the vessels to the system.
    MakeRHS(**kwargs)
        Assembles the RHS according to boundary conditions in **kwargs.
    """
 
    def __init__(self, **kwargs):
        """Create an instance of the 'Tissue' class,

        Keyword arguments
        -----------------
        Vessels : VascularNetwork
            The vascular network to be embedded in the tissue.
        ccoFile : str
            A .cco file where the vascular network is stored.
            Used if Vessels is not given.
        w : float
            The thickness of the vessels' walls.
            Default to 1micron.        
        
        """

        Vessels = kwargs.get('Vessels', False)
        ccoFile = kwargs.get('ccoFile', False)
        endotheliumThickness = kwargs.get('w', 1e-3)
        dimensions = kwargs.get('dimensions', None)
        spacing    = kwargs.get('spacing', 3*[endotheliumThickness])
        if Vessels:
            self.Vessels = Vessels
            self.Vessels.w = endotheliumThickness 
        elif ccoFile:
            print(f"Reading vascular network from '{ccoFile}'.")
            self.ImportVessels(ccoFile, **kwargs)
            print(f'Labeling mesh with wall thickness {endotheliumThickness}.')
        else:
            raise ValueError("Provide either a VascularNetwork' or a valid .cco file.")
        
        self.C3, self.C4, self.I4 = self.Vessels.LabelMesh(self.endotheliumThickness)
        return

    # Various useful property getters
    @property
    def nx(self):
        return self.Vessels.mesh.nCells
    @property
    def dx(self):
        return self.Vessels.mesh.spacing
    @property
    def v(self):
        return self.Vessels.mesh.v
    @property
    def labels(self):
        return self.Vessels.mesh.labels
    @property
    def endotheliumThickness(self):
        return self.Vessels.w
    @property
    def nVol(self):
        return self.Vessels.mesh.nCellsTotal
    @property
    def nPoints(self):
        return self.Vessels.nNodes()
    @property
    def nSeg(self):
        return self.Vessels.nVessels()
    @property
    def Vessels(self):
        return self._Vessels
    @Vessels.setter
    def Vessels(self, newVessels):
        if isinstance(newVessels, VascularNetwork):
            self._Vessels = newVessels
        else:
            raise ValueError("Input 'newVessels' must be of type VascularNetwork.")
        return
    @property
    def Mesh(self):
        return self._Vessels.mesh
    
    def ImportVessels(self, ccoFileName : str, endotheliumThickness : float = 1e-3, **kwargs):
        ccoFile = kwargs.pop('ccoFile', None)
        units = kwargs.pop('units', {'length':'cm', 'time':'s'})
        self.Vessels = VascularNetwork(ccoFileName, units, **kwargs)
        self.Vessels.w = endotheliumThickness
        return

    def MakeMassTransfer(self, U : float, saveIn : Optional[str] = None):
        """Assembles the mass transfer coefficients between vascular nodes and endothelial grid cells.

        The CellToSegment matrix is of the form [-I C], where -I links
        a vascular node with itself and C encodes the connectivity of
        the vascular nodes with endothelial cells.

        Parameters
        ----------
        U : float
            Transmembrane permeability coefficient (in L/T). 
        saveIn : str
            Optional, saves the matrix in the file in .npz format.
        """

        print("Assembling the mass transfer coefficients matrix", end='....')
            
        ## Use this one to cancel mass exchange with the inlet(s) and the 'backbone'
        permeability = U/self.endotheliumThickness # U = K_w/alpha with alpha solubility of O2
        # Gamma = np.array([permeability/self.Vessels.G[next(self.Vessels.G.predecessors(n))][n]['length']
        #                   if (self.Vessels.G.predecessors(n) and self.Vessels.G.nodes[n]['stage']>=-1)
        #                   else 0 for n in self.Vessels.G.nodes()])


        Gamma = np.array([1 if self.Vessels.G.predecessors(n)
                          else 0 for n in self.Vessels.G.nodes()])*permeability
        
        M = sp.diags(Gamma, offsets=0, format='csr', dtype=np.float32)
        
        M = sp.diags(Gamma*U/self.endotheliumThickness,
                     offsets=0, format='csr', dtype=np.float32)
        del Gamma
        # M = self.C3[1:,1:].T.dot(M.dot(self.C3[1:, 1:]))
        # M = self.C3.T.dot(M.dot(self.C3))
        M = self.CellToSegment.T.dot(M.dot(self.CellToSegment))
        
        # Add the equations linking vascular cells with vascular nodes
        # M = sp.lil_matrix((self.nPoints+self.nVol, self.nPoints+self.nVol))
        # nodesPosition = np.array([self.Vessels.G.nodes[n]['position']
        #                           if self.Vessels.G.predecessors(n)
        #                           else np.array((np.inf, np.inf, np.inf))
        #                           for n in range(self.nPoints)])

        # inletNodes = self.Vessels.inletNodes
        # for index3D, label in self.labels.elements.items():
        #     if label==2:
        #         cellPos = self.Mesh.CellCenter(index3D)
        #         closestNode = np.argmin(np.linalg.norm(nodesPosition-cellPos, axis=1))
        #         if closestNode in inletNodes:
        #             closestNode = next(self.Vessels.G.successors(closestNode))

        #         if self.Vessels.G.nodes[closestNode]['stage']>=-1:
        #             flatId = self.nPoints+self.Mesh.ToFlatIndexFrom3D(index3D)
        #             M[flatId, closestNode] = -permeability
        #             M[flatId, flatId] = permeability
        #             M[closestNode, closestNode] += permeability
        #             M[closestNode, flatId] -=permeability

                
        M = M + sp.vstack([sp.csr_matrix((self.nPoints, self.nPoints+self.nVol)), sp.hstack([-self.C4, self.I4])])
        
        # del self.C4
        # del self.C3
        # del self.I4

        if saveIn:
            sp.save_npz(saveIn, M)
            sp.save_npz(saveIn[:-4]+"_Gamma.npz", M)

        try:                                                                                 
            def foo(x): # An empty function to test if A is defined                          
                pass
            foo(self.A)
            self.A = self.A+M
        except AttributeError: # If A not defined, create it                                 
            self.A = M

        print(' Done.')        
        return

    def MakeReactionDiffusion(self, D : float, kt : float,
                              saveIn : Optional[str] = None, method=2):
        """Assembles the Reaction-Diffusion matrix for tissue cells.

        Parameters
        ----------
        D : float
            Oxygen diffusion coefficient in tissue.
        kt : float
            Oxygen consumption rates in tissue.
        saveIn : str
            Optional, saves the matrix in the file in .npz format.

        TODO: code the possibility for non-uniform grid.
        """
        print("Assembling the reaction-diffusion matrix", end='....')
        print(f"\n\tReaction rate {kt=}")
        print(f"\tDiffusion rate {D=}")
        spacing = self.dx
        v = 1.0 #np.prod(self.Mesh.dimensions) # Dimensions of the slab of tissue 
        nx, ny,nz  = self.nx
        dx,dy,dz = spacing

        # N.B.: this coefficients correspond to a discretization of the Laplacian operator -div(D*grad(f))

        coeffs = [D/dz/dz, D/dy/dy, D/dx/dx, # Lower diagonals
                  0,        # Main diagonal, coeff is set below
                  D/dz/dz, D/dy/dy, D/dx/dx] # Upper diagonals
        coeffs[3] = -sum(coeffs)
        offsets = [-nx*ny, -nx, -1,  # Lower diagonals
                   0,                                     # Main diagonal
                   nx*ny, nx, 1]     # Upper diagonals
        
        if method == 1:

            t = time()
            M = self.Mesh.MakePoissonWithNeumannBC(D)
            print("\tTime to generate the Reac-Diff operator:", time()-t)
            DeleteCells = np.ones(M.shape[0]) #sp.eye(nx*ny*nz, format='lil')
            
            offsets.pop(3)
            coeffs.pop(3)
            t = time()
            R = np.ones(M.shape[0]) # sp.eye(M.shape[0], format='lil') # The reaction term
            for index3D, label in self.labels.elements.items():
                flatId = self.Mesh.ToFlatIndexFrom3D(index3D)
                R[flatId] = 0.0
                if label==1:
                    DeleteCells[flatId] = 0.0
                    
                    # Delete diffusion from vascular cells
                    for neighbour, flux in zip((flatId + offset for offset in offsets), coeffs):
                        M[neighbour, flatId] = 0.0
                        M[neighbour, neighbour] += flux

            t1 = time()
            DeleteCells = sp.diags(DeleteCells, 0, format='csr')
            M = -DeleteCells.dot(M - sp.diags(R)*kt*v)
            print("\tTime to make the multiplication", time()-t)
            print("\tTime to delete diffusion in vessels",time()-t1)
            if saveIn:
                sp.save_npz(saveIn, M)

            t = time()
            M = sp.vstack((sp.lil_matrix( (self.nPoints, self.nVol), dtype=np.float32), M)).tolil()
            M = sp.hstack((sp.lil_matrix( (self.nVol+self.nPoints, self.nPoints), dtype=np.float32), M)).tolil()
            print("\tTime to pad the matrix:", time()-t)
            try:
                def foo(x): # An empty function to test if A is defined
                    pass
                foo(self.A)
                self.A = self.A+M
            except AttributeError: # If A not defined, create it
                self.A = M

            del M
            del DeleteCells
                
        if method == 2:

            M = self.Mesh.MakePoissonWithNeumannBC_Kronecker(D)
            # Adjust equations to account for internal vascular voxels (no diffusion nor reaction)
            # and endothelial cells (no reaction)
            DeleteCells = sp.eye(nx*ny*nz, format='lil')
            R = sp.eye(M.shape[0], format='lil') # The reaction term
            offsets.pop(3)
            coeffs.pop(3)
            for index3D, label in self.labels.elements.items():
                flatId = self.Mesh.ToFlatIndexFrom3D(index3D)
                R[flatId, flatId] = 0.0                               
                if label==1:
                    DeleteCells[flatId, flatId] = 0.0

                    # Delete diffusion from vascular cells
                    for neighbour, flux in zip((flatId + offset for offset in offsets), coeffs):
                        M[neighbour, flatId] = 0.0
                        M[neighbour, neighbour] += flux
                        
            M = -DeleteCells.dot(M) - R*kt*v
                
            M = sp.vstack((sp.lil_matrix( (self.nPoints, self.nVol), dtype=np.float32), M))
            M = sp.hstack((sp.lil_matrix( (self.nVol+self.nPoints, self.nPoints), dtype=np.float32), M))

            try:
                def foo(x): # An empty function to test if A is defined
                    pass
                foo(self.A)
            except AttributeError: # If A not defined, create it
                self.A = M

            del M            
            del DeleteCells
                           
        if saveIn:
            sp.save_npz(saveIn, self.A.tocsr()[self.nPoints:, self.nPoints:])

        print(' Done.')
        return
    
    def MakeConvection(self, inletBC : Dict[str, float]={'pressure':50},
                       outletBC : Dict[str, float]={'pressure':26},
                       saveIn : Optional[str] = None):
        """Assembles the convection matrix for intravascular transport of oxygen.

        The convection operator -div(f.c_v) is discretized using an
        upwind scheme. At vascular node i with links to j and k, the
        discretization reads:
        f_{i,j}(c_i-c_j)/l_{i,j} + f_{i,k}(c_i-c_k)/l_{i,k}.
        The method solves for blood flow in the vascular network.
        Boundary conditions for the haemodynamics simulations
        can be specified as dictionnaries {'pressure' or 'flow':value}.

        Parameters
        ----------
        inletBC : Dict[str, float]
            Inlet boundary condition, either 'pressure' or 'flow'.
        outletBC : Dict[str, float]
            Outlet boundary condition, either 'pressure' or 'flow'.
        saveIn : str
            Optional, saves the matrix in npz format.
        """
        
        print("Assembling the convection matrix")
        self.Vessels.SetLinearSystem(inletBC, outletBC)
        self.Vessels.SolveFlow()
        
        # Possible gain in time by going through inlet node (i.e., in_degree==0)
        # then make another loop for the non inlets.
        # K = -1.0 * networkx.incidence_matrix(self.Vessels.G, oriented=True).T
        # for i in self.Vessels.inletNodes:
        #     K.indptr = np.insert(K.indptr, K.indptr[i],i)
        #     K._shape = (K.shape[0]+1, K.shape[1])
        #     K[i,i] = 1.0

        # K = K.tolil()
        #data, rows = K.data, K.rows        
        # for i, edge in enumerate(zip(data,rows)):
        #     k, cols = np.array(edge[0]), np.array(edge[1])
        #     if 0<=i<=3:
        #         print(i, k, cols)
        #     # k = -1 for start of vessel, 1 otherwise
        #     try:
        #         n1, n2 = cols[k==1][0], cols[k==-1][0]
        #         v, l = self.Vessels.GetVelocity((n1,n2)), self.Vessels.G[n1][n2]['length']
        #         K.data[i] = [-val*v/l for val in K.data[i]]
        #     except IndexError:
        #         print(f"Inlet node at row {i}, node {cols}")
        #         pass

        K = sp.lil_matrix((self.nPoints, self.nPoints))
        inlets = self.Vessels.inletNodes
        for n1 in self.Vessels.G.nodes():
            if n1 in inlets:
                K[n1,n1] = 1.0
                
            for n2 in self.Vessels.G.successors(n1):
                v, l = self.Vessels.GetVelocity((n1,n2)), self.Vessels.G[n1][n2]['length']
                K[n2,n2] = -v/l
                K[n2,n1] = v/l
            
        K = K.tocsr()
        M = sp.csr_matrix((K.data,
                           K.indices,
                           np.pad(K.indptr, (0, self.nVol), "edge")),
                          shape=(self.nPoints+self.nVol, self.nPoints+self.nVol))
        print('Done.')
        
        if saveIn:
            sp.save_npz(saveIn, M.tocsr()[:self.nPoints,:self.nPoints])

        try:
            def foo(x): # An empty function to test if A is defined
                pass
            foo(self.A)
            self.A = self.A + M
        except AttributeError: # If A not defined
            print("Initiating matrix self.A...")
            self.A = M

        return

    def MakeRHS(self, cv : float, saveIn : Optional[str] = None):
        """Assembles the RHS.

        At the tissue boundary, no-flux conditions are applied. For
        the vascular O2 convection problem, concentration at inlet is
        needed.

        Parameters
        ----------
        cv : float
             Inlet concentration of blood oxygen.
        saveIn : str
            Optional, saves the matrix in npz format.
        """

        print("Assembling the right-hand-side vector", end='....')
        self.rhs = sp.lil_matrix((self.nPoints+self.nVol,1), dtype=np.float32)
        for node in self.Vessels.G.nodes():
            if not self.Vessels.G.pred[node]:
                # No parent found
                self.rhs[node,0] = cv
        # NOTE: the no-flux BC for tissue does not add anything to the rhs
        self.rhs = self.rhs.tocoo()
        if saveIn:
            sp.save_npz(saveIn, self.rhs)
        print(' Done.')
        return

    def Solve(self, checkForEmptyRows : bool=False, solver='pgmres', preconditioner='bjacobi', maxIter=5e3):

        self.A = self.A.tocsr()
        self.A.eliminate_zeros()

        if checkForEmptyRows:
            # Slicing rows is not implemented yet for scipy sparse_arrays
            # so we transform self.A into a sparse_matrix
            self.A = sp.csr_matrix((self.A.data, self.A.indices, self.A.indptr),
                                   shape = self.A.shape, dtype=float) 
            emptyRows = []            
            for i in range(self.A.shape[0]):
                if not list(self.A[i,:].data):
                    emptyRows.append(i)
            print(f"Found {len(emptyRows)} empty rows: {emptyRows}.")
            print("Their labels: ", [self.labels[self.Vessels.mesh.FlatIndexTo3D(cellFlatIndex-self.nPoints)] for cellFlatIndex in emptyRows])

        t = time()
        # print("Solving with scipy.sparse direct solver...")
        # x = scipy.sparse.linalg.spsolve(self.A, self.rhs)

        # This wrapper may be faster? Though no multi-threading
        # It is also not working for some unknown reason...
        # print("Solving with mkl solver...")
        # b = self.rhs.toarray()[:,0].astype(np.float32)
        # A = self.A.astype(np.float32)
        # x = sparse_qr_solve_mkl(A, b)

        # # Preconditioner
        # print("Solving with gmres...", end='')
        # print("\n\tMaking preconditioner (incomplete LU)")
        # self.A = self.A.tocsc()
        # ilu = scipy.sparse.linalg.spilu(self.A)
        # M_x = lambda x: ilu.solve(x)
        # n,m = self.A.shape
        # M = scipy.sparse.linalg.LinearOperator((n,n), M_x)
        # x, info = scipy.sparse.linalg.gmres(self.A, self.rhs.toarray(), M=M)
        # print("gmrest exited with code:", info)

        # Using PETSc
        def monitor(ksp, its, rnorm):
                if its%100 == 0:
                    print('%5d      %20.15g / %2.2g'%(its,rnorm, ksp.getTolerances()[0]))

        # Sanity check
        if self.A.getformat() != 'csr':
            self.A = self.A.tocsr()

        comm = mpi_comm
        petsc_mat = PETSc.Mat().createAIJ(size=self.A.shape,
                                          csr=(self.A.indptr,
                                               self.A.indices,
                                               self.A.data), comm=comm)

        solverType = solver #'bcgs'
        precondType = preconditioner #'ilu' # 'bjacobi'
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setType(solverType)
        ksp.setMonitor(monitor)
        pc = ksp.getPC()
        pc.setType(precondType)
        ksp.setTolerances(rtol=1e-6, max_it=maxIter)
        ksp.view()
        ksp.setConvergenceHistory()

        print(f"Solving with PETSc and {ksp.getType()} (preconditionner: {pc.getType()})")

        print(f"Iter.      residual norm / rtol")
        ksp.setOperators(petsc_mat)
        xpetsc = PETSc.Vec().create(comm=comm)
        b = PETSc.Vec().createWithArray(self.rhs.toarray()[:,0], comm=comm)
        xpetsc.setSizes(self.A.shape[0], None)
        xpetsc.setUp()
        ksp.solve(b, xpetsc)
        x = xpetsc.getArray()

        print("\tConvergence details")
        print(f"\t\tTime for solver ({ksp.getIterationNumber()} iter.): {time()-t}")
        print(f"\t\t{ksp.getConvergedReason()=} {ksp.getResidualNorm()=} ")

        plt.semilogy(ksp.getConvergenceHistory(), '-.')
        plt.title(f"Convergence history of PETSc {ksp.getType()} solver\nwith {pc.getType()} precondictionning.")
        
        del ksp
        del petsc_mat
        del b
        del xpetsc
        
        for n in self.Vessels.G.nodes():
            self.Vessels.G.nodes[n]['PO2'] = x[n]
        
        return x[list(networkx.topological_sort(self.Vessels.G))], x[self.nPoints:]
    
    
    def VesselsToVTK(self, VTKFileName : str):
        self.Vessels.VesselsToVTK(VTKFileName)
        return

    def MeshToVTK(self, VTKFileName : str):
        self.Vessels.MeshToVTK(VTKFileName)
        return

    def ToVTK(self, VTKFileName : str, X : np.ndarray):
        """
        Saves the oxygen array X (1D) to VTKFileName in
        VTK_STRUCTURED_POINTS format.
        """

        with open(VTKFileName, 'w') as f:
            print("Writing solution to", VTKFileName)

            f.write("# vtk DataFile Version 3.0\n")
            f.write("A mesh for the computation of oxygen perfusion.\n")
            # f.write("BINARY\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {self.Mesh.nCells[0]+1} {self.Mesh.nCells[1]+1} {self.Mesh.nCells[2]+1}\n")
            f.write(f"ORIGIN {self.Mesh.origin[0]} {self.Mesh.origin[1]} {self.Mesh.origin[2]}\n")
            f.write(f"SPACING {self.Mesh.spacing[0]} {self.Mesh.spacing[1]} {self.Mesh.spacing[2]}\n")
            
            # Writing the data
            f.write(f"CELL_DATA {self.nVol}\n")
            f.write(f"SCALARS labels int 1\n")
            f.write(f"LOOKUP_TABLE default")
            f.write("\n")
            f.write("\n".join(tqdm(self.labels, desc=f"Writing labels to {VTKFileName}", total=self.nVol)))

            f.write(f"SCALARS PO2 float 1\n")
            f.write(f"LOOKUP_TABLE default")
            f.write("\n")
            f.write("\n".join(tqdm((str(xi) for xi in X), desc=f"Writing PO2 to {VTKFileName}", total=self.nVol)))

        return     

            
