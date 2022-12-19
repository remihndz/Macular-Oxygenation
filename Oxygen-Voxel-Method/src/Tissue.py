from VascularNetwork import VascularNetwork
# from Mesh import UniformGrid
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
from typing import Union
import numpy as np
from typing import Dict, Optional

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
    A : scipy.sparse.sparse_array
    rhs : scipy.sparse.sparse_array
        The right hand side of the equation. Includes boundary conditions.
    CellToSegment : scipy.sparse.sparse_array
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
            Default to 1e-3 (1micron in mm).        
        
        """

        Vessels = kwargs.get('Vessels', False)
        ccoFile = kwargs.get('ccoFile', False)
        endotheliumThickness = kwargs.get('w', 1e-3)
        if Vessels:
            self.Vessels = Vessels
            self.Vessels.w = endotheliumThickness
        elif ccoFile:
            print(f"Reading vascular network from '{ccoFile}'.")
            self.ImportVessels(ccoFile, endotheliumThickness)
            print(f'Labeling mesh with wall thickness {endotheliumThickness}mm.')
        else:
            raise ValueError("Provide either a VascularNetwork' or a valid .cco file.")

        self.CellToSegment = self.Vessels.LabelMesh(self.endotheliumThickness)        
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
    
    def ImportVessels(self, ccoFileName : str, endotheliumThickness : float = 1e-3):
        self.Vessels = VascularNetwork(ccoFileName, spacing=endotheliumThickness)
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
            Transmembrane permeability coefficient.
        saveIn : str
            Optional, saves the matrix in the file in .npz format.
        """
        
        print("Assembling the mass transfer coefficients matrix", end='....')
        vesselData = self.Vessels.GetVesselData(['radius', 'length'])
        # Compute (curved) surface area of each vessel
        # as an estimate of the contact area between
        # intravascular and endothelial cells.
        # The results is scaled by the number of
        # endothelial cells attached to the vessel,
        # namely the number of 1s in a row -1 for
        # the column corresponding to the node itself.

        # A = 2*np.pi*vesselData['radius']*vesselData['length']/( self.CellToSegment.sum(axis=1) - 1 )
        A = vesselData['radius'].max()*vesselData['length'].max()*( self.CellToSegment.sum(axis=1)-1 )
        # del A
        M = sp.diags(A*U/self.endotheliumThickness,
                     offsets=0, format='csr', dtype=np.float32)
        M = self.CellToSegment.T @ M # Here @ is the matrix-matrix product for scipy sparse_arrays
        M = M @ self.CellToSegment
        
        try:
            #print(type(A), type(M))
            self.A -= M
        except:
            self.A = sp.lil_array((self.nPoints+self.nVol, self.nPoints+self.nVol), dtype=np.float32)
            self.A = -M
        print(' Done.')

        if saveIn:
            sp.save_npz(saveIn, M)
        
        return

    def MakeReactionDiffusion(self, D : float, kt : float,
                              saveIn : Optional[str] = None):
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
        M = sp.lil_array((self.nVol, self.nVol), dtype=np.float32)
        spacing = self.dx
        v = np.prod(spacing) # Volume of each cells
        nCells  = self.nx
        dx,dy,dz = spacing
        for cellFlatIndex in range(self.nVol):
            cell3DIndex = np.array(self.Vessels.mesh.FlatIndexTo3D(cellFlatIndex))
            cellLabel = self.labels[(ind for ind in cell3DIndex)]

            if cellLabel in [0,2]: # Diffusion does not happen in intravascular elements
                # Add the flux of each face.
                # Flux is 0 on boundary faces.
                for m in range(3):
                    for n in [-1, 1]:
                        neighbour = cell3DIndex + np.array([n if l==m else 0 for l in range(3)])
                        if self.Vessels.mesh.IsInsideMesh(neighbour):
                            neighbourFlatIndex = self.Vessels.mesh.ToFlatIndexFrom3D(neighbour)
                            flux = D*spacing[m-1]*spacing[m-2]/spacing[m]
                            M[cellFlatIndex, cellFlatIndex] -= flux
                            M[cellFlatIndex, neighbourFlatIndex] = flux

                # Add oxygen consumption if it is a tissue cell
                if cellLabel == 0:
                    M[cellFlatIndex, cellFlatIndex] -= kt*v

        try:
            #print(type(self.A), type(M))
            self.A[self.nPoints:, self.nPoints:] += M
        except:
            self.A = sp.lil_array((self.nPoints+self.nVol, self.nPoints+self.nVol), dtype=np.float32)
            self.A[self.nPoints:, self.nPoints:] = M
        print(' Done.')

        if saveIn:
            sp.save_npz(saveIn, M.tocsr())
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
        M = sp.lil_array((self.nPoints, self.nPoints), dtype=np.float32)
        for node in self.Vessels.G.nodes():
            successors = self.Vessels.G.successors(node)

            if not self.Vessels.G.pred[node]: 
                # It is a inlet node
                print(f"Inlet node equation is at row {node}")
                M[node, node] = 1
            else:
                for otherNode in successors:
                    flow = self.Vessels.G[node][otherNode]['flow']
                    length = self.Vessels.G[node][otherNode]['length']
                    M[node, otherNode] = -flow/length
                    M[otherNode, node] = flow/length
                    M[node, node] += flow/length

        try:
            #print(type(A), type(M))
            self.A[:self.nPoints, :self.nPoints] += M
        except:
            self.A = sp.lil_array((self.nPoints+self.nVol, self.nPoints+self.nVol), dtype=np.float32)
            self.A[:self.nPoints, :self.nPoints] = M
        print('Done.')

        if saveIn:
            sp.save_npz(saveIn, M.tocsr())

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
        self.rhs = sp.lil_array((self.nPoints+self.nVol,1), dtype=np.float32)
        for node in self.Vessels.G.nodes():
            if not self.Vessels.G.pred[node]:
                # No parent found
                self.rhs[node,0] = cv
        # NOTE: the no-flux BC for tissue does not add anything to the rhs
        self.rhs = self.rhs.tocsr()
        if saveIn:
            sp.save_npz(saveIn, self.rhs)
        print(' Done.')
        return

    def Solve(self) -> sp._arrays.csr_array:
        self.A = self.A.tocsr()
        self.A.eliminate_zeros()
        x = scipy.sparse.linalg.spsolve(self.A, self.rhs)
        return x[:self.nPoints], x[self.nPoints:]
    
    def VesselsToVTK(self, VTKFileName : str):
        self.Vessels.VesselsToVTK(VTKFileName)
        return

    def MeshToVTK(self, VTKFileName : str):
        self.Vessels.MeshToVTK(VTKFileName)
        return
