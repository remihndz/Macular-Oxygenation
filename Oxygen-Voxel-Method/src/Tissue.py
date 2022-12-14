from VascularNetwork import VascularNetwork
# from Mesh import UniformGrid
import networkx as nx
import scipy.sparse as sp


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
    Mc : scipy.sparse._arrays
        Matrix discretizing the O2 convection in vessels.
    Md : scipy.sparse._arrays
        Matrix discretizing the O2 diffusion in tissue.
    Gamma : scipy.sparse._arrays
        Matrix of the endothelium-to-tissue mass transfer flux balance.
    R : scipy.sparse._arrays
        Diagonal matrix of the (constant) reaction rates and cuboid volume.
    rhs : scipy.sparse._arrays
        The right hand side of the equation. Includes boundary conditions.
    CellToSegment : scipy.sparse._arrays
        The connectivity matrix between cells and their associated vascular segments.
    
    Methods:
    --------
    ImportVessels(ccoFileName)
        Reads the vascular network to be used from a .cco file.
    MakeMassTransfer()
        Assembles the matrix Gamma of mass transfer flux balance.
    MakeDiffusion()
        Assembles the matrix Md of the O2 diffusion in tissue.
    MakeConvection()
        Assembles the matrix Mc of the O2 convection in vessels.
    MakeReaction()
        Assembles the diagonal matrix R of oxygen consumption in tissue.
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

        self.CellToSegment = self.Vessels.LabelMesh(self.endotheliumThickness).tocsr(copy=False)
        return

    # Various attribute getters
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
        return self.Vessels.mesh.nCellsTotal()
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
    
    def ImportVessels(self, ccoFileName : str, endotheliumThickness : float = 1e-3):
        self.Vessels = VascularNetwork(ccoFileName, spacing=endotheliumThickness)
        self.Vessels.w = endotheliumThickness
        return

    def MakeMassTransfer(self, U : float):
        """Assembles the mass transfer coefficients between vascular nodes and endothelial grid cells.

        Parameters
        ----------
        U : float
            Transmembrane permeability coefficient.
        """
        vesselData = self.Vessels.GetVesselData(['radius', 'length'])
        # Compute (curved) surface area of each vessel
        # as an estimate of the contact area between
        # intravascular and endothelial cells.
        # The results is scaled by the number of
        # endothelial cells attached to the vessel.
        
        A = 2*np.pi*vesselData['radius']*vesselData['length']/self.CellToSegment.sum(axis=1)
        # del A
        M = sp.diags(A*U/self.endotheliumThickness,
                     offsets=0, format='csr', dtype=np.float32)
        M = CellToSegment.T @ M # Here @ is the matrix-matrix product for scipy sparse_arrays
        M = M @ CellToSegment
        self.Mc = M
        return

    def MakeReactionDiffusion(self, D : float, kt : float):
        """Assembles the Reaction-Diffusion matrix for tissue cells.

        Parameters
        ----------
        D : float
            Oxygen diffusion coefficient in tissue.
        kt : float
            Oxygen consumption rates in tissue.
        """
        M = sp.dok_array((self.nVol, self.nVol), dtype=np.float32)
        spacing = self.dx
        dx,dy,dz = spacing
        for cellFlatIndex in range(self.nVol):
            cell3DIndex = self.Vessels.mesh.FlatIndexTo3D(cellFlatIndex)
            if self.labels[cell3DIndex] == 0: # If in tissue, consumption rate non-nill
                # Here assume cubic control volumes (i.e., dx=dy=dz)
                M[cellFlatIndex, cellFlatIndex] = -2*D*(dx*dy/dz + dy*dz/dx + dx*dz/dy) - kt*self.v
                for m in [-1, 1]:
                    for l in range(3):
                        neighbourCell = cell3DIndex
                        neighbourCell[l] += m
                        neighbourFlatIndex = self.Vessels.mesh.3DToFlatIndex(neighbourCell)
                        M[cellFlatIndex, neighbourFlatIndex] = D*spacing[j-1]*spacing[j-2]/spacing[j]
        self.Md = M.tocsr(copy=True)
        del M
        return
    
    def MakeConvection(self):
        # TODO
        return
    
