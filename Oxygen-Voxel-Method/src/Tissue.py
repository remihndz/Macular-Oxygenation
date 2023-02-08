from VascularNetwork import VascularNetwork
# from Mesh import UniformGrid
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg
from NDSparseMatrix import SparseRowIndexer
import numpy as np
from typing import Dict, Optional, Union
# from sparse_dot_mkl import sparse_qr_solve_mkl
from petsc4py import PETSc
import matplotlib.pyplot as plt
from time import time

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
            Default to 1e-3 (1micron in mm).        
        
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
            print(f'Labeling mesh with wall thickness {endotheliumThickness}mm.')
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
        units = kwargs.pop('units', 'mm')
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

        Gamma = np.ones(self.nPoints, dtype=np.float32)*np.sum(2*np.pi*vesselData['radius']*vesselData['length'])
        # This might not be necessary because the connectivty matrix between endothelial and nodes
        # links endothelial cells with the end node of a segment (i.e., the inlet nodes are never connected).
        # Gamma[self.Vessels.inletNodes] = 0.0 # No mass exchange with the inlet nodes.
        
        M = sp.diags(Gamma*U/self.endotheliumThickness,
                     offsets=0, format='csr', dtype=np.float32)
    
        del Gamma
        # M = self.C3[1:,1:].T.dot(M.dot(self.C3[1:, 1:]))
        M = self.C3.T.dot(M.dot(self.C3))
        # M = self.CellToSegment.T @ M @ self.CellToSegment # Here @ is the matrix-matrix product for scipy sparse_arrays
        
        # Add the equations linking vascular cells with vascular nodes
        M = M + sp.vstack([sp.csr_matrix((self.nPoints, self.nPoints+self.nVol)), sp.hstack([-self.C4, self.I4])])
        
        del self.C4
        del self.C3
        del self.I4

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
        v = np.prod(self.Mesh.dimensions) # Dimensions of the slab of tissue 
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

            M = self.Mesh.MakePoissonWithNeumannBC(D)
            DeleteCells = sp.eye(nx*ny*nz, format='lil')
            
            offsets.pop(3)
            coeffs.pop(3)
            R = sp.eye(M.shape[0], format='lil') # The reaction term
            for index3D, label in self.labels.elements.items():
                flatId = self.Mesh.ToFlatIndexFrom3D(index3D)
                R[flatId, flatId] = 0.0
                if label==1:
                    DeleteCells[flatId, flatId] = 0.0
                    
                    # Delete diffusion from vascular cells
                    for neighbour, flux in zip((flatId + offset for offset in offsets), coeffs):
                        M[neighbour, flatId] = 0.0
                        M[neighbour, neighbour] += flux
            
            M = -DeleteCells.dot(M - R*kt*v)
            
            if saveIn:
                sp.save_npz(saveIn, M)

            M = sp.vstack((sp.lil_matrix( (self.nPoints, self.nVol), dtype=np.float32), M)).tolil()
            M = sp.hstack((sp.lil_matrix( (self.nVol+self.nPoints, self.nPoints), dtype=np.float32), M)).tolil()
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
        M = sp.lil_matrix((self.nPoints+self.nVol, self.nPoints+self.nVol), dtype=np.float32)
        for node in self.Vessels.G.nodes():
            successors = self.Vessels.G.successors(node)
            
            if not self.Vessels.G.pred[node]: 
                # It is an inlet node
                print(f"Inlet node equation is at row {node}")
                M[node, node] = 1
            
            for otherNode in successors:
                flow = self.Vessels.G[node][otherNode]['flow']
                flow = 0.0529 # In mum^3/s
                length = self.Vessels.G[node][otherNode]['length']
                # Upwind scheme?
                M[otherNode, node] = -flow/length
                M[otherNode, otherNode] = flow/length

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

    def Solve(self, checkForEmptyRows : bool=False):
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
        print("Solving with PETSc")
        # Sanity check
        if self.A.getformat() != 'csr':
            self.A = self.A.tocsr()
    
        comm = PETSc.COMM_WORLD
        petsc_mat = PETSc.Mat().createAIJ(size=self.A.shape,
                                          csr=(self.A.indptr,
                                               self.A.indices,
                                               self.A.data), comm=comm)

        solverType = 'bcgs' #'pgmres'
        precondType = None #'ilu
        ksp = PETSc.KSP().create(comm=comm)
        ksp.setType(solverType)
        pc = ksp.getPC()
        pc.setType(precondType)
        ksp.setFromOptions()
        ksp.setOperators(petsc_mat)
        xpetsc = PETSc.Vec().create(comm=comm)
        b = PETSc.Vec().createWithArray(self.rhs.toarray()[:,0], comm=comm)
        xpetsc.setSizes(self.A.shape[0], None)
        xpetsc.setUp()
        ksp.solve(b, xpetsc)
        x = xpetsc.getArray()
        del ksp
        del petsc_mat
        del b
        del xpetsc       
        
        print("Time for solver: ", time()-t)
        return x[list(nx.topological_sort(self.Vessels.G))], x[self.nPoints:]
                
        
        
    
    def VesselsToVTK(self, VTKFileName : str):
        self.Vessels.VesselsToVTK(VTKFileName)
        return

    def MeshToVTK(self, VTKFileName : str):
        self.Vessels.MeshToVTK(VTKFileName)
        return
