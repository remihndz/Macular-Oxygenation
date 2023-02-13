from Cell import Cell
import numpy as np
from NDSparseMatrix import NDSparseMatrix # A custom sparse matrix storage. No mathematical operation implemented
import vtk
from typing import Union, Tuple, List
from tqdm import tqdm
import scipy.sparse as sp
from astropy import units as u

class UniformGrid(object):
    """
    A class representing an uniform grid.

     Attributes
    ----------
    dimensions : numpy.ndarray((3,))
        an array of the x,y,z dimensions of the cuboid
    origin : numpy.ndarray((3,))
        the lower left corner of the cuboid
    nCells : numpy.ndarray((3,))
        the number of cells along each axis  
    nCellsTotal : int
        the total number of cells in the grid
    spacing : numpy.ndarray((3,))
        the length of the cells for each axis
    labels : ndarray((nx,ny,nz))
        a (sparse) 3D array of labels for each cell
    v : float
        The volume of a cell in the (uniform) grid.

    Methods
    -------
    IsInsideMesh(tuple or np.ndarray)
        Returns True if the (i,j,k) input is within the mesh.
    PointToCell(point)
        returns the cell to which point belongs
    CellCenter([i,j,k] or int)
        returns the coordinates of the center of the cell indexed by ijk
    Dist(cell1, cell2) 
        returns the distance (in i,j,k coordinates) between the cells.
    ToVTK(str)
        saves the mesh, with labels, in vtk structured points format.
    """
    # def __init__(self, dimensions=[100.0,100.0,100.0],
    #     origin=[0.0,0.0,0.0], # The bottom left corner of the cuboid
    #     nCells=[20,20,20],
    #     spacing=None):

    def __init__(self, dimensions=u.Quantity([1.0,1.0,1.0], u.cm),
                 origin=u.Quantity([0.0,0.0,0.0], u.cm),
                 nCells=[20,20,20],
                 spacing=None,
                 units='mm'):

        self.lengthScale = u.Unit(units)
        
        self.dimensions = dimensions
        self.origin = origin

        if not spacing is None:
            if isinstance(spacing, float):
                spacing = u.Quantity([spacing, spacing, spacing], u.cm)
                self.spacing = spacing.value
            else:
                self.spacing = u.Quantity(spacing, self.lengthScale).value
            n = np.ceil(self.dimensions/self.spacing).astype(int)
            print(self.spacing, self.dimensions, n)
            self.nCells = n
        else:
            self.nCells = nCells 
            spac = self._dimensions/self._nCells
            self.spacing = spac

        self.labels = NDSparseMatrix(shape=self.nCells, defaultValue=0) # Initialize an empty sparse array, i.e., full of zeros
        
        print(self)
    

    @property # Volume of the tissue
    def v(self) -> float:
        return self._v
    @v.setter
    def v(self, newVolume : float):
        self._v = newVolume

    @property # Volume of a cell
    def vCell(self) -> float:
        return self._vCell
        
    @property
    def dimensions_withUnit(self):
        return u.Quantity(self._dimensions, self.lengthScale)
    @property
    def dimensions(self):
        return self._dimensions
    @dimensions.setter
    def dimensions(self, dims : Union[Tuple[float], List[float], np.ndarray]):
        dims = u.Quantity(dims, self.lengthScale).value
        if np.all(np.array(dims) > 0.0) and np.array(dims).size==3:
            self._dimensions = u.Quantity(dims, self.lengthScale).value.reshape((3,))
        else:
            raise ValueError("Please enter valid dimensions.")

    @property
    def origin_withUnit(self):
        return u.Quantity(self._origin, self.lengthScale)
    @property
    def origin(self):
        return self._origin
    @origin.setter
    def origin(self, orig : Union[Tuple[float], List[float], np.ndarray]):
        orig = u.Quantity(orig, self.lengthScale).value
        if np.array(orig).size==3:
            self._origin = np.array(orig).reshape((3,))
        else:
            raise ValueError("Please enter valid point for origin.")

    @property
    def LabelsDistribution(self):
        count = self.nCellsTotal
        count -= self.labels.nonzeros
        distrib = {'Tissue':0, 'Intravascular':0, 'Endothelial':0}
        distrib['Tissue'] = count
        for label in self.labels.elements.values():
            if label==1:
                distrib['Intravascular'] +=1
            elif label==2:
                distrib['Endothelial'] +=1
            else:
                raise ValueError(f"Label '{label}' found in "
                                 "the label matrix is not a valid"
                                 " label.")
        assert sum(distrib.values())==self.nCellsTotal, "The total number of cells found does not match the total number of cells."
        return distrib
    
    @property
    def spacing_withUnits(self):
        return u.Quantity(self._spacing, self.lengthScale)    
    @property
    def spacing(self):
        return self._spacing    
    @spacing.setter
    def spacing(self, spac : Union[Tuple[float], List[float], np.ndarray]):
        spac = u.Quantity(spac, self.lengthScale).value
        if np.all(np.array(spac) > 0.0):
            if np.array(spac).size==3:
                self._spacing = np.array(spac).reshape((3,))
            elif np.array(spac).size == 1 and type(spac)==float:
                self._spacing = np.array([spac, spac, spac]).reshape((3,))
            else:
                raise ValueError("Please enter valid spacings.")
        else:
            raise ValueError("Please enter valid spacings.")
        self._vCell = np.prod(self._spacing)
        self._hmax  = max(self._spacing)

    @property
    def hmax(self):
        return self._hmax

    @property
    def nCells(self):
        return self._nCells
    @nCells.setter
    def nCells(self, n):
        if (isinstance(n, int)) and n<1:
            self._nCells = np.array([n,n,n]).reshape((3,))
        elif np.array(n).size==3:
            self._nCells = np.array(n).reshape((3,))
        else:
            print("Please enter a valide positive integer.")

    @property
    def nCellsTotal(self):
        return np.prod(self._nCells)            
    
    @property
    def labels(self):
        return self._labels
   
    @labels.setter
    def labels(self, newLabels):
        assert isinstance(newLabels, NDSparseMatrix), "New labels must be an NDSparseMatrix."
        self._labels = newLabels
        return

    def SetLabelOfCell(self, newLabel : int, cellId : Union[Tuple[int], List[int], np.ndarray]) -> bool:
        '''
        Returns False if labels[cellId] has not been updated.
        '''
        oldLabel = self.labels.readValue(cellId)
        updateValue = False
        # Vessel label takes priority over other labels
        if newLabel == 0 or oldLabel==1:
            return False
        elif newLabel == 1 and oldLabel != 1:
            updateValue = True
            self._labels.addValue(cellId, newLabel)
        # Endothelial label takes priority over tissue label
        elif newLabel == 2 and oldLabel == 0:
            updateValue = True
            self._labels.addValue(cellId, newLabel)
        return updateValue

    def ToFlatIndexFrom3D(self, ijk : Union[Tuple[int], List[int], np.ndarray]) -> int:
        return self.nCells[0]*self.nCells[1]*ijk[2] + self.nCells[0]*ijk[1] + ijk[0]

    def FlatIndexTo3D(self, idx : int) -> Tuple[int]:
        k = idx // (self.nCells[0]*self.nCells[1])
        j = (idx - k*self.nCells[0]*self.nCells[1]) // self.nCells[0]
        i = idx - self.nCells[0] * (j + self.nCells[1]*k)
        return (i,j,k)        

    def IsInsideMesh(self, ijk : Union[Tuple[int], List[int], np.ndarray]) -> bool:
        """Check if an (i,j,k) is a valid index for the mesh.

        Parameters
        ----------
        ijk : tuple
            The index to check.

        Returns
        -------
        bool
            True if inside the mesh.
        """
        for m in range(3):
            if ijk[m] < 0:
                return False
            if ijk[m] > self.nCells[m]-1:
                return False
        return True

    def PointToCell1(self, X : Union[List[float], Tuple[float], np.ndarray]) -> np.ndarray:
        # This works fine but sometimes the Repartition function gets stuck in an infinite loop
        # The other method seems to return lower indices? May be a rounding problem.
        xarr = np.array(X).reshape((3,))
        if (np.any(xarr < self.origin) or np.any(xarr > self.origin + self.dimensions)):
            raise ValueError(f"Point {X.tolist()} out of bounds for the cuboid between {self.origin} and {self.origin + self.dimensions}.")
        
        xCentered = xarr - self.origin
        cell = np.floor(np.divide(xCentered, self.spacing)).astype(int)
        return cell

    def PointToCell(self, X) -> np.ndarray:
        # Center and normalize
        xarr = np.divide(np.array(X).reshape((3,)) - self.origin, self.dimensions) 

        if (np.any(xarr < 0) or np.any(xarr > 1)):
            raise ValueError(f"Point {X} out of bounds for the cuboid between {self.origin} and {self.origin + self.dimensions}.")

        xVoxelSpace = np.multiply(xarr, self.nCells) # Project onto voxel space
        return np.floor(xVoxelSpace).astype(int)
        
    
    def CellCenter(self, ijk : Union[np.ndarray, List[int], int, Tuple[int]]) -> np.ndarray:
        if isinstance(ijk, int):
            ijkarr = np.array(self.FlatIndexTo3D(ijk))
        else:
            ijkarr = np.array(ijk).reshape((3,)) 
        if np.any(ijkarr-self.nCells > 0):
            raise ValueError(f"Indices {ijkarr.tolist()} out of bounds for the grid.")        
        cellCenter = self.origin + self.spacing * ijkarr
        return cellCenter

    @staticmethod
    def Dist(cell1 : Union[List[int], Tuple[int], np.ndarray],
             cell2 : Union[List[int], Tuple[int], np.ndarray]) -> int:
        return int(np.sum(np.abs(np.array(cell1)-np.array(cell2))))

    def _BoundingBoxOfVessel(self, p1 : np.ndarray, p2 : np.ndarray,
                             r : float) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Finds the bounding box for the vessel with end points p1 and p2
        in cell coordinates (i,j,k indices).
        For formulas, see https://iquilezles.org/articles/diskbbox/
        '''
        n = (p1-p2) # The axis of the cylinder
        n = r*(1-(n/np.linalg.norm(n))**2)**0.5 # The direction, orthogonal to the axis,
                                                # where to pick the bounding box corners
        # Find bounding box for the cylinder as the bounding box of the
        # bounding boxes of its caps (the disk faces)
        bboxes = np.array([p1 - n, p1 + n, p2 - n, p2 + n])
        cellMin, cellMax = self.PointToCell(bboxes.min(axis=0)), self.PointToCell(bboxes.max(axis=0))
        return cellMin, cellMax

    def ToNumpy(self) -> np.ndarray:
        arr = np.zeros((self.nCells))
        for i,j,k in ((x,y,z) for z in range(self.nCells[2])
                      for y in range(self.nCells[1])
                      for x in range(self.nCells[0])):
            arr[i,j,k] = self.labels[(i,j,k)]
        return arr


    def ToVTK(self, VTKFileName : str):
        """Save the mesh with its label in vtk format for visualisation.

        Parameters
        ----------
        VTKFileName : str
            File to store the labels.
        """

        with open(VTKFileName, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("A mesh for the computation of oxygen perfusion.\n")
            # f.write("BINARY\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {self.nCells[0]+1} {self.nCells[1]+1} {self.nCells[2]+1}\n")
            f.write(f"ORIGIN {self.origin[0]} {self.origin[1]} {self.origin[2]}\n")
            f.write(f"SPACING {self.spacing[0]} {self.spacing[1]} {self.spacing[2]}\n")
            
            # Writing the data
            f.write(f"CELL_DATA {self.nCellsTotal}\n")
            f.write(f"SCALARS labels int 1\n")
            f.write(f"LOOKUP_TABLE default")

            f.write("\n")
            f.write("\n".join(tqdm(self.labels, desc=f"Writing labels to {VTKFileName}"))) #, total=self.nCellTotal)))        
        return     

    def MakePoissonWithNeumannBC(self, D):
        
        nx,ny,nz = self.nCells
        dx,dy,dz = self.spacing
        
        coeffs = [D/dz/dz, D/dy/dy, D/dx/dx, # Lower diagonals
                  0,        # Main diagonal, coeff is set below
                  D/dz/dz, D/dy/dy, D/dx/dx] # Upper diagonals
        coeffs[3] = -sum(coeffs)
        offsets = [-nx*ny, -nx, -1,  # Lower diagonals
                   0,                                     # Main diagonal
                   nx*ny, nx, 1]     # Upper diagonals

        M = sp.diags(coeffs, offsets, shape=(self.nCellsTotal, self.nCellsTotal), format='lil')
        
        # Diffusion with homogeneous Neumann BC
        nx,ny,nz = self.nCells
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cellId = i + nx*j + nx*ny*k
                    if i==0:
                        if cellId-1>=0:
                            M[cellId, cellId-1] = 0.0
                        M[cellId, cellId+1] *= 2
                    elif i==nx-1:
                        if cellId+1<M.shape[1]:
                            M[cellId, cellId+1] = 0.0
                        M[cellId, cellId-1] *= 2
                    if j==0:
                        if cellId-nx>=0:
                            M[cellId, cellId-nx] = 0.0
                        M[cellId, cellId+nx] *= 2.0 
                    elif j==ny-1:
                        if cellId+nx<M.shape[1]:
                            M[cellId, cellId+nx] = 0.0
                        M[cellId, cellId-nx] *= 2.0
                    if k==0:
                        if cellId-nx*ny>=0:
                            M[cellId, cellId-nx*ny] = 0.0
                        M[cellId, cellId+nx*ny] *= 2.0
                    elif k==nz-1:
                        if cellId+nx*ny<M.shape[1]:
                            M[cellId, cellId+nx*ny] = 0.0
                        M[cellId, cellId-nx*ny] *= 2.0
                        
        return M.tolil()  # Return -div(Dgrad)

    def MakePoissonWithNeumannBC_Kronecker(self, D):

        # Uses the Kronecker representation to scale-up from 1D to 3D.
        # Need to work-out on paper whether the coefficients are correct
        # Should be OK if dx=dy=dz
        # Update: not working. Maybe use it to build the symmetric matrix and
        # add the BC manually. Or work out the math to see if that can be used directly
        def Make1DDiffWithoutBC(nx, dx):
            M = sp.diags([D/dx/dx, -2*D/dx/dx, D/dx/dx], [-1, 0, 1], shape=(nx,nx), format='lil')
            return M

        def Make3DFrom1DWithoutBC(n, h):
            nx,ny,nz = n
            dx,dy,dz = h
            Dx, Dy, Dz = Make1DDiffWithoutBC(nx,dx), Make1DDiffWithoutBC(ny,dy), Make1DDiffWithoutBC(nz,dz)
            return sp.kron(Dx, sp.eye(ny*nz)) + sp.kron(sp.eye(nx), sp.kron(Dy, sp.eye(nz))) + sp.kron(sp.eye(nx*ny), Dz)            
    
        M = Make3DFrom1DWithoutBC(self.nCells, self.spacing).tolil()
        # Diffusion with homogeneous Neumann BC
        nx,ny,nz = self.nCells
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    cellId = i + nx*j + nx*ny*k
                    if i==0:
                        if cellId-1>=0:
                            M[cellId, cellId-1] = 0.0
                        M[cellId, cellId+1] *= 2
                    elif i==nx-1:
                        if cellId+1<M.shape[1]:
                            M[cellId, cellId+1] = 0.0
                        M[cellId, cellId-1] *= 2
                    if j==0:
                        if cellId-nx>=0:
                            M[cellId, cellId-nx] = 0.0
                        M[cellId, cellId+nx] *= 2.0 
                    elif j==ny-1:
                        if cellId+nx<M.shape[1]:
                            M[cellId, cellId+nx] = 0.0
                        M[cellId, cellId-nx] *= 2.0
                    if k==0:
                        if cellId-nx*ny>=0:
                            M[cellId, cellId-nx*ny] = 0.0
                        M[cellId, cellId+nx*ny] *= 2.0
                    elif k==nz-1:
                        if cellId+nx*ny<M.shape[1]:
                            M[cellId, cellId+nx*ny] = 0.0
                        M[cellId, cellId-nx*ny] *= 2.0
                                                
        return M.tolil() # Return -div(Dgrad)

    def __str__(self):
        return f"""
        Mesh:
            Length scale: {self.lengthScale}
            Dimensions: {self.dimensions}
            Origin: {self.origin}
            Number of cells: {self.nCells}
            Spacing: {self.spacing}
            Total number of cells: {self.nCellsTotal}
            """

    def __repr__(self):
        return f"UniformGrid(dimensions={self.dimensions}, " \
            f"origin={self.origin}, " \
            f"nCells={self.nCells}," \
            f"spacing={self.spacing})"
