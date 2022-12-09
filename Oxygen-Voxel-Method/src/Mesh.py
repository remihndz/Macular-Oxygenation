from Cell import Cell
import numpy as np
from NDSparseMatrix import NDSparseMatrix # A custom sparse matrix storage. No mathematical operation implemented
import vtk 

class UniformGrid:
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
    
    Methods
    -------
    PointToCell(point)
        returns the cell to which point belongs
    CellCenter([i,j,k] or int)
        returns the coordinates of the center of the cell indexed by ijk
    Dist(cell1, cell2) 
        returns the distance (in i,j,k coordinates) between the cells.
    ToVTK(str)
        saves the mesh, with labels, in vtk structured points format.
    """
    def __init__(self, dimensions=[100.0,100.0,100.0],
        origin=[0.0,0.0,0.0], # The bottom left corner of the cuboid
        nCells=[20,20,20],
        spacing=None):
        
        self.dimensions = dimensions
        self.origin = origin

        if spacing:
            if isinstance(spacing, float):
                self.spacing = np.array([spacing, spacing, spacing])
            else:
                self.spacing = np.array(spacing)
            n = np.ceil(self.dimensions/self.spacing).astype(int)
            print(self.spacing, self.dimensions, n)
            self.nCells = n
        else:
            self.nCells = nCells 
            spac = self._dimensions/self._nCells
            self.spacing = spac

        # self.labels = 0 # 0 for tissue, 1 for intravascular and 2 for endothelium
        self.labels = NDSparseMatrix(size=self.nCells, defaultValue=0) # Initialize an empty sparse array, i.e., full of zeros
        
        print(self)
    
        
    @property
    def dimensions(self):
        return self._dimensions       
    @dimensions.setter
    def dimensions(self, dims):
        if np.all(np.array(dims) > 0.0) and np.array(dims).size==3:
            self._dimensions = np.array(dims).reshape((3,))
        else:
            raise ValueError("Please enter valid dimensions.")

    @property
    def origin(self):
        return self._origin          
    @origin.setter
    def origin(self, orig):
        if np.array(orig).size==3:
            self._origin = np.array(orig).reshape((3,))
        else:
            raise ValueError("Please enter valid point for origin.")

    
    @property
    def spacing(self):
        return self._spacing    
    @spacing.setter
    def spacing(self, spac):
        if np.all(np.array(spac) > 0.0):
            if np.array(spac).size==3:
                self._spacing = np.array(spac).reshape((3,))
            elif np.array(spac).size == 1 and type(spac)==float:
                self._spacing = np.array([spac, spac, spac]).reshape((3,))
            else:
                raise ValueError("Please enter valid spacings.")
        else:
            raise ValueError("Please enter valid spacings.")


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

    def SetLabelOfCell(self, newLabel : int, cellId : tuple):
        '''
        Returns False if labels[cellId] has not been updated.
        '''
        oldLabel = self.readValue(cellId)
        updateValue = False
        # Vessel label takes priority over other labels
        if newLabel == 1 and (oldLabel == 0 or oldLabel == 2):
            updateValue = True
        # Endothelial label takes priority over tissue label
        elif newLabel == 2 and oldLabel == 0:
            updateValue = True
        if updateValue:
            self._labels.addValue(cellId, newLabel)
        return updateValue

    def 3DToFlatIndex(self, ijk : tuple):
        return self.nCells[0]*self.nCells[1]*ijk[2] + self.nCells[0]*ijk[1] + i
    def FlatIndexTo3D(self, idx : int):
        k = idx // (self.nCells[0]*self.nCells[1])
        j = (idx - k*self.nCells[0]*self.nCells[1]) // self.nCells[0]
        i = idx - self.nCells[0] * (j + self.nCells[1]*k)
        return (i,j,k)        
    
    def __str__(self):
        return f"""
        Mesh:
            Dimensions: {self.dimensions}
            Origin: {self.origin}
            Number of cells: {self.nCells}
            Spacing: {self.spacing}
        """

    def __repr__(self):
        return f"UniformGrid(dimensions={self.dimensions}, " \
            f"origin={self.origin}, " \
            f"nCells={self.nCells}," \
            f"spacing={self.spacing})"
    
    def PointToCell(self, X):
        xarr = np.array(X).reshape((3,))
        if (np.any(xarr < self.origin) or np.any(xarr > self.origin + self.dimensions)):
            raise ValueError(f"Point {X.tolist()} out of bounds for the cuboid between {self.origin} and {self.origin + self.dimensions}.")
        
        xCentered = xarr - self.origin
        return np.floor(np.divide(xCentered, self.spacing)).astype(int)
    
    def CellCenter(self, ijk):
        if isinstance(ijk, int):
            ijkarr = np.array(self.FlatIndexTo3D(ijk))
        else:
            ijkarr = np.array(ijk).reshape((3,)) 
        if np.any(ijkarr-self.nCells > 0):
            raise ValueError(f"Indices {ijkarr.tolist()} out of bounds for the grid.")        
        cellCenter = self.origin + self.spacing * ijkarr
        return cellCenter

    @staticmethod
    def Dist(cell1, cell2):
        return int(np.sum(np.abs(np.array(cell1)-np.array(cell2))))

    def _BoundingBoxOfVessel(self, p1, p2, r):
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

    def ToNumpy(self):
        arr = np.zeros((self.nCells))
        for i,j,k in [(x,y,z) for z in range(self.nCells[2])
                      for y in range(self.nCells[1])
                      for x in range(self.nCells[0])]:
            arr[i,j,k] = self.labels[(i,j,k)]

        return arr
        
    
    def ToVTK(self, VTKFileName : str):
        '''
        Save the mesh with its label in vtk format for visualisation.
        '''        
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

            for i,j,k in [(x,y,z) for z in range(self.nCells[2])
                          for y in range(self.nCells[1])
                          for x in range(self.nCells[0])]:
                f.write(f"\n{int(self.labels[(i,j,k)])}")
        
        return     
    

