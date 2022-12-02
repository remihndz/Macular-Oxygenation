from Cell import Cell
import numpy as np



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
    labels : numpy.ndarray((nx,ny,nz))
        a 3D array of labels for each cell
    
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
            self.spacing = spacing
            n = np.ceil(self.dimensions/self.spacing).astype(int)
            print(self.spacing, self.dimensions, n)
            self.nCells = n
        else:
            self.nCells = nCells 
            spac = self._dimensions/self._nCells
            self.spacing = spac

        self.labels = 0 # 0 for tissue, 1 for intravascular and 2 for endothelium
            
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
        if np.array(newLabels).size == self.nCellsTotal:
            self._labels = np.array(newLabels).reshape(self.nCells)
        elif np.array(newLabels).size == 1:
            self._labels = np.ones(self.nCells) * newLabels
        else:
            raise ValueError("The number of labels don't match the number of cells.")

    def SetLabelOfCell(self, newLabel, cellId):
        if not cellId is None:
            if not isinstance(newLabel, (int, float)):
                raise ValueError("newLabel must be an int or float.")
            i,j,k = cellId
            self._labels[i,j,k] = newLabel
        return
    
    
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
        # TODO add the case where ijk is an int (i.e., the indice in a flattened array)
        if isinstance(ijk, int):
            k,remainder = divmod(ijk, self.nCells[0]*self.nCells[1])
            j, i = divmod(remainder, self.nCells[1])
            ijkarr = np.array([i,j,k])
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

    
    def ToVTK(self, vtkFileName : str):
        '''
        Save the mesh with its label in vtk format for visualisation.
        '''        
        with open(vtkFileName, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("A mesh for the computation of oxygen perfusion.\n")
            # f.write("BINARY\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {self.nCells[0]} {self.nCells[1]} {self.nCells[2]}\n")
            f.write(f"ORIGIN {self.origin[0]} {self.origin[1]} {self.origin[2]}\n")
            f.write(f"SPACING {self.spacing[0]} {self.spacing[1]} {self.spacing[2]}\n")
            # f.write(f"ORIGIN 0 0 0\n")
            # f.write(f"SPACING 1 1 1\n")
            
            # Writing the data
            f.write(f"POINT_DATA {self.nCellsTotal}\n")
            f.write(f"SCALARS labels int 1\n")
            f.write(f"LOOKUP_TABLE default")

            for i,j,k in [(x,y,z) for z in range(self.nCells[2])
                          for y in range(self.nCells[1])
                          for x in range(self.nCells[0])]:
                #f.write(f"\n{bin(int(self.labels[i,j,k]))}")
                f.write(f"\n{int(self.labels[i,j,k])}")

            # for label in self.labels.ravel(): # Unsure in what order the 3d-array is ravelled 
            #     f.write(f"\n{bin(int(label))}")
                                   
        return
            
    

    

    

    
    
    

     


# class Mesh:
#     """
#     A class representing a cartesian mesh of a cuboid.

#     Attributes
#     ----------
#     _cells : list
#     ub : np.array((3,))
#         the upper bound of the meshed geometry
#     lb : np.array((3,))
#         the lower bound of the meshed geometry  
#     nCells : int
#         the number of cells in the mesh  
#     Methods
#     -------
#     GenerateCells(list n or list dx)
#         generate the cartesian grid
#     @staticmethod
#     Dist(cell1, cell2) 
#         returns the distance (in i,j,k coordinates) between the cells.
#     """
    

    # def __init__(self, UB, LB, cells = [], n = [10,10,10]):
    #     self._cells = cells
    #     if not np.all(UB>LB):
    #         raise ValueError("Invalid arguments: lower bound & upper bound.")
        
    #     self.ub = UB
    #     self.lb = LB
    #     self.nCells = len(cells)

    #     if cells==[]:
    #         self.GenerateCells(n)

    # # TODO finish this.
    # @classmethod
    # def FromListOfCells(cls, cells):
    #     centers = np.array((len(cells),))
    #     for cell in cells:
    #         centers[cell.id] = np.sum(cell.center)
        
    #     # Problem: this could return different ID for each direction, given that 
    #     # all cells in a row have the same position along an axis
    #     ubIdx = np.argmax(centers) # Finds the id of the cell with highest x,y,z pos
    #     lbIdx = np.argmin(centers) # Finds the id of the cell with lowest x,y,z pos

    #     ub = [0.0, 0.0, 0.0]
    #     lb = [0.0, 0.0, 0.0]

    #     return cls(ub, lb, cells=cells)
    

    # def GenerateCells(self, n):
    #     """
    #     Generates (nx x ny x nz) cells for the cuboid, 
    #     traversing the x axis, then y axis then z axis. 
    #     """
    #     dx = (self.ub-self.lb)/np.array(n)
    #     print(f"Creating a grid with {dx=} and {n=}.")
    #     for k in range(1, n[2]+1):
    #         for j in range(1,n[1]+1):
    #             for i in range(1, n[0]+1):
    #                 center = self.lb + np.array([(i-0.5)*dx[0], (j-0.5)*dx[1], (k-0.5)*dx[2]])
    #                 self._cells.append(Cell(center, dx))
        
    #     self.nCells = n[0]*n[1]*n[2]     
        


    # def __str__(self):
    #     return f"""
    #     Mesh:
    #         Lower bound: {self.lb}
    #         Upper bound: {self.ub}
    #         Number of cells: {self.nCells}
    #     """

    # def __repr__(self):
    #     return f"Mesh(ub={self.ub}, " \
    #         f"lb={self.lb}, " \
    #         f"nCells={self.nCells})" 

    
