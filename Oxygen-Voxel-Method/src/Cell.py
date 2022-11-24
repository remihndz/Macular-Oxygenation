import numpy as np 

class Cell:
    """
    A class representing a cuboid cell.
    
    Attributes
    ----------
    id : int
        the id of the cell
    _center : list
        the x,y,z coordinates of the center of the cell
    _dx : list
        the length of the cell in each direction (x,y,z).
    _label : int
        the type of cell (0 for vascular, 1 for endothelial, 2 for tissue)

    Methods
    -------
    SetId(int)
        sets id for the cell
    SetLabel(int)
        sets _label for the cell
    GetLabel()
        getter of _label
    """

    def __init__(self, center, dx):
        if len(center) !=3:
            raise ValueError(f"Invalid 'center' argument. {len(center)=}!=3.")
        self._center = np.array(center).reshape((3,)).astype(float)
        if len(dx) !=3:
            raise ValueError(f"Invalid 'dx' argument. {len(dx)=}!=3.")
        for delta in dx:
            if delta <= 0:
                raise ValueError(f"Invalide 'dx' argument: negative value for the edge length.")
        self._dx = np.array(dx).astype(float)
        self._label=None



    def __str__(self):
        return f"""
        Cell:
            Cell center: {self._center}
            Edge lengths: {self._dx}
            Cell label: {self._label}
        """

    def __repr__(self):
        return f"Cell(center={self._center}, " \
            f"dx={self._dx})" 

    def GetLabel(self):
        return self._label

    def SetLabel(self, label):
        self._label = label

    def SetId(self, id):
        self._id = id