import numpy as np
import matplotlib.pyplot as plt
import Node
import pandas as pd
import networkx as nx
from Mesh import UniformGrid
from math import isclose
import scipy.sparse as sp
import scipy.sparse.linalg
import vtk
from typing import Dict, Union, Tuple, List
import matplotlib.pylab as plt

class VascularNetwork(object):
    """A class storing a vascular network given in a .cco format.
    TODO: add different input files than .cco.
    Attributes:
    -----------
    G : networkx.DiGraph
       the directed graph of the network, contains location, size and flow (if yet computed) information.
    mesh : Mesh.UniformGrid
       a mesh of the tissue surrounding the vasculature.
    w : float
       Vessel wall thickness.
    Flow_matrix : numpy.ndarray
       the left hand side matrix of the blood flow model.
    pProx : float
       inlet pressure value.
    qProx : float
       inlet flow value.
    pDist : float
       outlet pressure value.
    qDist : float
       outlet flow value.
       
    Methods:
    --------
    LabelMesh(endotheliumThickness, repartition=True)
        Labels the mesh according to the vascular vessel. 
    MeshToVTK(VTKFileName)
        Saves the mesh in .vtk legacy format.
    VesselsToVTK(VTKFileName)
        Saves the vascular network in legacy vtp format.
    Repartition(maxDist=1)
        Split vessel segments to fit mesh size
    SetLinearSystem(**kwargs)
        Assembles the linear system of blood flow model.
    SolveFlow()
        Solves blood flow with the boundary conditions set.
    MakeMc()
        Returns the O2 convection in vessels subsystem.
    nNodes()
        Returns the number of nodes in the network.
    nVessels()
        Returns the number of vessels in the network.
    BoundingBox()
        Return the bounding box of the network.
    PlotGraph(Graph)
        Plots a graph.
    """
    
    def __init__(self, ccoFile : str, units : str='mm', **kwargs):        
        """Creator for the class VascularNetwork

        Parameters
        ----------
        ccoFile : str
            the input network in .cco format
        units : str
            the length units. Must be one of
            {'mm','cm','micron','mum'}.
        **kwargs : 
            additional mesh parameters
            {'dimensions','origin','spacing'}

        """
        self.G = self.CreateGraph(ccoFile)
        self.C = -1.0 * nx.incidence_matrix(self.G, oriented=True).T # (nVessel, nNodes) matrix
        self._lengthConversionDict = {'mm':1e1,
                                      'cm':1.0,
                                      'micron':1e4,
                                      'mum':1e4}
        self.units = units
        self._isPartitionned = False

        # self.PlotGraph(self.G)

        # Pad the cuboid with the radius of the larger vessel to
        # prevent having vessel cylinder outside the cuboid
        bb = self.BoundingBox()
        maxRad = self.GetVesselData(['radius'])['radius'].max()
        minRad = self.GetVesselData(['radius'])['radius'].min()
        origin     = kwargs.get('origin', bb[0]) - 2*maxRad
        dimensions = kwargs.get('dimensions', bb[1]-bb[0]) + 4*maxRad
        spacing    = kwargs.get('spacing', [minRad/4]*3)
        
        if dimensions[-1]==0:
            dimensions[-1] = 1
        nCells = kwargs.get('nCells', [20,20,20])

        self.mesh = UniformGrid(dimensions = dimensions,
                                origin = origin,
                                nCells = nCells,
                                spacing = spacing)

        # Initialize linear system to None for error handling
        self.Flow_matrix = self.Flow_rhs = None
        self.Flow_loss = None
        self.inletBC = {}
        self.outletBC = {}
        return

    def LabelMesh(self, endotheliumThickness : float, repartition :bool=True):
        """Labels the tissue surrounding the vessels. Spliting of the vessel segments is performed prior to labelling unless specified otherwise.

        Parameters
        ----------
        endotheliumThickness : float
            Thickness of the endothelium.
        repartition : bool, default=True
            Set to 'True' to split vessels to fit mesh size.
        returnIntravascularConnectivity : bool, default=False
            If 'True', returns the connectivity matrix for the cells
            of the mesh labelled as intravascular.
        
        Returns
        -------
        ConnVascNodesToEndothelialCells : scipy.sparse_array
            The connectivity matrix of vascular nodes to endothelial cells.
        """


        if repartition:
            self.Repartition(maxDist=1) # Split the vessels to the size of the mesh
        # self.mesh.labels = 0
        self.mesh.labels.EmptyMatrix()
        self.w = endotheliumThickness

        # The empty connectivity matrix
        ConnVascNodesToEndothelialCells = sp.dok_array((self.nNodes(), self.nNodes()+self.mesh.nCellsTotal), dtype=np.int8)
                                          
        for n1, n2, data in self.G.edges(data=True):
            
            p1, p2 = self.G.nodes[n1]['position'], self.G.nodes[n2]['position']
            r,l = data['radius'], data['length']
            vectorDirection = (p1-p2)/l # Unit vector (direction)

            l_new, l_old = np.linalg.norm(p1-p2), l
            if not l_new == l_old: f"The length stored in the graph's edge data is incorrect: {l_old=} {l_new=} for vessel {(n1,n2)=}: {self.G.nodes[n1]=}, {self.G.nodes[n2]=}."
            P = np.outer(vectorDirection, vectorDirection) # Matrix of the orthogonal projection onto the vessel axis
            O = np.identity(3)-P                           # O*y = y-P*y is orthogonal to the axis

            ## Find the bounding box
            # To ensure full enclosure of the vessel, the bounding box is should
            # bound the vessel extended by its radius in each direction
            p1, p2 = p1 - r * vectorDirection, p2 + r * vectorDirection  
            cellMin, cellMax = self.mesh._BoundingBoxOfVessel(p1, p2, r)
            # Iterate through the cells within the bounding box
            for cellId in [(x,y,z) for x in range(cellMin[0], cellMax[0]+1)
                                   for y in range(cellMin[1], cellMax[1]+1)
                                   for z in range(cellMin[2], cellMax[2]+1)]:
                # Assign new label
                HasUpdatedValue, newLabel = self._LabelCellWithCylinder(O, p1, p2, r, cellId, endotheliumThickness)
                
                # Add to connectivity matrix if the cell label changed from tissue to endothelial
                if HasUpdatedValue and newLabel==2:
                    ConnVascNodesToEndothelialCells[n1, self.mesh.ToFlatIndexFrom3D(cellId)] =  1
            # Add connectivity of the vascular node to itself
            ConnVascNodesToEndothelialCells[n1, n1] = -1
                    
        print("Labelling successfully completed.")
        return ConnVascNodesToEndothelialCells.tocsr()
        
    def _LabelCellWithCylinder(self, O : np.ndarray, p1 : np.ndarray, p2 : np.ndarray, r : float,
                               cellId : tuple, endotheliumThickness : float):
        """
        Labels a cell within the bounding box of a cylinder.
        The new label is 0 for tissue, 1 for intravascular and 2 for endothelium.
        Parameters:
        -----------
            O : numpy.ndarray
                (I-P) with P the projection matrix onto the vessel's axis (centerline)
            p1, p2 : numpy.ndarray
                the end points of the vessel.
            r : float
                the vessel's radius.
            cellId : tuple
                the indices i,j,k locating the cell in the mesh.
            endotheliumThickness : float
                the thickness of the endothelial layer.
        """
        # If we have already labeled it, don't redo it.
        i,j,k = cellId
        
        cellCenter = self.mesh.CellCenter(cellId)
        
        d = np.linalg.norm( O.dot(p1-cellCenter) ) # The radial distance between the vessel axis and the cell center

        if (d < r - endotheliumThickness/2.0):
            newLabel = 1
        elif (d < r+endotheliumThickness/2.0):
            newLabel = 2
        else:
            newLabel = 0

        # print(f'{newLabel=}: {d=} {r=} {endotheliumThickness=}')
        
        # # Just a check using another method
        # old_d = d
        # v1, v2 = p2-p1, cellCenter-p1
        # if np.array_equal(p1, cellCenter) or np.array_equal(p2, cellCenter):
        #     d = 0
        # else:
        #     H = np.linalg.norm(v2)
        #     c = np.inner(v1,v2)/(np.linalg.norm(v1)*H)
        #     d = H * (1-c*c)**0.5
        # assert isclose(d, old_d, rel_tol=1e-5), f"Both methods don't compute the same radial distance {old_d=}, {d=} for {p1=}, {p2=} and {cellCenter=}."

        if (d < r - endotheliumThickness/2.0):
            otherLabel = 1
        elif (d < r+endotheliumThickness/2.0):
            otherLabel = 2
        else:
            otherLabel = 0

        # labelDict = {0:'tissue', 1:'vessel', 2:'endothelium'} 
        updatedValue = self.mesh.SetLabelOfCell(newLabel, cellId)        
        return updatedValue, newLabel

    def MeshToVTK(self, VTKFileName):
        self.mesh.ToVTK(VTKFileName)
        return
    
    def VesselsToVTK(self, VTKFileName):
        # Create list of points (nodes)
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(self.nNodes())
        for n, data in self.G.nodes(data=True):
            points.SetPoint(n, data['position'])

        # Create list of lines (vessels)
        # with list of radius
        lines = vtk.vtkCellArray()
        radius = vtk.vtkDoubleArray()
        flow = vtk.vtkDoubleArray()
        flow.SetName('flow')
        radius.SetName("radius")
        for n1, n2, data in self.G.edges(data=True):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, n1)
            line.GetPointIds().SetId(1, n2)
            lines.InsertNextCell(line)
            radius.InsertNextValue(data['radius'])
            try:
                flow.InsertNextValue(data['flow'])
            except:
                flow.InsertNextValue(0.0)

        # Create the polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetCellData().AddArray(radius)
        polydata.GetCellData().AddArray(flow)
        polydata.SetLines(lines)

        # Write the polydata
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(VTKFileName)
        writer.SetInputData(polydata)
        writer.SetDataModeToBinary()
        writer.Update()
        writer.Write()

        return
            
    def Repartition(self, maxDist=1):
        '''
        Split vessels so that end nodes are at most maxDist cells away
        from each other.
        '''
        nSplit = 0
        vesselsToSplit = []
        for e in self.G.edges():
            # e is a tuple (n1,n2)
            x1, x2 = self.G.nodes[e[0]]['position'], self.G.nodes[e[1]]['position']
            cell1, cell2 = self.mesh.PointToCell(x1), self.mesh.PointToCell(x2)
            
            if self.mesh.Dist(cell1, cell2) > maxDist:
                # print(f"edge {e} added to the list of vessels to split.")
                vesselsToSplit.append(e)

        # First split those vessels
        while vesselsToSplit:
            newVesselsToSplit = []
            
            for e in vesselsToSplit:
                # print(f"Splitting {e}.")
                newEdges = self._SplitVessel(e)
                nSplit+=1

                # Check whether the new vessels need to be split again
                for newEdge in newEdges:
                    x1, x2 = self.G.nodes[newEdge[0]]['position'], self.G.nodes[newEdge[1]]['position']
                    cell1, cell2 = self.mesh.PointToCell(x1), self.mesh.PointToCell(x2)
                    if self.mesh.Dist(cell1, cell2) > maxDist:
                        newVesselsToSplit.append(newEdge)
            # Repeat the process with the newly created vessels                    
            vesselsToSplit = newVesselsToSplit

        print(f"Vascular repartion has required {nSplit} splitings.")       


    def _SplitVessel(self, edge : tuple):
        '''
        Split edge into two segments by adding a node in the middle.
        edge is a tuple (n1,n2) of the nodes forming the segment.
        '''
        ## Create new node
        newNode = self.nNodes()
        ## Should be an unused name
        # assert not newNode in list(self.G.nodes), f"Node name {newNode} already exists."

        # Add the new node to the list of nodes
        newNodePos = (self.G.nodes[edge[0]]['position'] + self.G.nodes[edge[1]]['position'])/2.0
        self.G.add_node(newNode, position=newNodePos)

        # print(f'Added node {newNode} {self.G[newNode]=}')

        ## Update the connectivity
        dataDict = self.G[edge[0]][edge[1]]
        self.G.remove_edge(edge[0], edge[1])
        # print(f"Removed {edge=}.") 
        # First segment
        dataDict['length'] = np.linalg.norm(newNodePos-self.G.nodes[edge[0]]['position'])
        self.G.add_edge(edge[0], newNode, **dataDict)
        # Second segment
        dataDict['length'] = np.linalg.norm(newNodePos-self.G.nodes[edge[1]]['position'])
        self.G.add_edge(newNode, edge[1], **dataDict)

        # print(f"Created edges {(edge[0], newNode)} and {(newNode, edge[1])} with length {self.G[edge[0]][newNode]['length']}")
        
        return (edge[0], newNode), (newNode, edge[1])
        
        

    def GetVesselData(self, keys : List[str], returnAList : bool = False):
        dataDict = dict()
        for key in keys:
            tmpContainer = []
            for n1, n2, data in self.G.edges.data():
                tmpContainer.append(data.get(key, None))
            dataDict[key] = np.array(tmpContainer)
        if returnAList:
            return dataDict.values()
        return dataDict

    def SetLinearSystem(self, inletBC : Dict[str, float]={'pressure':100},
                        outletBC : Dict[str, float]={'pressure':26}):
        ## TODO: add the plasma skimming effect
        
        # Update the incidence matrix in case splitting has occured
        self.C = -1.0 * nx.incidence_matrix(self.G, oriented=True).T
        
        ## Resistance matrix
        R = []
        for n1, n2, data in self.G.edges(data=True):
            r = data['radius']
            l = data['length']
            mu = self.Viscosity(r, hd=data['hd']) * 1e-3 # Converts to Pa.s
            self.G[n1][n2]['viscosity'] = mu
            R.append( (8*mu*l)/(np.pi*(r**4)))
        R = sp.dia_array(([R], [0]), shape=(len(R), len(R)))

        ## Boundary conditions

        self.inletBC = inletBC
        self.outletBC = outletBC
        unit = {'pressure':'mmHg', 'flow':self.units+'^3/s'}
        
        nodeInlets = self.inletNodes
        nodeOutlets = self.outletNodes
        # print(f'{nodeInlets=}\n{nodeOutlets=}')
        
        D = np.zeros((self.nNodes(),)) # Decision matrix
        qBar = np.zeros((self.nNodes(),)) # RHS
        pBar = np.zeros((self.nNodes(),)) # RHS

        ## Define inlet boundary condition
    
        bcType, bcValue = next(iter(self.inletBC.items()))
        print(f'\tInlet boundary condition is {bcType}={bcValue}{unit[bcType]}')
        # bcType is checked to be 'pressure' or 'flow' in the setter function 
        if bcType=='pressure':
            bcValue *= 133.3224 # Assumes pressure was given in mmHg
            D[nodeInlets] = 1.0
            pBar[nodeInlets] = bcValue
        else:
            qBar[nodeInlets] = bcValue
            
        # Define outlet boundary condition
        bcType, bcValue = next(iter(self.outletBC.items()))
        print(f'\tOutlet boundary condition is {bcType}={bcValue}{unit[bcType]}')
        if bcType=='pressure':
            bcValue *= 133.3224 # Assumes pressure was given in mmHg
            D[nodeOutlets] = 1.0
            pBar[nodeOutlets] = bcValue
        else:
            qBar[nodeOutlets] = bcValue

        D = sp.dia_array(([D],[0]), shape = (self.nNodes(), self.nNodes()),
                         dtype=np.float32)
        I = sp.dia_array(sp.eye(D.shape[0]), dtype=np.float32)        
        
        self.Flow_matrix = sp.vstack([sp.hstack([R, -self.C], dtype=np.float32),
                                      sp.hstack([(I-D) @ self.C.T, D])],
                                     format='csr', dtype=np.float32)

        self.Flow_rhs    = np.concatenate([np.zeros(self.nVessels()),
                                           (I-D).dot(qBar) + D.dot(pBar)])
        return

    def SolveFlow(self):

        if (self.Flow_matrix is None) or (self.Flow_rhs is None):
            raise ValueError("Linear system has not been set yet.")

        x = scipy.sparse.linalg.spsolve(self.Flow_matrix, self.Flow_rhs)
        f,p = x[:self.nVessels()], x[self.nVessels():]
        dp  = self.C.dot(p)

        # assert f.size==self.nVessels(), f"Segment flow vector has wrong size. Expected {self.nVessels()} and got {f.size}."
        # assert p.size==self.nNodes(), f"Nodal pressure vector has wrong size. Expected {self.nNodes()} and got {p.size}."


        # Compute error of the solver
        self.Flow_loss = self.C.T.dot(f).sum()
        print(f"Flow loss (C*f).sum() = {self.Flow_loss} with q_in={f.max()}.")
        self.Flow_loss = 0
        for i,e in enumerate(self.G.edges()): # Add the flow to vessel data
            n1, n2 = e
            self.G[n1][n2].update(flow=f[i], dp=dp[i])
            if not self.G.pred[n1]:
                # An inlet
                self.Flow_loss += f[i]
            elif not self.G.succ[n2]:
                # An outlet
                self.Flow_loss -= f[i]
        print(f"Flow loss f_in-f_out = {self.Flow_loss}")
        
        return f,p,dp
    
    @property
    def inletBC(self):
        return self._inletBC
    @inletBC.setter
    def inletBC(self, newInletBC : Dict[str,float]):
        try:
            bcType = next(iter(newInletBC.keys()))
            if bcType in ['pressure', 'flow']:
                self._inletBC = newInletBC
            else:
                raise KeyError(f"'{bcType}' is not a"
                               "valid boundary condition type. "
                               f"Valide types are: {['pressure','flow']}.")
        except:
            self._inletBC = None

    @property
    def inletNodes(self) -> List[int]:
        return [x for x in self.G.nodes() if self.G.in_degree(x)==0]
    @property
    def outletNodes(self) -> List[int]:
        return [x for x in self.G.nodes() if self.G.out_degree(x)==0]

    @property
    def outletBC(self):
        return self._outletBC
    @outletBC.setter
    def outletBC(self, newOutletBC : Dict[str,float]):
        try:
            bcType = next(iter(newOutletBC.keys()))
            if bcType in ['pressure', 'flow']:
                self._outletBC = newOutletBC
            else:
                raise KeyError(f"'{bcType}' is not a"
                               "valid boundary condition type. "
                               f"Valide types are: {['pressure','flow']}.")
        except:
            self._outletBC = None

    @property
    def Flow_matrix(self):
        return self._Flow_matrix
    @Flow_matrix.setter
    def Flow_matrix(self, newFlowMatrix):
        if isinstance(newFlowMatrix, np.ndarray) and  newFlowMatrix.shape != (self.nVessels()+self.nNodes(),
                                                            self.nVessels()+self.nNodes()):
            raise ValueError(f"Flow matrix with shape {newFlowMatrix.shape} inconsistent with vascular network with {self.nVessels()} vessels and {self.nNodes()} nodes.")
        self._Flow_matrix = newFlowMatrix

    @property
    def Flow_rhs(self):
        return self._Flow_rhs
    @Flow_rhs.setter
    def Flow_rhs(self, newRHS):
        if isinstance(newRHS, np.ndarray) and newRHS.shape[0] != self.nVessels()+self.nNodes():
            raise ValueError(f"RHS with shape {newRHS.shape} inconsistent with vascular network with {self.nVessels()} vessels and {self.nNodes()} nodes.")
        self._Flow_rhs = newRHS
        
    @property
    def qProx(self):
        return self._qProx
    @qProx.setter
    def qProx(self, newValue):
        if newValue!=None and newValue <= 0:
            raise ValueError("Proximal flow (qProx) cannot be negative or 0.")
        self._qProx = newValue
    @property
    def qDist(self):
        return self._qDist
    @qDist.setter
    def qDist(self, newValue):
        # TODO add an option to define different distal flow for each outlet
        if newValue!=None and newValue <= 0:
            raise ValueError("Distal flow (qDist) cannot be negative or 0.")
        self._qDist = newValue

    @property
    def pProx(self):
        return self._pProx
    @pProx.setter
    def pProx(self, newValue):
        if newValue!=None and newValue <= 0:
            raise ValueError("Proximal pressure (pProx) cannot be negative or 0.")
        self._pProx = newValue

    @property
    def pDist(self):
        return self._pDist
    @pDist.setter
    def pDist(self, newValue):
        # TODO add an option to define different distal pressure for each outlet
        if newValue!=None and newValue <= 0:
            raise ValueError("Distal pressure (pDist) cannot be negative or 0.")
        self._pDist = newValue

        
    def nNodes(self):
        return self.G.number_of_nodes()
    def nVessels(self):
        return self.G.number_of_edges()
        
    @property
    def mesh(self):
        return self._mesh
    @mesh.setter
    def mesh(self, newMesh):
        if isinstance(newMesh, UniformGrid):
            self._mesh = newMesh
            self._isPartitionned = False
        else:
            raise ValueError("Provide a valid mesh type (i.e., an instance of the UniformGrid class).")
        
    @property
    def w(self) -> float:
        return self._w
    @w.setter
    def w(self, endotheliumThickness : float):
        self._w = endotheliumThickness
        
    @property
    def units(self):
        return self._units
    @units.setter
    def units(self, newUnits):
        try:                    
            self._lengthConversionFactor = self._lengthConversionDict[newUnits]
        except:
            raise KeyError(f"Wrong key '{newUnits}' for unit conversion. Valid keys are {self._lengthConversionDict.keys()}.")
        self._units = newUnits


    def BoundingBox(self):
        nodes = np.array([self.G.nodes[n]['position'] for n in self.G.nodes])
        return [np.min(nodes, axis=0), np.max(nodes, axis=0)]        
        

    @staticmethod
    def CreateGraph(ccoFile : str, convertUnitsTo='mm') -> nx.DiGraph:

        lengthConversionDict = {'mm':1e1,
                                'cm':1.0,
                                'micron':1e4,
                                'mum':1e4}
        try:
            lengthConversion = lengthConversionDict[convertUnitsTo]
        except:
            raise KeyError(f"Wrong key '{convertUnitsTo}' for unit conversion. Valid keys are {lengthConversionDict.keys()}")
        
        G = nx.DiGraph()
        
        with open(ccoFile, 'r') as f:

            token = f.readline()
            token = f.readline().split() # Tree information

            G.add_node(0, position=np.array([float(xi) for xi in token[:3]]) * lengthConversion)
            
            f.readline() # Blank line
            f.readline() # *Vessels
            nVessels = int(f.readline())
            print(f'The tree has {nVessels} vessels.')

            vesselRadii = dict()
            vesselLength = dict()

            for i in range(nVessels):

                vessel = f.readline().split()
                vesselId = int(vessel[0])
                vesselRadii[vesselId] = float(vessel[12]) * lengthConversion
                vesselLength[vesselId] = np.linalg.norm(np.array([float(x) for x in vessel[1:4]])
                                                         -np.array([float(x) for x in vessel[4:7]])) * lengthConversion
                G.add_node(vesselId+1, position=np.array([float(xi) for xi in vessel[4:7]]) * lengthConversion)

            f.readline() # Blank line
            f.readline() # *Connectivity

            edges = []
            for i in range(nVessels):
                
                vessel = f.readline().split()

                if int(vessel[1]) == -1: # If root
                    edges.append((0, 1, {'radius':vesselRadii[0], 'length':vesselLength[0], 'hd':0.45}))
                    
                else:
                    vessel, parent = int(vessel[0]), int(vessel[1])
                    edges.append((parent+1, vessel+1, {'radius':vesselRadii[vessel], 'length':vesselLength[vessel], 'hd':0.45}))

            G.add_edges_from(edges)
        return G

    
    @staticmethod
    def PlotGraph(Graph):
        nx.draw_networkx_edges(Graph, {n:Graph.nodes[n]['position'][:-1] for n in Graph.nodes})
        plt.show()

    def Viscosity(self, radius, hd=0.45, Type='Pries'):
        '''
        Radius should be in microns.
        '''
        d = 2 * self._lengthConversionDict['micron']/self._lengthConversionDict[self.units]
        
        if Type=='Constant':
            return 3.6 # cP
        if Type=='Haynes':
            # Both muInf and delta are taken from Takahashi's model
            muInf = 1.09*np.exp(0.024*hd)
            delta = 4.29
            return muInf/( (1+delta/(d/2.0))**2 )
        else:
            mu045 = 220*np.exp(-1.3*d) + 3.2 - 2.44*np.exp(-0.06*d*0.645)
            C = (0.8 + np.exp(-0.075*d))*(-1+(1+10**-11*(d)**12)**-1)+(1+10**-11*(d)**12)**-1
            return ( 1 + (mu045-1)*((1-hd)**C-1)/((1-0.45)**C-1) )
        

    def __str__(self):
        return f"""
        Vessels:
            Number of nodes: {self.nNodes()}
            Number of segments: {self.nVessels()}
            Bounding box: {[bb.tolist() for bb in self.BoundingBox()]}
            Inlet boundary condition: {self.inletBC}
            Outlet boudary condition: {self.outletBC}
        """
