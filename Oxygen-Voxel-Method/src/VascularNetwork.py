import numpy as np
import matplotlib.pyplot as plt
import Node
import pandas as pd
import networkx as nx
from Mesh import UniformGrid, RectilinearGrid
from math import isclose
import scipy.sparse as sp
import scipy.sparse.linalg
import vtk
from typing import Dict, Union, Tuple, List
import matplotlib.pylab as plt
from tqdm import tqdm
import multiprocessing
import sys
from astropy import units as u

class DAG(nx.DiGraph):
    def __init__(self, units : Dict[str,str], incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self.units = units
    
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

        try:
            convertL = unitsL/self._unitsL
            convertT = unitsT/self._unitsT
            newPos = {n:{'position':pos*convertL} for n,pos in self.nodes(data='position')}
            newEdgeData = {(n1,n2):{'length':data['length']*convertL, 
                            'flow':data['flow'] / convertT * (convertL**3),
                            'radius':data['radius']*convertL} for n1,n2,data in self.edges(data=True)}
            nx.set_edge_attributes(self, newEdgeData)
            nx.set_node_attributes(self, newPos)
            del newPos
            del newEdgeData
        except AttributeError:
            # This is the first setting of units.
            pass 

        self._units = {'length':unitsL, 'time':unitsT}
        self._lengthConversionFactor   = u.cm.to(unitsL)
        self._TorrConversionFactor     = u.torr.to(u.g / unitsL / (unitsT**2))
        self._cPConversionFactor       = u.cP.to(u.g / unitsL / unitsT)

    @property
    def nNodes(self) -> int:
        return self.number_of_nodes()
    @property
    def nVessels(self) -> int:
        return self.number_of_edges()
    
    def nodes_ordered(self, data = False):
        '''
        An ordered traversal of the DAG.
        '''
        if isinstance(data, str):
            return ((n, self.nodes[n][data]) for n in nx.topological_sort(self))
        elif data==True:
            return ((n, self.nodes[n]) for n in nx.topological_sort(self))
        else:
            return nx.topological_sort(self)

    @property
    def C(self):
        if hasattr(self, '_C'):
            pass
        else:
            self._C = sp.lil_matrix((self.nNodes, self.nVessels))
            for i, (n1,n2) in enumerate(self.edges()):
                self._C[n1, i] = 1.0
                self._C[n2, i] = -1.0
        return self._C.T

    @property
    def inletNodes(self) -> List[int]:
        return [x for x in self.nodes() if self.in_degree(x)==0]
    @property
    def outletNodes(self) -> List[int]:
        return [x for x in self.nodes() if self.out_degree(x)==0]
    @property
    def Resistances(self):
        ## Resistance matrix
        R = []
        for n1, n2, data in self.edges(data=True):
            r = data['radius']
            l = data['length']
            mu = self.Viscosity(r, hd=data['hd'])
            self[n1][n2]['viscosity'] = mu
            mu *= self._cPConversionFactor  # Converts to Torr.(self.unitsT)            
            R.append( (8*mu*l)/(np.pi*(r**4)))
        R = sp.dia_array(([R], [0]), shape=(len(R), len(R)))
        return R

    def GetVelocity(self, segment : tuple) -> float:
        n1,n2 = segment
        return self[n1][n2]['flow']/((self[n1][n2]['radius']**2)*np.pi)
    
    def SetLinearSystem(self, inletBC : Dict[str, float]={'pressure':100},
                        outletBC : Dict[str, float]={'pressure':26}):
        ## TODO: add the plasma skimming effect
        print("Setting the linear system for blood flow...")
        ## Boundary conditions

        self.inletBC = inletBC
        self.outletBC = outletBC
        unit = {'pressure':'mmHg', 'flow':self.unitsL**3/self.unitsT}
        
        nodeInlets = self.inletNodes
        nodeOutlets = self.outletNodes
        # print(f'{nodeInlets=}\n{nodeOutlets=}')
        
        D = np.zeros((self.nNodes,)) # Decision matrix
        qBar = np.zeros((self.nNodes,)) # RHS
        pBar = np.zeros((self.nNodes,)) # RHS

        ## Define inlet boundary condition
    
        bcType, bcValue = next(iter(self.inletBC.items()))
        print(f'\tInlet boundary condition is {bcType}={bcValue}{unit[bcType]}')
        # bcType is checked to be 'pressure' or 'flow' in the setter function 
        if bcType=='pressure':
            bcValue *= self._TorrConversionFactor # Assumes pressure was given in mmHg=Torr
            D[nodeInlets] = 1.0
            pBar[nodeInlets] = bcValue
        else:
            qBar[nodeInlets] = bcValue
            
        # Define outlet boundary condition
        bcType, bcValue = next(iter(self.outletBC.items()))
        print(f'\tOutlet boundary condition is {bcType}={bcValue}{unit[bcType]}')
        if bcType=='pressure':
            bcValue *= self._TorrConversionFactor # Assumes pressure was given in mmHg
            D[nodeOutlets] = 1.0
            pBar[nodeOutlets] = bcValue
        else:
            qBar[nodeOutlets] = bcValue

        D = sp.dia_matrix(([D],[0]), shape = (self.nNodes, self.nNodes),
                         dtype=np.float32)
        I = sp.dia_matrix(sp.eye(D.shape[0]), dtype=np.float32) 
        R = self.Resistances      
        self.Flow_matrix = sp.vstack([sp.hstack([R, -self.C], dtype=np.float32),
                                      sp.hstack([(I-D).dot(self.C.T), D])],
                                     format='csr', dtype=np.float32)

        self.Flow_rhs    = np.concatenate([np.zeros(self.nVessels),
                                           (I-D).dot(qBar) + D.dot(pBar)])

        del R
        del D
        del I
        return

    def SolveFlow(self):
        if not (hasattr(self, 'Flow_matrix') and hasattr(self, 'Flow_rhs')):
            print("Set the linear system first with 'SetLinearSystem({inletBC:value}, {outletBC:value})'.")
            return
        
        x = scipy.sparse.linalg.spsolve(self.Flow_matrix, self.Flow_rhs)
        f,p = x[:self.nVessels], x[self.nVessels:]/self._TorrConversionFactor
        dp  = self.C.dot(p)

        # assert f.size==self.nVessels(), f"Segment flow vector has wrong size. Expected {self.nVessels()} and got {f.size}."
        # assert p.size==self.nNodes, f"Nodal pressure vector has wrong size. Expected {self.nNodes} and got {p.size}."


        # Compute error of the solver
        self.Flow_loss = 0
        for i,e in enumerate(self.edges()): # Add the flow to vessel data
            n1, n2 = e
            self[n1][n2].update(flow=f[i], dp=dp[i])
            if not self.pred[n1]:
                # An inlet
                self.Flow_loss += f[i]
            elif not self.succ[n2]:
                # An outlet
                self.Flow_loss -= f[i]

        v = np.array([self.GetVelocity(e) for e in self.edges()])
        print(f"\tFlow loss with max/min velocity ({self.unitsL/self.unitsT})={v.max()}/{v.min()}\n\tmax/min flow ({self.unitsL**3/self.unitsT})={f.max()}/{f.min()}\n\tmax/min pressure (mmHg)={p.max()}/{p.min()}:")
        print(f"\t\tFlow loss f_in-f_out = {self.Flow_loss}")
        print(f"\t\tFlow loss (C*f).sum() = {self.C.T.dot(f).sum()}")        
                        
        return (f,p,dp)
        
    def Repartition(self, mesh:RectilinearGrid, maxDist=1):
        '''
        Split vessels so that end nodes are at most maxDist cells away
        from each other.
        '''
        nSplit = 0
        if maxDist < 1:
            maxDist = 1
        lmin = min(mesh.spacing)/3.0
        
        vesselsToSplit = []
        for e in list(self.edges()):
            # e is a tuple (n1,n2)
            x1, x2 = self.nodes[e[0]]['position'], self.nodes[e[1]]['position']
            stage1, stage2 = self.nodes[e[0]]['stage'], self.nodes[e[1]]['stage']
            if stage1<-1 or stage2<-1: # skip that vessel if it is part of the 'backbone'
                pass
            
            try:
                cell1, cell2 = mesh.PointToCell(x1), mesh.PointToCell(x2)
            except ValueError: # Skip that vessel, it is out of the domain
                continue 
            
            if mesh.Dist(cell1, cell2) > maxDist:
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
                    x1, x2 = self.nodes[newEdge[0]]['position'], self.nodes[newEdge[1]]['position']
                    cell1, cell2 = mesh.PointToCell(x1), mesh.PointToCell(x2)
                    
                    if mesh.Dist(cell1, cell2) > maxDist and np.linalg.norm(x1-x2)>lmin:
                        newVesselsToSplit.append(newEdge)
                    
            # Repeat the process with the newly created vessels                    
            vesselsToSplit = newVesselsToSplit

        print(f"Vascular repartion has required {nSplit} splitings.")      
        tmpLabels, newLabels = {}, {}
        for i,n in enumerate(nx.topological_sort(self)):
            tmpLabels[n]=-n
            newLabels[-n]=i
        nx.relabel_nodes(self, tmpLabels, copy=False)
        nx.relabel_nodes(self, newLabels, copy=False)

    def _SplitVessel(self, edge : tuple):
        '''
        Split edge into two segments by adding a node in the middle.
        edge is a tuple (n1,n2) of the nodes forming the segment.
        '''
        ## Create new node
        newNode = self.nNodes+1
        ## Should be an unused name
        assert not newNode in list(self.nodes), f"Node name {newNode} already exists."

        # Add the new node to the list of nodes
        newNodePos = (self.nodes[edge[0]]['position'] + self.nodes[edge[1]]['position'])/2.0
        newNodeStage = self.nodes[edge[0]]['stage'] # inherits the stage from upstream node
        self.add_node(newNode, position=newNodePos, stage=newNodeStage)

        # print(f'Added node {newNode} {self[newNode]=}')

        ## Update the connectivity
        dataDict = self[edge[0]][edge[1]]
        self.remove_edge(edge[0], edge[1])
        # print(f"Removed {edge=}.") 
        # First segment
        dataDict['length'] = np.linalg.norm(newNodePos-self.nodes[edge[0]]['position'])
        self.add_edge(edge[0], newNode, **dataDict)
        # Second segment
        dataDict['length'] = np.linalg.norm(newNodePos-self.nodes[edge[1]]['position'])
        self.add_edge(newNode, edge[1], **dataDict)
        
        return (edge[0], newNode), (newNode, edge[1])
        
    def BoundingBox(self):
        nodes = np.array([self.nodes[n]['position'] for n in self.nodes])
        return (np.min(nodes, axis=0), np.max(nodes, axis=0))  
    
    def CreateGraph(self, ccoFile : str) -> nx.DiGraph:

        lengthConversionDict = {'m':1e-2,
                                'mm':1e1,
                                'cm':1.0,
                                'micron':1e4,
                                'mum':1e4,
                                'um':1e4,
                                'microns':1e4}
        lengthConversion = lengthConversionDict[str(self.unitsL)]
        
        self.clear
        with open(ccoFile, 'r') as f:
            token = f.readline()
            token = f.readline().split() # Tree information            
            f.readline() # Blank line
            f.readline() # *Vessels
            nVessels = int(f.readline())
            print(f'The tree has {nVessels} vessels.')

            edges = dict()            
            for i in range(nVessels):
                vessel = f.readline().split()
                vesselId = int(vessel[0])
                x1,x2 = np.array([float(x) for x in vessel[1:4]])*lengthConversion, np.array([float(x) for x in vessel[4:7]])*lengthConversion
                r = float(vessel[12]) * lengthConversion
                l = np.linalg.norm(x1-x2) * lengthConversion
                edges[vesselId] = {'radius':r, 'length':l,'start':x1,'end':x2,'stage':vessel[-1]}
            f.readline() # Blank line
            f.readline() # *Connectivity

            rootId = None
            for i in range(nVessels):                
                vessel = f.readline().split()
                vesselId = int(vessel[0])
                edges[vesselId]['parent'] = int(vessel[1])
                if int(vessel[1])==-1:
                    rootId = vesselId
                edges[vesselId]['descendants'] = [int(descendant) for descendant in vessel[2:]]
            vesselId, node, nodep = rootId, 0, 1 # Start with the root            
            
            def AddVesselToGraph(vesselId, startNode):
                endNode = self.number_of_nodes()
                vessel = edges.pop(vesselId)
                self.add_node(endNode, position=vessel['end'], stage=int(vessel.pop('stage')))
                self.add_edge(startNode, endNode, radius=vessel['radius'], length=vessel['length'], hd=vessel.pop('hd',0.45))

                for descendant in vessel['descendants']:
                    AddVesselToGraph(descendant, endNode)

            self.add_node(0, position=edges[rootId]['start'], stage=-2)
            AddVesselToGraph(rootId, 0)                
            nodesToRemove = [n for n,stage in self.nodes(data='stage')
                            if stage==-2 ]
            for node in nodesToRemove:
                self.remove_node(node)

        tmpLabels, newLabels = {}, {}
        for i,n in enumerate(nx.topological_sort(self)):
            tmpLabels[n]=-n
            newLabels[-n]=i
        nx.relabel_nodes(self, tmpLabels, copy=False)
            #nx.convert_node_labels_to_integers(self)

    def VesselsToVTK(self, VTKFileName):
        # Create list of points (nodes)
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(self.nNodes)
        for n, data in self.nodes(data=True):
            points.SetPoint(n, data['position'])

        # Create list of lines (vessels)
        # with list of radius
        lines = vtk.vtkCellArray()
        radius = vtk.vtkDoubleArray()
        flow = vtk.vtkDoubleArray()
        PO2  = vtk.vtkDoubleArray()
        
        flow.SetName(f'flow [{(self.unitsL**3)/self.unitsT}]')
        radius.SetName(f"radius [{self.unitsL}]")
        PO2.SetName(f"PO2 [mmHg]")
        
        for n1, n2, data in tqdm(self.edges(data=True), desc=f"Writing vessel data to {VTKFileName}"):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, n1)
            line.GetPointIds().SetId(1, n2)
            lines.InsertNextCell(line)
            radius.InsertNextValue(data['radius'])
            flow.InsertNextValue(data.get('flow', 0.0))

        for n, p in (self.nodes(data='PO2')):
            PO2.InsertNextValue(p)

        # Create the polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetCellData().AddArray(radius)
        polydata.GetCellData().AddArray(flow)
        polydata.GetPointData().AddArray(PO2)
        polydata.SetLines(lines)

        # Write the polydata
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(VTKFileName)
        writer.SetInputData(polydata)
        writer.SetDataModeToBinary()
        writer.Update()
        writer.Write()

        return

    def LabelMesh(self, mesh : UniformGrid, endotheliumThickness : float, repartition=True):
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
        NodesToEndothelialCells : scipy.sparse_array
            The connectivity matrix of vascular nodes to endothelial cells.
        NodesToVascularCells : scipy.sparse_array
            The connectivity matrix of vascular nodes to vascular cells.
        VascularCellsToNodes : scipy.sparse_array
            The connectivity matrix of vascular cells to vascular nodes.
        """
        

        if repartition:
            self.Repartition(mesh, maxDist=1) # Split the vessels to the size of the mesh

        mesh.labels.EmptyMatrix()
        self.w = endotheliumThickness

        # The empty connectivity matrix
        def _LabelSerial(edge, _C2v, _C4, _I4):

            n1,n2,data = edge
            
            p1, p2 = self.nodes[n1]['position'], self.nodes[n2]['position']                    
            
            r,l = data['radius'], data['length']
            vectorDirection = (p1-p2)/l # Unit vector (direction)

            # l_new, l_old = np.linalg.norm(p1-p2), l
            # if  np.isclose(l_new, l_old):
            #     print(f"The length stored in the graph's edge data is incorrect: {l_old=} {l_new=} for vessel {(n1,n2)=}: {self.nodes[n1]=}, {self.nodes[n2]=}.")
            P = np.outer(vectorDirection, vectorDirection) # Matrix of the orthogonal projection onto the vessel axis
            O = np.identity(3)-P                           # O*y = y-P*y is orthogonal to the axis

            ## Find the bounding box
            # To ensure full enclosure of the vessel, the bounding box should
            # bound the vessel extended by its radius in each direction
            p1, p2 = p1 - r * vectorDirection, p2 + r * vectorDirection  
            cellMin, cellMax = mesh._BoundingBoxOfVessel(p1, p2, r)

            # Iterate through the cells within the bounding box
            for cellId in ((x,y,z) for x in range(cellMin[0], cellMax[0]+1)
                                   for y in range(cellMin[1], cellMax[1]+1)
                                   for z in range(cellMin[2], cellMax[2]+1)):
                
                # if any([i>j or i<0 for i,j in zip(cellId, mesh.nCells)]):
                #     continue
                # print(cellId, mesh.nCells, [i>j or i<0 for i,j in zip(cellId, mesh.nCells)])
                # Assign new label
                HasUpdatedValue, newLabel = self._LabelCellWithCylinder(mesh, O, p1, p2, r, cellId, endotheliumThickness)
                # Add to connectivity matrix if the cell label changed from tissue to endothelial
                if HasUpdatedValue:
                    flatId = mesh.ToFlatIndexFrom3D(cellId)
                    if newLabel==1:
                        _C4[flatId, n2] = 1.0
                        _I4[flatId,flatId] = 1.0
                        _C2v.rows[flatId].clear()
                        _C2v.data[flatId].clear()
                    elif newLabel==2:
                        _C2v[flatId, n2] = 1.0
                        
    
        C2v = sp.lil_matrix((mesh.nVol, self.nNodes))
        C4  = sp.lil_matrix((mesh.nVol, self.nNodes))
        I4  = sp.lil_matrix((mesh.nVol,mesh.nVol))

        for edge in self.edges(data=True):

            n1,n2, data = edge
            p1, p2 = self.nodes[n1]['position'], self.nodes[n2]['position']
            d = (p1-p2)/data['length']
            for t in np.linspace(-mesh.hmax/data['length'], mesh.hmax/data['length']+1, endpoint=True, num=10):
                try:
                    cellId = mesh.PointToCell(p1 + t*data['length']*d)
                    HasUpdated = mesh.SetLabelOfCell(1, cellId)        
                    if HasUpdated:
                        flatId = mesh.ToFlatIndexFrom3D(cellId)
                        C4[flatId, n2] = 1.0
                        I4[flatId,flatId] = 1.0
                except ValueError:
                    pass

        for edge in tqdm((edge for edge in self.edges(data=True)), total=self.nVessels, desc="Labelling in progress"):
            _LabelSerial(edge, C2v,C4,I4)
        return C2v.tocsr(), C4.tocsr(), I4.tocsr()
    
    def _LabelCellWithCylinder(self, mesh, O : np.ndarray, p1 : np.ndarray, p2 : np.ndarray, r : float,
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
        try:                   
            cellCenter = mesh.CellCenter(cellId)
        except ValueError: # The cell is not in the cuboid (vessel outside domain, thus ignored)
            return False, 0 
        d = np.linalg.norm( O.dot(p1-cellCenter) ) # The radial distance between the vessel axis and the cell center           
        
        if (d < r-endotheliumThickness/2.0):
            newLabel = 1
        elif (d < r - endotheliumThickness/2.0 + 0.87*mesh.hmax): # 0.87~=sqrt(3)/2 which is the max distance to the center of the unit cube
        #elif (d<r+endotheliumThickness/2.0):
            newLabel = 2
        else:
            newLabel = 0

        # labelDict = {0:'tissue', 1:'vessel', 2:'endothelium'} 
        updatedValue = mesh.SetLabelOfCell(newLabel, cellId)        
        return updatedValue, newLabel
    
    def Viscosity(self, radius, hd=0.45, Type='Pries'):
        '''
        Radius should be in microns.
        '''
        d = (2 * radius * self.unitsL).to_value(u.um)
        
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
            Number of nodes: {self.nNodes}
            Number of segments: {self.nVessels()}
            Length scale: {self.unitsL}
            Time scale: {self.unitsT}
            Bounding box: {[bb.tolist() for bb in self.BoundingBox()]}
            Maximum/Minimum radii: {self.maxRad}/{self.minRad}
        """

class VascularNetwork(object):
    """A class storing a vascular network given in a .cco format.
    TODO: add different input files than .cco.
    Attributes:
    -----------
    nNodes
        Returns the number of nodes in the network.
    nVessels
        Returns the number of vessels in the network.
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
    BoundingBox()
        Return the bounding box of the network.
    PlotGraph(Graph)
        Plots a graph.
    """
    
    def __init__(self, ccoFile : str,
                 units : Dict[str, str]={'length':'mm','time':'s'},
                 **kwargs):        
        """Creator for the class VascularNetwork

        Parameters
        ----------
        ccoFile : str
            the input network in .cco format
        units : dict(str,str)
            the time and length units. Length must be one of
            {'mm','cm','micron','um'}.
        **kwargs : 
            additional mesh parameters
            {'dimensions','origin','spacing'}

        """

        self._lengthConversionDict = {'mm':1e1,
                                      'cm':1.0,
                                      'micron':1e4,
                                      'um':1e4,
                                      'mum':1e4}
        self.units = units
        self._isPartitionned = False

        print(f"cm are converted to {self.unitsL} by multiplying by {self._lengthConversionFactor}")

        self.G = self.CreateGraph(ccoFile, convertUnitsTo=self.unitsL)
        self.C = -1.0 * nx.incidence_matrix(self.G, oriented=True).T # (nVessel, nNodes) matrix

        # self.PlotGraph(self.G)

        # Pad the cuboid with the radius of the larger vessel to
        # prevent having vessel cylinder outside the cuboid
        bb = self.BoundingBox()
        print("Vessels' kwargs:", kwargs)
        self.maxRad = self.GetVesselData(['radius'])['radius'].max()
        self.minRad = self.GetVesselData(['radius'])['radius'].min()
        origin     = kwargs.get('origin', bb[0]) - 2*self.maxRad
        dimensions = kwargs.get('dimensions', bb[1]-bb[0]) + 4*self.maxRad
        spacing    = kwargs.get('spacing', [self.minRad/4]*3)

        if dimensions[-1]==0:
            dimensions[-1] = 1
        nCells = kwargs.get('nCells', [20,20,20])

        print(f"Creating mesh with {dimensions=}\n{origin=}\n{spacing=}")
        self.mesh = UniformGrid(dimensions = dimensions,
                                origin = origin,
                                nCells = nCells,
                                spacing = spacing,
                                units=self.unitsL)

        # Initialize linear system to None for error handling
        self.Flow_matrix = self.Flow_rhs = None
        self.Flow_loss = None
        self.inletBC = {}
        self.outletBC = {}

        print(self)
        
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
        NodesToEndothelialCells : scipy.sparse_array
            The connectivity matrix of vascular nodes to endothelial cells.
        NodesToVascularCells : scipy.sparse_array
            The connectivity matrix of vascular nodes to vascular cells.
        VascularCellsToNodes : scipy.sparse_array
            The connectivity matrix of vascular cells to vascular nodes.
        """
        
        global _Label

        if repartition:
            self.Repartition(maxDist=1) # Split the vessels to the size of the mesh

        self.mesh.labels.EmptyMatrix()
        self.w = endotheliumThickness

        # The empty connectivity matrix

        def _LabelSerial(edge, NodesToEndothelialCells, VascularCellsToNodes, NodesToVascularCells):

            n1,n2,data = edge
            
            p1, p2 = self.G.nodes[n1]['position'], self.G.nodes[n2]['position']                    
            
            r,l = data['radius'], data['length']
            vectorDirection = (p1-p2)/l # Unit vector (direction)

            # l_new, l_old = np.linalg.norm(p1-p2), l
            # if  np.isclose(l_new, l_old):
            #     print(f"The length stored in the graph's edge data is incorrect: {l_old=} {l_new=} for vessel {(n1,n2)=}: {self.G.nodes[n1]=}, {self.G.nodes[n2]=}.")
            P = np.outer(vectorDirection, vectorDirection) # Matrix of the orthogonal projection onto the vessel axis
            O = np.identity(3)-P                           # O*y = y-P*y is orthogonal to the axis

            ## Find the bounding box
            # To ensure full enclosure of the vessel, the bounding box should
            # bound the vessel extended by its radius in each direction
            p1, p2 = p1 - r * vectorDirection, p2 + r * vectorDirection  
            cellMin, cellMax = self.mesh._BoundingBoxOfVessel(p1, p2, r)

            # Iterate through the cells within the bounding box
            for cellId in ((x,y,z) for x in range(cellMin[0], cellMax[0]+1)
                                   for y in range(cellMin[1], cellMax[1]+1)
                                   for z in range(cellMin[2], cellMax[2]+1)):
                # Assign new label
                HasUpdatedValue, newLabel = self._LabelCellWithCylinder(O, p1, p2, r, cellId, endotheliumThickness)
                # Add to connectivity matrix if the cell label changed from tissue to endothelial
                if HasUpdatedValue:
                    if newLabel==1:
                        VascularCellsToNodes[self.mesh.ToFlatIndexFrom3D(cellId), n2] = 1
                        NodesToVascularCells[self.mesh.ToFlatIndexFrom3D(cellId)] = 1
                        NodesToEndothelialCells.rows[self.nNodes()+self.mesh.ToFlatIndexFrom3D(cellId)].clear()
                        NodesToEndothelialCells.data[self.nNodes()+self.mesh.ToFlatIndexFrom3D(cellId)].clear()
                    elif newLabel==2:
                        NodesToEndothelialCells[self.nNodes()+self.mesh.ToFlatIndexFrom3D(cellId), n2] 
                        
        ## TODO: add multiprocessing labelling (how much of a gain in time though?)
        if True: 
            print("Labelling with 1 processor.")
    
            NodesToEndothelialCells = sp.diags([-1], [0], shape=(self.nPoints, self.nVol + self.nPoints), format='lil').T
            VascularCellsToNodes = sp.lil_matrix((self.nVol, self.nPoints))
            NodesToVascularCells = np.zeros(self.nVol)

            for edge in self.G.edges(data=True):

                n1,n2, data = edge
                p1, p2 = self.G.nodes[n1]['position'], self.G.nodes[n2]['position']
                d = (p1-p2)/data['length']
                for t in np.linspace(-self.mesh.hmax/data['length'], self.mesh.hmax/data['length']+1, endpoint=True, num=10):
                    try:
                        cellId = self.mesh.PointToCell(p1 + t*data['length']*d)
                    except ValueError:
                        continue
                    HasUpdated = self.mesh.SetLabelOfCell(1, cellId)        
                    if HasUpdated:
                        VascularCellsToNodes[self.mesh.ToFlatIndexFrom3D(cellId), n2] = 1
                        NodesToVascularCells[self.mesh.ToFlatIndexFrom3D(cellId)] = 1

            for edge in tqdm((edge for edge in self.G.edges(data=True)), total=self.nVessels(), desc="Labelling in progress"):
                _LabelSerial(edge, NodesToEndothelialCells, VascularCellsToNodes, NodesToVascularCells)

        NodesToEndothelialCells = NodesToEndothelialCells.T
        for cellId, label in zip(list(self.mesh.labels.elements.keys()), list(self.mesh.labels.elements.values())):
            if label==1:
                node = VascularCellsToNodes[self.mesh.ToFlatIndexFrom3D(cellId)].rows[0]
                if len(node)>1:
                    for n in node[1:]:
                        VascularCellsToNodes[self.mesh.ToFlatIndexFrom3D(cellId), n] = 0
                    node = node[0]

                for i,n in ((k,m) for k in range(3) for m in (-1,1)):
                    neighbourId = list(cellId)
                    neighbourId[i] += n
                    neighbourId = tuple(neighbourId)

                    try:
                        if self.mesh.labels[neighbourId]==0:
                            # Update the endothelial cells' connectivity matrix
                            NodesToEndothelialCells[node, self.nPoints + self.mesh.ToFlatIndexFrom3D(neighbourId)] = 1.0
                            HasUpdated = self.mesh.SetLabelOfCell(2, neighbourId) # A vascular cell should be surrounded by vascular cells or endothelial cells
                            # Sanity check
                            #assert HasUpdated, f"The cell {neighbourId} has not been updated"
                    except KeyError:
                        #print(f"Neighbour {neighbourId} of cell {cellId} out of bounds of the tissue ({self.mesh.nCells=}).")
                        pass

        return NodesToEndothelialCells.tocsr(), VascularCellsToNodes.tocsr(), sp.diags([NodesToVascularCells], [0], format='csr')
    
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
        
        if (d < r-endotheliumThickness/2.0):
            newLabel = 1
        elif (d < r - endotheliumThickness/2.0 + 0.87*self.mesh.hmax): # 0.87~=sqrt(3)/2 which is the max distance to the center of the unit cube
        #elif (d<r+endotheliumThickness/2.0):
            newLabel = 2
        else:
            newLabel = 0

        # labelDict = {0:'tissue', 1:'vessel', 2:'endothelium'} 
        updatedValue = self.mesh.SetLabelOfCell(newLabel, cellId)        
        return updatedValue, newLabel

    def MeshToVTK(self, VTKFileName):
        self.mesh.ToVTK(VTKFileName)
        return
    
    def VesselsToVTK(self, VTKFileName):
        # Create list of points (nodes)
        points = vtk.vtkPoints()
        points.SetNumberOfPoints(self.nPoints)
        for n, data in self.G.nodes(data=True):
            points.SetPoint(n, data['position'])

        # Create list of lines (vessels)
        # with list of radius
        lines = vtk.vtkCellArray()
        radius = vtk.vtkDoubleArray()
        flow = vtk.vtkDoubleArray()
        PO2  = vtk.vtkDoubleArray()
        
        flow.SetName(f'flow [{(self.unitsL**3)/self.unitsT}]')
        radius.SetName(f"radius [{self.unitsL}]")
        PO2.SetName(f"PO2 [mmHg]")
        
        for n1, n2, data in tqdm(self.G.edges(data=True), desc=f"Writing vessel data to {VTKFileName}"):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, n1)
            line.GetPointIds().SetId(1, n2)
            lines.InsertNextCell(line)
            radius.InsertNextValue(data['radius'])
            flow.InsertNextValue(data.get('flow', 0.0))

        for n, p in (self.G.nodes(data='PO2')):
            PO2.InsertNextValue(p)

        # Create the polydata
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.GetCellData().AddArray(radius)
        polydata.GetCellData().AddArray(flow)
        polydata.GetPointData().AddArray(PO2)
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
        if maxDist < 1:
            maxDist = 1
        lmin = min(self.mesh.spacing)/3.0
        
        vesselsToSplit = []
        for e in list(self.G.edges()):
            # e is a tuple (n1,n2)
            x1, x2 = self.G.nodes[e[0]]['position'], self.G.nodes[e[1]]['position']
            stage1, stage2 = self.G.nodes[e[0]]['stage'], self.G.nodes[e[1]]['stage']
            if stage1<-1 or stage2<-1: # skip that vessel if it is part of the 'backbone'
                pass
            
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
                    
                    if self.mesh.Dist(cell1, cell2) > maxDist and np.linalg.norm(x1-x2)>lmin:
                        newVesselsToSplit.append(newEdge)
                    
            # Repeat the process with the newly created vessels                    
            vesselsToSplit = newVesselsToSplit

        # tmpLabels, newLabels = {}, {}
        # for i,n in enumerate(nx.topological_sort(self.G)):
        #     tmpLabels[n]=-n
        #     newLabels[-n]=i
        # self.G=nx.relabel_nodes(self.G, tmpLabels, copy=False)
        # self.G=nx.relabel_nodes(self.G, newLabels, copy=False)
        print(f"Vascular repartion has required {nSplit} splitings.")       


    def _SplitVessel(self, edge : tuple):
        '''
        Split edge into two segments by adding a node in the middle.
        edge is a tuple (n1,n2) of the nodes forming the segment.
        '''
        ## Create new node
        newNode = self.nPoints
        ## Should be an unused name
        assert not newNode in list(self.G.nodes), f"Node name {newNode} already exists."

        # Add the new node to the list of nodes
        newNodePos = (self.G.nodes[edge[0]]['position'] + self.G.nodes[edge[1]]['position'])/2.0
        newNodeStage = self.G.nodes[edge[0]]['stage'] # inherits the stage from upstream node
        self.G.add_node(newNode, position=newNodePos, stage=newNodeStage)

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
            mu = self.Viscosity(r, hd=data['hd'])*self._cPConversionFactor  # Converts to Torr.(self.unitsT)
            self.G[n1][n2]['viscosity'] = mu
            R.append( (8*mu*l)/(np.pi*(r**4)))
        R = sp.dia_array(([R], [0]), shape=(len(R), len(R)))

        ## Boundary conditions

        self.inletBC = inletBC
        self.outletBC = outletBC
        unit = {'pressure':'mmHg', 'flow':self.unitsL**3/self.unitsT}
        
        nodeInlets = self.inletNodes
        nodeOutlets = self.outletNodes
        # print(f'{nodeInlets=}\n{nodeOutlets=}')
        
        D = np.zeros((self.nPoints,)) # Decision matrix
        qBar = np.zeros((self.nPoints,)) # RHS
        pBar = np.zeros((self.nPoints,)) # RHS

        ## Define inlet boundary condition
    
        bcType, bcValue = next(iter(self.inletBC.items()))
        print(f'\tInlet boundary condition is {bcType}={bcValue}{unit[bcType]}')
        # bcType is checked to be 'pressure' or 'flow' in the setter function 
        if bcType=='pressure':
            bcValue *= self._TorrConversionFactor # Assumes pressure was given in mmHg=Torr
            D[nodeInlets] = 1.0
            pBar[nodeInlets] = bcValue
        else:
            qBar[nodeInlets] = bcValue
            
        # Define outlet boundary condition
        bcType, bcValue = next(iter(self.outletBC.items()))
        print(f'\tOutlet boundary condition is {bcType}={bcValue}{unit[bcType]}')
        if bcType=='pressure':
            bcValue *= self._TorrConversionFactor # Assumes pressure was given in mmHg
            D[nodeOutlets] = 1.0
            pBar[nodeOutlets] = bcValue
        else:
            qBar[nodeOutlets] = bcValue

        D = sp.dia_array(([D],[0]), shape = (self.nPoints, self.nPoints),
                         dtype=np.float32)
        I = sp.dia_array(sp.eye(D.shape[0]), dtype=np.float32)        
        
        self.Flow_matrix = sp.vstack([sp.hstack([R, -self.C], dtype=np.float32),
                                      sp.hstack([(I-D) @ self.C.T, D])],
                                     format='csr', dtype=np.float32)

        self.Flow_rhs    = np.concatenate([np.zeros(self.nVessels()),
                                           (I-D).dot(qBar) + D.dot(pBar)])

        # print(self)        
        return

    def SolveFlow(self):

        print(f"Solving for blood flow and pressure with boundary conditions:\n"
              f"\t-Inlet {self.inletBC}\n"
              f"\t-Outlet {self.outletBC}")
        
        if (self.Flow_matrix is None) or (self.Flow_rhs is None):
            raise ValueError("Linear system has not been set yet.")

        x = scipy.sparse.linalg.spsolve(self.Flow_matrix, self.Flow_rhs)
        f,p = x[:self.nVessels()], x[self.nVessels():]/self._TorrConversionFactor
        dp  = self.C.dot(p)

        # assert f.size==self.nVessels(), f"Segment flow vector has wrong size. Expected {self.nVessels()} and got {f.size}."
        # assert p.size==self.nNodes, f"Nodal pressure vector has wrong size. Expected {self.nNodes} and got {p.size}."


        # Compute error of the solver
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

        v = np.array([self.GetVelocity(e) for e in self.G.edges()])
        print(f"\tFlow loss with max/min velocity={v.max()}/{v.min()}, max/min flow={f.max()}/{f.min()}, max/min pressure {p.max()}/{p.min()}:")
        print(f"\t\tFlow loss f_in-f_out = {self.Flow_loss}")
        print(f"\t\tFlow loss (C*f).sum() = {self.C.T.dot(f).sum()}")        
                        
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
        if isinstance(newFlowMatrix, np.ndarray) and  newFlowMatrix.shape != (self.nVessels()+self.nPoints,
                                                            self.nVessels()+self.nPoints):
            raise ValueError(f"Flow matrix with shape {newFlowMatrix.shape} inconsistent with vascular network with {self.nVessels()} vessels and {self.nPoints} nodes.")
        self._Flow_matrix = newFlowMatrix

    @property
    def Flow_rhs(self):
        return self._Flow_rhs
    @Flow_rhs.setter
    def Flow_rhs(self, newRHS):
        if isinstance(newRHS, np.ndarray) and newRHS.shape[0] != self.nVessels()+self.nPoints:
            raise ValueError(f"RHS with shape {newRHS.shape} inconsistent with vascular network with {self.nVessels()} vessels and {self.nPoints} nodes.")
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
        # try:                    
        #     self._lengthConversionFactor = self._lengthConversionDict[unitsL]
        # except:
        #     raise KeyError(f"Wrong key '{unitsL}' for unit conversion. Valid keys are {self._lengthConversionDict.keys()}.")
        self._units = {'length':unitsL, 'time':unitsT}
        self._lengthConversionFactor   = u.cm.to(unitsL)
        self._TorrConversionFactor     = u.torr.to(u.g / unitsL / (unitsT**2))
        self._cPConversionFactor       = u.cP.to(u.g / unitsL / unitsT)

    @property
    def nVol(self):
        return self.mesh.nCellsTotal
    @property
    def nPoints(self):
        return self.nNodes()

    def GetVelocity(self, segment : tuple) -> float:
        n1,n2 = segment
        return self.G[n1][n2]['flow']/((self.G[n1][n2]['radius']**2)*np.pi)

    def BoundingBox(self):
        nodes = np.array([self.G.nodes[n]['position'] for n in self.G.nodes])
        return [np.min(nodes, axis=0), np.max(nodes, axis=0)]        
        

    @staticmethod
    def CreateGraph(ccoFile : str, convertUnitsTo) -> nx.DiGraph:

        lengthConversionDict = {'mm':1e1,
                                'cm':1.0,
                                'micron':1e4,
                                'mum':1e4,
                                'um':1e4,
                                'microns':1e4}
        try:
            lengthConversion = lengthConversionDict[str(convertUnitsTo)]
            print(f"{lengthConversion=}")
        except:
            raise KeyError(f"Wrong key '{convertUnitsTo}' for unit conversion. Valid keys are {lengthConversionDict.keys()}")
        
        G = nx.DiGraph()
        
        with open(ccoFile, 'r') as f:

            token = f.readline()
            token = f.readline().split() # Tree information

            # G.add_node(0, position=np.array([float(xi) for xi in token[:3]]) * lengthConversion)
            
            f.readline() # Blank line
            f.readline() # *Vessels
            nVessels = int(f.readline())
            print(f'The tree has {nVessels} vessels.')


            edges = dict()
            
            for i in range(nVessels):

                vessel = f.readline().split()
                vesselId = int(vessel[0])
                x1,x2 = np.array([float(x) for x in vessel[1:4]])*lengthConversion, np.array([float(x) for x in vessel[4:7]])*lengthConversion
                r = float(vessel[12]) * lengthConversion
                l = np.linalg.norm(x1-x2) * lengthConversion

                edges[vesselId] = {'radius':r, 'length':l,'start':x1,'end':x2,'stage':vessel[-1]}
                
                # G.add_node(vesselId+1, position=np.array([float(xi) for xi in vessel[4:7]]) * lengthConversion)

            f.readline() # Blank line
            f.readline() # *Connectivity

            rootId = None
            for i in range(nVessels):
                
                vessel = f.readline().split()
                vesselId = int(vessel[0])

                edges[vesselId]['parent'] = int(vessel[1])
                if int(vessel[1])==-1:
                    rootId = vesselId
                edges[vesselId]['descendants'] = [int(descendant) for descendant in vessel[2:]]
                
                # if int(vessel[1]) == -1: # If root
                #     edges.append((0, 1, {'radius':vesselRadii[0], 'length':vesselLength[0], 'hd':0.45}))
                    
                # else:
                #     vessel, parent = int(vessel[0]), int(vessel[1])
                #     edges.append((parent+1, vessel+1, {'radius':vesselRadii[vessel], 'length':vesselLength[vessel], 'hd':0.45}))

            # G.add_edges_from(edges)
            vesselId, node, nodep = rootId, 0, 1 # Start with the root
            
            
            def AddVesselToGraph(vesselId, startNode):
                endNode = G.number_of_nodes()
                vessel = edges.pop(vesselId)
                G.add_node(endNode, position=vessel['end'], stage=int(vessel.pop('stage')))
                G.add_edge(startNode, endNode, radius=vessel['radius'], length=vessel['length'], hd=vessel.pop('hd',0.45))

                for descendant in vessel['descendants']:
                    AddVesselToGraph(descendant, endNode)

            G.add_node(0, position=edges[rootId]['start'], stage=-2)
            AddVesselToGraph(rootId, 0)                
            
            nodeToRemove = [n for n,stage in G.nodes(data='stage')
                            if stage==-2 ]
            for node in nodeToRemove:
                G.remove_node(node)

        return nx.convert_node_labels_to_integers(G)

    
    @staticmethod
    def PlotGraph(Graph):
        nx.draw_networkx_edges(Graph, {n:Graph.nodes[n]['position'][:-1] for n in Graph.nodes})
        plt.show()

    def Viscosity(self, radius, hd=0.45, Type='Pries'):
        '''
        Radius should be in microns.
        '''
        d = (2 * radius * self.unitsL).to_value(u.um)
        
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
            Number of nodes: {self.nPoints}
            Number of segments: {self.nVessels()}
            Length scale: {self.unitsL}
            Time scale: {self.unitsT}
            Bounding box: {[bb.tolist() for bb in self.BoundingBox()]}
            Maximum/Minimum radii: {self.maxRad}/{self.minRad}
            Inlet boundary condition: {self.inletBC}
            Outlet boudary condition: {self.outletBC}
        """
