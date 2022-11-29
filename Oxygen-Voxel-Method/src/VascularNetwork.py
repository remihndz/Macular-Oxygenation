import numpy as np
import matplotlib.pyplot as plt
import Node
import pandas as pd
import networkx as nx
from Mesh import UniformGrid
from scipy.linalg import solve

class VascularNetwork:

    def __init__(self, ccoFile, units='mm'):

        self.G = self.CreateGraph(ccoFile)
        self.C = -1.0 * nx.incidence_matrix(self.G, oriented=True).toarray().T # (nVessel, nNodes) matrix
        self._lengthConversionDict = {'mm':1e1,
                                      'cm':1.0,
                                      'micron':1e4,
                                      'mum':1e4}
        self.units = units
        self._isPartitionned = False

        # self.PlotGraph(self.G)

        bb = self.BoundingBox()
        print(f'{bb=}')
        dimensions = bb[1]-bb[0]
        if dimensions[-1]==0:
            dimensions[-1] = 0.1
        self.mesh = UniformGrid(dimensions = dimensions,
                                origin = bb[0])

        # Initialize linear system to None for error handling
        self.Flow_matrix = self.Flow_rhs = None
        self.Flow_loss = None


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

            dist = self.mesh.Dist(cell1, cell2)
            assert isinstance(dist, int), f"{dist=} is not an int."

            if dist > maxDist:
                print(f"edge {e} added to the list of vessels to split.")
                vesselsToSplit.append(e)

        # First split those vessels
        while vesselsToSplit:
            newVesselsToSplit = []
            
            for e in vesselsToSplit:
                print(f"Splitting {e}.")
                newEdges = self.SplitVessel(e)
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


    def SplitVessel(self, edge : tuple):
        '''
        Split edge into two segments by adding a node in the middle.
        edge is a tuple (n1,n2) of the nodes forming the segment.
        '''
        ## Create new node
        newNode = self.nNodes()
        # Should be an unused name
        assert not newNode in list(self.G.nodes), f"Node name {newNode} already exists."

        # Add the new node to the list of nodes
        newNodePos = (self.G.nodes[edge[0]]['position'] + self.G.nodes[edge[1]]['position'])/2.0
        self.G.add_node(newNode, position=newNodePos)
        print(f'Added node {newNode} {self.G[newNode]=}')

        ## Update the connectivity
        dataDict = self.G[edge[0]][edge[1]]
        self.G.remove_edge(*edge)
        # First segment
        dataDict['length'] = np.linalg.norm(newNodePos-self.G.nodes[edge[0]]['position'])
        self.G.add_edge(edge[0], newNode, **dataDict)
        # Second segment
        dataDict['length'] = np.linalg.norm(newNodePos-self.G.nodes[edge[1]]['position'])
        self.G.add_edge(newNode, edge[1], **dataDict)

        print(f'Created edges {(edge[0], newNode)} and {(newNode, edge[1])}')
        
        return (edge[0], newNode), (newNode, edge[1])
        
        

    def GetVesselData(self, keys : list):
        dataDict = dict()
        for key in keys:
            tmpContainer = []
            for n1, n2, data in self.G.edges.data():
                tmpContainer.append(data.get(key, None))
            dataDict[key] = np.array(tmpContainer)
        return dataDict

    def SetLinearSystem(self, **kwargs):
        ## TODO: add the plasma skimming effect

        # Update the incidence matrix in case splitting has occured
        self.C = -1.0 * nx.incidence_matrix(self.G, oriented=True).toarray().T
        
        ## Resistance matrix
        R = []
        for n1, n2, data in self.G.edges(data=True):
            r = data['radius']
            l = data['length']
            mu = self.Viscosity(r, hd=data['hd']) * 1e-3 # Converts to Pa.s
            R.append( (8*mu*l)/(np.pi*(r**4)))
        R = np.diag(R)
        ## Boundary conditions
        
        nodeInlets = [x for x in self.G.nodes() if self.G.in_degree(x)==0]
        nodeOutlets = [x for x in self.G.nodes() if self.G.out_degree(x)==0]
        # print(f'{nodeInlets=}\n{nodeOutlets=}')
        
        D = np.zeros((self.nNodes(),)) # Decision matrix
        qBar = np.zeros((self.nNodes(),)) # RHS
        pBar = np.zeros((self.nNodes(),)) # RHS

        # Define inlet boundary condition
        if 'qProx' in kwargs:
            # self.qProx = kwargs.get('qProx', # Default value is 15 muL/min
            #                         15.0 * 1e9 / (60 * pow(self._lengthConversionDict['micron']
            #                                                /self._lengthConversionDict[self.units], 3)))            
            self.qProx = kwargs['qProx']
            self.pProx = None
            qBar[nodeInlets] = self.qProx

            print(f'Using {self.qProx}{self.units}^3/s as inflow boundary condition.')

        else:
            self.qProx = None
            self.pProx = kwargs.get('pProx', # Default value is 50mmHg
                                    50 * 133.3224)
            D[nodeInlets] = 1.0
            pBar[nodeInlets] = self.pProx
            
            print(f'Using {self.pProx/133.3224}mmHg as inlet pressure.')

        # Define outlet boundary condition
        if 'qDist' in kwargs:
            self.qDist = kwargs['qDist']
            self.pDist = None
            qBar[nodeOutlets] = self.qDist
            
            print(f'Using {self.qProx}{self.units}^3/s as outflow boundary condition.')
        else:
            self.pDist = kwargs.get('pDist', # Default value is 25mmHg
                                    25 * 133.3224)
            self.qDist = None
            pBar[nodeOutlets] = self.pDist
            D[nodeOutlets] = 1.0

            print(f'Using {self.pDist/133.3224}mmHg as outlet pressure.')
        D = np.diag(D)
        I = np.identity(D.shape[0])
        
        self.Flow_matrix = np.block([
            [R, -self.C],
            [(I-D).dot(self.C.T), D]
            ])
        self.Flow_rhs    = np.concatenate([np.zeros(self.nVessels()),
                                           (I-D).dot(qBar) + D.dot(pBar)])
        
        return

    def SolveFlow(self):

        if (self.Flow_matrix is None) or (self.Flow_rhs is None):
            raise ValueError("Linear system has not been set yet.")

        x = solve(self.Flow_matrix, self.Flow_rhs)
        f,p = x[:self.nVessels()], x[self.nVessels():]
        dp  = self.C.dot(p)

        assert f.size==self.nVessels(), f"Segment flow vector has wrong size. Expected {self.nVessels()} and got {f.size}."
        assert p.size==self.nNodes(), f"Nodal pressure vector has wrong size. Expected {self.nNodes()} and got {p.size}."


        # Compute error of the solver
        self.Flow_loss = self.C.T.dot(f).sum()
        print(f"Flow loss (C*f).sum() = {self.Flow_loss} with q_in={f.max()}.")

        for i,e in enumerate(self.G.edges()):
            self.G[e[0]][e[1]].update(flow=f[i], dp=dp[i])
        
        return f,p,dp
    

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


    # def Repartition(self):
        
        

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

            G.add_node(0, position=np.array([float(xi) for xi in token[:3]]))
            
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
        
# class Node:
#     """
#     A class to construct a linked list.

#     Attributes:
#     -----------
#     pos : numpy.ndarray
#        the position of the node in 3D cartesian coordinates
#     next : list
#        pointers to the daughter nodes
#     """
#     def __init__(self, pos, Next=None):
#         self.data = pos
#         self.next = Next

#     @property
#     def next(self):
#         return self._next       
#     @next.setter
#     def next(self, Next):
#         self._next = Next

# class VascularNetwork:
#     """
#     A class to represent a network of vessels embedded in a tissue slab.
    
#     Attributes
#     ----------
#     C : np.array((nVessels, nNodes))
#        represents the nodes and edges of the graph (oriented)
#     R : np.array((nVessels,))
#        the radius of the vessel segments (=edges)
#     nodes : dict
#        a dictionnary of nodes location
#     edges : list
#        a list of the edges, each formed by two nodes
#     mesh : Mesh
#        the Cartesian mesh within which the network is embedded
#     isRepartitionned : bool
#        has the network been repartitionned
#     nVessels : int
#        the number of vessels in the network

#     Methods
#     -------
#     RepartitionVasculature()
#         splits vessel segments until similar characteristic length as the mesh's cells 
#     CheckCharacteristicLengths()
#         returns whether the characteristic lengths of the vessels and mesh is similar
#     GetNVessels()
#         returns the number of vessels.
#     Label3DMesh()
#         labels the cells of the mesh 
#     """

#     def __init__(self, ccoFile : str):
        
#         C, R, D, nodes = ReadCCO(ccoFile)
        





#     @classmethod
#     def ReadCCO(CCOFile):
        
#         SegmentsData = []
#         DistalNodes  = dict()
        
#         nodeName = -1

#         with open(CCOFile, 'r') as f:
                
#             print(f.readline().strip())
#             token = f.readline().split()
#             inletNode = ([float(xi) for xi in token[:3]])
#             print(f"Inlet at {inletNode}")

#             f.readline()
#             print(f.readline().strip())
#             nVessels = int(f.readline())
#             print(f'The tree has {nVessels} vessels.')
            
#             for i in range(nVessels):
                
#                 row = (f.readline()).split()
                
#                 nodeName+=1
#                 DistalNodes[int(row[0])] = nodeName
                    
#                 if treeType=='Object':
#                     # Uncomment to keep vessels added in specific stages
#                     if True:
#                     # if int(row[-1]) > -2: 

#                         # Id, xProx, xDist, radius, length, flow computed by CCO, distalNode, stage
#                         SegmentsData.append([int(row[0]),
#                         np.array([float(x) for x in row[1:4]])*1e4, 
#                         np.array([float(x) for x in row[4:7]])*1e4,
#                         float(row[12])*1e4,
#                         np.linalg.norm(np.array([float(x) for x in row[1:4]])*1e4
#                                     -np.array([float(x) for x in row[4:7]])*1e4),
#                         float(row[13]),
#                         nodeName,
#                         int(row[-1])])
                        

#                 else:
#                     # Id, xProx, xDist, radius, length, flow computed by CCO, distalNode, vesselId (for multiple segments vessels?)
#                     SegmentsData.append([int(row[0]),
#                     np.array([float(x) for x in row[1:4]])*1e4, 
#                     np.array([float(x) for x in row[4:7]])*1e4,
#                     float(row[8])*1e4,
#                     float(row[10])*1e4,
#                     float(row[11]),
#                     nodeName,
#                     int(row[-2])])
                    
#             if treeType=='Object':            
#                 df = pd.DataFrame(SegmentsData, columns=['Id', 'xProx', 'xDist', 'Radius', 'Length', 'Flow','DistalNode','Stage'])
#             else:
#                 df = pd.DataFrame(SegmentsData, columns=['Id', 'xProx', 'xDist', 'Radius', 'Length', 'Flow','DistalNode','Stage'])

#             df['Inlet'] = False
#             df['Outlet'] = False
#             df = df.set_index('Id', drop=False)
#             df['ParentId'] = -1
#             df['BranchesId'] = [[] for i in range(df.shape[0])]
        
#             f.readline()
#             print(f.readline().strip())
            
#             NodesConnections = []
#             SegNewName = -1
#             for i in range(nVessels):
#                 row = (f.readline()).split()
#                 SegmentId, ParentId, BranchesIds = int(row[0]), int(row[1]), [int(x) for x in row[2:]]

#                 if SegmentId in DistalNodes:    
#                     ProximalNode = DistalNodes[SegmentId]
#                     branchesId = []
#                     for connection in BranchesIds:
#                         DistalNode = DistalNodes[connection]
#                         SegNewName +=1
#                         NodesConnections.append((ProximalNode, DistalNode, SegNewName, connection))
#                         branchesId.append(connection)
#                     df.at[SegmentId, 'BranchesId'] = branchesId    
                    
#                     if not BranchesIds:
#                         df.at[SegmentId, 'Outlet'] = True
                    
#                     if not ParentId in DistalNodes: # Inlet node, need to add the proximal node to the tree
#                         df.at[SegmentId, 'Inlet'] = True
#                         nodeName+=1
#                         SegNewName +=1
#                         NodesConnections.append((nodeName, DistalNodes[SegmentId], SegNewName, SegmentId))
#                     else:
#                         df.at[SegmentId, 'ParentId'] = ParentId

        
#         ## Gives each inlet and its downstream branches a number
#         def AssignBranchNumberToDownstreamVessels(InletIdx, branchNumber):
            
#             # branchNumber = df.at[InletIdx, 'BranchNumber']
#             branches = df.at[InletIdx, 'BranchesId']
            
#             for branchId in branches:
#                 branchIdx = df[df.Id==branchId].index[0]
#                 df.at[branchIdx, 'BranchNumber'] = branchNumber
#                 df.at[branchIdx, 'Bifurcation']  = df.at[InletIdx,'Bifurcation']+1 
#                 AssignBranchNumberToDownstreamVessels(branchIdx, branchNumber)       
            
#         df['BranchNumber'] = -1
#         df['Bifurcation']  = 0
#         for i,row in enumerate(df[df.Inlet].iterrows()):
#             df.at[row[0], 'BranchNumber'] = i        
#             AssignBranchNumberToDownstreamVessels(row[0],i)
                        
#         # Project from the plane to the sphere
#         if project:
#             radiusEyeball = 23e3
#             # xProx = ProjectOnSphere(np.vstack(df['xProx'].to_numpy().ravel()), r=radiusEyeball)
#             # xDist = ProjectOnSphere(np.vstack(df['xDist'].to_numpy().ravel()), r=radiusEyeball)
#             xProx = StereographicProjection(np.vstack(df.xProx.to_numpy().ravel()))
#             xDist = StereographicProjection(np.vstack(df.xDist.to_numpy().ravel()))
#             df['xProx'] = [xProx[i,:] for i in range(xProx.shape[0])] 
#             df['xDist'] = [xDist[i,:] for i in range(xDist.shape[0])]
#             df['Length'] = np.linalg.norm(xProx-xDist, axis=1)

#         ## Create the matrices for the solver
#         ConnectivityMatrix = np.zeros((nodeName+1, len(SegmentsData)))
#         Radii  = np.zeros((len(SegmentsData),))
#         Length = np.zeros((len(SegmentsData),))
#         df['SegName'] = df.index
#         df['Id'] = df.index
#         df['Boundary'] = 0
#         D = np.zeros((nodeName+1,nodeName+1)) # Decision matrix

        
#         nodesLoc = dict()

#         for proxNode, distNode, SegmentName, SegmentId in NodesConnections:
            
#             ConnectivityMatrix[proxNode, SegmentName] = 1
#             ConnectivityMatrix[distNode, SegmentName] = -1
#             Radii[SegmentName] = df.at[SegmentId, 'Radius']
#             xProx, xDist = df.at[SegmentId, 'xProx'], df.at[SegmentId, 'xDist']
#             Length[SegmentName] = np.linalg.norm(xProx-xDist)

#             nodesLoc[proxNode] = np.array(xProx).astype(float).reshape((3,))
#             nodesLoc[distNode] = np.array(xDist).astype(float).reshape((3,))

#             df.at[SegmentId, 'SegName'] = SegmentName
#             if df.at[SegmentId, 'Inlet']:
#                 df.at[SegmentId,'Boundary'] = 1
#                 D[proxNode, proxNode] = 1
#             elif df.at[SegmentId, 'Outlet']:
#                 df.at[SegmentId,'Boundary'] = -1
#                 D[distNode, distNode] = -1
               
#         df = df.set_index('SegName')
  
#         return ConnectivityMatrix.T, Radii, D, nodesLoc

