import numpy as np
import Node
import pandas as pd


class VascularNetwork:
    """
    A class to represent a network of vessels embedded in a tissue slab.
    
    Attributes
    ----------
    C : np.array((nVessels, nNodes))
       represent the nodes and edges of the graph (oriented)
    R : np.array((nVessels,))
       the radius of the vessel segments (=edges)
    nodes : list
       a list of instances of the Node class
    edges : list
       a list of the edges, each formed by two nodes
    mesh : Mesh
       the Cartesian mesh within which the network is embedded
    isRepartitionned : bool
       has the network been repartitionned
    nVessels : int
       the number of vessels in the network

    Methods
    -------
    RepartitionVasculature()
        splits vessel segments until similar characteristic length as the mesh's cells 
    CheckCharacteristicLengths()
        returns whether the characteristic lengths of the vessels and mesh is similar
    GetNVessels()
        returns the number of vessels.
    Label3DMesh()
        labels the cells of the mesh 
    """

    def __init__(self, ccoFile:'str to ccoFile'):
        
        C,R,





    @classmethod
    def ReadCCO(CCOFile):
        
        SegmentsData = []
        DistalNodes  = dict()
        
        nodeName = -1

        with open(CCOFile, 'r') as f:
                
            print(f.readline().strip())
            token = f.readline().split()
            inletNode = ([float(xi) for xi in token[:3]])
            print(f"Inlet at {inletNode}")

            f.readline()
            print(f.readline().strip())
            nVessels = int(f.readline())
            print(f'The tree has {nVessels} vessels.')
            
            for i in range(nVessels):
                
                row = (f.readline()).split()
                
                nodeName+=1
                DistalNodes[int(row[0])] = nodeName
                    
                if treeType=='Object':
                    # Uncomment to keep vessels added in specific stages
                    if True:
                    # if int(row[-1]) > -2: 

                        # Id, xProx, xDist, radius, length, flow computed by CCO, distalNode, stage
                        SegmentsData.append([int(row[0]),
                        np.array([float(x) for x in row[1:4]])*1e4, 
                        np.array([float(x) for x in row[4:7]])*1e4,
                        float(row[12])*1e4,
                        np.linalg.norm(np.array([float(x) for x in row[1:4]])*1e4
                                    -np.array([float(x) for x in row[4:7]])*1e4),
                        float(row[13]),
                        nodeName,
                        int(row[-1])])
                        

                else:
                    # Id, xProx, xDist, radius, length, flow computed by CCO, distalNode, vesselId (for multiple segments vessels?)
                    SegmentsData.append([int(row[0]),
                    np.array([float(x) for x in row[1:4]])*1e4, 
                    np.array([float(x) for x in row[4:7]])*1e4,
                    float(row[8])*1e4,
                    float(row[10])*1e4,
                    float(row[11]),
                    nodeName,
                    int(row[-2])])
                    
            if treeType=='Object':            
                df = pd.DataFrame(SegmentsData, columns=['Id', 'xProx', 'xDist', 'Radius', 'Length', 'Flow','DistalNode','Stage'])
            else:
                df = pd.DataFrame(SegmentsData, columns=['Id', 'xProx', 'xDist', 'Radius', 'Length', 'Flow','DistalNode','Stage'])

            df['Inlet'] = False
            df['Outlet'] = False
            df = df.set_index('Id', drop=False)
            df['ParentId'] = -1
            df['BranchesId'] = [[] for i in range(df.shape[0])]
        
            f.readline()
            print(f.readline().strip())
            
            NodesConnections = []
            SegNewName = -1
            for i in range(nVessels):
                row = (f.readline()).split()
                SegmentId, ParentId, BranchesIds = int(row[0]), int(row[1]), [int(x) for x in row[2:]]

                if SegmentId in DistalNodes:    
                    ProximalNode = DistalNodes[SegmentId]
                    branchesId = []
                    for connection in BranchesIds:
                        DistalNode = DistalNodes[connection]
                        SegNewName +=1
                        NodesConnections.append((ProximalNode, DistalNode, SegNewName, connection))
                        branchesId.append(connection)
                    df.at[SegmentId, 'BranchesId'] = branchesId    
                    
                    if not BranchesIds:
                        df.at[SegmentId, 'Outlet'] = True
                    
                    if not ParentId in DistalNodes: # Inlet node, need to add the proximal node to the tree
                        df.at[SegmentId, 'Inlet'] = True
                        nodeName+=1
                        SegNewName +=1
                        NodesConnections.append((nodeName, DistalNodes[SegmentId], SegNewName, SegmentId))
                    else:
                        df.at[SegmentId, 'ParentId'] = ParentId

        
        ## Gives each inlet and its downstream branches a number
        def AssignBranchNumberToDownstreamVessels(InletIdx, branchNumber):
            
            # branchNumber = df.at[InletIdx, 'BranchNumber']
            branches = df.at[InletIdx, 'BranchesId']
            
            for branchId in branches:
                branchIdx = df[df.Id==branchId].index[0]
                df.at[branchIdx, 'BranchNumber'] = branchNumber
                df.at[branchIdx, 'Bifurcation']  = df.at[InletIdx,'Bifurcation']+1 
                AssignBranchNumberToDownstreamVessels(branchIdx, branchNumber)       
            
        df['BranchNumber'] = -1
        df['Bifurcation']  = 0
        for i,row in enumerate(df[df.Inlet].iterrows()):
            df.at[row[0], 'BranchNumber'] = i        
            AssignBranchNumberToDownstreamVessels(row[0],i)
                        
        # Project from the plane to the sphere
        if project:
            radiusEyeball = 23e3
            # xProx = ProjectOnSphere(np.vstack(df['xProx'].to_numpy().ravel()), r=radiusEyeball)
            # xDist = ProjectOnSphere(np.vstack(df['xDist'].to_numpy().ravel()), r=radiusEyeball)
            xProx = StereographicProjection(np.vstack(df.xProx.to_numpy().ravel()))
            xDist = StereographicProjection(np.vstack(df.xDist.to_numpy().ravel()))
            df['xProx'] = [xProx[i,:] for i in range(xProx.shape[0])] 
            df['xDist'] = [xDist[i,:] for i in range(xDist.shape[0])]
            df['Length'] = np.linalg.norm(xProx-xDist, axis=1)

        ## Create the matrices for the solver
        ConnectivityMatrix = np.zeros((nodeName+1, len(SegmentsData)))
        Radii  = np.zeros((len(SegmentsData),))
        Length = np.zeros((len(SegmentsData),))
        df['SegName'] = df.index
        df['Id'] = df.index
        df['Boundary'] = 0
        D = np.zeros((nodeName+1,nodeName+1)) # Decision matrix

        
        nodesLoc = dict()

        for proxNode, distNode, SegmentName, SegmentId in NodesConnections:
            
            ConnectivityMatrix[proxNode, SegmentName] = 1
            ConnectivityMatrix[distNode, SegmentName] = -1
            Radii[SegmentName] = df.at[SegmentId, 'Radius']
            xProx, xDist = df.at[SegmentId, 'xProx'], df.at[SegmentId, 'xDist']
            Length[SegmentName] = np.linalg.norm(xProx-xDist)

            nodesLoc[proxNode] = np.array(xProx).astype(float).reshape((3,))
            nodesLoc[distNode] = np.array(xDist).astype(float).reshape((3,))

            df.at[SegmentId, 'SegName'] = SegmentName
            if df.at[SegmentId, 'Inlet']:
                df.at[SegmentId,'Boundary'] = 1
                D[proxNode, proxNode] = 1
            elif df.at[SegmentId, 'Outlet']:
                df.at[SegmentId,'Boundary'] = -1
                D[distNode, distNode] = -1
               
        df = df.set_index('SegName')
  
        return ConnectivityMatrix.T, Radii, D, nodesLoc
 