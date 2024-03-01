import matplotlib.pyplot as plt
# import numba

import scipy.sparse as sp
import numpy as np

from typing import Union, Tuple, List
from itertools import groupby
import graph_tool as gt
from graph_tool.generation import lattice
from graph_tool.draw import graph_draw
from graph_tool.util import find_edge
from tqdm import tqdm
import cProfile as profile
from numpy import ma
import itertools
import time

import cProfile as profile
from matplotlib.collections import LineCollection


def get_prop_type(value, key=None):
    """
    Performs typing and value conversion for the graph_tool PropertyMap class.
    If a key is provided, it also ensures the key is in a format that can be
    used with the PropertyMap. Returns a tuple, (type name, value, key)
    """
    
    if key=='position':
        tname = 'vector<float>' # Somehow using vectors doesn't allow for substraction
    elif key == 'plexus':
        tname = 'short'
        value = int(value)
    elif key == 'stage':
        tname = 'short'
        try: 
            value = int(value)
        except ValueError:
            value = 10
    elif key=='nodeType':
        tname = 'string'
        value = str(value)
    elif key in ['radius','pressure','length','hd','flow']:
        tname = 'float'
        try:
            value = float(value)
        except ValueError:
            value = 0.0

    else:    

        if isinstance(key, str):
            # Encode the key as ASCII
            tname = 'string'
            key = key.encode('ascii', errors='replace')

        # Deal with the value
        if isinstance(value, bool):
            tname = 'bool'

        elif isinstance(value, int):
            tname = 'float'
            value = float(value)

        elif isinstance(value, float):
            tname = 'float'

        elif isinstance(value, str):
            tname = 'string'
            value = str(value.encode('ascii', errors='replace'))

        elif isinstance(value, dict):
            tname = 'object'

        else:
            tname = 'string'
            value = str(value)

    return str(tname), value, key

class Grid(object):
    def __init__(self, origin:tuple[float], sideLengths:tuple[float], shape:tuple[int])->None:
        """
        Initialize a 3D grid with the specified shape.

        Parameters:
        - origin (tuple): The location of the bottom, left, front corner (i.e., index 0 o the grid), e.g., (0,0,0)
        - sideLengths (tuple): The length of each side of the grid (e.g., (1.0,1.2,1.5))
        - shape (tuple): The shape of the grid (e.g., (3, 3, 3)).
        """
        self.n = np.asarray(shape)
        self.origin = np.asarray(origin)
        self.l = np.asarray(sideLengths)
        self.h = np.asarray([l/(n-1) for l,n in zip(self.l, self.n)])
        self.labels = {}

    def __getitem__(self, idx)->float:
        return self.grid[idx]
    def __iter__(self):
        for i in range(self.size):
            try:
                yield self.labels[i]
            except KeyError:
                yield 0

    @property
    def shape(self)->tuple[int]:
        return self.n
    @property
    def size(self)->int:
        return np.prod(self.shape)

    def GetItem(self, flatIndex:int)->float:
        '''Returns grid entry based on a flat index.'''
        return self.flatGrid[flatIndex]    

    def GetNeighbors(self, i,j,k)->np.ndarray:
        """Get the direct neighbors of a cell at the specified indices."""
        neighbors = []
        for di in range(-1, 2):
            for dj in range(-1, 2):
                for dk in range(-1, 2):
                    if not 0<sum(abs(d) for d in (di,dj,dk))<=1:
                        continue # Skip diagonal neighbours and call to cell itself
                    ni, nj, nk = i + di, j + dj, k + dk
                    if 0 <= ni < self.shape[0] and 0 <= nj < self.shape[1] and 0 <= nk < self.shape[2]:
                        neighbors.append(self.grid[ni,nj,nk])
        return np.asarray(neighbors)
    
    def ToFlatIndex(self, i:int,j:int,k:int)->int:
        '''Returns the index in the flattened array.'''
        #return self.shape[1] * (i*self.shape[0] + j) + k
        return np.ravel_multi_index((i,j,k), dims=self.shape, mode='wrap')
    def To3DIndex(self, idx:int)->tuple[int]:
        '''Returns the 3D index.'''
        # i = idx // (self.shape[0]*self.shape[1])
        # j = (idx - i*self.shape[0]*self.shape[1]) // self.shape[0]
        # k = idx - self.shape[1] * (j + self.shape[0]*i)
        # return (i,j,k)
        return np.unravel_index(idx, shape=self.shape)


    def ProbeAlongLine(self, start:tuple[float], end:tuple[float]):
        '''
        TODO, see https://code.activestate.com/recipes/578112-bresenhams-line-algorithm-in-n-dimensions/
        '''
        print("Not implemented yet.")
        return
    
    def Dist(self,
             cell1 : Union[List[int], Tuple[int], np.ndarray],
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
        return np.maximum(cellMin, [0,0,0]), np.minimum(cellMax, self.n-1)
    
    def PointToCell(self, X: Union[List[float], Tuple[float], np.ndarray]) -> np.ndarray:
        # Center and normalize
        X = np.divide(np.asarray(X).reshape((3,-1)) - self.origin[:,np.newaxis], self.l[:,np.newaxis]) 
        xVoxelSpace = np.multiply(X, self.n[:,np.newaxis]) # Project onto voxel space
        return np.floor(xVoxelSpace).astype(int)
    
    def CellCenter(self, ijk : Union[np.ndarray, List[int], int, Tuple[int]]) -> np.ndarray:
        if isinstance(ijk, int):
            ijkarr = np.asarray(self.To3DIndex(ijk))
        ijkarr = np.asarray(ijk).reshape((3,-1)) 
        if ((ijkarr-self.n[:,np.newaxis]) > 0).any():
            raise ValueError(f"Indices {ijkarr[((ijkarr-self.n[:,np.newaxis])>0).any(1)]} out of bounds for the grid.")        
        cellCenter = self.origin[:,np.newaxis] + self.l[:,np.newaxis] * ((0.5+ijkarr)/self.n[:,np.newaxis])
        return np.squeeze(cellCenter)
    
class VascularGraph(gt.Graph):
    
    def __init__(self, graphFile:str=None) -> None:
        # super(VascularGraph, self).__init__(directed=True)
        if graphFile:
            G = self._ReadFromFile(graphFile)
            sort = gt.topology.topological_sort(G)
            G.vp['sort'] = G.new_vertex_property('int', vals=sort)
            super(VascularGraph, self).__init__(G, directed=True, vorder=G.vp['sort'])
            self.graphFile = graphFile
            self.vp['sort'] = self.new_vertex_property('int')
            self.vp['sort'].a = sort
        else:
            super(VascularGraph, self).__init__(directed=True)
    
    def _ReadFromFile(self, graphFile:str):
        def ifFloat(x):
            try:
                return float(x)
            except ValueError:
                return x
        
        G = gt.Graph()
        with open(graphFile, 'r') as f:
            line = f.readline()
            while not line.startswith('# Nodes'):
                line = f.readline()
            nNodes = int(line.split(' ')[-1])
            attributes = f.readline().strip('\n').split(',')[1:] # First item is node name
            nodes = []
            for n in range(nNodes):
                node = f.readline().strip('\n').split(',')
                nodes.append((node[0], [ifFloat(node[i+1]) if attributes[i]!='position'
                                        else np.array(list(map(float, node[i+1].strip('[').strip(']').split()))) 
                                        for i in range(len(attributes))]))
            nodeAttributes = [n for n in attributes] # Copy
            line = f.readline()
            while not line.startswith('# Edges'):
                line = f.readline()
            nEdges = int(line.split(' ')[-1])
            attributes = f.readline().strip('\n').split(',')[2:] # First two items are node names
            edges = []
            for n in range(nEdges):
                edge = f.readline().split(',')
                edges.append((edge[0], edge[1], [ifFloat(edge[i+2]) for i in range(len(attributes))]))
            edgeAttributes = [n for n in attributes] # Copy
            
        # Convert to integers before adding to graph
        nodesToInt = {n[0]:i for i,n in enumerate(nodes)}
        nodes = [[nodesToInt[n], [get_prop_type(value, key)[1] for key, value in zip(nodeAttributes, d)]] for n,d in nodes]    
        edges = [(nodesToInt[n1], nodesToInt[n2], *[get_prop_type(value, key)[1] for key, value in zip(edgeAttributes, d)]) for n1,n2,d in edges]
        G.add_edge_list(edges,
                    eprops=[get_prop_type(value, key)[::-2] for key, value in zip(edgeAttributes, edges[0][2:])],        
                    )
        del edges, nodesToInt
        
        vprops = [G.new_vertex_property(get_prop_type(value,key)[0]) for key, value in zip(nodeAttributes, nodes[0][1])]
        for i, vprop in enumerate(vprops):
            G.vp[nodeAttributes[i]] = vprop
            if vprop.value_type() not in ['python::object','string', 'vector<double>', 'vector<long double>']:
                vprop.a = [d[i] for _,d in nodes]
            elif vprop.value_type() in ['string', 'python::object']:
                for n,d in nodes:
                    vprop[n] = d[i]
            else:
                vprop.set_2d_array(np.array([d[i] for _,d in nodes]).T, pos=np.array([n for n,_ in nodes]))
        del nodes

        return G
    
    def SplitVesselsToMeshSize(self, h:float, vessels:list):
        print("Not implemented yet.")
        pass

class ControlVolumes(object):
    '''
    Creates the control volumes within the Grid and stores the connections and their types
    between the grid and the control volumes (endothelial and (optional) vascular voxels)
    and between the vascular nodes and the control volumes.
    '''
    def __init__(self, grid:Grid, vessels:VascularGraph):
        # Compute distance to grid.origin along each axis
        d = vessels.new_vertex_property("bool")

        graph = vessels.copy()
        vorder = graph.new_vertex_property("int")
        vorder.a = np.arange(graph.num_vertices())[gt.topology.topological_sort(graph)]
        self._vessels = gt.Graph(graph, directed=True, vorder=vorder)
        # pos = self._vessels.vp['position']
        # #inFOV = (np.abs(np.array([pos[u] for u in vessels.iter_vertices()])-grid.origin)<=grid.l)
        # positions = vessels.vp['position'].get_2d_array([0,1,2]) - grid.origin.reshape((3,1))        
        points = self._vessels.vp['position'].get_2d_array([0,1,2]) 
        d.a = np.logical_and((points<=grid.origin[:,np.newaxis]+grid.l[:,np.newaxis]), (points>=grid.origin[:,np.newaxis])).all(0)
        self.grid = grid
        self.VAG = gt.GraphView(self._vessels, vfilt=d)

    def _RotationMatrix(self, n:np.ndarray):
        """Rotate a capsule onto the vessel axis n."""
        if (n==[0,0,1]).all():
            return np.eye(3)
        if (n==[0,0,-1]).all():
            R = np.array([[0,0,0],[0,0,0],[0,0,-1]])                                
        R = (n[2]-1)/(n[0]**2+n[1]**2) * np.array([[n[0]**2, n[0]*n[1], 0],
                                                    [n[0]*n[1], n[1]**2, 0],
                                                    [0, 0, n[0]**2+n[1]**2]])                      
        R+= np.array([[1, 0, -n[0]],
                        [0, 1, -n[1]],
                        [n[0], n[1], 1]])                              
        return R

    def _LabelRefCapsule(self, r:float, l:float, w:float,):
                        # nx:int=50, ny:int=50, nz:int=100):
        """
        Samples point around the refrence capsule 
        (a vertical cylinder of length l and radius r)
        and labels them according to the lumen radius and vessel wall thickness (w).
        """    
        # Creates a 3D grid, large enough for the capsule to fit in
        # pos    = np.meshgrid(np.linspace(-1.1*(r-w), 1.1*(r+w), endpoint=True, num=nx),
        #                     np.linspace(-1.1*(r-w), 1.1*(r+w), endpoint=True, num=ny),
        #                     np.linspace(0, l, endpoint=True, num=nz)) 
        width = r+w # Width of the cylinder
        pos = np.meshgrid(np.arange(-width, width, w),
                            np.arange(-width, width, w),
                            np.arange(-r, l+r, min(l, min(self.grid.h))))
        pos = np.vstack(list(map(lambda x:x.ravel(), pos))) # (3,nx*ny*nz) array of locations
        dists = np.linalg.norm(pos[:2], axis=0) # Distance from the cylinder axis
        labels = np.where(dists<r-w/2, 1, np.where(dists<r+w/2, 2, 0))
        # labels = np.where(dists<r+w, 2, 0)
        # labels[dists<r-w] = 1
        return pos, labels
    
    def _LabelAroundVessel(self, e:list, w:float):
        ''' 
        TODO: look for the closest node in the edge to assign the connectivity to. Should then return 4 sets (2 for each node). 
        ### Parameters:
        - e: list
            An edge.
        - w: float
            The endothelium thickness (in cm).
        '''

        p0, p1 = [self.VAG.vp['position'][n].a for n in e]
        r, l   = self.VAG.ep['radius'][e], self.VAG.ep['length'][e] 
        R = self._RotationMatrix((p1-p0)/l) # Rotation operator to the vessel axis

        pos, labels = self._LabelRefCapsule(r, l, w) # Return a grid with its labels on a vertical capsule
        m0 = pos[-1,:]<l/2 # The cells that should be attached to the proximal node's control volume
        pos = p0[:, np.newaxis] + R.T.dot(pos) # Transpose and translate the capsule to the real vessel's axis
        gridDomain = self.grid.PointToCell(pos) # Maps the points in the reference capsule to voxels in our grid
        
        mask = (gridDomain<self.grid.n[:,np.newaxis]).all(0) # Some points may be outside the grid after rotation, so we mask them
        mask = mask & (gridDomain>0).all(0)
        labels = labels[mask]
        m0 = m0[mask]
        ijk = self.grid.ToFlatIndex(*gridDomain)[mask] # Maps the ijk indices to flat array indices        
        # try:
        endothelium0 = ijk[ma.masked_where(m0, labels, 0)==2]
        endothelium1 = ijk[ma.masked_where(~m0, labels, 0)==2]
        vascular0    = ijk[ma.masked_where(m0, labels, 0)==1]
        vascular1    = ijk[ma.masked_where(~m0, labels, 0)==1]
        # except IndexError:
        #     print(ijk.shape, labels.shape, labels[np.logical_and(mask, m0)].shape, labels[np.logical_and(mask, ~m0)].shape)
        #     raise IndexError("")
        vascular0 = set(vascular0) # The vascular labels
        endothelium0 = set(endothelium0) # The endothelium labels
        vascular1 = set(vascular1) # The vascular labels
        endothelium1 = set(endothelium1) # The endothelium labels
        vascular0 = vascular0-endothelium0 # Endothelium label is preffered over vascular  
        vascular1 = vascular1-endothelium1 # Endothelium label is preffered over vascular  
        del pos, ijk, labels, gridDomain # Clean up          
        return endothelium0, vascular0, endothelium1, vascular1 
        
    def LabelMesh(self, w:float=1e-4)->None:
        '''
        TODO: see _LabelAroundVessel.
        Labels the mesh and saves the connectivity between the control volumes, the mesh and the vascular nodes. 
        ### Parameters
        - w: float
            The endothelium thickness (in cm).
        '''

        LabelVessel = self._LabelAroundVessel
        # endo = self.VAG.new_vertex_property("python::object", val=set())
        # vasc = self.VAG.new_vertex_property("python::object", val=set())

        G = self.VAG.copy()
        G.clear_edges()
        G.shrink_to_fit()
        Voxels = np.fromiter(G.add_vertex(self.grid.size), dtype=int)
        eType = G.new_edge_property("bool") # 0 for vascular, 1 for endothelial connections
        G.ep['eType'] = eType

        for e in tqdm(self.VAG.iter_edges(), 
                        total=self.VAG.num_edges(), 
                        desc="Labelling...", unit="edge"): # Loop through the vessels in the FOV.
            for u, voxelsSet, t in zip((e[0],e[0], e[1], e[1]), LabelVessel(e,w), (1,0,1,0)):
                G.add_edge_list(((u, v, t) for v in voxelsSet), eprops=[eType])
            # print(f"{G.num_edges()=}, {G.num_vertices()=}")
        del voxelsSet

        edgesToKeep = []
        Voxels = Voxels[G.get_total_degrees(Voxels)>0]
        for v in tqdm(Voxels):
            edges = G.get_all_edges(v, eprops=[eType])
            if (edges[:,-1]==1).any():
                edge = edges[edges[:,-1]==1][0]
            else:
                edge = edges[0]
            edgesToKeep.append(edge)
        # Make it an internal property
        self.VAG.vp['endothelium'] = self.VAG.new_vertex_property("python::object", val=[])
        self.VAG.vp['vascular']    = self.VAG.new_vertex_property("python::object", val=[])

        vasc = find_edge(G, eType, 0)
        endo = find_edge(G, eType, 1)
        vertex_index = G.vertex_index
        for cv, voxel in tqdm(vasc, desc="Adding vascular voxels to graph"):
            self.VAG.vp['vascular'][cv].append(vertex_index[voxel])
        for cv, voxel in tqdm(endo, desc="Adding endothelium voxels to graph"):
            self.VAG.vp['endothelium'][cv].append(vertex_index[voxel])

        # self.VAG.vp['endothelium'] = endo
        # self.VAG.vp['vascular']    = vasc
        return G

    def bar(self): # To profile the time for labelling.
        profile.runctx('self.LabelMesh()', globals(), locals())

    def LabelsToGrid(self):
        # Write the labels to the grid for plotting.
        try:
            # self.grid.labels = {i:1 for idx in tqdm(itertools.chain(self.VAG.vp['vascular']), total=self.VAG.num_vertices(), unit='vertices') for i in idx}
            self.grid.labels = {idx:1 for idx in itertools.chain.from_iterable(self.VAG.vp['vascular'])}
            self.grid.labels = {**self.grid.labels, 
                                **{idx:2 for idx in itertools.chain.from_iterable(self.VAG.vp['endothelium'])}}        
        except KeyError:
            self.grid.labels = {} # No labels.
        
    def ToVTK(self, VTKFileName:str="gridLabels.vtk"): 
        self.LabelsToGrid()
        with open(VTKFileName, 'w') as f:
            f.write("# vtk DataFile Version 5.1\n")
            f.write("A mesh for the computation of oxygen perfusion.\n")
            #f.write("BINARY\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write("DIMENSIONS {:d} {:d} {:d}\n".format(*(self.grid.n+1)))
            f.write("ORIGIN {:f} {:f} {:f}\n".format(*self.grid.origin))
            f.write("SPACING {:f} {:f} {:f}\n".format(*self.grid.h))
        
            # Writing the data
            f.write(f"CELL_DATA {self.grid.size}\n")
            f.write("SCALARS labels int 1\n")
            f.write("LOOKUP_TABLE default\n")

            a = np.array(
                [
                    self.grid.labels.get(i,0) for i in 
                        tqdm(
                            # Change the order to x change the fastest and z the slowest for paraview.
                                np.arange(self.grid.size).reshape(self.grid.shape).ravel('F'), 
                            unit='labels',
                            desc=f"Writing labels to {VTKFileName}",
                            total=self.grid.size)
                ],
                dtype=np.int8
                )
            a.tofile(f, sep='\n', format='%d')
            del a

            # f.write(('\n'.join(i.__str__() for i in 
            #                    tqdm(self.grid, desc=f"Writing labels to {VTKFileName}", 
            #                         total=self.grid.size, unit=' Labels'))))   
        return          

    def _PlotPlexus(self, plexus:int, output=None):
        _G = gt.GraphView(self.VAG, vfilt=self.VAG.vp['plexus'].fa==plexus)
        if _G.num_vertices()==0:
                print(f"No vertices in the " + ("SVP" if plexus==0 else
                                            "ICP" if plexus==1 else
                                            "DCP") + " layer. Not plotting.")
                return
        graph_draw(_G, pos=self.VAG.vp['position'], bg_color='white', output=output)
        del _G
    def PlotSVP(self, output=None):
        self._PlotPlexus(plexus=0, output=output)             
    def PlotICP(self, output=None):
        self._PlotPlexus(plexus=1, output=output)
    def PlotDCP(self, output=None):
        self._PlotPlexus(plexus=2, output=output)           

def AddIntermediateVertices(graph:VascularGraph, edge:list[int], num_intermediate:int):
    if num_intermediate<1:
        return
    source, target = edge
    originalEdgeProp = {key: graph.ep[key][edge] for key in graph.ep if key!='length'}
    originalNodeProp = {key: graph.vp[key][source] for key in graph.vp if key!='position'}
    p0, p1 = graph.vp['position'][source].a, graph.vp['position'][target].a
    l = np.linalg.norm(p0-p1)/(num_intermediate+1)
    n = (p1-p0)/np.linalg.norm(p0-p1)
    graph.remove_edge(edge)
    newNodes = graph.add_vertex(num_intermediate)
    for i, newNode in enumerate(newNodes if num_intermediate>1 else [newNodes]):   
        for key, value in originalNodeProp.items():
            graph.vp[key][newNode] = value  
        graph.vp['position'][newNode] = p0 + n*l*(i+1)     
        newEdge = graph.add_edge(source, newNode)
        for key, value in originalEdgeProp.items():
            graph.ep[key][newEdge] = value 
        graph.ep['length'][newEdge] = l
        source = newNode
    newEdge = graph.add_edge(newNode, target)
    for key, value in originalEdgeProp.items():
        graph.ep[key][newEdge] = value 
    graph.ep['length'][newEdge] = l

def SplitVessels_NoReordering(graph:VascularGraph, h:list[float]):
    edges = list(graph.iter_edges())
    positions = graph.vp['position']
    num_intermediates = abs(np.array([[positions[u].a-positions[v].a] for u,v in edges]))/np.asarray(h)
    num_intermediates = np.squeeze(num_intermediates, 1).max(-1).astype(int)    
    for edge, num_intermediate in tqdm(zip(edges, num_intermediates), desc='Refining vascular mesh', total=len(edges), unit='edges'):
        if num_intermediate<1:
            continue
        AddIntermediateVertices(graph, edge, num_intermediate+1)
    return

def SplitVessels(CV:ControlVolumes, h:list[float]):
    graph = CV._vessels.copy()
    vfilter, isInverted = CV.VAG.get_vertex_filter()
    if isInverted:
        vfilter.a = np.invert(vfilter)    
    graph.vp['vfilt'] = graph.new_vertex_property("bool", vfilter.a)

    edges = list(CV.VAG.iter_edges())
    positions = graph.vp['position']
    # Estimate number of splittings necessary to have l<h
    num_intermediates = abs(np.array([[positions[u].a-positions[v].a] for u,v in edges]))/np.asarray(h) 
    num_intermediates = np.squeeze(num_intermediates, 1).max(-1).astype(int)    
    for edge, num_intermediate in tqdm(zip(edges, num_intermediates), desc='Refining vascular mesh', total=len(edges), unit='edges'):
        if num_intermediate<1:
            continue
        AddIntermediateVertices(graph, edge, num_intermediate+1)

    # Reorder the vessels and replace the CV internal graph
    vfilter, isInverted = CV.VAG.get_vertex_filter()
    if isInverted:
        vfilter.a = np.invert(vfilter)
    vorder = graph.new_vertex_property("int")
    for i,v in enumerate(gt.topology.topological_sort(graph)):
        vorder[v] = i
    CV._vessels = gt.Graph(graph, directed=True, vorder=vorder)
    CV._vessels.vp['sort'] = CV._vessels.new_vertex_property("int", vorder.a)
    CV.VAG = gt.GraphView(CV._vessels, vfilt=CV._vessels.vp['vfilt'])    
    del CV._vessels.vp['vfilt'], CV.VAG.vp['vfilt']
    return 

class gmres_counter(object):
    def __init__(self, disp=10):
        self._disp = disp
        self.niter = 0
        self.rks = []

    def __call__(self, rk=None):
        self.niter += 1
        if self.niter%self._disp==0:
            if isinstance(rk, np.ndarray):
                rk = rk.get()
            self.rks.append(rk)
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

    def __enter__(self):
        self.t1 = time.time()
        return self
    def __exit__(self, type, value, traceback):
        t = (time.time() - self.t1)/60
        plt.plot(self._disp * (1+np.arange(len(self.rks))), self.rks)
        plt.xlabel("Number of iterations")
        plt.ylabel("Residual (relative?)")
        plt.title(f"Solving time: {t}min")
        plt.show()


def PositivePart(M):
    # Positive part of matrix
    return .5*(M + abs(M))