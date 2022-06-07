'''
Tools for the computation of 
Horton-Strahler stream order.
'''

import matplotlib.pyplot as plt
import numpy as np
import copy

def ReadTree(ccoFile):
    print("Reading", ccoFile)

    with open(ccoFile, 'r') as f:
        # Unused '*Tree' information
        row = f.readline()
        row = f.readline()
        row = f.readline()

        # Vessels data
        f.readline().strip()
        nVessels = int(f.readline())

        treeData = []
        for i in range(nVessels):
            row = (f.readline()).split() # Split all columns in a list
            Id, xProx, xDist, r, q, stage = int(row[0]), row[1:4], row[4:7], float(row[12]), float(row[13]), int(row[-1])
            l = sum([(float(a)-float(b))**2 for a,b in zip(xProx, xDist)])**.5
            #treeData.append([Id, r*10, l*10, q, stage]) # Convert to mm
            treeData.append([Id, r*1e4, q, l*1e4, 
            		      float(xProx[0])*1e4,            		      float(xProx[1])*1e4, float(xProx[2])*1e4, 
            		      float(xDist[0])*1e4,        		      float(xDist[1])*1e4, float(xDist[2])*1e4])        
        row = f.readline()
        f.readline().strip()

        treeConnectivity = []
        for i in range(nVessels):
            row = (f.readline()).split()
            if (len(row)==2):
                Id, parentId = int(row[0]), int(row[1])
                treeConnectivity.append([Id, parentId, []])
            else:
                Id, parentId, children = int(row[0]), int(row[1]), [int(i) for i in row[2:]]
                treeConnectivity.append([Id, parentId, children])         

        print("\tNumber of vessels in the tree:", nVessels, "\n")
            
    return treeData, treeConnectivity


def BifurcationOrder(treeData, treeConnectivity, orderMax=1e6):
    
    treeConnectivity.sort()
    treeData.sort()
    orders = np.zeros((len(treeConnectivity),))+1

    def updateTree(vessel, currentOrder=0):
        # Go up the tree and update all orders
        parentId = int(vessel[1])

        orders[vessel[0]] = max(currentOrder, orders[vessel[0]])
        if parentId:
            parent = treeConnectivity[parentId]                
            if len(parent[-1])==2:
                updateTree(parent, min(currentOrder+1,orderMax))
            else:
                updateTree(parent, min(currentOrder, orderMax))
        else:
            orders[parentId] = min(currentOrder, orderMax)
            return
    
    for v in treeConnectivity:
        if len(v[-1])==0:       # if terminal segment
            updateTree(v, currentOrder=0)

    # Add stream order to vessel data
    treeDataNoRoot = []
    count = 0
    for vessel in treeData:
        Id = int(vessel[0])
        order = orders[Id]
        treeData[Id].append(order)

        if treeData[Id][-2]>=-1:
            Id, radius, length, flow, stage, order = treeData[Id]
            treeDataNoRoot.append([count, radius, length, flow, stage, order])
            count+=1

    return treeDataNoRoot


def StrahlerOrder(treeData, treeConnectivity):

    treeConnectivity.sort()
    currentOrder = 0
    vessel = treeConnectivity[0]
    vesselDict = {vessel[0]: [vessel[1], vessel[2], currentOrder]}
    for vessel in treeConnectivity[1:]:
        if len(vessel)>2:
            IdVessel, IdParent, ChildrenList = vessel
        else:
            IdVessel, IdParent = vessel
            ChildrenList = []
        vesselDict[IdVessel] = [IdParent, ChildrenList, currentOrder]
    prunedTree = copy.deepcopy(vesselDict)
    
    def CurrentLeaves(prunedTree):
        leavesList = []
        for Id, vessel in prunedTree.items():
            if vessel[-1] == 0 and len(vessel[-2])==0:
                leavesList.append(Id)
        return leavesList

    def UpdateAndPrune(prunedTree, leaves):        
        # Assign an order to the current leaves
        copyTree = copy.deepcopy(prunedTree)
        increment= False

        for leaf in leaves:
            Children = treeConnectivity[leaf][-1]
            if len(Children)==2 and copyTree[Children[0]][-1]==currentOrder and copyTree[Children[1]][-1]==currentOrder:
                copyTree[leaf][-1] = currentOrder+1
                increment = True
                
            else:
                childrenOrders = [0]
                for child in Children:
                    childrenOrders.append(copyTree[child][-1])
                ''' 
                Two different rules, the max rule is the one from Wikipedia and gives fewer orders.
                Both give the right results for the test tree.
                '''
                copyTree[leaf][-1] = currentOrder # max(1,max(childrenOrders)) 

        # Prune
        for leaf in leaves:
            parentId = copyTree[leaf][0]
            if parentId>=0:
                copyTree[parentId][1].remove(leaf)
        return copyTree, increment

    currentOrder = 1
    computeForTheRoot = iter([True, False]) # Just used to run one more time once the root is reached
    while prunedTree[0][-1]==0 or next(computeForTheRoot):
        leafList = CurrentLeaves(prunedTree)
        prunedTree, increment = UpdateAndPrune(prunedTree, leafList)
        if increment: currentOrder+=1
        
        if 0 in leafList:
            break

    # Add stream order to vessel data
    treeDataNoRoot = []
    count = 0
    for vessel in treeData:
        Id = int(vessel[0])
        order = prunedTree[Id][-1]
        treeData[Id].append(order)

        if treeData[Id][-2]>=-1:
            Id, radius, length, flow, stage, order = treeData[Id]
            treeDataNoRoot.append([count, radius, length, flow, stage, order])
            count+=1
    return treeDataNoRoot

def TreeStatistics(orderedTree, treeData):
    return #stats

def PlotTreeStatistics(orderedTree, outputImageFile=None):

    maxOrder = np.array(orderedTree)[:,-1].astype(int).max()
    data = np.array(orderedTree)

    dataRadius = np.zeros((maxOrder, len(orderedTree)))
    dataLength = np.zeros((maxOrder, len(orderedTree)))
    dataVolume   = np.zeros((maxOrder, len(orderedTree)))
    dataAspectRatio   = np.zeros((maxOrder, len(orderedTree)))
    orderDistribution = np.zeros((maxOrder,))
    
    for Id, radius, length, flow, stage, order in orderedTree:
        r,l = radius*1e5, length*1e5 # Convert to microns
        i,j = int(order)-1, int(Id)
        dataRadius[i, j] = radius
        dataLength[i, j] = length
        dataVolume[i, j] = radius*radius*length*np.pi
        dataAspectRatio[i, j] = length/radius
        orderDistribution[i] += 1
        
    meanRadius = np.mean(dataRadius, axis = 1, where=dataRadius>0)
    stdRadius  = np.std(dataRadius, axis = 1, where=dataRadius>0)
    meanLength = np.mean(dataLength, axis = 1, where=dataRadius>0)
    stdLength  = np.std(dataLength, axis = 1, where=dataRadius>0)
    meanVolume = np.mean(dataVolume, axis = 1, where=dataRadius>0)
    stdVolume  = np.std(dataVolume, axis = 1, where=dataRadius>0)
    meanAspectRatio = np.mean(dataAspectRatio, axis = 1, where=dataRadius>0)
    stdAspectRatio  = np.std(dataAspectRatio, axis = 1, where=dataRadius>0)
    
    x = np.flip(np.arange(maxOrder) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
    fig.set_figheight(7.2)
    fig.set_figwidth(8.8)
    scale='linear'
    ax1.errorbar(x, 2*meanRadius, stdRadius*2, marker='s', fillstyle='full')
    ax1.set_yscale('log')
    ax1.set_title('Diameter vs. Strahler order')
    ax2.errorbar(x, meanLength, stdLength, marker='s')
    ax2.set_yscale(scale)
    ax2.set_title('Length vs. Strahler order')
    ax3.errorbar(x, meanVolume, stdVolume, marker='s')
    ax3.set_yscale(scale)
    ax3.set_title('Volume vs. Strahler order')
    ax4.errorbar(x, meanAspectRatio, stdAspectRatio, marker='s')
    ax4.set_yscale(scale)
    ax4.set_title('Aspect ratio vs. Strahler order')

    fig2, axes = plt.subplots(2,2, sharex='col')
    fig2.set_figheight(7.2)
    fig2.set_figwidth(8.8)
    scale = 'linear'
    axes[0,0].boxplot(dataRadius.transpose())
    axes[0,0].set_yscale(scale)
    axes[0,0].set_title('Radius vs. Strahler order')
    axes[0,1].boxplot(dataLength.transpose())
    axes[0,1].set_yscale(scale)
    axes[0,1].set_title('Length vs. Strahler order')
    axes[1,0].boxplot(dataVolume.transpose())
    axes[1,0].set_yscale(scale)
    axes[1,0].set_title('Volume vs. Strahler order')
    axes[1,1].boxplot(dataAspectRatio.transpose())
    axes[1,1].set_yscale(scale)
    axes[1,1].set_title('Aspect ratio vs. Strahler order')


    if outputImageFile:
        print('\nWriting outputs (image+data) in', outputImageFile[:-4] + '.png/.dat')
        data = np.vstack((x, meanRadius, stdRadius, meanLength, stdLength, meanVolume, stdVolume, meanAspectRatio, stdAspectRatio, orderDistribution))
        np.savetxt(outputImageFile[:-4] + '.dat', data)
        plt.savefig(outputImageFile)
    plt.show()
    
    return


def BifurcationDiameterRatio(treeData, treeConnectivity, plot=False):
    '''
    Return the ratio of the smaller daughter vessel to the larger (dsdl)
    against the ratio of the larger daughter to the parent vessel (dldp)
    '''
    
    copyData = copy.deepcopy(treeData)
    copyData.sort()
    data = []
    for Id, parentId, children in treeConnectivity:
        if parentId>0 and len(children)==2: # parentId=-1 if its root
            
            dp = copyData[parentId][1]
            dchildren = [copyData[children[0]][1], copyData[children[1]][1]]
            ds = min(dchildren)
            dl = max(dchildren)

            # dsdl.append(ds/dl)
            # dldp.append(dl/dp)
            data.append([ds/dl, dl/dp, parentId])

    data.sort()
    data = np.array(data)
    dsdl, dldp, Ind = data[:,0], data[:,1], data[:,2]

    if plot:
        aggregatedData = []
        # x = np.linspace(data[:,0].min(),1, 10)
        x = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for i,xi in enumerate(x[1:]):
            xim = x[i]
            mask = (dsdl>xim) & (dsdl<xi)
            aggregatedData.append(dldp[mask])

        plt.boxplot(aggregatedData, positions=x[1:])        
        # plt.scatter(dsdl, dldp, label='This work')

        m = 2.85
        x = np.linspace(0.2,1,500)
        y = np.linspace(0.2,1,500)
        X,Y = np.meshgrid(x,y)

        f,g = Y**(-m), 1+X**m
        z = f-g
        cs1 = plt.contour(X,Y, z, [0], linestyles='--', linewidths=1.)
        m = 3
        f,g = Y**(-m), 1+X**m
        cs2 = plt.contour(X,Y,(f-g), [0], linestyles='-.', linewidths=1.)

        cs1.collections[0].set_label('m=2.85')
        cs2.collections[0].set_label('m=3')
                
        plt.xlim(0,1.2)
        # plt.ylim(0,1.2)
        # plt.yscale('log')
    
        plt.legend()
        plt.show()

    return [a for a in dsdl], [a for a in dldp]
  
def VBC(treeData, treeConnectivity, plot=False):

    # Copy the radii, sorted by vessel ID
    radii = np.zeros((len(treeData),))
    for v in treeData:
        radii[v[0]] = v[1]
    
    VBC = []
    for v in treeConnectivity:
        if len(v[-1])==2:
            p, d1, d2 = radii[v[0]], radii[v[-1][0]], radii[v[-1][1]]
            VBC.append( (d1**2 + d2**2)/(p**2) )

    VBC = np.array(VBC)
    
    if plot:
        split = 10
        x = np.linspace(VBC.min(), VBC.max(), split+1)
        y = np.zeros((split+1,))
        sortedVBC = np.sort(VBC)
        
        j = 0
        for i in range(split):
            count = 0
            xi = x[i]
            while xi >= sortedVBC[j]:
                count+=1
                j+=1
            y[i] = count
        y[-1] = VBC.size-j

        plt.bar(x[:-1],y[:-1], width=(x[1]-x[0])/2.0)
        plt.xticks(x[:-1], rotation=20)
        plt.xlabel('Vessel branching coefficient')
        plt.ylabel('Number of vessels')
        plt.show()
        
    return VBC
                
