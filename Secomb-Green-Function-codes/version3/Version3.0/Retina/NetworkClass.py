import numpy as np
from scipy.linalg import solve


class Network:

    def __init__(self, Nodes, Vessels):
        ''' 
        Nodes is a nx4 numpy array with the first column being a node ID and the three
        following being the coordinates of the node (i.e. x,y and z)
        If Nodes.shape==(n,3) (i.e. the file only contains the coordinates without IDs),
        then the nodes ID is the line's number.
        Vessels is a mx4 numpy array with columns:
        Node_in Node_out Radius Length
        which describe the starting and ending node constituing a vessel 
        of given Length and Radius.
        If Length is zero, it will be computed later
        '''            

        self._Nodes = Nodes
        self.NbNodes = Nodes.shape[0]
        self._VesselsList = Vessels

        # If the node ID is not given in the input file, an ID is assigned to each node
        if Nodes.shape[1] == 3:
            self._Nodes = np.hstack((np.arange(self.NbNodes).reshape(-1,1), Nodes))

        self._IsBuilt = False                  
        self._IsBuilt = self._Build(Vessels)
        self.P = None


    def _Build(self, Vessels):
        '''
        Build:
        - A adjency matrix:
        _M_Adjency[i,j]   = 1 if there is a vessel connecting nodes i and j
                          = 0 otherwise
        - A matrix of lengths:
        _M_Lengths[i,j]   = L_ij the length of the vessel linking nodes i and j
                          = 0    if _M_Adjency[i,j] = 0
        - A matrix of radii:
        _M_Radii[i,j]     = r_ij the radius of the vessel linking nodes i and j
                          = 0    if _M_Adjency[i,j] = 0
        - A matrix of viscosities:
        _M_Viscosity[i,j] = nu_ij the dynamic viscosity of the blood in the vessel 
        - A matrix containing the 'resistance coefficient' of each vessel: _M_Gmat
        - A list of the inlet and outlet nodes for which pressure is specified: _bndry_nodes        
        '''       
        
        self.NbVessels = Vessels.shape[0]
        
        self._M_Adjency    = np.zeros((self.NbNodes, self.NbNodes))
        self._M_Lengths    = np.zeros((self.NbNodes, self.NbNodes))
        self._M_Radii      = np.zeros((self.NbNodes, self.NbNodes))
        self._M_Viscosity  = np.zeros((self.NbNodes, self.NbNodes))
        self._M_Gmat       = np.zeros((self.NbNodes, self.NbNodes))
        self._bndry_nodes  = []
        
        for k in range(self.NbVessels):

            i, j, r, L = Vessels[k, :]
            i,j = int(i), int(j)

            if not L>0:         # In case length is not given for the vessel, compute it
                x1, y1, z1 = self._Nodes[i, 1:]
                x2, y2, z2 = self._Nodes[j, 1:]
                L = np.sqrt( (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2 )

            mu = self._ViscosityModel(r)
            g  = np.pi*r**4/(8*mu*L)
            
            self._M_Adjency[i,j]   = 1
            self._M_Adjency[j,i]   = 1
            
            self._M_Lengths[i,j]   = L
            self._M_Lengths[j,i]   = L
            
            self._M_Radii[i,j]     = r
            self._M_Radii[j,i]     = r
            
            self._M_Viscosity[i,j] = mu
            self._M_Viscosity[j,i] = mu

            self._M_Gmat[i,j]      = g
            self._M_Gmat[j,i]      = g

        for i in range(self.NbNodes):
            if np.sum(self._M_Adjency[i,:])==2:
                self._bndry_nodes.append(i)

        return True


    def _ViscosityModel(self, radius, Hct=0.45):
        ''' 
        Compute the dynamic viscosity as a function of the radius and hematocrit.
        The model for the viscosity is taken from:
        Haynes RH., "Physical basis of the dependence of blood viscosity on tube radius.",
          Am J Physiol., 1960.
        '''
        delta = 4.29            
        return 1.09*np.exp(0.024*Hct)/( (1.0 + delta/radius)**2 )
        
        

    def _LinearSystem(self, P_in, P_out):
        ''' 
        Builds the linear system resulting from the Poiseuille law for the given
        inlet/outlet pressure in P_in/P_out imposed at the boundary nodes.
        The unknowns are
          - the nodal pressures at each node (including known inlet and outlet nodes),
                p_j with j=0,...,self.NbNodes-1 
        '''

        A = np.zeros((self.NbNodes, self.NbNodes))
        b = np.zeros(self.NbNodes)

        # Build the system's matrix without boundary conditions
        for i in range(self.NbNodes):
            for j in np.flatnonzero(self._M_Adjency[i,:]):
                A[i,j] = (-1.0)*self._M_Gmat[i,j]
            A[i,i] = np.sum(self._M_Gmat[i,:])

        # Add the boundary conditions
        self.found_inlet_node = False
        for i in self._bndry_nodes:
            # Find the inlet node for the network used by Yidan Xue (master student, Apr. 2021)
            x,y,z = self._Nodes[i,1:]
            if i==self._bndry_nodes[0]:

                self.found_inlet_node = True                
                self.InletNode = i
                # The equation for pressure at the inlet node P_0 is P_0 = P_in
                A[i,:] = 0.0*A[i,:]
                A[i,i] = 1.0
                b[i]   = P_in

                # The contribution of the inlet node to the pressure at its
                # neigbhors node is added to the right-hand-side vector
                j = np.flatnonzero(self._M_Adjency[i,:])[0]
                A[j,i] = 0.0
                b[j]  -= P_in*self._M_Gmat[i,j]                

            # Repeat for the outlet nodes
            else:
                # The equation at an outlet node P_i is P_i = P_out
                A[i,:] = 0.0*A[i,:]
                A[i,i] = 1.0
                b[i]   = P_out

                # The contribution of the outlet node in the equation at the
                # neighboring node is added to the rhs
                j      = np.flatnonzero(self._M_Adjency[i,:])[0]
                A[j,i] = 0.0
                b[j]  += P_out*self._M_Gmat[i,j]

        self.A, self.b = A,b

        return A, b           
            
                
    def Solve_for_Pressure(self, Inlet_Pressure = 40, Outlet_Pressure = 0):
        '''
        Assemble, store, then solve the linear system.
        Inlet_Pressure and Outlet_Pressure are the specified pressures at 
        boundary nodes.
        '''
        
        self._LinearSystem(Inlet_Pressure, Outlet_Pressure)

        self.P = solve(self.A, self.b)
        
        return self.P
            

    def SavePressure(self, fileName):

        File = open(fileName, 'w')
        header  = "# This file contains nodal pressure information\n"
        header += "#    {:15s} {:15s} {:15s} {:15s}\n".format('X', 'Y', 'Z', 'Pressure (mmHg)') 
        
        File.write(header)
        for i in range(self.NbNodes):
            x,y,z = self._Nodes[i,1:]
            p     = self.P[i]
            File.write('{:15f} {:15f} {:15f} {:15f}\n'.format(x,y,z,p))
        
        File.close()

    def SaveFlow(self, fileName):

        Flow_matrix = np.zeros(self._M_Radii.shape)
        for i in range(self.NbNodes):
            for j in np.flatnonzero(self._M_Adjency[i,:]):
                Flow_matrix[i,j] = 10000*(self.P[i]-self.P[j])*(self._M_Radii[i,j]**4)*np.pi/(self._M_Viscosity[i,j]*8.0*self._M_Lengths[i,j])
                #print(f'Flow for vessel between {i} and {j}: {self.P[i]} and {self.P[j]}')
        np.savetxt(fileName, Flow_matrix)
        
        
        
        
    def ComputeParameters(self, fileName=None):
        '''
        Compute different parameters of the vascular network using the computed pressure:
        - Blood flow
        - Pressure drop in a vessel
        - Wall shear stress
        - Shear rate at wall surface
        
        Save the data in fileName if provided.
        '''

        # Flow
        self.Q      = np.zeros((self.NbVessels,1))
        # Pressure drop
        self.DeltaP = np.zeros((self.NbVessels,1))
        # Shear rate at wall surface
        self.gamma  = np.zeros((self.NbVessels,1))
        # Wall shear stress
        self.tau    = np.zeros((self.NbVessels,1))
        # Radius
        self.radius = np.zeros((self.NbVessels,1))
        # Length
        self.length = np.zeros((self.NbVessels,1))
        # Viscosity
        self.mu     = np.zeros((self.NbVessels,1))
        
        for k in range(self.NbVessels):

            node_in, node_out, r = self._VesselsList[k, 0:3]
            node_in, node_out = int(node_in), int(node_out)
            L = self._M_Lengths[node_in, node_out]
            DeltaP = self.P[node_in] - self.P[node_out] 

            self.radius[k] = r
            self.length[k] = L
            self.mu[k]     = self._M_Viscosity[node_in, node_out]
            self.DeltaP[k] = DeltaP
            self.Q[k]      = (np.pi*r**4/(8.0*L))*DeltaP
            self.gamma[k]  = 4.0*self.Q[k]/(np.pi*r**3)
            self.tau[k]    = self._M_Viscosity[node_in, node_out]*self.gamma[k]
            
        

        if fileName is None:
            return

        else:
            File = open(fileName, 'w')
            header  = "# This file contains hemodynamic parameters\n"
            header += "#  {:15s} {:15s} {:15s} {:15s} {:15s} {:15s} {:15s}\n".format('Radius', 'Length', 'Viscosity', 'Pressure drop', 'Flow', 'Shear rate', 'Wall shear stress') 
            
            File.write(header)
            for k in range(self.NbVessels):
                r, L, mu, DeltaP, Q, gamma, tau = self.radius[k][0], self.length[k][0], self.mu[k][0], self.DeltaP[k][0], self.Q[k][0], self.gamma[k][0], self.tau[k][0]
                File.write('{:15f} {:15f} {:15f} {:15f} {:15f} {:15f} {:15f}\n'.format(r, L, mu, DeltaP, Q, gamma, tau))
                
            File.close()
