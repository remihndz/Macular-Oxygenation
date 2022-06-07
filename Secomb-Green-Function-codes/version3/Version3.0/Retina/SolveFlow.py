from NetworkClass import Network
import numpy as np
import os

'''
Node_co-ordinates.txt's columns are:
"Node ID" "x coordinate" "y coordinate" "z coordinate", in meters,
contain the information on the nodes location.
Connection_joints.txt's columns are:
"Node_in" "Node_out" "Length(m)" "Radius(m)"
contains the information on the vessels.
'''

# Nodes_coordinate = np.loadtxt('Node_co-ordinates.txt')
# Connectivity     = np.loadtxt('Connection_joints.txt', usecols=(0,1,2,3))

base_dir    = './Student_Network/'
# base_dir    = './WKEB_Network/'
results_dir = base_dir+'Results/'

try:
    os.mkdir(results_dir)
except OSError as error:
    pass

coordinate_file   = 'node_coordinates.csv'
connectivity_file = 'connection.csv'

Nodes        = np.loadtxt(base_dir + coordinate_file)
Connectivity = np.loadtxt(base_dir + connectivity_file, usecols=(0,1,2))
Connectivity = np.hstack( (Connectivity, np.zeros((Connectivity.shape[0], 1))) )

# Create the capillary network with the input data and find nodal pressures
CapNet = Network(Nodes, Connectivity)
CapNet.Solve_for_Pressure()

# Post-Processing
CapNet.SavePressure(results_dir+'pressure.dat')
CapNet.ComputeParameters(results_dir+'hemodynamic_parameters.dat')
CapNet.SaveFlow(results_dir+'flow_matrix.dat')

print('Results saved in', results_dir)