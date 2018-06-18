#-----Compute basis
# input: sampled (state, action) pair
# output: phi (column vector of length equal to total number of basis functions, each entry is corresponding basis function evaluated at the (s,a) pair)
# Definition: Computes the basis functions for given (state,action) pair(s)
#
# How? tbd

import numpy

def calculate_basis(state,action):
	phi = []
	phi = np.array(phi)
	return phi

def calculate_basis(samples):
	phi = []
	for state,action in samples:
		phi.append(calculate_basis(state,action)) 
	
	phi = np.array(phi)
	return phi