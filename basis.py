#-----Compute basis
# input: sampled (state, action) pair
# output: phi (column vector of length equal to total number of basis functions, each entry is corresponding basis function evaluated at the (s,a) pair)
# Definition: Computes the basis functions for given (state,action) pair(s)
#
# How? tbd

import numpy as np

def calculate_basis(state,action):
	#phi = np.insert(state,0,action)
	#state = [0.2 0.3 0.1 0.4]
	
	if action==1:
		phi = state*1
	if action==0:
		phi = state*-1
		
	phi = np.array(phi)
	
	return phi

def calculate_bases(samples):
	phi = []
	
	for state,action,done,next_state in samples:
		phi.append(calculate_basis(state,action)) 
	
	phi = np.array(phi)
	return phi