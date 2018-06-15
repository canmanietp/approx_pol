#----Policy Evaluation (LSQ) see: https://www.cs.rutgers.edu/~mlittman/courses/robots03/lspi.pdf
# input: samples, policy, new_policy
# output: w (weights) (of policy)
# Definition: Use samples to approximate Q-values of old policy and new_policy
# ???approximate phi, R, P (R is reward function, P is transition function)???
#
# Loop through all samples
#
# compute phi (basis functions) for current (state,action)
# get next action for next state via policy(next_state)
# compute next_phi for (next_state,next_action)
#
# update A and b
# A = A + phi*(phi-next_phi*new_policy.discount) 
# b = b + phi*reward (of current sample)
#
# end loop
#
# Solve linear system Aw = b to find w (may need SVD to invert A)

import policy
import numpy as np

def LSQ(samples, pi, new_pi):
	k = 1 #should initialize to k x k = size of basis
	A = np.zeros(k,k)
	b = np.zeros(k,1)
	
	for state,action,reward,next_state in samples:
		phi = basis(state,action)
		next_action = Policy.act(pi,state)
		next_phi = basis(next_state,next_action)
		
		A = A + phi*(phi-next_phi*new_pi.discount)
		b = b + phi*reward
		
	w = np.linalg.solve(a,b)

#-----Compute basis
# input: sampled (state, action) pair
# output: phi (column vector of length equal to total number of basis functions, each entry is corresponding basis function evaluated at the (s,a) pair)
# Definition: Computes the basis functions for given (state,action) pair
#
# How? tbd

def basis(state,action):
	basis = []
	return basis