#----Policy Evaluation (LSQ) see: https://www.cs.rutgers.edu/~mlittman/courses/robots03/lspi.pdf
# input: samples, policy, new_policy
# output: w (weights) (of policy)
# Definition: Use samples to approximate Q-values of old policy and new_policy
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
import basis
import numpy as np

def calc_weights(samples, k, phi,pi):
	#k x k = size of basis
	A = np.zeros((k,k))
	b = np.zeros(k)
	
	for state,action,reward,next_state in samples:
		phi = basis.calculate_basis(state,action)
		next_action = policy.Policy.act(pi,state)
		next_phi = basis.calculate_basis(next_state,next_action)
		
		A += np.outer(phi,phi-next_phi*pi.discount)
		b += phi*reward
	
	w = np.linalg.solve(A,b)
	pi.weights = w

