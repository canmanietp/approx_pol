#------Policy Iteration (Least-Squares)
# input: samples, basis
# output: "good" policy
# Definition: Finds a good policy 
#
# while (iteration < max_iterations) and (distance > epsilon) 
# evaluate policy (produces weights for the policy)
#	 ---use evaluation algorithm such as LSQ above---
# compute distance between current and all previous policies
#    ---compute the L2 and L-inf norm of the difference between weights of current policy and policy being checked
# store the current policy in list of policies

import policy
import LSQ
import sampler
import basis
from numpy as np
import math

def LSPI(pi,policies):
	epsilon = 0.01 #convergence criterion
	index = 0
	max_iterations = 100
	distance = math.inf
	
	while index < max_iterations and distance > epsilon:
		samples = sample(10) #num of episodes to simulate and get samples from (need to sample from these according to prob. dist.?)
	
		phi = calculate_basis(samples)
		k,x = basis.ndim
	
			for check_pi in policies:
				LSQ(samples,k,phi,pi) ## Least squares approximation of Q function
			
				l1 = len(pi.weights)
				l2 = len(check_pi.weights) #??
			
				if l1==l2:
					diff = pi.weights - check_pi.weights
					Linf_norm = np.linalg.norm(diff, np.inf)
					L2_norm = np.linalg.norm(diff) 
				else:
					Linf_norm = 
					L2_norm = 
				
				distance = L2_norm #print Linf_norm
			
		