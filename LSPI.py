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

def LSPI(pi): #pi = initial policy with initial weights (can be zero)
	epsilon = 0.01 #convergence criterion
	iteration = 0
	max_iterations = 100
	distance = math.inf
	
	while iteration < max_iterations and distance > epsilon:
		samples = sample(10) #num of episodes to simulate and get samples from (need to sample from these according to prob. dist.?)
	
		phi = calculate_basis(samples)
		k,x = basis.ndim
		
		old_pi = pi #old weights of pi
	
		LSQ(samples,k,phi,pi) ## Least squares approximation of Q function, calculate and set new weights for pi based on samples
			
		l1 = len(old_pi.weights)
		l2 = len(pi.weights) 
		
		#compare weights of old_pi and new weights of pi (based on new samples)
		if l1==l2:
			diff = old_pi.weights - pi.weights
			Linf_norm = np.linalg.norm(diff, np.inf)
			L2_norm = np.linalg.norm(diff) 
		else:
			Linf_norm = np.absolute(np.linalg.norm(old_pi.weights,inf) - np.linalg.norm(pi.weights,inf))
			L2_norm = np.absolute(np.linalg.norm(old_pi.weights) - np.linalg.norm(pi.weights))
				
		distance = L2_norm #print Linf_norm
		iteration+=1
			
	return pi