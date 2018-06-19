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
import numpy as np
import copy

def run_LSPI():
	epsilon = 0.01 #convergence criterion
	iteration = 0
	max_iterations = 100
	distance = float('Inf')
	first_time = True
	pi = []
	
	sample_n = 10 # num of episodes to simulate
	
	while iteration < max_iterations and distance > epsilon:
		samples = sampler.sample(sample_n) # get samples from simulation (need to sample from these according to prob. dist.?)
	
		phi = basis.calculate_bases(samples)
		k = 5 ##FIX THIS HARD CODE phi.ndim #dimensions of basis phi (needs to be altered for matrix)
		
		if first_time:
			pi = policy.zero_policy(k)  # initial policy with initial weights zero
		
		
		old_pi = copy.copy(pi) #old weights of pi

		LSQ.calc_weights(samples,k,phi,pi) ## Least squares approximation of Q function, calculate and set new weights for pi based on samples
		
		l1 = len(old_pi.weights)
		l2 = len(pi.weights)
		
		print(old_pi.weights)
		print(pi.weights)
		
		#compare weights of old_pi and new weights of pi (based on new samples)
		if l1==l2:
			diff = old_pi.weights - pi.weights
			Linf_norm = np.linalg.norm(diff, np.inf)
			L2_norm = np.linalg.norm(diff) 
		else:
			Linf_norm = np.absolute(np.linalg.norm(old_pi.weights,inf) - np.linalg.norm(pi.weights,inf))
			L2_norm = np.absolute(np.linalg.norm(old_pi.weights) - np.linalg.norm(pi.weights))
				
		distance = L2_norm #print Linf_norm
		
		print("DISTANCE IS")
		print(distance)
		
		iteration+=1
		first_time = False
			
	return pi