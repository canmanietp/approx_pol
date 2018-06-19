#------class Policy
# Policy is implicitly represented through a set of weights w

# Holds the discount factor and the weights associated with the policy, and the list of discrete action possibilities
#
# Given a state, returns an action.

# Calculate actions by pi(state) = max_{a in A} Q(state,a) = max_{a in A} phi(state,a)*w

import numpy as np
import basis
import sampler

class Policy:
	
	#BAD PRACTICE
	discount = 0.98
	actions = [0,1]  #hard-coded for CartPole-v2
	
	def __init__(self,weights):
		self.weights = weights #should be initialized to zeros 
	
	def act(self, state):
		
		max_act = -float('Inf')
		max_func = -float('Inf')
		
		for a in self.actions:
			val_func = basis.calculate_basis(state,a)*self.weights
			
			norm_val = np.linalg.norm(val_func)
			norm_max = np.linalg.norm(max_func)
			
			if norm_val > norm_max: ##not sure it makes sense to compare norms
				max_func = val_func
				max_act = a
		
		if max_act==-float('Inf'):
			max_act = sampler.random_action()
				
		return max_act
		

def zero_policy(size):
	pi = Policy(np.zeros((size,1)))
	return pi