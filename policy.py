#------class Policy
# Policy is implicitly represented through a set of weights w

# Holds the discount factor and the weights associated with the policy, and the list of discrete action possibilities
#
# Given a state, returns an action.

# Calculate actions by pi(state) = max_{a in A} Q(state,a) = max_{a in A} phi(state,a)*w

import numpy as np
import basis
import sampler
import sys
import random

class Policy:
	
	#BAD PRACTICE
	discount = 0.9
	#actions = [0,1]  #hard-coded for CartPole-v2
	
	def __init__(self,weights,tried_actions):
		self.weights = weights #should be initialized to zeros 
		self.tried_actions = tried_actions
	
	def act(self, state):		
		max_act = -float('Inf')
		max_func = -float('Inf')
		
		if (random.randint(1,100)<2):
			self.add_tried_action(sampler.random_action())
		
		if len(self.tried_actions) > 500:
			self.tried_actions = self.tried_actions[-500:]		
			
		for a in self.tried_actions:
			
			phi = basis.calculate_basis(state,a)
			val_func = np.dot(np.transpose(phi),self.weights)
			
			if val_func > max_func:
				max_func = val_func
				max_act = a
		
		#self.tried_actions.append(max_act)
		
		if max_func==-float('Inf'):
			max_act = sampler.random_action()
		return max_act
		
	def add_tried_action(self,a):
		self.tried_actions.append(a)

def zero_policy(size):
	pi = Policy(np.zeros((size,1)),[])
	return pi