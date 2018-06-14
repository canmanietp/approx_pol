class Policy:
	def __init__(self, policy, weights, l2, linf):
		self.policy = policy #dict
		self.weights = weights 
		self.l2 = l2
		self.linf = linf
		
	def act(self,state):
	# returns action for given state according to policy
		return self.policy[state]
		
	## function which computes random policy?
		