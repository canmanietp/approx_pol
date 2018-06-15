#------class Policy
# Policy is implicitly represented through a set of weights w
# pi(state) = max_{a in A} Q(state,a) = max_{a in A} phi(state,a)*w
# Holds the discount factor, weights (when calculated in the iteration) and the L-inf and L2 norms of the weights (when calculated)
# and the basis associated with the policy
# and the total number of (discrete) actions [need to mod for continuous)
# Given a state, returns an action.

class Policy:
	def __init__(self, discount, basis, weights, l2, linf):
		self.discount = discount #dict
		self.basis = basis
		self.weights = weights 
		self.l2 = l2
		self.linf = linf
		
	## function which computes random policy?
	
	#def act(self, state):
		# if EXPLORE
		# action = a random action
		# actionphi = compute basis phi of given state and action above
		# else
		# action = action with maximum Q value (calculated by phi*weights)
		# actionphi = ^
		#return action, actionphi