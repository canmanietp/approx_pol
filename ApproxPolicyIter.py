
#-----Run simulator
# Run one episode according to policy (first policy is random) to select actions starting at initial_state 
# episode terminates when reaches max_steps or terminal condition (walker falls down or gets to the end)

#-----Compute basis
# input: sampled (state, action) pair
# output: phi (column vector of length equal to total number of basis functions, each entry is corresponding basis function evaluated at the (s,a) pair)
# Definition: Computes the basis functions for given (state,action) pair
#
# How? tbd

#----Policy Evaluation (LSQ) see: https://www.cs.rutgers.edu/~mlittman/courses/robots03/lspi.pdf
# input: samples, policy, new_policy
# output: w (weights), A, b (matrices) (of policy)
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

#------Policy Iteration (Least-Squares)
# input: samples, basis
# output: "good" policy
# Definition: Finds a good policy 
#
# epsilon is minimum distance that indicates convergence
#
# while (iteration < max_iterations) and (distance < epsilon) 
# evaluate policy (produces weights for the policy)
#	 ---use evaluation algorithm such as LSQ above---
# compute distance between current and all previous policies
#    ---compute the L2 and L-inf norm of the difference between weights of current policy and policy being checked
# store the current policy in list of policies

#------Approximate Q of policy
# Approx Q is just phi*weights