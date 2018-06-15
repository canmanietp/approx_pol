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