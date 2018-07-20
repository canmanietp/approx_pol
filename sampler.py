import gym
import gym.spaces
import random
import numpy as np
import policy

# 'CartPole-v1'
# actions Discrete(2)
# states Box(4,)
# reward range (-inf,inf)

# 'BipedalWalker-v2'
# actions box(4,) (not sure yet how to use continuous actions without discretising, may have to use NN)
# states box(24,) (or 14 if you ignore the LIDAR measurements)


#-----Create samples
# Collect samples of (s,a,r,s') for n episodes of simulator
# Return list of (state, action, reward, next state) tuples
def sample(n,pi):
	env = gym.make('CartPole-v1')
	MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
	
	sample = ()
	samples = []
	
	for i in range(n):
		prev_s = env.reset()
		if pi==[]:
			a = random_action()
		else:
			a = pi.act(prev_s) # random_action() 
		next_s,r,done,info = env.step(a)
		sample = (prev_s,a,r,next_s)
		samples.append(sample)
		prev_s = next_s
		step_count = 0
		while step_count < MAX_STEPS and not done:
			if pi==[]:
				a = random_action()
			else:
				a = pi.act(prev_s) # random_action() 
			next_s,r,done,info = env.step(a)
			sample = (prev_s,a,r,next_s)
			samples.append(sample)
			prev_s = next_s
			step_count+=1
	
	print("finished sampling " + str(n) + " episodes")
	env.close()
	return samples 
	
def random_action():
	return int(random.getrandbits(1)) #random 0 or 1 action
	
def use_policy(pi):
	env = gym.make('CartPole-v1')
	MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')
	total_reward = 0
	
	prev_s = env.reset()
	env.render()
	a = pi.act(prev_s)
	next_s,r,done,info = env.step(a)
	total_reward+=r
	prev_s = next_s
	step_count = 0
	
	while step_count < MAX_STEPS and not done:
		a = pi.act(prev_s) #act based on policy
		next_s,r,done,info = env.step(a)
		total_reward+=r
		env.render()
		prev_s = next_s
		step_count+=1
		
	print("lasted " + str(step_count) + " steps")
	print(total_reward)
	env.close()