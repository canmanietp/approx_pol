import gym
import gym.spaces
import random

# 'CartPole-v0'
# actions Discrete(2)
# states Box(4,)
# reward range (-inf,inf)

# 'BipedalWalker-v2'
# actions box(4,) (not sure yet how to use continuous actions without discretising, may have to use NN)
# states box(24,) (or 14 if you ignore the LIDAR measurements)

env = gym.make('CartPole-v1')
MAX_STEPS = env.spec.tags.get('wrapper_config.TimeLimit.max_episode_steps')

#-----Create samples
# Collect samples of (s,a,r,s') for n episodes of simulator
# Return list of (state, action, reward, next state) tuples
def sample(n):
	sample = ()
	samples = []
	
	for i in range(n):
		prev_s = env.reset()
		a = bool(random.getrandbits(1)) #random 0 or 1 action
		next_s,r,done,info = env.step(a)
		sample = (prev_s,a,r,next_s)
		samples.append(sample)
		prev_s = next_s
		step_count = 0
		while step_count < MAX_STEPS and not done:
			a = bool(random.getrandbits(1)) #random 0 or 1 action
			next_s,r,done,info = env.step(a)
			sample = (prev_s,a,r,next_s)
			samples.append(sample)
			step_count+=1
	return samples