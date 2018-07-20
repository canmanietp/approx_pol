# import sampler
# 
# samples = sampler.sample(50)
# print(samples)

import LSPI
import matplotlib.pyplot as plt
import sampler
import pickle

pi,distances = LSPI.run_LSPI()

plt.plot(distances)
plt.show()

pickle.dump(pi, open( "saved_policy.p", "wb" ) )

# pi = pickle.load( open( "saved_policy.p", "rb" ) )
# sampler.use_policy(pi)
