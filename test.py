# import sampler
# 
# samples = sampler.sample(50)
# print(samples)

import LSPI
import matplotlib.pyplot as plt
import sampler

pi,distances = LSPI.run_LSPI()

plt.plot(distances)
plt.show()

sampler.use_policy(pi)