# import sampler
# 
# samples = sampler.sample(50)
# print(samples)

import LSPI
import matplotlib.pyplot as plt

policy,distances = LSPI.run_LSPI()

plt.plot(distances)
plt.show()