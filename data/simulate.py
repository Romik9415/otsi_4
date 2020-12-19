from simulators import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

noise_gen = NoiseSimulator()
arr = np.zeros(100)
x = next(noise_gen)
for i, n in enumerate(x):
	#print(i)
	arr[i] = n

plt.plot(arr)
plt.show()