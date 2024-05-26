# put a binary array here
import numpy as np

length = 20000
np.random.seed(42)

binary_data = np.random.randint(0,2,length)
print(binary_data)
np.save("binary_data.npy", binary_data)