# put a binary array here
import numpy as np

length = 2_000_000
np.random.seed(42)

binary_data = np.random.randint(0,2,length)
# print(binary_data)
np.save("Data_files/binary_data.npy", binary_data)