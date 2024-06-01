from ldpc_jossy.py import ldpc
import numpy as np 

c = ldpc.code('802.16', '1/2', 54)
print(c.K)

raw_bin_data = np.load("Data_files/binary_data.npy")[:648]
print(raw_bin_data[:20])
ldpc_encoded_data = c.encode(raw_bin_data)
print(ldpc_encoded_data[:20]) 

# generate coded data from the binary data. 