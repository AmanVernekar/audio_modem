from ldpc_jossy.py import ldpc
import numpy as np 

import parameters

z = parameters.ldpc_z
k = parameters.ldpc_k
c = ldpc.code('802.16', '1/2', z)

raw_bin_data = np.load("Data_files/example_file_data.npy")

def encode_data(_raw_bin_data): 
    # The code must have an input of 648 to compute the encoded data,
    # therefore the raw binary data is first zero padded to ensure it's a multiple of 648. 
    mod_k = (len(_raw_bin_data) % k)                             # Finds how much we should pad by 
    zeros = k - mod_k
    if zeros == k: 
        zeros = 0
    _raw_bin_data = np.pad(_raw_bin_data, (0,zeros), 'constant')      # Pads by num of zeros 
    chunks_num = int(len(_raw_bin_data) / k)
    raw_bin_chunks = np.array(np.array_split(_raw_bin_data, chunks_num))
     
    # Generates a sequence of coded bits and appends to the list
    ldpc_list = []
    for i in range(chunks_num): 
        ldpc_encoded_chunk = c.encode(raw_bin_chunks[i])
        ldpc_list.append(ldpc_encoded_chunk)
    
    ldpc_encoded_data = np.concatenate(ldpc_list)

    # Returns the data encoded in blocks of k 
    return ldpc_encoded_data, _raw_bin_data, chunks_num

coded_info_sequence = encode_data(raw_bin_data)[0]
raw_bin_data_extended = encode_data(raw_bin_data)[1]


# Now add bit flips 

# def flip_bits(bit_array, num_flips):
#     # Ensure bit_array is a NumPy array
#     assert isinstance(bit_array, np.ndarray), "Input must be a NumPy array"
    
#     # Check that the number of flips does not exceed the length of the array
#     assert num_flips <= len(bit_array), "Number of flips cannot exceed the length of the array"
    
#     # Choose random indices to flip
#     indices = np.random.choice(len(bit_array), num_flips, replace=False)

#     # Flip the bits at the chosen indices
#     bit_array[indices] = 1 - bit_array[indices]

#     return bit_array

# noisy_data = list(flip_bits(coded_info_sequence, 116))

noisy_data = np.load("Data_files/received_hard_decided_bits.npy")
noisy_data = list(noisy_data)
print(len(noisy_data))

mult_vals = [7]

for j in mult_vals:
    y = []
    for i in range(len(noisy_data)): 
        y.append( j * (.5 - noisy_data[i]))
    
    y = np.array(y)
    app, it = c.decode(y)
    app = app[:648]

    compare1 = raw_bin_data_extended
    compare2 = noisy_data[:648]

    app = np.where(app < 0, 1, 0)
    compare3 = app

    def error(compare1, compare2, test): 
        wrong = 0
        for i in range(len(compare1)): 
            if int(compare1[i]) != compare2[i]: 
                wrong = wrong + 1
        print("wrong: ", wrong)
        print(test, " : ", (wrong/ len(compare1))*100)

    error(compare1, compare2, '1 against 2')
    error(compare1, compare3, '1 against 3')

