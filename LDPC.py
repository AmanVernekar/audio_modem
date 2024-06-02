from ldpc_jossy.py import ldpc
import numpy as np 
from math import sqrt 
import random
import parameters 

# Generate a code 
z = parameters.ldpc_z
k = parameters.ldpc_k
c = ldpc.code('802.16', '1/2', z)

raw_bin_data = np.load("Data_files/binary_data.npy")[:97200]  

def encode_data(_raw_bin_data): 
    # The code must have an input of 648 to compute the encoded data,
    # therefore the raw binary data is first zero padded to ensure it's a multiple of 648. 
    mod_k = (len(_raw_bin_data) % k)                             # Finds how much we should pad by 
    zeros = k - mod_k
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

# Takes raw binary data and uses an ldpc code to add redundancy
ldpc_encoded_data = encode_data(raw_bin_data)[0]
ldpc_encoded_data = ldpc_encoded_data[1296*5:1296*6]   # Takes only block six of encoded data 

def qpsk_modulator(binary_sequence):
    # if binary_sequence has odd number of bits, add 0 at the end
    if len(binary_sequence) % 2 != 0:
        binary_sequence = np.append(binary_sequence, 0)
    
    # Initialize an empty array to store modulated symbols
    modulated_sequence = np.empty(len(binary_sequence) // 2, dtype=complex)
    
    # Mapping to complex symbols using QPSK   !! Do we care about energy of each symbol? !!
    for i in range(0, len(binary_sequence), 2):
        bit_pair = binary_sequence[i:i+2]
        # 00 => 1+j
        if np.array_equal(bit_pair, [0, 0]):
            modulated_sequence[i//2] = 1 + 1j
        # 01 => -1+j
        elif np.array_equal(bit_pair, [0, 1]):
            modulated_sequence[i//2] = -1 + 1j
        # 11 => -1-j
        elif np.array_equal(bit_pair, [1, 1]):
            modulated_sequence[i//2] = -1 - 1j
        # 11 => 1-j
        elif np.array_equal(bit_pair, [1, 0]):
            modulated_sequence[i//2] = 1 - 1j
    
    # print(f"QPSK Modulated sequence: {modulated_sequence}")
    return modulated_sequence 

modulated_sequence = qpsk_modulator(ldpc_encoded_data)   # Block six modulated data

mean = 0
sigma = 0.9  # Example standard deviation

# Generate Gaussian noise for both real and imaginary parts
noise_real = np.random.normal(mean, sigma, modulated_sequence.shape)
print("Real noise: ", noise_real[:10])
noise_imag = np.random.normal(mean, sigma, modulated_sequence.shape)

# Combine real and imaginary parts to form the complex noise
noise = noise_real + 1j * noise_imag

# Add the noise to the original complex array
noisy_modulated_sequence = modulated_sequence + noise
print("noisy modulated data length: ", len(noisy_modulated_sequence))

def LLRs(complex_vals, c_k, sigma_square, A): 
    LLR_list = []
    for i in range(len(complex_vals)): 
        c_conj = c_k[i].conjugate()
        L_1 = (A*c_k[i]*c_conj*sqrt(2)*complex_vals[i].imag) / (sigma_square)
        LLR_list.append(L_1)
        L_2 = (A*c_k[i]*c_conj*sqrt(2)*complex_vals[i].real) / (sigma_square)
        LLR_list.append(L_2)

    return LLR_list

def decode_data(LLRs, chunks_num): 
    LLRs_split = np.array(np.array_split(LLRs, chunks_num))
     
    decoded_list = []
    for i in range(chunks_num): 
        decoded_chunk, it = c.decode(LLRs_split[i])
        decoded_list.append(decoded_chunk)
    
    decoded_data = np.concatenate(decoded_list)
    threshold = 0.0
    decoded_data = (decoded_data < threshold).astype(int)

    decoded_data_split = np.array(np.array_split(decoded_data, chunks_num))[:, : 648]
    decoded_raw_data = np.concatenate(decoded_data_split)

    return decoded_raw_data

c_k = [1]*len(modulated_sequence)
sigma_square = sigma*sigma
A = 1
LLRs_block_1 = LLRs(noisy_modulated_sequence, c_k, sigma_square, A)
decoded_raw_data = decode_data(LLRs_block_1, chunks_num = 1)

compare1 = raw_bin_data[648*5:6*648]
compare2 = decoded_raw_data[:648]

def error(compare1, compare2, test): 
    wrong = 0
    for i in range(len(compare1)): 
        if int(compare1[i]) != compare2[i]: 
            wrong = wrong + 1
    print("wrong: ", wrong)
    print(test, " : ", (wrong/ len(compare1))*100)

error(compare1, compare2, '1 against 2')





# Makes fake LLRs values to check it's working 
# noise_std = 0.01
# received_data = ldpc_encoded_data
# received_data = np.where(received_data == 0, 1, -1)
# fake_LLRs = received_data  + np.random.normal(0, noise_std,len(received_data)) 
# print(len(fake_LLRs))
# print(fake_LLRs[:10])

# app, it = c.decode(fake_LLRs)
# threshold = 0.0
# decoded_data = (app < threshold).astype(int)
# print(decoded_data[:10])
# result = all(raw_bin_data[648*5:6*648] == decoded_data[:648])
# print(result)
# compare1 = raw_bin_data[648*5:6*648]
# compare2 = decoded_data[:648]

# def error(compare1, compare2, test): 
#     wrong = 0
#     for i in range(len(compare1)): 
#         if int(compare1[i]) != compare2[i]: 
#             wrong = wrong + 1
#     print("wrong: ", wrong)
#     print(test, " : ", (wrong/ len(compare1))*100)

# error(compare1, compare2, '1 against 2')



# # Decoding 
# # decodes the LLRs to find belief values. Then apply when app > 0 choose 0 and when app < 0 choose 1. 

# def decode_data(LLRs, chunks_num): 
#     LLRs_split = np.array(np.array_split(LLRs, chunks_num))
     
#     decoded_list = []
#     for i in range(chunks_num): 
#         decoded_chunk, it = c.decode(LLRs_split[i])
#         decoded_list.append(decoded_chunk)
    
#     decoded_data = np.concatenate(decoded_list)
#     threshold = 0.0
#     decoded_data = (decoded_data < threshold).astype(int)

#     decoded_data_split = np.array(np.array_split(decoded_data, chunks_num))[:, : 648]
#     decoded_raw_data = np.concatenate(decoded_data_split)

#     return decoded_raw_data

# decoded_raw_data = decode_data(fake_LLRs, chunks_num)

# # The decoded_raw_data just outputs a length of k so I assume we just take the first half as it is systematic?
# result1 = all(raw_bin_data_extend[:648] == decoded_raw_data[:648])
# result2 = all(raw_bin_data_extend[648:2*648] == decoded_raw_data[648:2*648])
# result3 = all(raw_bin_data_extend[648*5:6*648] == decoded_raw_data[648*5:6*648])
# print(result1, result2, result3)



# Check not using LLRs
# noise_std = 0.1
# ldpc_encoded_data = ldpc_encoded_data  + np.random.normal(0, noise_std, len(ldpc_encoded_data))


# app, it = c.decode(ldpc_encoded_data[648*2*5:6*2*648])
# threshold = 0.5 
# decoded_data = (app > threshold).astype(int)
# print(decoded_data[:10])

# result = all(raw_bin_data_extend[648*5:6*648] == decoded_data[:648])
# print(result)


# num_bits_to_flip = 30
# indices_to_flip = random.sample(range(len(ldpc_encoded_data)), num_bits_to_flip)

# # Flip the bits at the selected indices
# for index in indices_to_flip:
#     ldpc_encoded_data[index] = 1 - ldpc_encoded_data[index] 