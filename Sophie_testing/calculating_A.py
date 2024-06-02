# Given an ofdm symbol we want to find the average magnitude 
# We will be given a numpy array of complex values. 

import numpy as np

def average_magnitude(complex_array):
    # Calculate the magnitudes of the complex numbers
    magnitudes = np.abs(complex_array)
    
    # Calculate the average of the magnitudes
    average_mag = np.mean(magnitudes)
    
    return average_mag

# Now we need a function to calculate sigma, for now though we can say equal to 1 


def LLRs(complex_vals, c_k, sigma_square, A): 
    LLR_list = []
    for i in range(len(complex_vals)): 
        c_conj = c_k[i].conjugate()
        L_1 = (A*c_k[i]*c_conj*np.sqrt(2)*complex_vals[i].imag) / (sigma_square)
        LLR_list.append(L_1)
        L_2 = (A*c_k[i]*c_conj*np.sqrt(2)*complex_vals[i].real) / (sigma_square)
        LLR_list.append(L_2)

    return LLR_list

def decode_data(LLRs, chunks_num): 
    LLRs_split = np.array(np.array_split(LLRs, chunks_num))
     
    decoded_list = []
    for i in range(chunks_num): 
        decoded_chunk, it = ldpc_c.decode(LLRs_split[i])
        decoded_list.append(decoded_chunk)
    
    decoded_data = np.concatenate(decoded_list)
    threshold = 0.0
    decoded_data = (decoded_data < threshold).astype(int)

    decoded_data_split = np.array(np.array_split(decoded_data, chunks_num))[:, : 648]
    decoded_raw_data = np.concatenate(decoded_data_split)

    return decoded_raw_data

c_k = channel_coefficients
sigma_square = 1
A = 10
LLRs_block_1 = LLRs(ofdm_datachunks[symbol_index], c_k, sigma_square, A)
decoded_raw_data = decode_data(LLRs_block_1, chunks_num = 1)
print(decoded_raw_data)
