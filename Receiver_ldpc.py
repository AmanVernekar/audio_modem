import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import correlate
from ldpc_jossy.py import ldpc
from math import sqrt

from Transmit_and_receive import parameters 
from Transmit_and_receive import our_chirp


datachunk_len = parameters.datachunk_len             # length of the data in the OFDM symbol
prefix_len = parameters.prefix_len                   # length of cyclic prefix
symbol_len = parameters.symbol_len                   # total length of symbol
sample_rate = parameters.sample_rate                 # samples per second
rec_duration = parameters.rec_duration               # duration of recording in seconds
chirp_duration = parameters.chirp_duration           # duration of chirp in seconds
chirp_start_freq = parameters.chirp_start_freq       # chirp start freq
chirp_end_freq = parameters.chirp_end_freq           # chirp end freq
chirp_type = parameters.chirp_type                   # chirp type
recording_data_len = parameters.recording_data_len   # number of samples of data (HOW IS THIS FOUND)
lower_bin = parameters.lower_bin
upper_bin = parameters.upper_bin
symbol_count = parameters.symbol_count
num_data_bins = upper_bin-lower_bin
num_known_symbols = 5

# STEP 1: Generate transmitted chirp and record signal
chirp_sig = our_chirp.chirp_sig

# Using real recording 
# We are recording 1s of silence followed by a chirp with a prefix and suffix 
# recording = sd.rec(sample_rate*rec_duration, samplerate=sample_rate, channels=1, dtype='float32')
# sd.wait()

# recording = recording.flatten()  # Flatten to 1D array if necessary
# np.save(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy", recording)

#  Using saved recording
# recording = np.load(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy")
recording = np.load(f"Data_files/best_so_far/{symbol_count}symbol_recording_to_test_with_w_noise.npy")

# STEP 2: initially synchronise

# The matched_filter_output has been changed to full as this is the default method and might be easier to work with
# this means that the max index detected is now at the end of the chirp
matched_filter_output = correlate(recording, chirp_sig, mode='full')

# Create plots of the recording and matched filter response note that the x axes are different. 
t_rec = np.arange(0, len(recording))
t_mat = np.arange(0, len(matched_filter_output))


detected_index = np.argmax(matched_filter_output)
print(detected_index)

# Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)
chirp_sample_count = int(sample_rate*chirp_duration)   # number of samples of the chirp 
detected_chirp = recording[detected_index-chirp_sample_count:detected_index]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

# channel impulse before resynchronisation
# plt.plot(abs(channel_impulse))  
# plt.show()

# STEP 3: resynchronise and compute channel coefficients from fft of channel impulse response 
# functions to choose the start of the impulse
def impulse_start_10_90_jump(channel_impulse):   
    channel_impulse_max = np.max(channel_impulse)
    channel_impulse_10_percent = 0.1 * channel_impulse_max
    channel_impulse_90_percent = 0.6 * channel_impulse_max

    impulse_start = 0

    for i in range(len(channel_impulse) - 1):
        if channel_impulse[i] < channel_impulse_10_percent and channel_impulse[i + 5] > channel_impulse_90_percent:
            impulse_start = i + 5
            break

    if impulse_start > len(channel_impulse) / 2:
        impulse_start = impulse_start - len(channel_impulse)

    return impulse_start


def impulse_start_max(channel_impulse):
    impulse_start = np.argmax(abs(channel_impulse))
    print(impulse_start)
    if impulse_start > len(channel_impulse) / 2:
        impulse_start = impulse_start - len(channel_impulse)
    print(impulse_start)
    return impulse_start


impulse_shift = impulse_start_max(channel_impulse)
impulse_shift = 0


data_start_index = detected_index+impulse_shift + prefix_len
recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]
# load in the file sent to test against
source_mod_seq = np.load(f"Data_files/mod_seq_{symbol_count}symbols.npy")[num_known_symbols*num_data_bins:]
print(len(source_mod_seq))
# STEP 5: cut into different blocks and get rid of cyclic prefix
num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 
print(f"Num of OFDM symbols: {num_symbols}")

time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]
sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_w_noise.npy')
sent_without_chirp = sent_signal[-symbol_count*symbol_len:]
sent_datachunks = np.array(np.array_split(sent_without_chirp, symbol_count))[:, prefix_len:]

ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
def estimate_channel_from_known_ofdm(_num_known_symbols):
    channel_estimates = np.zeros((_num_known_symbols, datachunk_len), dtype='complex')
    for i in range(_num_known_symbols):
        channel_fft = ofdm_datachunks[i]/fft(sent_datachunks[i])
        channel_estimates[i] = channel_fft
    
    average_channel_estimate = np.mean(channel_estimates, axis=0)
    print(channel_estimates.shape)
    print(average_channel_estimate.shape)
    return average_channel_estimate
channel_estimate = estimate_channel_from_known_ofdm(num_known_symbols)

ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin] # Selects the values from the data bins 
# data = data.flatten()
# data = data[:len(source_mod_seq)]  # as the binary data isn't an exact multiple of 511*2 we have zero padded this gets rid of zeros
# makes list of colours corresponding to the original modulated data


colors = np.where(source_mod_seq == (1+1j), "b", #"b"
            np.where(source_mod_seq == (-1+1j), "c", #"c"
            np.where(source_mod_seq == (-1-1j), "m", #"m"
            np.where(source_mod_seq == (1-1j), "y",  #"y"
            "Error"))))

fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
# Generate and plot data for each subplot
for i in range(2):
    for j in range(5):
        # Generate random data
        # index = 10*(i*5 + j) + 9
        index = i*5 + j
        _data = data[index]
        x = _data.real
        y = _data.imag
        _colors = colors[index*num_data_bins:(index+1)*num_data_bins]
        _source_mod_seq = source_mod_seq[index*num_data_bins:(index+1)*num_data_bins]
        # Plot on the corresponding subplot
        ax = axes[i, j]
        # ax.scatter(x[_colors==mask_col], y[_colors==mask_col], c = _colors[_colors==mask_col])
        ax.scatter(x, y, c = _colors)
        ax.axvline(0)
        ax.axhline(0)
        ax.set_xlim((-30,30))
        ax.set_ylim((-30,30))
        ax.set_aspect('equal')

        errors = 0
        for n, val in enumerate(_data):
            sent = _source_mod_seq[n]
            if val.real/sent.real < 0 or val.imag/sent.imag < 0:
                errors = errors + 1
        ax.set_title(f'OFDM Symbol {index + 1}\nerror % = {round(errors*100/len(_data), 2)}')
        # ax.text(10, 10, f"error % = {errors/len(_data)}")
# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# Decoding attempt 2 

z = parameters.ldpc_z
k = parameters.ldpc_k
c = ldpc.code('802.16', '1/2', z)

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

c_k = channel_estimate
sigma_square = 1
A = 10
LLRs_block_1 = LLRs(data[0], c_k, sigma_square, A)
decoded_raw_data = decode_data(LLRs_block_1, chunks_num = 1)

compare1 = np.load("Data_files/binary_data.npy")[648*5:6*648]
compare2 = decoded_raw_data[:648]

def error(compare1, compare2, test): 
    wrong = 0
    for i in range(len(compare1)): 
        if int(compare1[i]) != compare2[i]: 
            wrong = wrong + 1
    print("wrong: ", wrong)
    print(test, " : ", (wrong/ len(compare1))*100)

error(compare1, compare2, '1 against 2')












































# DECODING 
# Hard decoding: 
# def hard_decision(one_block_of_data): 
#     one_block_hard_decode = []
#     for i in range(len(data[0])): 
#         if one_block_of_data[i].real > 0 and one_block_of_data[i].imag > 0:
#             one_block_hard_decode = one_block_hard_decode + [0,0]
#         elif one_block_of_data[i].real < 0 and one_block_of_data[i].imag > 0:
#             one_block_hard_decode = one_block_hard_decode + [0,1]
#         elif one_block_of_data[i].real < 0 and one_block_of_data[i].imag < 0:
#             one_block_hard_decode = one_block_hard_decode + [1,1]
#         else:
#             one_block_hard_decode = one_block_hard_decode + [1,0]

#     one_block_hard_decode = np.array(one_block_hard_decode)
#     return one_block_hard_decode

# print(data.shape)
# block_1_hard_decode = hard_decision(data[0])
# block_2_hard_decode = hard_decision(data[1])

# compare3 = block_1_hard_decode[:648] 
# compare4 = raw_bin_data = np.load("Data_files/binary_data.npy")[num_known_symbols*num_data_bins:(num_known_symbols*num_data_bins)+num_data_bins]   # num_known_symbols*num_data_bins
# compare5 = block_1_hard_decode
# compare6 = encoded_sent_data = np.load("Data_files/105ldpc_encoded_data.npy")[num_known_symbols*num_data_bins*2:(num_known_symbols*num_data_bins*2)+(num_data_bins*2)]

# def error(compare1, compare2, test): 
#     wrong = 0
#     for i in range(len(compare1)): 
#         if int(compare1[i]) != compare2[i]: 
#             wrong = wrong + 1
#     print("wrong: ", wrong)
#     print(test, " : ", (wrong/ len(compare1))*100)

# error(compare3, compare4, '3 against 4')
# error(compare5, compare6, '5 against 6')

# # noise_std = 0.1
# # one_block_hard_decode_noise = block_1_hard_decode  + np.random.normal(0, noise_std, len(block_1_hard_decode))

# # z = parameters.ldpc_z
# # k = parameters.ldpc_k
# # c = ldpc.code('802.16', '1/2', z)

# # app, it = c.decode(one_block_hard_decode_noise)
# # threshold = 0.5 
# # decoded_data = (app > threshold).astype(int)
# # print(decoded_data[:10])

# # compare7 = decoded_data[:648]
# # compare8 = raw_bin_data = np.load("Data_files/binary_data.npy")[648*5:648*6]

# # error(compare7, compare8, '7 against 8')


# # So it works if we just take the first half of of the data as it is a systematic code and for some reason the code is making it worse
# # I could try 


# noise_std = 0.01
# received_data = np.where(block_1_hard_decode == 0, 1, -1)
# fake_LLRs = received_data  + np.random.normal(0, noise_std, len(block_1_hard_decode))
# print(len(fake_LLRs))

# z = parameters.ldpc_z
# k = parameters.ldpc_k
# c = ldpc.code('802.16', '1/2', z)

# app, it = c.decode(fake_LLRs)
# threshold = 0.0
# decoded = (app < threshold).astype(int)

# compare1 = decoded[:648]
# compare2 = raw_bin_data = np.load("Data_files/binary_data.npy")[num_known_symbols*num_data_bins:(num_known_symbols*num_data_bins)+num_data_bins]   # num_known_symbols*num_data_bins

# error(compare1, compare2, '1 against 2')


# # def decode_data(LLRs, chunks_num): 
# #     LLRs_split = np.array(np.array_split(LLRs, chunks_num))
     
# #     decoded_list = []
# #     for i in range(chunks_num): 
# #         decoded_chunk, it = c.decode(LLRs_split[i])
# #         decoded_list.append(decoded_chunk)
    
# #     decoded_data = np.concatenate(decoded_list)
# #     threshold = 0.0
# #     decoded_data = (decoded_data < threshold).astype(int)

# #     decoded_data_split = np.array(np.array_split(decoded_data, chunks_num))[:, : 648]
# #     decoded_raw_data = np.concatenate(decoded_data_split)

# #     return decoded_raw_data

# # decoded_raw_data = decode_data(fake_LLRs, chunks_num = 1)

# compare1 = decoded_raw_data
# compare2 = raw_bin_data = np.load("Data_files/binary_data.npy")[num_known_symbols*num_data_bins:(num_known_symbols*num_data_bins)+num_data_bins]   # num_known_symbols*num_data_bins

# error(compare1, compare2, '1 against 2')



# # Attempting to decode with LLRs
# # Create LLRs
# # Step 1: Find sigma 
# # Step 2: Find A
# # Step 3: For one data block 


# #  Using LLRs
# def LLRs(complex_vals, c_k, sigma_square, A): 
#     LLR_list = []
#     for i in range(len(complex_vals)): 
#         c_conj = c_k[i].conjugate()
#         L_1 = (A*c_k[i]*c_conj*sqrt(2)*complex_vals[i].imag) / (sigma_square)
#         LLR_list.append(L_1)
#         L_2 = (A*c_k[i]*c_conj*sqrt(2)*complex_vals[i].real) / (sigma_square)
#         LLR_list.append(L_2)

#     return LLR_list

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

# c_k = channel_estimate[lower_bin:upper_bin]
# sigma_square = 1
# A = 10
# LLRs_block_1 = LLRs(data[0], c_k, sigma_square, A )
# decoded_raw_data = decode_data(LLRs_block_1, chunks_num = 1)

# # now we need to test if decoded_raw_date equals the sent data. 
# # The first 5 ofdm blocks are used for estimation so we need the 6th block of data to compare

# compare1 = decoded_raw_data[:648]
# compare2 = raw_bin_data = np.load("Data_files/binary_data.npy")[648*5:(648*5)+648]

