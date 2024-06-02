import numpy as np
from numpy.fft import fft, ifft
import sounddevice as sd
import soundfile as sf
from ldpc_jossy.py import ldpc

import parameters
import our_chirp

#-----------------------------------------------------
# Import parameters
#-----------------------------------------------------

prefix_len = parameters.prefix_len         # cyclic prefix length
datachunk_len = parameters.datachunk_len        # length of data  
sample_rate = parameters.sample_rate        # sample rate 
lower_bin = parameters.lower_bin
upper_bin = parameters.upper_bin
binary_len = (upper_bin-lower_bin)
symbol_count = parameters.symbol_count

# Generate a code 
z = parameters.ldpc_z
k = parameters.ldpc_k
c = ldpc.code('802.16', '1/2', z)

# We want a coded info sequence of symbol_count * binary_len * 2 so we need raw_data of half this 
raw_bin_data = np.load("Data_files/binary_data.npy")[:symbol_count*binary_len] 
print("length of raw data: ", len(raw_bin_data)) 

def encode_data(_raw_bin_data): 
    # The code must have an input of 648 to compute the encoded data,
    # therefore the raw binary data is first zero padded to ensure it's a multiple of 648. 
    mod_k = (len(_raw_bin_data) % k)                             # Finds how much we should pad by 
    zeros = k - mod_k
    if zeros == k: 
        zeros = 0    # Stops it adding an extra block of zeros
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
print("length of ldpc encoded data: ", len(ldpc_encoded_data))
np.save(f'Data_files/{symbol_count}ldpc_encoded_data.npy', ldpc_encoded_data)


# From the above the part of the encoded data that we care about is from [known_symbol_num * num_data_bins * 2 : same + (num_data_bins *2)]
# This is what is transmitted and hard decoded so check this first. 

# STEP 2: Modulate as complex symbols using QPSK
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

modulated_sequence = qpsk_modulator(ldpc_encoded_data) 
print("length of modulated encoded data: ", len(modulated_sequence))
np.save(f"Data_files/mod_seq_{symbol_count}symbols.npy", modulated_sequence)

# STEP 3: insert QPSK complex values into as many OFDM datachunks as required 
def create_ofdm_datachunks(modulated_sequence, chunk_length, lower_bin, upper_bin):
    #  calculate number of information bins
    num_information_bins = (upper_bin - lower_bin) #    733 - 85 = 648

    # append with 0s if not modulated_sequence is not multiple of num_information_bins
    num_zeros = num_information_bins - (len(modulated_sequence) % num_information_bins)
    print(num_zeros)

    if num_zeros != num_information_bins:
        zero_block = np.zeros(num_zeros, dtype=complex)
        modulated_sequence = np.append(modulated_sequence, zero_block)
    print(len(modulated_sequence))
    # create new array containing modulated_sequence, where each row corresponds to an OFDM data chunk
    separated_mod_sequence = np.reshape(modulated_sequence, (-1, num_information_bins)) 
    # separated_mod_sequence = np.array(np.array_split(modulated_sequence, 5))
    print(f"modulated data array shape: {separated_mod_sequence.shape}") 

    # create a complex array of ofdm data chunks, where each symbol is an array filled with 0s of length chunk_length
    num_of_symbols = separated_mod_sequence.shape[0]
    random_noise = np.random.choice(np.array([1+1j, -1+1j, -1-1j, 1-1j]), (num_of_symbols, chunk_length//2 - 1))
    ofdm_datachunk_array = np.ones((num_of_symbols, chunk_length), dtype=complex)  
    ofdm_datachunk_array[:, 1:chunk_length//2] = random_noise
    ofdm_datachunk_array[:, chunk_length//2 + 1 :] = np.fliplr(np.conjugate(random_noise))

    # insert information in OFDM blocks: 
    ofdm_datachunk_array[:, lower_bin:upper_bin] = separated_mod_sequence  # populates first half of block
    ofdm_datachunk_array[:, chunk_length-(upper_bin-1):chunk_length-(lower_bin-1)] = np.fliplr(np.conjugate(separated_mod_sequence))  # second half of block
 
    return ofdm_datachunk_array  # returns array of OFDM blocks

ofdm_datachunks = create_ofdm_datachunks(modulated_sequence, datachunk_len, lower_bin, upper_bin)
print("ofdm datachunks shape", ofdm_datachunks.shape)

# STEP 4: IDFT each OFDM symbol
time_domain_datachunks = ifft(ofdm_datachunks, axis=1)  # applies ifft to each row
time_domain_datachunks = time_domain_datachunks.real  # takes real part of ifft

# STEP 5: add cyclic prefix to each part
def add_cyclic_prefix(ofdm_datachunks, prefix_length):
    block_prefixes = ofdm_datachunks[:, -prefix_length:]
    ofdm_symbols = np.concatenate((block_prefixes, ofdm_datachunks), axis=1)
    return ofdm_symbols

ofdm_symbols = add_cyclic_prefix(time_domain_datachunks, prefix_len)

# STEP 6: flatten all time domain blocks into one array
concatenated_blocks = ofdm_symbols.flatten()

# STEP 7:convert to audio file to transmit across channel. add chirp beforehand etc.
def convert_data_to_audio(data, sampling_rate):
    # normalise to between -1 and 1:
    max_absolute_val = np.max(np.abs(data))
    waveform = data / max_absolute_val

    # play the waveform
    # sd.play(waveform, sampling_rate)
    # sd.wait()
    # np.save("rep_waveform.npy", waveform)
    return waveform 

chirp_sig = our_chirp.chirp_sig
chirp_w_prefix_suffix = our_chirp.chirp_w_prefix_suffix

waveform = convert_data_to_audio(concatenated_blocks, sample_rate)
overall_sig = our_chirp.start_sig + chirp_w_prefix_suffix + list(waveform)
print(len(waveform))

# Play the audio data
# sd.play(overall_sig, sample_rate)
# sd.wait()  # Wait until the sound has finished playing

np.save(f'Data_files/{symbol_count}symbol_overall_w_noise.npy', overall_sig)

output_file = f'Data_files/{symbol_count}symbol_audio_to_test_with_w_noise.wav'
sf.write(output_file, overall_sig, sample_rate)

print(f"Samples of data: {len(concatenated_blocks)}")
