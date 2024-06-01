import numpy as np
from numpy.fft import fft, ifft
import sounddevice as sd
import soundfile as sf
from scipy.signal import chirp

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
binary_len = (upper_bin-lower_bin+1)*2
symbol_count = parameters.symbol_count


#-----------------------------------------------------
# STEP 1: Convert file to binary (including preamble)
#-----------------------------------------------------

#-----------------------------------------------------
# STEP 1.1: Load file information (binary data, file name, file size)
#-----------------------------------------------------

file_path = 'Transmit_and_receive/files_to_transmit/example.txt'

# Extract file name
file_name = file_path.split('/')[-1]  # Extracts the file name from the file path

# File as binary array
with open(file_path, 'rb') as file:
    # Read the file content into a binary string
    file_content = file.read()
    file_size_bytes = len(file_content)
    # Convert the binary content to a numpy array
    file_as_binary_array = np.unpackbits(np.frombuffer(file_content, dtype=np.uint8))

# Calculate file size in bits
file_size_bits = file_size_bytes * 8

# # Print file information
# print("File Name:", file_name)
# print("File Size (bytes):", file_size_bytes)
# print("File Size (bits):", file_size_bits)
# print("Binary array:", file_as_binary_array)

#-----------------------------------------------------
# STEP 1.2: Add preamble
#-----------------------------------------------------

# currently a WIP, see 'files_to_binary.py'
# return 'bitsream'

#-----------------------------------------------------
# STEP 2: Encode the binary data using LDPC
#-----------------------------------------------------

# takes in input 'bitsream'
# return 'encoded_bitsream'

#-----------------------------------------------------
# for testing purposes:
#-----------------------------------------------------
encoded_bitstream = np.load("Data_files/binary_data.npy")[:symbol_count*binary_len]

#-----------------------------------------------------
# STEP 3: Modulate as complex symbols using QPSK
#-----------------------------------------------------

def qpsk_modulator(binary_sequence):
    mult = 20
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
    
    return modulated_sequence * mult

modulated_sequence = qpsk_modulator(encoded_bitstream) 
# saving modulated sequence as npy file
np.save(f"Data_files/mod_seq_{symbol_count}symbols.npy", modulated_sequence)

#-----------------------------------------------------
# STEP 4: Create OFDM datachunks and insert information 
#-----------------------------------------------------

def create_ofdm_datachunks(modulated_sequence, chunk_length, lower_bin, upper_bin):
    mult = 20
    #  calculate number of information bins
    num_information_bins = (upper_bin - lower_bin) + 1

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
    print(f"yo {separated_mod_sequence.shape}") 

    # create a complex array of ofdm data chunks, where each symbol is an array filled with 0s of length chunk_length
    num_of_symbols = separated_mod_sequence.shape[0]
    random_noise = np.random.choice(mult*np.array([1+1j, -1+1j, -1-1j, 1-1j]), (num_of_symbols, chunk_length//2 - 1))
    ofdm_datachunk_array = np.ones((num_of_symbols, chunk_length), dtype=complex)  # change this so not zeros
    ofdm_datachunk_array[:, 1:chunk_length//2] = random_noise
    ofdm_datachunk_array[:, chunk_length//2 + 1 :] = np.fliplr(np.conjugate(random_noise))

    # insert information in OFDM blocks: 
    ofdm_datachunk_array[:, lower_bin:upper_bin+1] = separated_mod_sequence  # populates first half of block
    ofdm_datachunk_array[:, chunk_length-upper_bin:(chunk_length-lower_bin)+1] = np.fliplr(np.conjugate(separated_mod_sequence))  # second half of block
 
    return ofdm_datachunk_array  # returns array of OFDM blocks

ofdm_datachunks = create_ofdm_datachunks(modulated_sequence, datachunk_len, lower_bin, upper_bin)
# print(ofdm_datachunks.shape)

#-----------------------------------------------------
# STEP 5: Convert OFDM datachunks to time signal (including cyclic prefix)
#-----------------------------------------------------

#-----------------------------------------------------
# STEP 5.1: IDFT each OFDM symbol
#-----------------------------------------------------

time_domain_datachunks = ifft(ofdm_datachunks, axis=1)  # applies ifft to each row
time_domain_datachunks = time_domain_datachunks.real  # takes real part of ifft

#-----------------------------------------------------
# STEP 5.2: add cyclic prefix to each part
#-----------------------------------------------------

def add_cyclic_prefix(ofdm_datachunks, prefix_length):
    block_prefixes = ofdm_datachunks[:, -prefix_length:]
    ofdm_symbols = np.concatenate((block_prefixes, ofdm_datachunks), axis=1)
    return ofdm_symbols

ofdm_symbols = add_cyclic_prefix(time_domain_datachunks, prefix_len)

#-----------------------------------------------------
# STEP 5.3: flatten all time domain blocks into one array
#-----------------------------------------------------

concatenated_blocks = ofdm_symbols.flatten()

#-----------------------------------------------------
# STEP 5.4: Convert array to waveform !!DO WE DO THIS HERE?????
#-----------------------------------------------------

def convert_data_to_waveform(data):
    # normalise to between -1 and 1:
    max_absolute_val = (1/0.9)*np.max(np.abs(data))
    waveform = data / max_absolute_val

    # play the waveform
    # sample_rate = parameters.sample_rate
    # sd.play(waveform, sampling_rate)
    # sd.wait()
    # np.save("rep_waveform.npy", waveform)
    return waveform 

waveform = convert_data_to_waveform(concatenated_blocks, sample_rate)

#-----------------------------------------------------
# STEP 6: Add chirps and known OFDM blocks as per BEEEP standard
#-----------------------------------------------------

chirp_sig = our_chirp.chirp_sig
chirp_w_prefix_suffix = our_chirp.chirp_w_prefix_suffix


overall_sig = our_chirp.start_sig + chirp_w_prefix_suffix + list(waveform)
print(len(waveform))

#-----------------------------------------------------
# STEP 7: Save the final signal as a wav file
#-----------------------------------------------------

# Play the audio data
# sd.play(overall_sig, sample_rate)
# sd.wait()  # Wait until the sound has finished playing

np.save(f'Data_files/{symbol_count}symbol_overall_w_noise.npy', overall_sig)

output_file = f'Data_files/{symbol_count}symbol_audio_to_test_with_w_noise.wav'
sf.write(output_file, overall_sig, sample_rate)

print(f"Samples of data: {len(concatenated_blocks)}")
