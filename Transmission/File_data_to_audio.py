import numpy as np

#in this transmitter program:
#step 1: encode file as binary data (e.g. LDPC)
coded_info_sequence = np.array([1,0,1])  #placeholder coded sequence

#step 2: Modulate as complex symbols using QPSK
def qpsk_modulator(binary_sequence):
    # if binary_sequence has odd number of bits, add 0 at the end
    if len(binary_sequence) % 2 != 0:
        binary_sequence = np.append(binary_sequence, 0)
    
    # Initialize an empty array to store modulated symbols
    modulated_sequence = np.empty(len(binary_sequence) // 2, dtype=complex)
    
    # Mapping to complex symbols using QPSK
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

modulated_sequence = qpsk_modulator(coded_info_sequence)

#step 3: insert QPSK symbols into as many OFDM symbols as required (only in correct numbers of bins)

# function for calculating bin values (optional)
def calculate_bins(sample_rate, lower_freq, upper_freq, ofdm_block_length):
    lower_bin = np.ceil((lower_freq / sample_rate) * ofdm_block_length).astype(int)  # round up
    upper_bin = np.floor((upper_freq / sample_rate) * ofdm_block_length).astype(int)  # round down

    print(f"""
    for the parameters: sample rate = {sample_rate}Hz
                        information bandlimited to {lower_freq} - {upper_freq}Hz
                        OFDM block length = {ofdm_block_length}
                lower bin is {lower_bin}
                upper bin is {upper_bin}
    """)
    return lower_bin, upper_bin

bin_vals = calculate_bins(44100, 1000, 8000, 1024)

lower_bin = bin_vals[0]
upper_bin = bin_vals[1]

def create_ofdm_blocks(modulated_sequence, block_length, lower_bin, upper_bin):
    #  calculate number of information bins
    num_information_bins = (upper_bin - lower_bin) + 1

    # append with 0s if not modulated_sequence is not multiple of num_information_bins
    num_zeros = num_information_bins - (len(modulated_sequence) % num_information_bins)

    if num_zeros != 0:
        zero_block = np.zeros(num_zeros, dtype=complex)
        modulated_sequence = np.append(modulated_sequence, zero_block)

    # create new array containing modulated_sequence, where each row corresponds to an OFDM block
    modulated_blocks = np.reshape(modulated_sequence, (-1, num_information_bins))  

    # create a complex array of ofdm blocks, where each block is an array filled with 0s of length block_length
    num_of_blocks = modulated_blocks.shape[0]
    ofdm_block_array = np.zeros((num_of_blocks, block_length), dtype=complex)

    # insert information in OFDM blocks: 
    ofdm_block_array[:, lower_bin:upper_bin+1] = modulated_blocks  # populates first half of block
    ofdm_block_array[:, block_length-upper_bin:(block_length-lower_bin)+1] = np.fliplr(np.conjugate(modulated_blocks))  # second half of block
 
    return ofdm_block_array  # returns array of OFDM blocks

ofdm_blocks = create_ofdm_blocks(modulated_sequence, 1024, 2, 511)
# print(ofdm_blocks)

#step 4: IDFT each OFDM symbol


#step 5: add cyclic prefix to each part
def add_cyclic_prefix(ofdm_block, prefix_length):
    return "ofdm block with cyclic prefix"

# ^^ use this function on each of the blocks, or create a similar function to do it all at once

#step 6: concatenate all time domain blocks. Do we want to space out these blocks? 
# beyond just the prefix? (probs no)
def concatenate_blocks(data):
    return "string of real values"


#step 7:convert to audio file to transmit across channel. add chirp beforehand etc.
def convert_values_to_audio(data):
    return "audio file"

#for separate decoder program:
#step 1: perform channel estimation and synchronisation steps
#step 2: crop audio file to the data
#step 3: cut into different blocks and get rid of cyclic prefix
#step 4: take the DFT
#step 5: divide by channel coefficients determined in step 1
#step 6: choose complex values corresponding to information bits
#step 7: map each value to bits using QPSK decision regions
detected_coded_seq = np.array([1,0,0])

#step 8: decode recieved bits to information bits
#step 9: convert information bits to file using standardised preamble.


#for testing: evaluate performance using percentage of coded bits successfully detected.
accuracy = np.mean(coded_info_sequence == detected_coded_seq) *100  # percentage of matching bits
print(f"Percentage of accurately detected bits: {accuracy:.5f}%")