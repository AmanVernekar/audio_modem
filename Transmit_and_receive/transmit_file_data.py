import numpy as np
from numpy.fft import fft, ifft

# step 1: encode file as binary data (e.g. LDPC)
coded_info_sequence = np.array([1,0,1])  #placeholder coded sequence

# step 2: Modulate as complex symbols using QPSK
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

modulated_sequence = qpsk_modulator(coded_info_sequence) 

# step 2.5: function for calculating bin values (optional)
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

# step 3: insert QPSK symbols into as many OFDM symbols as required 
# !! from here onwards, there is a confusing naming with ofdm symbol/block/ifft. 
# We should change these to make this more clear
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

# step 4: IDFT each OFDM symbol
ifft_ofdm_blocks = ifft(ofdm_blocks, axis=1)  # applies ifft to each row

# step 5: add cyclic prefix to each part
def add_cyclic_prefix(ofdm_blocks, prefix_length):
    block_prefixes = ofdm_blocks[:, -prefix_length:]
    ofdm_blocks_w_prefixes = np.concatenate((block_prefixes, ofdm_blocks), axis=1)
    return ofdm_blocks_w_prefixes

ofdm_blocks_w_prefixes = add_cyclic_prefix(ifft_ofdm_blocks, 2)

# step 6: flatten all time domain blocks into one array
concatenated_blocks = ofdm_blocks_w_prefixes.flatten()

# step 7:convert to audio file to transmit across channel. add chirp beforehand etc.
def convert_values_to_audio(data):
    return "audio file"