import numpy as np

#in this transmitter program:
#step 1: encode file as binary data (e.g. LDPC)
coded_info_sequence = np.array([1,0,1])  #placeholder coded sequence

#step 2: Modulate as complex symbols using QPSK
def qpsk_modulator(binary_sequence):
    return "return the modulated sequence"

#step 3: insert QPSK symbols into as many OFDM symbols as required (only in correct numbers of bins)
#  - this should ensure that the OFDM symbol has conjugate symmetry

# lets use frequency bins for information of 1kHz to 8kHz (approximately)
# 44.1 kHz sampling rate
# => maybe create a function to evaluate correct bins? from lower cutoff, upper cutoff, sampling freq

def create_ofdm_blocks(modulated_sequence, block_length, lower_bin, upper_bin):
    return "arrray of OFDM blocks, with information in lower bin to upper bin"  # e.g. bins 1-511

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