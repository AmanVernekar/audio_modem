import numpy as np

#-----------------------------------------------------
# Global parameters
#-----------------------------------------------------

prefix_len = 1024                            # length of cyclic prefix
sample_rate = 48000                          # samples per second
num_known_symbols = 1

#-----------------------------------------------------
# Chirp parameters
#-----------------------------------------------------

chirp_start_freq = 761.72                    # chirp start freq (Hz)
chirp_end_freq = 8824.22                     # chirp end freq (Hz)
chirp_type = "linear"                        # chirp type
chirp_duration = 1.365                       # duration of chirp in seconds

#-----------------------------------------------------
# OFDM parameters
#-----------------------------------------------------

datachunk_len = 4096                        # length of the data in the OFDM symbol (DFT length)
lower_bin = 85
upper_bin = 732

symbol_len = datachunk_len + prefix_len     # total length of symbol

#-----------------------------------------------------
# LDPC parameters
#-----------------------------------------------------

ldpc_type = '802.16'
ldpc_rate = '1/2'
ldpc_z = 54
ldpc_k = 648

#-----------------------------------------------------
# Parameters for testing
#-----------------------------------------------------

symbol_count = 105
recording_data_len = symbol_count * symbol_len
rec_duration_test = 30                          # duration of recording in seconds
rec_duration_real = 60


#-----------------------------------------------------
# LDPC Channel Estimation parameters (probably not needed as ldpc and symbol length make sense together)
#-----------------------------------------------------
num_symbols_for_ce = 5
num_estimates_obtained = 4
alpha = 0.5


#-----------------------------------------------------
# Known OFDM symbol
#-----------------------------------------------------
np.random.seed(1)
random_integers = np.random.randint(0,4,2047)
random_complex_values = np.sqrt(2) * np.exp(0 + random_integers * 1j * np.pi/2  + np.pi/4 * 1j)
known_datachunk = np.zeros((datachunk_len), dtype=complex)
known_datachunk[1 : datachunk_len//2] = random_complex_values
known_datachunk[datachunk_len//2  + 1:] = random_complex_values[::-1]
sent_known_datachunks = [known_datachunk]
