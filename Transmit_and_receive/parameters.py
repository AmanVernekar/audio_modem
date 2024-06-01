import numpy as np

#-----------------------------------------------------
# Global parameters
#-----------------------------------------------------

prefix_len = 512                           # length of cyclic prefix
sample_rate = 44100                         # samples per second

rec_duration = 19                            # duration of recording in seconds

#-----------------------------------------------------
# Chirp parameters
#-----------------------------------------------------

chirp_start_freq = 0.01                     # chirp start freq
chirp_end_freq = 22050                      # chirp end freq
chirp_type = "linear"                       # chirp type
chirp_duration = 5                          # duration of chirp in seconds

#-----------------------------------------------------
# OFDM parameters
#-----------------------------------------------------

datachunk_len = 4096                        # length of the data in the OFDM symbol (DFT length)
lower_bin = 85
upper_bin = 850

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