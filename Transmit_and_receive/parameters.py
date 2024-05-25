datachunk_len = 4096                        # length of the data in the OFDM symbol
prefix_len = 512                           # length of cyclic prefix
symbol_len = datachunk_len + prefix_len     # total length of symbol
sample_rate = 44100                         # samples per second
rec_duration = 7                            # duration of recording in seconds
chirp_duration = 5                          # duration of chirp in seconds
chirp_start_freq = 0.01                     # chirp start freq
chirp_end_freq = 22050                      # chirp end freq
chirp_type = "linear"                       # chirp type
recording_data_len = 4608                  # number of samples of data (HOW IS THIS FOUND)
lower_bin = 85
upper_bin = 850