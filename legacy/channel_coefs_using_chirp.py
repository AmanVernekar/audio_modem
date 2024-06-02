    #Recalculate the section of chirp we want
    detected_chirp = recording[detected_index-chirp_sample_count+shift:detected_index+shift]
    detected_fft = fft(detected_chirp)
    channel_fft = detected_fft/chirp_fft
    channel_impulse = ifft(channel_fft)

    # take the channel that is the length of the cyclic prefix, zero pad to get datachunk length and fft
    channel_impulse_cut = channel_impulse[:prefix_len]
    channel_impulse_full = list(channel_impulse_cut) + [0]*int(datachunk_len-prefix_len) # zero pad to datachunk length
    channel_coefficients = fft(channel_impulse_full)