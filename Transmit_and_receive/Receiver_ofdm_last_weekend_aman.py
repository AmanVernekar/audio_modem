import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate, find_peaks
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d
# from ldpc_jossy.py import ldpc

import parameters
import our_chirp


#-----------------------------------------------------
# Import parameters
#-----------------------------------------------------
datachunk_len = parameters.datachunk_len             # length of the data in the OFDM symbol
prefix_len = parameters.prefix_len                   # length of cyclic prefix
symbol_len = parameters.symbol_len                   # total length of symbol
sample_rate = parameters.sample_rate                 # samples per second
               
chirp_duration = parameters.chirp_duration           # duration of chirp in seconds
chirp_start_freq = parameters.chirp_start_freq       # chirp start freq
chirp_end_freq = parameters.chirp_end_freq           # chirp end freq
chirp_type = parameters.chirp_type                   # chirp type
chirp_sample_count = int(sample_rate*chirp_duration) + 2 * prefix_len
chirp_sig = our_chirp.chirp_sig

lower_bin = parameters.lower_bin                     # lower info bin
upper_bin = parameters.upper_bin                     # upper info bin
num_data_bins = upper_bin - lower_bin + 1            # calculate number of information bins
num_known_symbols = parameters.num_known_symbols     # number of known symbols used for channel estimation

alpha = parameters.alpha                             # weight for on the fly channel estimation
shifts = [0] #range(-200,200)

ldpc_z = parameters.ldpc_z
ldpc_k = parameters.ldpc_k
# ldpc_c = ldpc.code('802.16', '1/2', ldpc_z)

testing = True

# TODO
# edit this to contain the known chirp based on agreed seed/sent file if testing = True
# also create a separate testing pipeline in the end
# also import certain params only if testing = True (else infer from metadata)
if testing:
    symbol_count = parameters.symbol_count               # number of symbols used in the test
    source_mod_seq = np.load(f"Data_files/mod_seq_{symbol_count}symbols.npy")[num_known_symbols*num_data_bins:]
    sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_w_noise.npy')
    sent_without_chirps = sent_signal[sample_rate + chirp_sample_count : -chirp_sample_count]
    sent_datachunks = np.array(np.array_split(sent_without_chirps, symbol_count))[:, prefix_len:]
    sent_known_datachunks = sent_datachunks[:num_known_symbols]
    rec_duration = parameters.rec_duration_test
    recording_data_len = parameters.recording_data_len
    total_num_symbols = int(recording_data_len/symbol_len)

else:
    rec_duration = parameters.rec_duration_real
    sent_known_datachunks = parameters.sent_known_datachunks

existing_recording = True
recording_filename = f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy"





# Plots of the recording and matched filter output. Note that the x axes are different. 
def plot_matched_filter():
    t_rec = np.arange(0, len(recording))
    t_mat = np.arange(0, len(matched_filter_output))
    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(t_rec, recording, label='Recording', color='b')
    ax1.set_xlabel('X-axis 1')
    ax1.set_ylabel('Y-axis 1')
    ax1.legend()
    ax1.set_title('First Plot')

    ax2.plot(t_mat, matched_filter_output, label='Matched filter output', color='r')
    ax2.set_xlabel('X-axis 2')
    ax2.set_ylabel('Y-axis 2')
    ax2.legend()
    ax2.set_title('Second Plot')

    plt.tight_layout()

    plt.plot(abs(matched_filter_output))
    plt.show()

# functions to choose the start of the impulse
def impulse_start_10_90_jump(channel_impulse):   
    """Calculates index for 10-90% channel synchronisation method"""
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
    """Calculates index for argmax channel synchronisation method"""
    impulse_start = np.argmax(abs(channel_impulse))
    print(impulse_start)
    if impulse_start > len(channel_impulse) / 2:
        impulse_start = impulse_start - len(channel_impulse)
    print(impulse_start)
    return impulse_start

# TODO: need to change error functions to account for grey coding (calculate bit-error, not complex value error)
def error_in_symbol(symbol_index, received_data, source_data):
    received_symbol = received_data[symbol_index]
    source_symbol = source_data[symbol_index * num_data_bins : (symbol_index+1) * num_data_bins]
    correct = 0
    for i in range(num_data_bins):
        if np.sign(received_symbol[i].real) == np.sign(source_symbol[i].real) and (np.sign(received_symbol[i].imag) == np.sign(source_symbol[i].imag)):
            correct = correct + 1
    return num_data_bins - correct

# TODO: separate functions to plot full constellation and required symbols, with error rate
def plot_single_symbol(data, index):
    mult = 1
    x = data[index].real
    y = data[index].imag
    error = error_in_symbol(index, data, source_mod_seq)
    _source_mod_seq = source_mod_seq[index * num_data_bins : (index+1) * num_data_bins]
    colors = np.where(_source_mod_seq == mult*(1+1j), "b", #"b"
                np.where(_source_mod_seq == mult*(-1+1j), "c", #"c"
                np.where(_source_mod_seq == mult*(-1-1j), "m", #"m"
                np.where(_source_mod_seq == mult*(1-1j), "y",  #"y"
                "Error"))))
    plt.title(f"OFDM Symbol {index + 1}\nerror % = {error*100/num_data_bins}")
    plt.scatter(x, y, c=colors)
    plt.show()


def plot_constellation(symbol_indices, colors, data):
    # for mask_col in ["b", "c", "m", "y"]:
    # mask_col = "b"
    # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 8))
    # Generate and plot data for each subplot
    total_error = 0
    
    for i in range(2):
        for j in range(5):
            # Generate random data
            # index = 10*(i*5 + j) + 9
            index = i*5 + j + 5
            _data = data[index]
            x = _data.real
            y = _data.imag
            # _colors = colors[index*num_data_bins:(index+1)*num_data_bins]
            # Plot on the corresponding subplot
            # ax = axes[i, j]
            # # ax.scatter(x[_colors==mask_col], y[_colors==mask_col], c = _colors[_colors==mask_col])
            # ax.scatter(x, y, c = _colors)
            # ax.axvline(0)
            # ax.axhline(0)
            # ax.set_xlim((-50,50))
            # ax.set_ylim((-50,50))
            # ax.set_aspect('equal')

            _error_in_symbol = error_in_symbol(index, data, source_mod_seq)

            # ax.set_title(f'OFDM Symbol {index + 1}\nerror % = {round(errors*100/len(_data), 2)}')
            # ax.text(10, 10, f"error % = {errors/len(_data)}")

    # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

def detect_chirps(recording):
    matched_filter_output = correlate(recording, chirp_sig, mode='full') # mode = 'full' => max index detected is at the end of chirp
    indices, _ = find_peaks(matched_filter_output, height = 0.5*max(matched_filter_output), distance = sample_rate * 5)
    start_chirp_index, end_chirp_index = indices[0], indices[-1]
    data_start_index = start_chirp_index + prefix_len
    data_end_index = end_chirp_index - chirp_sample_count - prefix_len
    recording_data_len = data_end_index - data_start_index + 1
    excess_len = recording_data_len % symbol_len
    
    data_end_index = data_end_index + symbol_len - excess_len # always add samples and throw out a symbol later if required (based on metadata)
    return data_start_index, data_end_index

def coeffs_from_chirp(shift, start_chirp_index): # not used at present
    detected_chirp = recording[start_chirp_index-chirp_sample_count+shift : start_chirp_index+shift]
    detected_fft = fft(detected_chirp)
    channel_fft = detected_fft/fft(chirp_sig)
    channel_impulse = ifft(channel_fft)

    # take the channel that is the length of the cyclic prefix, zero pad to get datachunk length and fft
    channel_impulse_cut = channel_impulse[:prefix_len]
    channel_impulse_full = list(channel_impulse_cut) + [0]*int(datachunk_len-prefix_len) # zero pad to datachunk length
    channel_coefficients = fft(channel_impulse_full)
    return channel_coefficients

def estimate_channel_from_known_ofdm(ofdm_datachunks):
        channel_estimates = np.zeros((num_known_symbols, datachunk_len), dtype='complex')
        for i in range(num_known_symbols):
            channel_fft = ofdm_datachunks[i]/sent_known_datachunks[i]
            # plt.plot(fft(sent_known_datachunks[i]))

            plt.plot(np.abs(ifft(channel_fft)))
            plt.title("yo")
            plt.show()
            channel_estimates[i] = channel_fft
        
        average_channel_estimate = np.mean(channel_estimates, axis=0)
        return average_channel_estimate

def recalc_channel(shift, recording, data_start_index, data_end_index):
    data_start_index = data_start_index + shift
    data_end_index = data_end_index + shift
    recording_without_chirps = recording[data_start_index : data_end_index + 1]
    
    known_recording = recording_without_chirps[:num_known_symbols*symbol_len]
    known_time_domain_datachunks = np.reshape(known_recording, (num_known_symbols, symbol_len))[:, prefix_len:]
    ofdm_datachunks = fft(known_time_domain_datachunks) 
    channel_estimate = estimate_channel_from_known_ofdm(ofdm_datachunks)

    ofdm_datachunks = ofdm_datachunks/channel_estimate
    data = ofdm_datachunks[:, lower_bin:upper_bin+1]

    error = 0
    for i in range(num_known_symbols):
        error = error + error_in_symbol(i, data, source_mod_seq)
    return channel_estimate, error

def optimise_channel(shifts, recording, data_start_index, data_end_index):
    total_errors = np.array([recalc_channel(shift, recording, data_start_index, data_end_index)[1] for shift in shifts])
    opt_shift = shifts[np.argmin(total_errors)]
    opt_channel, opt_error = recalc_channel(opt_shift, recording, data_start_index, data_end_index)
    return opt_channel, opt_error, opt_shift

# TODO: Functions for LDPC

# function that takes bits (for one datachunk only) and converts it into one datachunk
def bits_to_datachunk(bits):
    return datachunk

# function that takes an ofdm datachunk and returns the bits that the datachunk represents
# reverse of above function
def datachunk_to_bits(datachunk):
    return bits

# use ldpc to recover sent bits from the received bits
def recover_bits(received_bits):
    return recovered_bits

def LLRs(complex_vals, c_k, sigma_square, A): 
    LLR_list = []
    for i in range(len(complex_vals)): 
        c_conj = c_k[i].conjugate()
        L_1 = (A*c_k[i]*c_conj*np.sqrt(2)*complex_vals[i].imag) / (sigma_square)
        LLR_list.append(L_1)
        L_2 = (A*c_k[i]*c_conj*np.sqrt(2)*complex_vals[i].real) / (sigma_square)
        LLR_list.append(L_2)

    return LLR_list

def decode_data(LLRs, chunks_num): 
    LLRs_split = np.array(np.array_split(LLRs, chunks_num))
     
    decoded_list = []
    for i in range(chunks_num): 
        decoded_chunk, it = ldpc_c.decode(LLRs_split[i])
        decoded_list.append(decoded_chunk)
    
    decoded_data = np.concatenate(decoded_list)
    threshold = 0.0
    decoded_data = (decoded_data < threshold).astype(int)

    decoded_data_split = np.array(np.array_split(decoded_data, chunks_num))[:, : 648]
    decoded_raw_data = np.concatenate(decoded_data_split)

    return decoded_raw_data


# TODO: Functions for file and metadata

# use the bitstream to extract metadata
def extract_metadata(recovered_bitstream):
    # \0\0FileName.type\0\0numBits\0\0
    
    return file_name, file_type, file_num_bits

# how many symbols does the given number of bits result in
def bits_to_num_symbols(num_bits):
    return int(num_bits/(2 * num_data_bins))

# how many bits does the given number of symbols result in
def num_symbols_to_bits(num_symbols):
    return num_data_bins * num_symbols * 2

# get just the file without metadata or end padding
def remove_metadata_and_padding(recovered_bitstream):
    return file_data

# save file so we can open it!
def save_file(file_name, file_type, file_data):
    pass




if __name__ == '__main__':
    if existing_recording:
        recording = np.load(recording_filename)
    else:
        recording = sd.rec(sample_rate*rec_duration, samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        recording = recording.flatten()  # Flatten to 1D array if necessary
        np.save(recording_filename, recording)
    

    data_start_index, data_end_index = detect_chirps(recording)
    # channel_coefficients, _, opt_shift = optimise_channel(shifts, recording, data_start_index, data_end_index)
    # plt.plot(np.abs(ifft(channel_coefficients)))
    # plt.show()
    opt_shift = 0
    
    data_start_index = data_start_index + opt_shift
    data_end_index = data_end_index + opt_shift

    recording_without_chirps = recording[data_start_index : data_end_index + 1]
    total_num_symbols = int(len(recording_without_chirps)/symbol_len)
    time_domain_datachunks = np.reshape(recording_without_chirps, (total_num_symbols, symbol_len))[:, prefix_len:]
    ofdm_datachunks = fft(time_domain_datachunks)

    channel_coefficients = ofdm_datachunks[0]/sent_known_datachunks[0]

    recovered_bitstream = np.array([])

    for symbol_index in range(1, total_num_symbols):
        received_datachunk = ofdm_datachunks[symbol_index]/channel_coefficients
        received_datachunks = ofdm_datachunks/channel_coefficients
        plot_single_symbol(received_datachunks[:, lower_bin:upper_bin+1], symbol_index)

        c_k = channel_coefficients
        sigma_square = 1
        A = 10
        LLRs_block_1 = LLRs(ofdm_datachunks[symbol_index], c_k, sigma_square, A)
        decoded_raw_data = decode_data(LLRs_block_1, chunks_num = 1)
        print(decoded_raw_data)
        exit()
        
        received_bits = datachunk_to_bits(received_datachunk)
        recovered_bits = recover_bits(received_bits)
        recovered_bitstream = np.append(recovered_bitstream, recovered_bits)
        
        recovered_datachunk = bits_to_datachunk(received_bits)
        new_coefficients = ofdm_datachunks[symbol_index]/recovered_datachunk
        channel_coefficients = alpha * channel_coefficients + (1 - alpha) * new_coefficients

    
    file_name, file_type, file_num_bits = extract_metadata(recovered_bitstream)
    required_num_symbols = bits_to_num_symbols(file_num_bits)

    # add or remove symbols and change bitstream accordingly (based on metadata)
    if required_num_symbols < total_num_symbols:
        extra_symbols_to_remove = total_num_symbols - required_num_symbols
        extra_bits_to_remove = num_symbols_to_bits(extra_symbols_to_remove)
        recovered_bitstream = recovered_bitstream[-extra_bits_to_remove:]

    elif required_num_symbols > total_num_symbols:
        extra_symbols_needed = required_num_symbols - total_num_symbols
        new_data_end_index = data_end_index + extra_symbols_needed * symbol_len
        extra_recording = recording[data_end_index : new_data_end_index + 1]
        extra_time_domain_datachunks = np.reshape(recording_without_chirps, (total_num_symbols, symbol_len))[:, prefix_len:]
        extra_ofdm_datachunks = fft(time_domain_datachunks)

        for symbol_index in range(1, extra_symbols_needed):
            received_datachunk = extra_ofdm_datachunks[symbol_index]/channel_coefficients
            received_bits = datachunk_to_bits(received_datachunk)
            recovered_bits = recover_bits(received_bits)
            recovered_bitstream = np.append(recovered_bitstream, recovered_bits)
            
            recovered_datachunk = bits_to_datachunk(received_bits)
            new_coefficients = extra_ofdm_datachunks[symbol_index]/recovered_datachunk
            channel_coefficients = alpha * channel_coefficients + (1 - alpha) * new_coefficients


    file_data = remove_metadata_and_padding(recovered_bitstream)
    save_file(file_name, file_type, file_data)








    # TODO: add code to calculate error rate and make plots if testing
    if testing:
        pass
        # colors = np.where(source_mod_seq == mult*(1+1j), "b", #"b"
        #         np.where(source_mod_seq == mult*(-1+1j), "c", #"c"
        #         np.where(source_mod_seq == mult*(-1-1j), "m", #"m"
        #         np.where(source_mod_seq == mult*(1-1j), "y",  #"y"
        #         "Error"))))
        # plot_constellation(range(6,16), colors)
