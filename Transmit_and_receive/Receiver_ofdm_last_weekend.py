import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

import parameters
import our_chirp


#-----------------------------------------------------
# Import parameters
#-----------------------------------------------------
datachunk_len = parameters.datachunk_len             # length of the data in the OFDM symbol
prefix_len = parameters.prefix_len                   # length of cyclic prefix
symbol_len = parameters.symbol_len                   # total length of symbol
sample_rate = parameters.sample_rate                 # samples per second
rec_duration = parameters.rec_duration               # duration of recording in seconds
chirp_duration = parameters.chirp_duration           # duration of chirp in seconds
chirp_start_freq = parameters.chirp_start_freq       # chirp start freq
chirp_end_freq = parameters.chirp_end_freq           # chirp end freq
chirp_type = parameters.chirp_type                   # chirp type
recording_data_len = parameters.recording_data_len   # number of samples of data
lower_bin = parameters.lower_bin                     # lower info bin
upper_bin = parameters.upper_bin                     # upper info bin
symbol_count = parameters.symbol_count               # number of symbols used in the test
num_data_bins = upper_bin-lower_bin+1                # calculate number of information bins
num_known_symbols = 5                                # number of known symbols used for channel estimation
total_num_symbols = int(recording_data_len/symbol_len)
existing_recording = True
recording_filename = f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy"
chirp_sample_count = int(sample_rate*chirp_duration) 
chirp_sig = our_chirp.chirp_sig
mult = 20
shifts = range(-200,200)

source_mod_seq = np.load(f"Data_files/mod_seq_{symbol_count}symbols.npy")[num_known_symbols*num_data_bins:]
sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_w_noise.npy')
sent_without_chirp = sent_signal[-symbol_count*symbol_len:]
sent_datachunks = np.array(np.array_split(sent_without_chirp, symbol_count))[:, prefix_len:]




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

def coeffs_from_chirp(shift):
    detected_chirp = recording[matched_filter_max_index-chirp_sample_count+shift:matched_filter_max_index+shift]
    detected_fft = fft(detected_chirp)
    channel_fft = detected_fft/fft(chirp_sig)
    channel_impulse = ifft(channel_fft)

    # take the channel that is the length of the cyclic prefix, zero pad to get datachunk length and fft
    channel_impulse_cut = channel_impulse[:prefix_len]
    channel_impulse_full = list(channel_impulse_cut) + [0]*int(datachunk_len-prefix_len) # zero pad to datachunk length
    channel_coefficients = fft(channel_impulse_full)
    return channel_coefficients

def estimate_channel_from_known_ofdm(_num_known_symbols):
        channel_estimates = np.zeros((_num_known_symbols, datachunk_len), dtype='complex')
        for i in range(_num_known_symbols):
            channel_fft = ofdm_datachunks[i]/fft(sent_datachunks[i])
            channel_estimates[i] = channel_fft
        
        average_channel_estimate = np.mean(channel_estimates, axis=0)
        print(channel_estimates.shape)
        print(average_channel_estimate.shape)
        return average_channel_estimate

def detect_data_from_synchronisation_index(index):
    return None

def recalc_channel(shift):
    data_start_index = matched_filter_max_index + shift + prefix_len
    recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]
    
    known_time_domain_datachunks = np.array(np.array_split(recording_without_chirp, total_num_symbols))[:num_known_symbols, prefix_len:]
    ofdm_datachunks = fft(known_time_domain_datachunks) 
    channel_estimate = estimate_channel_from_known_ofdm(num_known_symbols)

    ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
    data = ofdm_datachunks[:, lower_bin:upper_bin+1]

    error = 0
    for i in range(num_known_symbols):
        error = error + error_in_symbol(i, data, source_mod_seq)
    return channel_estimate, error

def optimise_channel(shifts, num_known_symbols):
    total_errors = np.array([recalc_channel(shift)[1] for shift in shifts])
    opt_shift = shifts[np.argmin(total_errors)]
    opt_channel, opt_error = recalc_channel(opt_shift)
    return opt_channel, opt_error

# need to change error functions to account for grey coding (calculate bit-error, not complex value error)
def error_in_symbol(symbol_index, received_data, source_data):
    received_symbol = received_data[symbol_index]
    source_symbol = source_data[symbol_index * num_data_bins : (symbol_index+1) * num_data_bins]
    is_correct = (np.sign(received_symbol.real) == np.sign(source_symbol.real)) and (np.sign(received_symbol.imag) == np.sign(source_symbol.imag))
    truth = np.where(is_correct, 1, 0)
    correct = np.count_nonzero(truth)
    return num_data_bins - correct

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




if __name__ == '__main__':
    #-----------------------------------------------------
    # STEP 1: Record + save audio or load pre-recorded file
    #-----------------------------------------------------

    if existing_recording:
        recording = np.load(recording_filename)
    else:
        recording = sd.rec(sample_rate*rec_duration, samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        recording = recording.flatten()  # Flatten to 1D array if necessary
        np.save(recording_filename, recording)
    

    #-----------------------------------------------------
    # STEP 2: Synchronisation with matched filter 
    #-----------------------------------------------------

    matched_filter_output = correlate(recording, chirp_sig, mode='full') # mode = 'full' => max index detected is at the end of chirp
    matched_filter_max_index = np.argmax(matched_filter_output)


    #-----------------------------------------------------
    # STEP 3: Initial re-synchronisation with argmax/cyclic shift method
    #-----------------------------------------------------

    # not being done currently

    
    #-----------------------------------------------------
    # STEP 4: Re-synchonise using known OFDM optimisation. Calculate initial channel coefficients
    #-----------------------------------------------------

    #-----------------------------------------------------
    # STEP 5: Run a loop to decode rest of file
    #-----------------------------------------------------

    #-----------------------------------------------------
    # STEP 5.1 Demodulate 5 OFDM symbols to obtain codeword bits.
    #-----------------------------------------------------

    #-----------------------------------------------------
    # STEP 5.2 Use the LDPC decoder to recover message bits from codeword bits.
    #-----------------------------------------------------

    #-----------------------------------------------------
    # STEP 5.3 Reconstruct 4 subcarrier sequences from recovered message bits.
    #-----------------------------------------------------

    #-----------------------------------------------------
    # STEP 5.4 Calculate new channel estimates and average over the 4 sequences.
    #-----------------------------------------------------

    #-----------------------------------------------------
    # STEP 5.5 Update channel estimate by taking a weighted average of new and current estimates.
    #-----------------------------------------------------

    channel_coefficients = optimise_channel(shifts, num_known_symbols)
    

    #-----------------------------------------------------
    # STEP 5.6: Optionally calculate/plot errors
    #-----------------------------------------------------
    colors = np.where(source_mod_seq == mult*(1+1j), "b", #"b"
            np.where(source_mod_seq == mult*(-1+1j), "c", #"c"
            np.where(source_mod_seq == mult*(-1-1j), "m", #"m"
            np.where(source_mod_seq == mult*(1-1j), "y",  #"y"
            "Error"))))
    plot_constellation(range(6,16), colors)


    #-----------------------------------------------------
    # STEP 6: Convert information bits to file using standardised preamble.
    #-----------------------------------------------------