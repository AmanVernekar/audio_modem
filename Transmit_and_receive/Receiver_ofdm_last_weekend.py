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


#-----------------------------------------------------
# STEP 1: Record + save audio or load pre-recorded file
#-----------------------------------------------------

# Using real recording 
recording = sd.rec(sample_rate*rec_duration, samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()

recording = recording.flatten()  # Flatten to 1D array if necessary
np.save(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy", recording)

#  Using saved recording
# recording = np.load(f"Data_files/{symbol_count}symbol_recording_to_test_with_w_noise.npy")

#-----------------------------------------------------
# STEP 2: Synchronisation with matched filter 
#-----------------------------------------------------

chirp_sig = our_chirp.chirp_sig

matched_filter_output = correlate(recording, chirp_sig, mode='full')
# Note: mode = 'full' => max index detected is now at the end of the chirp

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
# plot_matched_filter()

matched_filter_max_index = np.argmax(matched_filter_output)

# Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)
chirp_sample_count = int(sample_rate*chirp_duration)   # number of samples of the chirp 
# window recording to chirp:
detected_chirp = recording[matched_filter_max_index-chirp_sample_count:matched_filter_max_index] 
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

# plot channel impulse before resynchronisation
# plt.plot(abs(channel_impulse))  
# plt.show()

#-----------------------------------------------------
# STEP 3: Initial re-synchronisation with argmax/cyclic shift method
#-----------------------------------------------------

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

impulse_shift = impulse_start_max(channel_impulse)
print(f"Impulse shift = {impulse_shift}")
impulse_shift = 0

# CURRENTLY WE ARE NOT DOING THIS. NEITHER METHODS ARE BEING CALLED IN THE CODE

#-----------------------------------------------------
# STEP 4: Detection pipeline!
#-----------------------------------------------------



def estimate_channel_from_known_ofdm(_num_known_symbols):
        channel_estimates = np.zeros((_num_known_symbols, datachunk_len), dtype='complex')
        for i in range(_num_known_symbols):
            channel_fft = ofdm_datachunks[i]/fft(sent_datachunks[i])
            channel_estimates[i] = channel_fft
        
        average_channel_estimate = np.mean(channel_estimates, axis=0)
        print(channel_estimates.shape)
        print(average_channel_estimate.shape)
        return average_channel_estimate

def detect_data_from_synchronisation_index():
    return None


shifts = range(-200,200)
total_errors = np.zeros((len(shifts)))

source_mod_seq = np.load(f"Data_files/mod_seq_{symbol_count}symbols.npy")[num_known_symbols*num_data_bins:]

sent_signal = np.load(f'Data_files/{symbol_count}symbol_overall_w_noise.npy')
sent_without_chirp = sent_signal[-symbol_count*symbol_len:]
sent_datachunks = np.array(np.array_split(sent_without_chirp, symbol_count))[:, prefix_len:]

mult = 20

colors = np.where(source_mod_seq == mult*(1+1j), "b", #"b"
            np.where(source_mod_seq == mult*(-1+1j), "c", #"c"
            np.where(source_mod_seq == mult*(-1-1j), "m", #"m"
            np.where(source_mod_seq == mult*(1-1j), "y",  #"y"
            "Error"))))



#-----------------------------------------------------
# STEP 4.1: Re-compute channel channel coefficients
#-----------------------------------------------------

#-----------------------------------------------------
# STEP 4.1: Map each value to bits using QPSK decision regions
#-----------------------------------------------------

#-----------------------------------------------------
# STEP 4.1: Compensate for phase drift
#-----------------------------------------------------

for g, shift in enumerate(shifts):
    #Recalculate the section of chirp we want
    detected_chirp = recording[matched_filter_max_index-chirp_sample_count+shift:matched_filter_max_index+shift]
    detected_fft = fft(detected_chirp)
    channel_fft = detected_fft/chirp_fft
    channel_impulse = ifft(channel_fft)

    # take the channel that is the length of the cyclic prefix, zero pad to get datachunk length and fft
    channel_impulse_cut = channel_impulse[:prefix_len]
    channel_impulse_full = list(channel_impulse_cut) + [0]*int(datachunk_len-prefix_len) # zero pad to datachunk length
    channel_coefficients = fft(channel_impulse_full)

    # plt.plot(abs(channel_impulse))
    # plt.show()
    # plt.plot(abs(channel_coefficients))
    # plt.show()

    #-----------------------------------------------------
    # STEP 4: crop audio file to the data
    #-----------------------------------------------------
    data_start_index = matched_filter_max_index+shift+prefix_len
    recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]
    # load in the file sent to test against
    # print(len(source_mod_seq))

    #-----------------------------------------------------
    # STEP 5: cut into different blocks and get rid of cyclic prefix
    #-----------------------------------------------------
    num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 

    # print(f"Num of OFDM symbols: {num_symbols}")

    time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]

    ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
    # channel_estimate = ofdm_datachunks[0]/fft(sent_datachunks[0])

    channel_estimate = estimate_channel_from_known_ofdm(num_known_symbols)

    ofdm_datachunks = ofdm_datachunks[num_known_symbols:]/channel_estimate # Divide each value by its corrosponding channel fft coefficient. 
    data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

    # data = data.flatten()
    # data = data[:len(source_mod_seq)]  # as the binary data isn't an exact multiple of 511*2 we have zero padded this gets rid of zeros

    # makes list of colours corresponding to the original modulated data






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
            _source_mod_seq = source_mod_seq[index*num_data_bins:(index+1)*num_data_bins]
            # Plot on the corresponding subplot
            # ax = axes[i, j]
            # # ax.scatter(x[_colors==mask_col], y[_colors==mask_col], c = _colors[_colors==mask_col])
            # ax.scatter(x, y, c = _colors)
            # ax.axvline(0)
            # ax.axhline(0)
            # ax.set_xlim((-50,50))
            # ax.set_ylim((-50,50))
            # ax.set_aspect('equal')

            errors = 0
            for polka, val in enumerate(_data):
                sent = _source_mod_seq[polka]
                if val.real/sent.real < 0 or val.imag/sent.imag < 0:
                    errors = errors + 1
            total_error = total_error + errors
    total_errors[g] = total_error*10/len(_data)

            # ax.set_title(f'OFDM Symbol {index + 1}\nerror % = {round(errors*100/len(_data), 2)}')
            # ax.text(10, 10, f"error % = {errors/len(_data)}")

    # Adjust layout to prevent overlap
    # plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.show()

plt.plot(shifts, total_errors)
plt.axvline(shifts[np.argmin(total_errors)])
plt.ylabel("Bit error percentage (%)")
plt.xlabel("Index")
plt.show()

print(shifts[np.argmin(total_errors)])
print(np.min(total_errors))


# plt.scatter(data.real, data.imag, c=colors)
# plt.axvline(0)
# plt.axhline(0)
# plt.show()

#-----------------------------------------------------
# STEP 5: Resynchronise using known OFDM blocks
#-----------------------------------------------------

#-----------------------------------------------------
# STEP 6: Recalculate detected data using optimised synchronisation estimate
#-----------------------------------------------------

#-----------------------------------------------------
# STEP 7: Decode recieved bits to information bits (LDPC decoding)
#-----------------------------------------------------

#-----------------------------------------------------
# STEP 7.5 : Optionally calculate/plot errors
#-----------------------------------------------------

# recovered_values = np.where(data.real >= 0 and data.imag >= 0, 1+1j, 
#             np.where(data.real < 0 and data.imag >= 0, -1+1j, 
#             np.where(data.real < 0 and data.imag < 0, -1-1j, 
#             np.where(data.real >= 0 and data.imag < 0, 1-1j, 
#             "Error"))))

# errors = np.count_nonzero(recovered_values-source_mod_seq)
# print(errors/len(recovered_values))

#-----------------------------------------------------
# STEP 8: Convert information bits to file using standardised preamble.
#-----------------------------------------------------
