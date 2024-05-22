import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d


datachunk_len = 2048
prefix_len = 1024
symbol_len = datachunk_len + prefix_len
lower_freq = 1000
upper_freq = 11000
sample_rate = 44100  # samples per second
duration = 15
chirp_duration = 5
chirp_start_freq = 0.01
chirp_end_freq = 22050
chirp_type = "linear"

recording_data_len = 331776

# step 1: Generate transmitted chirp and record signal
def calculate_bins(sample_rate, lower_freq, upper_freq, ofdm_chunk_length):
    lower_bin = np.ceil((lower_freq / sample_rate) * ofdm_chunk_length).astype(int)  # round up
    upper_bin = np.floor((upper_freq / sample_rate) * ofdm_chunk_length).astype(int)  # round down

    print(f"""
    for the parameters: sample rate = {sample_rate}Hz
                        information bandlimited to {lower_freq} - {upper_freq}Hz
                        OFDM symbol length = {ofdm_chunk_length}
                lower bin is {lower_bin}
                upper bin is {upper_bin}
    """)
    return lower_bin, upper_bin

lower_bin, upper_bin = calculate_bins(sample_rate, lower_freq, upper_freq, datachunk_len)

t_total = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)

chirp_sig = chirp(t_chirp, f0=chirp_start_freq, f1=chirp_end_freq, t1=chirp_duration, method=chirp_type)
chirp_sig = list(chirp_sig)


# # Using real recording 
recording = sd.rec(sample_rate*duration, samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()

# step 2: perform channel estimation and synchronisation steps

# Apply the matched filter for synchronisation
recording = recording.flatten()  # Flatten to 1D array if necessary
np.save("rep_recording_to_test_with.npy", recording)

# Using saved recording
# recording = np.load("recording_to_test_with.npy")

# The matched_filter_output has been changed to full as this is the default method and might be easier to work with
# this means that the max index detected is now at the end of the chirp
matched_filter_output = correlate(recording, chirp_sig, mode='full')

# Create plots of the recording and matched filter response note that the x axes are different. 
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

# plt.plot(abs(matched_filter_output))
plt.show()

detected_index = np.argmax(matched_filter_output)
print(detected_index)

# Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)
# n = int(sample_rate*chirp_duration/2)
# detected_chirp = recording[detected_index-n:detected_index+n]
n = int(sample_rate*chirp_duration)
print(n)
detected_chirp = recording[detected_index-n:detected_index]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)
print(np.argmax(abs(channel_impulse)))

plt.plot(abs(channel_impulse))
plt.show()

def impulse_start_10_90_jump(channel_impulse):   
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
    impulse_start = np.argmax(abs(channel_impulse))
    print(impulse_start)
    if impulse_start > len(channel_impulse) / 2:
        impulse_start = impulse_start - len(channel_impulse)
    print(impulse_start)
    return impulse_start


def impulse_start_smoothing(channel_impulse):

    # Dataset
    x = np.arange(0, len(channel_impulse), 1)
    y = abs(channel_impulse)

    X_Y_Spline = make_interp_spline(x, y)

    # Returns evenly spaced numbers
    # over a specified interval.
    X_ = np.linspace(x.min(), x.max(), len(channel_impulse))
    Y_ = X_Y_Spline(X_)

    # Plotting the Graph
    plt.plot(X_, Y_)
    plt.title("Plot Smooth Curve Using the scipy.interpolate.make_interp_spline() Class")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()


def impulse_start_guassian(channel_impulse): 
    data = channel_impulse

    # Apply Gaussian smoothing
    # sigma determines the standard deviation of the Gaussian kernel
    # window size +- 3 data points can be interpreted as sigma = 1 for a standard Gaussian kernel
    smoothed_data = gaussian_filter1d(data, sigma=1)

    print("Original Data: ", data)
    print("Smoothed Data: ", smoothed_data)

impulse_shift = impulse_start_max(channel_impulse)


#Recalculate the section of chirp we want
detected_chirp = recording[detected_index-n+impulse_shift:detected_index+impulse_shift]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

channel_impulse_cut = channel_impulse[:datachunk_len]
channel_coefficients = fft(channel_impulse_cut)

plt.plot(abs(channel_impulse))
plt.show()

# step 2: crop audio file to the data
data_start_index = detected_index+impulse_shift
recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]
# load in the file sent to test against
source_mod_seq = np.load("rep_mod_seq.npy")
print(len(source_mod_seq))


# step 3: cut into different blocks and get rid of cyclic prefix

num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 

print(f"Num of OFDM symbols: {num_symbols}")

time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]

ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
ofdm_datachunks = ofdm_datachunks/channel_coefficients # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

data = data.flatten()
data = data[:len(source_mod_seq)]

colors = np.where(source_mod_seq == 1+1j, "b", 
            np.where(source_mod_seq == -1+1j, "c", 
            np.where(source_mod_seq == -1-1j, "m", 
            np.where(source_mod_seq == 1-1j, "y", 
            "Error"))))

plt.scatter(data.real, data.imag, c=colors)
plt.show()


# step 4: take the DFT
# step 5: divide by channel coefficients determined in step 1
# step 6: choose complex values corresponding to information bits
# step 7: map each value to bits using QPSK decision regions
# step 8: decode recieved bits to information bits
# step 9: convert information bits to file using standardised preamble.
