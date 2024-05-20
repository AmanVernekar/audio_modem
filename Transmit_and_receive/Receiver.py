import numpy as np
from numpy.fft import fft, ifft
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy.signal import chirp, correlate


datachunk_len = 2048
prefix_len = 1024
symbol_len = datachunk_len + prefix_len
lower_freq = 1000
upper_freq = 11000
sample_rate = 44100  # samples per second
duration = 5
chirp_duration = 2
chirp_start_freq = 0.01
chirp_end_freq = 22050
chirp_type = "linear"

recording_data_len = 67584

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

recording = sd.rec(sample_rate*duration, samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()

# step 2: perform channel estimation and synchronisation steps

# Apply the matched filter for synchronisation
recording = recording.flatten()  # Flatten to 1D array if necessary
matched_filter_output = correlate(recording, chirp_sig, mode='same')

plt.plot(matched_filter_output)
plt.show()

detected_index = np.argmax(matched_filter_output)

#Use matched filter to take out the chirp from the recording
chirp_fft = fft(chirp_sig)
n = int(sample_rate*chirp_duration/2)
detected_chirp = recording[detected_index-n:detected_index+n]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

plt.plot(abs(channel_impulse))
plt.show()

def impulse_start(channel_impulse):   
    channel_impulse_max = np.max(channel_impulse)
    channel_impulse_10_percent = 0.1 * channel_impulse_max
    channel_impulse_90_percent = 0.5 * channel_impulse_max

    impulse_start = 0

    for i in range(len(channel_impulse) - 1):
        if channel_impulse[i] < channel_impulse_10_percent and channel_impulse[i + 1] > channel_impulse_90_percent:
            impulse_start = i + 1
            break

    if impulse_start > len(channel_impulse) / 2:
        impulse_start = impulse_start - len(channel_impulse)

    return impulse_start

impulse_shift = impulse_start(channel_impulse)

#Recalculate the section of chirp we want
detected_chirp = recording[detected_index-n+impulse_shift:detected_index+n+impulse_shift]
detected_fft = fft(detected_chirp)
channel_fft = detected_fft/chirp_fft
channel_impulse = ifft(channel_fft)

plt.plot(abs(channel_impulse))
plt.show()

# step 2: crop audio file to the data
data_start_index = detected_index+n+impulse_shift
recording_without_chirp = recording[data_start_index : data_start_index+recording_data_len]
# load in the file sent to test against
source_mod_seq = np.load("../mod_seq.npy")


# step 3: cut into different blocks and get rid of cyclic prefix

num_symbols = int(len(recording_without_chirp)/symbol_len)  # Number of symbols 

time_domain_datachunks = np.array(np.array_split(recording_without_chirp, num_symbols))[:, prefix_len:]

ofdm_datachunks = fft(time_domain_datachunks)  # Does the fft of all symbols individually 
ofdm_datachunks = ofdm_datachunks/channel_fft  # Divide each value by its corrosponding channel fft coefficient. 
data = ofdm_datachunks[:, lower_bin:upper_bin+1] # Selects the values from 1 to 511

data = data.flatten()

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