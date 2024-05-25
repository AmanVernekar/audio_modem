import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp, find_peaks
from numpy.fft import fft, ifft

sample_rate = 44100  # samples per second
duration = 5
chirp_duration = 1
threshold = 0
start_freq = 0.01
end_freq = 22050
chirp_type = "linear" 
test_num = 1
prefix_length = 5000
chirp_count = 1

t_total = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)

chirp_sig = chirp(t_chirp, f0=start_freq, f1=end_freq, t1=chirp_duration, method=chirp_type)
chirp_sig = list(chirp_sig)

recording = sd.rec(sample_rate*duration, samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
output_file = 'recording.wav'

sd.play(recording, samplerate=sample_rate)
sd.wait()
# sf.write(output_file, recording, sample_rate)


# Apply the matched filter
recording = recording.flatten()  # Flatten to 1D array if necessary
matched_filter_output = correlate(recording, chirp_sig, mode='same') #chaang to scipy correlzte

peaks, properties = find_peaks(matched_filter_output, distance=sample_rate*chirp_duration-1000, height=1e5)
heights = properties['peak_heights']

plt.plot(matched_filter_output)
plt.plot(peaks, heights)
plt.show()

chirp_fft = fft(chirp_sig)
n = int(sample_rate*chirp_duration/2)
channels = np.ndarray((chirp_count, 2*n), "complex")

for i, peak in enumerate(peaks[:1]):
    detected_chirp = recording[peak-n:peak+n]
    detected_fft = fft(detected_chirp)
    channel_fft = detected_fft/chirp_fft
    channel = ifft(channel_fft)
    channels[i] = channel

plt.plot(abs(channels[0]))
plt.show()

def impulse_start_1(channel_impulse):    # Function that takes a recorded chirp section and finds where it jumps to start (not working)   
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

def impulse_start_2(channel_impulse):
    impulse_start = np.argmax(channel_impulse)

    if impulse_start > len(channel_impulse) / 2:
        impulse_start = impulse_start - len(channel_impulse)

    return int(impulse_start)

impulse_starts = []

for i in range(len(channels)):  # Finds the index at which we must shift for each recorded channel.
    impulse_starts.append(impulse_start_1(channels[i]))

print(impulse_starts)

for i, peak in enumerate(peaks[:1]):
    detected_chirp = recording[peak-n+impulse_starts[i]:peak+n+impulse_starts[i]]
    detected_fft = fft(detected_chirp)
    channel_fft = detected_fft/chirp_fft
    channel = ifft(channel_fft)
    channels[i] = channel

plt.plot(abs(channels[0]))
plt.show()

average_channel_impulse = np.mean(channels, axis=0)

channel_power = average_channel_impulse * np.conjugate(average_channel_impulse)
cum_power = np.cumsum(channel_power)
total_power = cum_power[-1]
for i, power in enumerate(cum_power):
    if power > 0.9*total_power:
        ninety_index = i
        break

plt.plot(average_channel_impulse)
plt.axvline(ninety_index)
plt.title("Averaged channel")
plt.show()

plt.plot(np.abs(average_channel_impulse))
plt.axvline(ninety_index)
plt.title("Averaged absolute channel")
plt.show()




# Find the maximum value in the matched filter output
# max_value = np.max(matched_filter_output)
# detected_index = np.argmax(matched_filter_output)
# detected_time = detected_index / sample_rate

# # Print detection results
# if max_value > threshold:
#     print(f"Detected signal at time: {detected_time:.2f} seconds with correlation value: {max_value:.2f}")
# else:
#     print("Signal not detected")

# Plot the results
# plt.figure(figsize=(12, 8))

# plt.subplot(3, 1, 1)
# plt.title(f"{chirp_type} Chirp Signal")
# plt.plot(t_chirp, chirp_sig)
# # plt.xlim(0, chirp_sig_duration)

# plt.subplot(3, 1, 2)
# plt.title("Recorded Audio Signal")
# plt.plot(t_total, recording)
# plt.axvline(x=detected_time, color='r', linestyle='--', label='Detected Midpoint')
# plt.axvline(x=detected_time - chirp_duration/2, color='r', linestyle='--', label='Detected Start Time')
# plt.axvline(x=detected_time + chirp_duration/2, color='r', linestyle='--', label='Detected End Time')
# # plt.xlim(0, duration)

# plt.subplot(3, 1, 3)
# plt.title("Matched Filter Output")
# plt.plot(t_total, matched_filter_output)
# plt.axvline(x=detected_time, color='r', linestyle='--', label='Detected Midpoint')
# plt.axvline(x=detected_time - chirp_duration/2, color='r', linestyle='--', label='Detected Start Time')
# plt.axvline(x=detected_time + chirp_duration/2, color='r', linestyle='--', label='Detected End Time')
# plt.legend()
# # plt.xlim(0, duration)

# plt.tight_layout()
# plt.show()


# plt.plot(chirp_fft)
# n = int(sample_rate*chirp_duration/2)
# # print(n)
# detected_chirp = recording[detected_index-n:detected_index+n]
# detected_fft = fft(detected_chirp)
# channel_fft = detected_fft/chirp_fft
# plt.plot(np.arange(0,1,1/(2*n)), np.abs(channel_fft))
# plt.show()

# plt.plot(np.arange(0,chirp_duration,1/(sample_rate)), np.abs(ifft(channel_fft)))
# # # file_name = f'{chirp_type}_f0_{start_freq}_f1_{end_freq}_time_{chirp_duration}_test_{test_num}_channel'
# # # plt.savefig(f"Sophie_testing/{file_name}")
# plt.show()

# with open("rec.txt", "+w") as f:
#     f.write(str(list(recording)))

# count = 2500
# start = int(detected_index + chirp_duration*sample_rate/2 + 0.5*sample_rate)
# test = recording[start-count:start+count]
# plt.plot(test)
# plt.axvline(x=count, color='r', linestyle='--', label='Detected End Time')

# file_name = f'{chirp_type}_f0_{start_freq}_f1_{end_freq}_time_{chirp_duration}_test_{test_num}_zoom_in'
# plt.savefig(f"Sophie_testing/{file_name}")

# plt.show()

# index1 = detected_index + n + prefix_length
# detected_tone = recording[index1:index1+2*n]
# plt.plot(detected_tone)
# plt.title("detected tone")
# plt.show()

# detected_tone_fft = fft(detected_tone)
# tone_fft = detected_tone_fft/channel_fft
# plt.plot(np.abs(tone_fft))
# plt.title("recovered tone fft")
# plt.show()

# recovered_tone = ifft(tone_fft)
# plt.plot(np.abs(recovered_tone))
# plt.title("recovered tone")
# plt.show()

