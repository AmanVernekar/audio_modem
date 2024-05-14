import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate, chirp
from numpy.fft import fft

sample_rate = 44100  # samples per second
duration = 7
chirp_duration = 2
threshold = 100000

t_total = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
t_chirp = np.linspace(0, chirp_duration, int(sample_rate * chirp_duration), endpoint=False)

chirp_sig = chirp(t_chirp, f0=0.1, f1=sample_rate/2, t1=chirp_duration, method='linear')
chirp_sig = list(chirp_sig)

recording = sd.rec(sample_rate*duration, samplerate=sample_rate, channels=1, dtype='int16')
sd.wait()
output_file = 'recording.wav'

sd.play(recording, samplerate=sample_rate)
sd.wait()
sf.write(output_file, recording, sample_rate)


# Apply the matched filter
recording = recording.flatten()  # Flatten to 1D array if necessary
matched_filter_output = correlate(recording, chirp_sig, mode='same') #chaang to scipy correlzte

# Find the maximum value in the matched filter output
max_value = np.max(matched_filter_output)
detected_index = np.argmax(matched_filter_output)
detected_time = detected_index / sample_rate

# Print detection results
if max_value > threshold:
    print(f"Detected signal at time: {detected_time:.2f} seconds with correlation value: {max_value:.2f}")
else:
    print("Signal not detected")

# Plot the results
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.title("Logarithmic Chirp Signal")
plt.plot(t_chirp, chirp_sig)
# plt.xlim(0, chirp_sig_duration)

plt.subplot(3, 1, 2)
plt.title("Recorded Audio Signal")
plt.plot(t_total, recording)
plt.axvline(x=detected_time, color='r', linestyle='--', label='Detected Midpoint')
plt.axvline(x=detected_time - chirp_duration/2, color='r', linestyle='--', label='Detected Start Time')
plt.axvline(x=detected_time + chirp_duration/2, color='r', linestyle='--', label='Detected End Time')
# plt.xlim(0, duration)

plt.subplot(3, 1, 3)
plt.title("Matched Filter Output")
plt.plot(t_total, matched_filter_output)
plt.axvline(x=detected_time, color='r', linestyle='--', label='Detected Midpoint')
plt.axvline(x=detected_time - chirp_duration/2, color='r', linestyle='--', label='Detected Start Time')
plt.axvline(x=detected_time + chirp_duration/2, color='r', linestyle='--', label='Detected End Time')
plt.legend()
# plt.xlim(0, duration)

plt.tight_layout()
plt.show()


chirp_fft = fft(chirp_sig)
detected_chirp = recording[detected_index-sample_rate:detected_index+sample_rate]
detected_fft = fft(detected_chirp)
channel = detected_fft/chirp_fft
plt.plot(np.abs(channel))
plt.show()

with open("rec.txt", "+w") as f:
    f.write(str(list(recording)))

count = 1000
start = int(detected_index + chirp_duration*sample_rate/2)
test = recording[start-count:start+count]
plt.plot(test)
plt.axvline(x=count, color='r', linestyle='--', label='Detected End Time')
plt.show()
