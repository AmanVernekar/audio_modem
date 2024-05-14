import numpy as np
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import correlate
from scipy.signal import chirp

sample_rate = 44100  # samples per second
duration = 7
total_t = 2

# Generate a NumPy array with the audio data (sine wave)
t1 = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
t2 = np.linspace(0, total_t, int(sample_rate * total_t), endpoint=False)

# audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
# audio_data = np.random.normal(-1, 1, int(sample_rate * duration))


# chirp_sig = [0]*sample_rate

# t = np.linspace(0, total_t, total_t*sample_rate)
chirp_sig = chirp(t2, f0=0.1, f1=22050, t1=total_t, method='linear')
# print(len(chirp_sig)/sample_rate)
chirp_sig = list(chirp_sig)
# chirp_sig.extend(chirp_sig)
# print(len(chirp_sig)/sample_rate)

# chirp_sig.extend(chirp_sig)





fs = 44100
numsamples = fs*duration

recording = sd.rec(numsamples, samplerate=fs, channels=1, dtype='int16')
sd.wait()
# print(recording)

output_file = 'yo.wav'

sd.play(recording, samplerate=fs)
sd.wait()
sf.write(output_file, recording, sample_rate)
# print(len(recording))

threshold = 100000

# Apply the matched filter
recording = recording.flatten()  # Flatten to 1D array if necessary
matched_filter_output = correlate(recording, chirp_sig, mode='same')

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
plt.title("chirp_sig Signal")
plt.plot(t2, chirp_sig)
# plt.xlim(0, chirp_sig_duration)

plt.subplot(3, 1, 2)
plt.title("Recorded Audio Signal")
plt.plot(t1, recording)
# plt.xlim(0, duration)

plt.subplot(3, 1, 3)
plt.title("Matched Filter Output")
plt.plot(t1, matched_filter_output)
plt.axvline(x=detected_time, color='r', linestyle='--', label='Detected Time')
plt.legend()
# plt.xlim(0, duration)

plt.tight_layout()
plt.show()
