import numpy as np
from scipy.signal import chirp
import sounddevice as sd

# Parameters
duration = 5.0  # seconds
frequency = 22000.0  # Hz (A4 note)
sample_rate = 44100  # samples per second

# Generate a NumPy array with the audio data (sine wave)
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# audio_data = 0.5 * np.sin(2 * np.pi * frequency * t)
# audio_data = np.random.normal(-1, 1, int(sample_rate * duration))

total_t = 2

start_sig = [0]*sample_rate

t = np.linspace(0, total_t, total_t*sample_rate)
chirp_sig = chirp(t, f0=100, f1=1000, t1=total_t, method='logarithmic')
print(len(chirp_sig)/sample_rate)
chirp_sig = list(chirp_sig)
chirp_sig.extend(chirp_sig*2)
print(len(chirp_sig)/sample_rate)

start_sig.extend(chirp_sig)

# Play the audio data
sd.play(start_sig, sample_rate)
sd.wait()  # Wait until the sound has finished playing
