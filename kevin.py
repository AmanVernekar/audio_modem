import numpy as np
from scipy.signal import chirp
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

# Tone Parameters
tone_duration = 1.0  # seconds
tone_frequency = 800.0  # Hz 
sample_rate = 44100  # samples per second
cyclic_prefix = 1024

# Generate a tone to check our chirp detection is correct. 
t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
tone = list(0.5 * np.sin(2 * np.pi * tone_frequency * t_tone))
tone_prefix = tone[-cyclic_prefix:]
tone = tone_prefix + tone 


# Chirp Parameters
chirp_duration = 2  # seconds
start_sig = [0]*sample_rate  # 1 second silence
half_s_samples = int(0.5*sample_rate) #generates half a second of samples 
# gap = [0]*half_s_samples
start_freq = 0.01
end_freq = 22050
chirp_type = "linear" 

t = np.linspace(0, chirp_duration, int(chirp_duration*sample_rate))  # time-values for chirp
chirp_sig = chirp(t, f0=start_freq, f1=end_freq, t1=chirp_duration, method=chirp_type)
chirp_sig = list(chirp_sig) 


chirp_prefix = chirp_sig[-cyclic_prefix:]
chirp_suffix = chirp_sig[:cyclic_prefix]
chirp_w_prefix_suffix = chirp_prefix + chirp_sig + chirp_suffix

waveform = list(np.load("waveform.npy"))

overall_sig = start_sig + chirp_w_prefix_suffix + waveform

# print(overall_sig)

# Play the audio data
# sd.play(overall_sig, sample_rate)
# sd.wait()  # Wait until the sound has finished playing

output_file = 'audio_to_test_with.wav'
sf.write(output_file, overall_sig, sample_rate)
