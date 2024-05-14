import numpy as np
from scipy.signal import chirp
import sounddevice as sd

# Tone Parameters
tone_duration = 1.0  # seconds
tone_frequency = 1000.0  # Hz 
sample_rate = 44100  # samples per second

# Generate a tone to check our chirp detection is correct. 
t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * tone_frequency * t_tone)


# Chirp Parameters
chirp_t = 2
start_sig = [0]*sample_rate

t = np.linspace(0, chirp_t, chirp_t*sample_rate)
chirp_sig = chirp(t, f0=0.1, f1=22050, t1=chirp_t, method='linear')
chirp_sig = list(chirp_sig)
chirp_sig.extend(chirp_sig)  # Generates a second chirp 

overall_sig = start_sig.extend(chirp_sig)  # Adds the 1s of nothing at the start to the double chirp
overall_sig.extend(tone)  # Adds the tone to the end of the signal so we can check if our detection is correct. 

# Play the audio data
sd.play(overall_sig, sample_rate)
sd.wait()  # Wait until the sound has finished playing
