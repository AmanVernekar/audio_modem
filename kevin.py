import numpy as np
from scipy.signal import chirp
import sounddevice as sd

# Tone Parameters
tone_duration = 1.0  # seconds
tone_frequency = 800.0  # Hz 
sample_rate = 44100  # samples per second

# Generate a tone to check our chirp detection is correct. 
t_tone = np.linspace(0, tone_duration, int(sample_rate * tone_duration), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * tone_frequency * t_tone)


# Chirp Parameters
chirp_duration = 2  # seconds
start_sig = [0]*sample_rate  # 1 second silence
gap = [0]*0.5*sample_rate  # 0.5s gap between chirp and tone

t = np.linspace(0, chirp_duration, chirp_duration*sample_rate)  # time-values for chirp
chirp_sig = chirp(t, f0=0.1, f1=22050, t1=chirp_duration, method='linear')
chirp_sig = list(chirp_sig) 
chirp_prefix = chirp_sig[-15000:]
chirp_w_prefix = chirp_prefix + chirp_sig
#chirp_w_prefix.extend(chirp_w_prefix)  # Generates a second chirp 

overall_sig = start_sig + chirp_w_prefix + gap + tone  # Adds the 1s of nothing at the start to the chirp with the prefix then 
                                                       # a small gap and then the tone. 


# Play the audio data
sd.play(overall_sig, sample_rate)
sd.wait()  # Wait until the sound has finished playing
