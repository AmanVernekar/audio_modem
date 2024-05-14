# import numpy as np
# import sounddevice as sd
# from scipy.signal import correlate
# import matplotlib.pyplot as plt

# # Parameters
# sample_rate = 44100  # samples per second
# duration = 1.0  # duration of the template in seconds
# frequency = 440.0  # Hz (A4 note)
# threshold = 0.8  # threshold for detection

# # Generate the template sound wave to detect
# t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
# template = 0.5 * np.sin(2 * np.pi * frequency * t)

# # Callback function to process audio in chunks
# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status)
#     global template, sample_rate, threshold
    
#     # Flatten the input data
#     audio_data = indata[:, 0]

#     # Apply the matched filter
#     matched_filter_output = correlate(audio_data, template, mode='same')

#     # Find the maximum value in the matched filter output
#     max_value = np.max(matched_filter_output)
#     if max_value > threshold:
#         detected_index = np.argmax(matched_filter_output)
#         detected_time = detected_index / sample_rate
#         print(f"Detected signal at time: {detected_time:.2f} seconds with correlation value: {max_value:.2f}")

#     # Plot the results for visualization (optional)
#     plt.clf()
#     plt.subplot(2, 1, 1)
#     plt.title("Audio Data")
#     plt.plot(audio_data)
#     plt.subplot(2, 1, 2)
#     plt.title("Matched Filter Output")
#     plt.plot(matched_filter_output)
#     plt.pause(0.01)

# # Stream audio from the microphone
# with sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate):
#     print("Listening for the signal...")
#     plt.figure()
#     plt.show(block=True)


import sounddevice as sd

fs = 44100
numsamples = fs*3

recording = sd.rec(numsamples, samplerate=fs, channels=1, dtype='int16')
sd.wait()
print(recording)

sd.play(recording, samplerate=fs)
sd.wait()