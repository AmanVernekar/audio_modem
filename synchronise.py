

channel_impulse = ifft(channel_fft)
plt.plot(np.abs(channel_impulse))

channel_impulse_max = np.max(channel_impulse)
channel_impulse_10_percent = 0.1 * channel_impulse_max
channel_impulse_90_percent = 0.9 * channel_impulse_max

channel_impulse_start = None

# Iterate through the array to find the transition point
for i in range(len(channel_impulse) - 1):
    if channel_impulse[i] < channel_impulse_10_percent and channel_impulse[i + 1] > channel_impulse_90_percent:
        transition_index = i + 1
        break

if transition_index is not None:
    print(f"The transition occurs at index {transition_index}.")
else:
    print("No transition point found.")

if transition_index > len(channel_impulse) / 2: 
    transition_index = transition_index - len(channel_impulse)


synchronised_chirp = recording[detected_index-n-transition_index:detected_index+n-transition_index]