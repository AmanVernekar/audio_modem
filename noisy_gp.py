import numpy as np
import matplotlib.pyplot as plt

# Parameters
start = 1  # starting value of the sequence
r = 0.99      # common ratio (should be between 0 and 1 for a decreasing sequence)
length = 512  # length of the sequence

# Generate the sequence
geometric_sequence = start * r**np.arange(length)

# Noise parameters
noise_mean = 0
noise_std = 0.05  # standard deviation of the noise

# Generate noise
noise = np.random.normal(noise_mean, noise_std, length)

# Add noise to the geometric sequence
noisy_sequence = geometric_sequence + noise

# Plot the sequences
plt.figure(figsize=(10, 6))
plt.plot(geometric_sequence, label='Original Geometric Sequence')
plt.plot(noisy_sequence)#, label='Noisy Sequence', linestyle='dashed')
plt.legend()
plt.title('Decreasing Geometric Sequence with Noise')
plt.xlabel('Index')
plt.ylabel('Value')
plt.show()
