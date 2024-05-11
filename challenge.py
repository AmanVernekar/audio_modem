import numpy as np  

channel = np.genfromtxt('dataset/channel.csv')  # Imports the channel file as an array. 

# files = []
# for i in range(1, 10):
#     files.append(np.genfromtxt(f'dataset/file{i}.csv'))
#     print(files[i-1][0])
#     print((files[i-1].shape[0])/1056.0)

block_len = 1024
prefix_len = 32
symbol_len = block_len + prefix_len
channel_len = len(channel)

f1 = np.genfromtxt('dataset/file1.csv') # Import the file1 data into an array. 
num_symbols = int(len(f1)/symbol_len)  # Number of symbols 
f1 = np.array(np.array_split(f1, num_symbols))[:, 32:]   # Splits into symbols sent by the channel (length 1056) and removes first 32. 
print(f1.shape)

f1 = np.fft.fft(f1)  # Does the fft of all symbols individually 
channel = np.fft.fft(np.concatenate((channel, [0]*(block_len - channel_len))))   # Zero pads the end of the channel pulse and takes fft. 
f1 = f1/channel  # Divide each value by its corrosponding channel fft coefficient. 
f1 = f1[:, 1:512] # Selects the values from 1 to 511
#print(f1.shape)
#print(f1[0])