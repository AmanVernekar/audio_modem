import numpy as np

channel = np.genfromtxt('dataset/channel.csv')

# files = []
# for i in range(1, 10):
#     files.append(np.genfromtxt(f'dataset/file{i}.csv'))
#     print(files[i-1][0])
#     print((files[i-1].shape[0])/1056.0)

block_len = 1024
prefix_len = 32
symbol_len = block_len + prefix_len
channel_len = len(channel)

f1 = np.genfromtxt('dataset/file1.csv')
num_symbols = int(len(f1)/symbol_len)
f1 = np.array(np.array_split(f1, num_symbols))[:, 32:]
# f1 = np.array(f1[32:])
# print(f1[0])
print(f1.shape)

f1 = np.fft.fft(f1)
channel = np.fft.fft(np.concatenate((channel, [0]*(block_len - channel_len))))
f1 = f1/channel
f1 = f1[:, 1:512]
print(f1.shape)
print(f1[0])