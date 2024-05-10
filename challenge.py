import numpy as np

channel = np.recfromcsv('dataset/channel.csv')

files = []
for i in range(1, 10):
    files.append(np.recfromcsv(f'dataset/file{i}.csv'))
    print(files[i-1].shape)
