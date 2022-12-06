import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# load z data
file = './Encoding/VAE/result/0902_1619_392_train_z.csv'
# file = '../../dataset/EEG Eye State.arff.csv'

data = pd.read_csv(file, header=None)
data = pd.DataFrame.to_numpy(data)
data = np.array(data[:, :-1], dtype=np.float64)
label = np.array(data[:, -1], dtype=np.int64)

colors = ['red', 'blue']

fig = plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], c=label, cmap=matplotlib.colors.ListedColormap(colors))

cb = plt.colorbar()
loc = np.arange(0, max(label), max(label) / float(len(colors)))
cb.set_ticks(loc)
cb.set_ticklabels(colors)