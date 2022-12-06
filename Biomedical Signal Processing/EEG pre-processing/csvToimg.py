import os
import sys
import csv
import numpy as np
import pandas as pd
import torch
import pywt
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram

import os
import csv
import cv2
import math
import torch
import numpy as np
import pandas as pd
import pywt
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, spectrogram

transpose = False
ch = 1
sr = 200
sec = 1
moving_rate = 0.5

dataPath = './EEG Epilepsy/img/'
fileName = './EEG Epilepsy/EEG_Epilepsy_random.csv'
cls = ['ictal', 'interictal', 'preictal']

for c in cls:
    os.makedirs(dataPath + c + '/', exist_ok=True)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y


t_data = pd.read_csv(fileName, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
data = pd.DataFrame.to_numpy(t_data)
eeg_data = data[:, :-1]
label_data = data[:, -1].tolist()

idx = 1
for eeg, label in zip(eeg_data, label_data):
    time = np.arange(eeg.size) / sr
    WinLength = int(0.5 * sr)
    step = int(0.025 * sr)
    Nsamples = int(np.floor(WinLength / 2))
    hz = np.linspace(0, sr // 2, Nsamples + 1)
    dfreq = hz[1]
    nsteps = int(np.floor((eeg.size - WinLength) / step))
    myamp = list()
    for i in range(nsteps):
        # signal duration 500 ms (512 data points)
        data = eeg[i * step:i * step + WinLength]

        FourierCoeff = np.fft.fft(data) / WinLength
        DC = [np.abs(FourierCoeff[0])]  # DC component
        amp = np.concatenate((DC, 2 * np.abs(FourierCoeff[1:])))

        amp = amp[:int(45 / dfreq)]
        myamp.append(amp)

    power = np.power(myamp, 2)

    fig, ax = plt.subplots(figsize=(10, 10))  # , constrained_layout=True)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # myparams = dict(nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')
    f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True,
                               mode='magnitude')

    # spectrum is a ContourSet object
    dt = 120 / nsteps  # 120 seconds in number of steps
    X = nseg
    Y = f
    Z = Sxx
    levels = 45
    spectrum = ax.contourf(X, Y, Z, levels, cmap='jet')  # ,'linecolor','none')
    imgName = 'eeg_spec' + str(idx) + '.png'

    if label == 1:
        plt.savefig(dataPath + cls[0] + '/' + imgName, bbox_inches='tight', pad_inches=0.0)
    elif label == 2:
        plt.savefig(dataPath + cls[1] + '/' + imgName, bbox_inches='tight', pad_inches=0.0)
    elif label == 3:
        plt.savefig(dataPath + cls[2] + '/' + imgName, bbox_inches='tight', pad_inches=0.0)

    idx += 1
    plt.close(fig)
    plt.clf()