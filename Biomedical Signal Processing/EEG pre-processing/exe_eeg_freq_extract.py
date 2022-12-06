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

fol_path = './EEG Epilepsy/test/'

os.makedirs(fol_path, exist_ok=True)
os.makedirs(fol_path + 'ictal/', exist_ok=True)
os.makedirs(fol_path + 'interictal/', exist_ok=True)
os.makedirs(fol_path + 'preictal/', exist_ok=True)


# pywt function
def wavedec(data, wavelet, mode='symmetric', level=1, axis=-1):
    data = np.asarray(data)

    if not isinstance(wavelet, pywt.Wavelet):
        wavelet = pywt.Wavelet(wavelet)

    # Initialization
    coefs, lengths = [], []

    # Decomposition
    lengths.append(len(data))
    for i in range(level):
        data, d = pywt.dwt(data, wavelet, mode, axis)

        # Store detail and its length
        coefs.append(d)
        lengths.append(len(d))

    # Add the last approximation
    coefs.append(data)
    lengths.append(len(data))

    # Reverse (since we've appended to the end of list)
    coefs.reverse()
    lengths.reverse()

    # return np.concatenate(coefs).ravel(), lengths
    return coefs, lengths


def detcoef(coefs, lengths, levels=None):

    if not levels:
        levels = range(len(lengths) - 2)

    if not isinstance(levels, list):
        levels = [levels]

    first = np.cumsum(lengths) + 1
    first = first[-3::-1]
    last = first + lengths[-2:0:-1] - 1

    x = []
    for level in levels:
        d = coefs[first[level - 1] - 1:last[level - 1]]
        x.append(d)

    if len(x) == 1:
        x = x[0]

    return x


def wrcoef(coefs, lengths, wavelet, level):

    def upsconv(x, f, s):
        # returns an extended copy of vector x obtained by inserting zeros
        # as even-indexed elements of data: y(2k-1) = data(k), y(2k) = 0.
        y_len = 2 * len(x) + 1
        y = np.zeros(y_len)
        y[1:y_len:2] = x

        # performs the 1-D convolution of the vectors y and f
        y = np.convolve(y, f, 'full')

        # extracts the vector y from the input vector
        sy = len(y)
        d = (sy - s) / 2.0
        y = y[int(math.floor(d)):(sy - int(math.ceil(d)))]

        return y

    if not isinstance(wavelet, pywt.Wavelet):
        wavelet = pywt.Wavelet(wavelet)

    data = detcoef(coefs, lengths, level)

    idx = len(lengths) - level
    data = upsconv(data, wavelet.rec_hi, lengths[idx])
    for k in range(level - 1):
        data = upsconv(data, wavelet.rec_lo, lengths[idx + k + 1])

    return data

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # print(b,a)
    y = lfilter(b, a, data)
    return y


# execute signal analysis in frequency domain

fileName = './EEG Epilepsy/data/EEG_Epilepsy_random.csv'

t_data = pd.read_csv(fileName, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
data = pd.DataFrame.to_numpy(t_data)
eeg_data = data[:, :-1]
label = data[:, -1].tolist()

# file_save = 'EEG_Epilepsy_spect.csv'
# spec = open(file_save, 'w', newline='')
# wSpec = csv.writer(spec)

"""
file_alpha = 'EEG_Epilepsy_random_alpha.csv'
file_beta = 'EEG_Epilepsy_random_beta.csv'
file_gamma = 'EEG_Epilepsy_random_gamma.csv'

Alpha = open(file_alpha, 'w', newline='')
Beta = open(file_beta, 'w', newline='')
Gamma = open(file_gamma, 'w', newline='')
wAlpha = csv.writer(Alpha)
wBeta = csv.writer(Beta)
wGamma = csv.writer(Gamma)

# eeg to freq band (csv)
for line, idx in zip(eeg, label):
    alpha = butter_bandpass_filter(line, 8.1, 12.0, 200)
    beta = butter_bandpass_filter(line, 16.0, 36.0, 200)
    gamma = butter_bandpass_filter(line, 36.1, 80, 200)
    # delta = butter_bandpass_filter(trainEEGData[0], 0.0, 4.0, 200)
    # sigma = butter_bandpass_filter(trainEEGData[0], 12.1, 16.0, 200)
    # theta = butter_bandpass_filter(trainEEGData[0], 4.1, 8.0, 200)

    alpha.tolist().append(idx)
    beta.tolist().append(idx)
    gamma.tolist().append(idx)

    # alpha.append(idx)
    # beta.append(idx)
    # gamma.append(idx)

    wAlpha.writerow(alpha)
    wBeta.writerow(beta)
    wGamma.writerow(gamma)
"""

"""
fig1, ax1 = plt.subplots(figsize=(10, 10))
fig2, ax2 = plt.subplots(figsize=(10, 10))
fig3, ax3 = plt.subplots(figsize=(10, 10))

# fig, ax = plt.subplots(figsize=(10, 10))
# ax.set_xlabel("EEG band")
# ax.set_ylabel("Mean band Amplitude")

ax1.set_title("ictal")
ax1.set_xlabel("EEG band")
ax1.set_ylabel("Mean band Amplitude")
ax2.set_title("interictal")
ax2.set_xlabel("EEG band")
ax2.set_ylabel("Mean band Amplitude")
ax3.set_title("preictal")
ax3.set_xlabel("EEG band")
ax3.set_ylabel("Mean band Amplitude")

for eeg, lab in zip(eeg_data, label):

    # fast fourier transform
    fft_vals = np.absolute(np.fft.rfft(eeg))
    fft_freq = np.fft.rfftfreq(len(eeg), 1.0/sr)

    eeg_bands = {'Delta': (0, 4),
                 'Theta': (4, 8),
                 'Alpha': (8, 12),
                 'Beta': (12, 30),
                 'Gamma': (30, 45)}

    # Take the mean of the fft amplitude for each EEG band
    eeg_band_fft = dict()

    for band in eeg_bands:
        freq_ix = np.where((fft_freq >= eeg_bands[band][0]) &
                           (fft_freq <= eeg_bands[band][1]))[0]
        eeg_band_fft[band] = np.mean(fft_vals[freq_ix])

    df = pd.DataFrame(columns=['band', 'val'])
    df['band'] = eeg_bands.keys()
    df['val'] = [eeg_band_fft[band] for band in eeg_bands]

    if lab == 1:
        # band_ict = band_ict.append(eeg_bands.keys())
        # val_ict = val_ict.append([eeg_band_fft[band] for band in eeg_bands])
        # ax1.plot(df['band'], df['val'])
        ax1.plot(fft_freq, fft_vals, color='red')
        # ax.plot(fft_freq, fft_vals, color='red')
    elif lab == 2:
        # band_int = band_int.append(eeg_bands.keys())
        # val_int = val_int.append([eeg_band_fft[band] for band in eeg_bands])
        # ax2.plot(df['band'], df['val'])
        ax2.plot(fft_freq, fft_vals, color='green')
        # ax.plot(fft_freq, fft_vals, color='green')

    elif lab == 3:
        # band_pre = band_pre.append(eeg_bands.keys())
        # val_pre = val_pre.append([eeg_band_fft[band] for band in eeg_bands])
        # ax3.plot(df['band'], df['val'])
        ax3.plot(fft_freq, fft_vals, color='blue')
        # ax.plot(fft_freq, fft_vals, color='blue')

# fig.savefig('eeg_freq.png')

fig1.savefig('eeg_freq_ict.png')
fig2.savefig('eeg_freq_int.png')
fig3.savefig('eeg_freq_pre.png')
step = "here"
"""

"""
# eeg to freq band (spectrogram)
idx = 1
for eeg, lab in zip(eeg_data, label):
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

    if lab == 1:
        fol_path + fol_path + 'ictal/'
    elif lab == 2:
        fol_path + fol_path + 'interictal/'
    elif lab == 3:
        fol_path + fol_path + 'preictal/'

    imgName = 'eeg_spec' + str(idx) + '.png'
    plt.savefig(fol_path + imgName, bbox_inches='tight', pad_inches=0.0)
    idx += 1
    plt.close(fig)
    plt.clf()
"""

idx = 1
fft_ictal = 0
fft_preictal = 0
fft_interictal = 0
spec_min = 0
spec_max = 0

for eeg, lab in zip(eeg_data, label):
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

    f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')  # relative

    # Sxx = np.log(Sxx)

    if lab == 1:
        fft_ictal += Sxx
    elif lab == 2:
        fft_interictal += Sxx
    elif lab == 3:
        fft_preictal += Sxx

    if spec_min > np.min(Sxx):
        spec_min = np.min(Sxx)
    if spec_max < np.max(Sxx):
        spec_max = np.max(Sxx)

    # wSpec.writerows(Sxx)
    # wSpec.writerow([lab])
    # wSpec.writerow('\n')

fft_ictal = fft_ictal / 450
fft_interictal = fft_interictal / 450
fft_preictal = fft_preictal / 450
master_f = 0

for j in range(len(fft_preictal[0])):
    master_f += fft_preictal[:, j]

master_f = master_f / len(fft_preictal[0])

"""
# save avg spectrogram & master function
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_title("preictal_master")
ax.set_xlabel('frequency')
ax.set_ylabel('Mean band Amplitude')
ax.plot(master_f)
fig.savefig('master_pre.png')

fig1, ax1 = plt.subplots(figsize=(10, 10))
ax1.set_title("ictal_avg")
ax1.set_xlabel('frequency')
ax1.set_ylabel('Mean band Amplitude')
ax1.plot(fft_ictal)
fig2, ax2 = plt.subplots(figsize=(10, 10))
ax2.set_title("interictal_avg")
ax2.set_xlabel('frequency')
ax2.set_ylabel('Mean band Amplitude')
ax2.plot(fft_interictal)
fig3, ax3 = plt.subplots(figsize=(10, 10))
ax3.set_title("preictal_avg")
ax3.set_xlabel('frequency')
ax3.set_ylabel('Mean band Amplitude')
ax3.plot(fft_preictal)
fig1.savefig('eeg_ict_avg.png')
fig2.savefig('eeg_int_avg.png')
fig3.savefig('eeg_pre_avg.png')
"""

for eeg, lab in zip(eeg_data, label):
    time = np.arange(eeg.size) / sr
    WinLength = int(0.5 * sr)
    step = int(0.025 * sr)
    Nsamples = int(np.floor(WinLength / 2))
    hz = np.linspace(0, sr // 2, Nsamples + 1)
    dfreq = hz[1]

    fig, ax = plt.subplots(figsize=(10, 10))  # , constrained_layout=True)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

    """
    # original spectrum function
    myparams = dict(nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')
    f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude') # original

    # spectrum is a ContourSet object
    dt = 120 / nsteps  # 120 seconds in number of steps
    X = nseg
    Y = f
    Z = Sxx
    levels = 45
    spectrum = ax.contourf(X, Y, Z, levels, cmap='jet')
    """

    # master function subtraction J
    f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')

    X = nseg
    Y = f
    # Y = np.log(f)

    # for j in range(len(Sxx[0])):
    #     Sxx[:, j] -= master_f
    Z = Sxx
    # Z = np.log(Sxx)

    # Z = Sxx - fft_preictal

    levels = 45
    # spectrum = ax.contourf(X, Y, Z, levels, cmap='jet')
    spectrum = ax.contourf(X, Y, Z, levels, vmin=spec_min, vmax=spec_max, cmap='jet') # spec_max = 155.xx

    """
    if lab == 1:
        save_path = fol_path + 'ictal/'
    elif lab == 2:
        save_path = fol_path + 'interictal/'
    elif lab == 3:
        save_path = fol_path + 'preictal/'

    imgName = 'eeg_spec' + str(idx) + '.png'
    plt.savefig(save_path + imgName, bbox_inches='tight', pad_inches=0.0)
    idx += 1
    plt.close(fig)
    plt.clf()
    """