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

def wavedec(data, wavelet, mode='symmetric', level=1, axis=-1):
    """
    Multiple level 1-D discrete fast wavelet decomposition
    Calling Sequence
    ----------------
    [C, L] = wavedec(data, wavelet, mode, level, axis)
    [C, L] = wavedec(data, wavelet)
    [C, L] = wavedec(data, 'sym3')
    Parameters
    ----------
    data: array_like
        Input data
    wavelet : Wavelet object or name string
        Wavelet to use
    mode : str, optional
        Signal extension mode, see Modes (default: 'symmetric')
    level : int, optional
        Decomposition level (must be >= 0). Default is 1.
    axis: int, optional
        Axis over which to compute the DWT. If not given, the
        last axis is used.
    Returns
    -------
    C: list
        Ordered list of flattened coefficients arrays (N=level):
        C = [app. coef.(N)|det. coef.(N)|... |det. coef.(1)]
    L: list
        Ordered list of individual lengths of coefficients arrays.
        L(1)   = length of app. coef.(N)
        L(i)   = length of det. coef.(N-i+2) for i = 2,...,N+1
        L(N+2) = length(X).
    Description
    -----------
    wavedec can be used for multiple-level 1-D discrete fast wavelet
    decomposition using a specific wavelet name or instance of the
    Wavelet class instance.
    The coefficient vector C contains the approximation coefficient at level N
    and all detail coefficient from level 1 to N
    The first entry of L is the length of the approximation coefficient,
    then the length of the detail coefficients are stored and the last
    value of L is the length of the signal vector.
    The approximation coefficient can be extracted with C(1:L(1)).
    The detail coefficients can be obtained with C(L(1):sum(L(1:2))),
    C(sum(L(1:2)):sum(L(1:3))),.... until C(sum(L(1:length(L)-2)):sum(L(1:length(L)-1)))
    The implementation of the function is based on pywt.wavedec
    with the following minor changes:
        - checking of the axis is dropped out
        - checking of the maximum possible level is dropped out
          (as for Matlab's implementation)
        - returns format is modified to Matlab's internal format:
          two separate lists of details coefficients and
          corresponding lengths
    Examples
    --------
    # >>> C, L = wavedec([3, 7, 1, 1, -2, 5, 4, 6], 'sym3', level=2)
    # >>> C
    array([  7.38237875   5.36487594   8.83289608   2.21549896  11.10312807
            -0.42770133   3.72423411   0.48210099   1.06367045  -5.0083641
            -2.11206142  -2.64704675  -3.16825651  -0.67715519   0.56811154
             2.70377533])
    # >>> L
    array([5, 5, 6, 8])
    """
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
    """
    1-D detail coefficients extraction
    Calling Sequence
    ----------------
    D = detcoef(C, L)
    D = detcoef(C, L, N)
    D = detcoef(C, L, [1, 2, 3])
    Parameters
    ----------
    coefs: list
        Ordered list of flattened coefficients arrays (N=level):
        C = [app. coef.(N)|det. coef.(N)|... |det. coef.(1)]
    lengths: list
        Ordered list of individual lengths of coefficients arrays.
        L(1)   = length of app. coef.(N)
        L(i)   = length of det. coef.(N-i+2) for i = 2,...,N+1
        L(N+2) = length(X).
    levels : int or list
        restruction level with N<=length(L)-2
    Returns
    ----------
    D : reconstructed detail coefficient
    Description
    -----------
    detcoef is for extraction of detail coefficient at different level
    after a multiple level decomposition. If levels is omitted,
    the detail coefficients will extract at all levels.
    The wavelet coefficients and lengths can be generated using wavedec.
    Examples
    --------
    # >>> x = range(100)
    # >>> w = pywt.Wavelet('sym3')
    # >>> C, L = wavedec(x, wavelet=w, level=5)
    # >>> D = detcoef(C, L, levels=len(L)-2)
    """
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
    """
    Restruction from single branch from multiple level decomposition
    Calling Sequence
    ----------------
    X = wrcoef(C, L, wavelet, level)
    Parameters
    ----------
    # type='a' is not implemented.
    # type : string
    #   approximation or detail, 'a' or 'd'.
    coefs: list
        Ordered list of flattened coefficients arrays (N=level):
        C = [app. coef.(N)|det. coef.(N)|... |det. coef.(1)]
    lengths: list
        Ordered list of individual lengths of coefficients arrays.
        L(1)   = length of app. coef.(N)
        L(i)   = length of det. coef.(N-i+2) for i = 2,...,N+1
        L(N+2) = length(X).
    wavelet : Wavelet object or name string
        Wavelet to use
    level : int
        restruction level with level<=length(L)-2
    Returns
    ----------
    X :
        vector of reconstructed coefficients
    Description
    -----------
    wrcoef is for reconstruction from single branch of multiple level
    decomposition from 1-D wavelet coefficients.
    The wavelet coefficients and lengths can be generated using wavedec.
    Examples
    --------
    # >>> x = range(100)
    # >>> w = pywt.Wavelet('sym3')
    # >>> C, L = wavedec(x, wavelet=w, level=5)
    # >>> X = wrcoef(C, L, wavelet=w, level=len(L)-2)
    """

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


"""
# wavelet trasform
wavelet_function = 'db8'
wavelet_level = 8
w = pywt.Wavelet(wavelet_function)

C, L = wavedec(trainEEGData[0], wavelet=w, level=wavelet_level)

D1 = wrcoef(C, L, wavelet=w, level=1)
D2 = wrcoef(C, L, wavelet=w, level=2)
D3 = wrcoef(C, L, wavelet=w, level=3)
D4 = wrcoef(C, L, wavelet=w, level=4)
D5 = wrcoef(C, L, wavelet=w, level=5)
D6 = wrcoef(C, L, wavelet=w, level=6)
D7 = wrcoef(C, L, wavelet=w, level=7)
D8 = wrcoef(C, L, wavelet=w, level=8)
A8 = wrcoef(C, L, wavelet=w, level=8)

x = list(range(0, 512, 1))
plt.plot(D1)
"""

"""
# fast fourier transform
fft_vals = np.absolute(np.fft.rfft(trainEEGData[0]))
fft_freq = np.fft.rfftfreq(len(trainEEGData[0]), 1.0/hz)

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
ax = df.plot.bar(x='band', y='val', legend=False)
ax.set_xlabel("EEG band")
ax.set_ylabel("Mean band Amplitude")
plt.show()
"""

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # print(b,a)
    y = lfilter(b, a, data)
    return y


fileName = './EEG Epilepsy/EEG_Epilepsy_random.csv'

file_alpha = 'EEG_Epilepsy_random_alpha.csv'
file_beta = 'EEG_Epilepsy_random_beta.csv'
file_gamma = 'EEG_Epilepsy_random_gamma.csv'

Alpha = open(file_alpha, 'w', newline='')
Beta = open(file_beta, 'w', newline='')
Gamma = open(file_gamma, 'w', newline='')
wAlpha = csv.writer(Alpha)
wBeta = csv.writer(Beta)
wGamma = csv.writer(Gamma)

t_data = pd.read_csv(fileName, header=None)  # data 도중 length -1 문제 발생 : header = 0 확인!
data = pd.DataFrame.to_numpy(t_data)
eeg_data = data[:, :-1]
label = data[:, -1].tolist()

"""
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
# eeg to freq band (spectrogram)
eeg = eeg_data[0, :]
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
# logpower = 10*np.log10(power)


# eeg spectrogram
fig, ax = plt.subplots(2, 1, figsize=(16, 8), constrained_layout=True)
# fig.suptitle('Time-frequency power via short-time FFT')

ax[0].plot(time, eeg, lw=1, color='C0')
ax[0].set_ylabel('Amplitude ($\mu V$)')
ax[0].set_title('EEG signal')

# spectrum is a ContourSet object
dt = 120 / nsteps  # 120 seconds in number of steps
X = np.arange(nsteps) * dt
Y = hz[:int(45 / dfreq)]
Z = np.array(myamp).T
levels = 45
spectrum = ax[1].contourf(X, Y, Z, levels, cmap='jet')  # ,'linecolor','none')

# get the colormap
cbar = plt.colorbar(spectrum)  # , boundaries=np.linspace(0,1,5))
cbar.ax.set_ylabel('Amplitude ($\mu$V)', rotation=90)
cbar.set_ticks(np.arange(0, 50, 10))

# A working example (for any value range) with five ticks along the bar is:

m0 = int(np.floor(np.min(myamp)))  # colorbar min value
m4 = int(np.ceil(np.max(myamp)))  # colorbar max value
m1 = int(1 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 1
m2 = int(2 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 2
m3 = int(3 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 3
cbar.set_ticks([m0, m1, m2, m3, m4])
cbar.set_ticklabels([m0, m1, m2, m3, m4])

# cbar.set_ticks(np.arange(0, 1.1, 0.5))

ax[1].axhline(y=8, linestyle='--', linewidth=1.5, color='white')
ax[1].axhline(y=12, linestyle='--', linewidth=1.5, color='white')
ax[1].set_ylim([0, 40])
ax[1].set_yticks(np.arange(0, 45, 5))
ax[1].set_ylabel('Frequency (Hz)')

for myax in ax:
    myax.set_xlim(0, 120)
    myax.set_xticks(np.arange(0, 121, 30))
    myax.set_xlabel('Time (sec.)')
    
"""
# fig, ax = plt.subplots(figsize=(10, 10)) #, constrained_layout=True)
# ax.grid(False)
# ax.set_xticks([])
# ax.set_yticks([])
# # ax.set_zticks([])
#
# # ax[0].plot(time, eeg, lw=1, color='C0')
# # ax[0].set_ylabel('Amplitude ($\mu V$)')
# # ax[0].set_title('EEG signal')
#
# # myparams = dict(nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')
# f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')
#
# # spectrum is a ContourSet object
# dt = 120 / nsteps  # 120 seconds in number of steps
# X = nseg
# Y = f
# Z = Sxx
# levels = 45
# spectrum = ax.contourf(X, Y, Z, levels, cmap='jet')  # ,'linecolor','none')
#
# """
# # get the colormap
# cbar = plt.colorbar(spectrum)  # , boundaries=np.linspace(0,1,5))
# cbar.ax.set_ylabel('Amplitude ($\mu$V)', rotation=90)
# cbar.set_ticks(np.arange(0, 50, 10))
#
# # A working example (for any value range) with five ticks along the bar is:
# m0 = int(np.floor(np.min(Sxx)))  # colorbar min value
# m4 = int(np.ceil(np.max(Sxx)))  # colorbar max value
# m1 = int(1 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 1
# m2 = int(2 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 2
# m3 = int(3 * (m4 - m0) / 4.0 + m0)  # colorbar mid value 3
# cbar.set_ticks([m0, m1, m2, m3, m4])
# cbar.set_ticklabels([m0, m1, m2, m3, m4])
# """
# # cbar.set_ticks(np.arange(0, 1.1, 0.5))
#
# # ax[1].axhline(y=8, linestyle='--', linewidth=1.5, color='white')
# # ax[1].axhline(y=12, linestyle='--', linewidth=1.5, color='white')
# # ax.set_ylim([0, 40])
# # ax.set_yticks(np.arange(0, 45, 5))
# # ax.set_ylabel('Frequency (Hz)')
# # ax.set_xlabel('Time (sec.)')
#
# # plt.show()
# plt.savefig('image1.png', bbox_inches='tight', pad_inches=0.0)

"""
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

    f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude') # relative

    if lab == 1:
        fft_ictal += Sxx
    elif lab == 2:
        fft_interictal += Sxx
    elif lab == 3:
        fft_preictal += Sxx

fft_ictal = fft_ictal/450
fft_interictal = fft_interictal/450
fft_preictal = fft_preictal/450
master_f = 0

for j in range(len(fft_preictal[0])):
    master_f += fft_preictal[:, j]

master_f = master_f/len(fft_preictal[0])

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

    # myparams = dict(nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')
    # f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude') # original

    """
    # spectrum is a ContourSet object
    dt = 120 / nsteps  # 120 seconds in number of steps
    X = nseg
    Y = f
    Z = Sxx
    levels = 45
    # spectrum = ax.contourf(X, Y, Z, levels, cmap='jet')  # ,'linecolor','none')
    spectrum = ax.contourf(X, Y, Z, levels, cmap='jet')
    """
    # master function subtraction J
    f, nseg, Sxx = spectrogram(eeg, fs=sr, nperseg=WinLength, noverlap=WinLength - step, return_onesided=True, mode='magnitude')
    
    X = nseg
    Y = f

    # for j in range(len(Sxx[0])):
    #     Sxx[:, j] -= master_f
    # Z = Sxx

    Z = Sxx - fft_preictal

    # if lab == 1:
    #     Z = Sxx - fft_ictal
    # elif lab == 2:
    #     Z = Sxx - fft_interictal
    # elif lab == 3:
    #     Z = Sxx - fft_preictal

    levels = 45
    spectrum = ax.contourf(X, Y, Z, levels, cmap='jet')

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