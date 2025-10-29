from scipy.signal import iirnotch, filtfilt
import numpy as np

def remove_line_noise(X, Fs, baseHz=60, Q=35):
    if X.shape[0] < X.shape[1]:
        X = X.T
    X_clean = X.copy()
    nyq = Fs / 2
    for f0 in np.arange(baseHz, nyq, baseHz):
        w0 = f0 / (Fs / 2)
        b, a = iirnotch(w0, w0 / Q)
        X_clean = filtfilt(b, a, X_clean, axis=0)
    return X_clean
