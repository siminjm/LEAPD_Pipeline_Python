import numpy as np

def detect_noisy_channels(X, labels, thresh=4):
    ch_std = np.std(X, axis=0)
    z = (ch_std - np.mean(ch_std)) / np.std(ch_std)
    bad_idx = np.where(np.abs(z) > thresh)[0]
    bad_labels = [labels[i] for i in bad_idx]
    return bad_idx, bad_labels, ch_std
