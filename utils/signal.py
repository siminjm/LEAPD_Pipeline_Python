from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch
from typing import List, Tuple

def create_filter(Fs: float, Filt: np.ndarray) -> np.ndarray:
    """Butterworth bandpass defined by Filt rows [[0, f1, 0], [f2, inf, 0]] (MATLAB-style)."""
    start_f = float(Filt[0,1])
    if (Filt.shape[0] == 1) or (int(Filt[1,2]) == 1):
        stop_f = Fs/2 - 1e-3
    else:
        stop_f = float(Filt[1,0])
    Order = 6
    Wp = [start_f/(Fs/2), stop_f/(Fs/2)]
    sos = butter(Order, Wp, btype='bandpass', output='sos')
    return sos

def filter_data(AllData, Filt: np.ndarray, Fs: float) -> list:
    """Bandpass + 60Hz notch + L2 normalize per subject."""
    sos = create_filter(Fs, Filt)
    wo = 60/(Fs/2)
    bw = wo/35
    b_notch, a_notch = iirnotch(wo, bw)
    Xf = []
    total = len(AllData) if isinstance(AllData, (list, tuple)) else int(AllData.shape[1])
    for i in range(total):
        X = AllData[i]
        X = np.asarray(X).astype(float).squeeze()
        if X.ndim > 1:
            X = X.mean(axis=1)
        X = sosfiltfilt(sos, X)
        X = sosfiltfilt(np.array([[1.0, 0.0, 0.0, 1.0]]), X) if False else X  # placeholder
        X = sosfiltfilt(np.array([[1.0, 0.0, 0.0, 1.0]]), X) if False else X  # keep same length placeholder
        # Apply notch (biquad in tf form); convert to sosfiltfilt via lfilterfilt equivalent not available -> use filtfilt? 
        # We approximate with direct filtering using lfilter via sosfiltfilt is not valid; simplify: apply notch with filtfilt-like zero-phase using np.flip
        # Practical compromise: forward-backward using lfilter not available; iirnotch returns (b,a). We'll implement simple forward-backward.
        from scipy.signal import lfilter
        X = lfilter(b_notch, a_notch, X)
        X = lfilter(b_notch, a_notch, X[::-1])[::-1]
        nrm = np.linalg.norm(X)
        if nrm > 0:
            X = X / nrm
        Xf.append(X)
    return Xf

def compute_lpc_librosa(x: np.ndarray, order: int) -> np.ndarray:
    """Compute LPC coefficients using librosa.lpc (returns a[1:])."""
    try:
        import librosa
    except Exception as e:
        raise RuntimeError("librosa is required for LPC (install via requirements.txt)") from e
    a = librosa.lpc(x.astype(float), order)
    # librosa returns [1, -a1, -a2, ...] depending on convention; align to MATLAB arburg sign where coefficients are [1 a2 a3 ...]
    # We'll return a[1:] with sign flipped to match MATLAB's 'a(2:end)'
    return -a[1:]


def compute_lpc_burg(x, order: int):
    r"""
    Pure Burg algorithm (MATLAB arburg-style) returning a[1:] coefficients.
    Reference: standard Burg recursion; matches sign convention used by MATLAB's arburg,
    where arburg returns [1, a2, a3, ...]. We return a[1:] to match your MATLAB usage.
    r"""
    import numpy as np
    x = np.asarray(x, dtype=float).reshape(-1)
    N = x.size
    if order >= N:
        raise ValueError("order must be < number of samples")
    # Initial forward/backward errors
    ef = x[1:].copy()
    eb = x[:-1].copy()
    # AR coeffs (including leading 1)
    a = np.zeros(order+1, dtype=float)
    a[0] = 1.0
    E = np.dot(x, x) / N  # initial prediction error
    for m in range(1, order+1):
        # reflection coefficient
        num = -2.0 * np.dot(eb, ef)
        den = np.dot(ef, ef) + np.dot(eb, eb)
        if den <= np.finfo(float).eps:
            k = 0.0
        else:
            k = num / den
        # update AR coefficients (Levinson-like)
        a_new = a.copy()
        a_new[1:m] = a[1:m] + k * a[m-1:0:-1]
        a_new[m] = k
        a = a_new
        # update prediction error
        E *= (1.0 - k*k)
        if m < order:
            ef_new = ef[1:] + k * eb[1:]
            eb_new = eb[:-1] + k * ef[:-1]
            ef, eb = ef_new, eb_new
    # Return a[1:] (consistent with MATLAB using a(2:end))
    return a[1:]
