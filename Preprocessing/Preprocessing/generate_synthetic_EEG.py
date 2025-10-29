import numpy as np

def generate_synthetic_EEG(duration_sec=5, nCh=8, Fs=500):
    t = np.arange(0, duration_sec, 1/Fs)
    alpha = np.sin(2*np.pi*10*t)
    theta = 0.7*np.sin(2*np.pi*6*t)
    beta  = 0.3*np.sin(2*np.pi*20*t)
    sources = np.vstack((alpha, theta, beta))
    mixWeights = np.random.randn(3, nCh)
    X = sources.T @ mixWeights
    X /= np.max(np.abs(X), axis=0, keepdims=True)
    X += 0.05*np.random.randn(*X.shape)
    X += 0.1*np.sin(2*np.pi*60*t)[:, None]
    X[:, int(0.6*nCh):] += 0.2*t[:, None]
    blink = 3*np.exp(-((t-2.5)**2)/(2*0.12**2))
    X[:, :2] += blink[:, None]
    X[:, -1] += 6*np.random.randn(len(t))
    labels = [f"Ch{i+1}" for i in range(nCh)]
    return X, Fs, labels, t
