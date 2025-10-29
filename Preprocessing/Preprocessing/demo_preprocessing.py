import numpy as np, matplotlib.pyplot as plt
from generate_synthetic_EEG import generate_synthetic_EEG
from pipeline_preprocessing import pipeline_preprocessing

print("\n=========================\nEEG Preprocessing Demo\n=========================\n")
X, Fs, labels, t = generate_synthetic_EEG()
print(f"Synthetic EEG: {X.shape[0]} samples × {X.shape[1]} channels")
X_clean, labels_clean, report, savePath = pipeline_preprocessing(X, Fs, labels)
ch = 0
plt.figure(figsize=(12,7))
plt.suptitle("EEG Preprocessing — Before vs After", fontsize=14, fontweight="bold")
plt.subplot(3,1,1); plt.plot(t, X[:,ch],'k'); plt.axvline(2.5,color='r',ls='--')
plt.title(f"Raw EEG ({labels[ch]})"); plt.ylabel("µV")
plt.subplot(3,1,2); plt.plot(t, X_clean[:,ch],'b')
plt.title(f"Cleaned EEG ({labels_clean[ch]})"); plt.ylabel("µV")
plt.subplot(3,1,3)
freqs = np.fft.rfftfreq(X.shape[0], 1/Fs)
plt.plot(freqs, np.abs(np.fft.rfft(X[:,ch])), 'k', label='Raw')
plt.plot(freqs, np.abs(np.fft.rfft(X_clean[:,ch])), 'b', label='Cleaned')
plt.legend(); plt.xlim([0,100]); plt.xlabel("Frequency (Hz)")
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()
print("\n--- Summary ---")
for k,v in report.items(): print(f"{k}: {v}")
print(f"Saved at: {savePath}")
