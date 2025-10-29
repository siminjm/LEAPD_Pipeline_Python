import os, numpy as np, mne, scipy.io as sio, datetime
from detect_noisy_channels import detect_noisy_channels
from remove_line_noise import remove_line_noise

def pipeline_preprocessing(X, Fs, labels,
                           remove_line=True, use_ica=True,
                           save=True, save_dir="cleaned_data"):
    os.makedirs(save_dir, exist_ok=True)
    report = {"line_noise_removed": 0, "artifact_ics": [], "bad_channels": []}

    # Step 1: Line noise removal
    if remove_line:
        X = remove_line_noise(X, Fs)
        report["line_noise_removed"] = 1

    # Step 2: Detect and remove noisy channels
    bad_idx, bad_labels, _ = detect_noisy_channels(X, labels)
    if len(bad_idx) > 0:
        X = np.delete(X, bad_idx, axis=1)
        labels = [lbl for i, lbl in enumerate(labels) if i not in bad_idx]
        report["bad_channels"] = bad_labels

    # Step 3: ICA-based artifact removal
    if use_ica:
        info = mne.create_info(labels, Fs, ch_types="eeg")
        raw = mne.io.RawArray(X.T, info)
        raw.set_montage("standard_1020", on_missing="ignore")

        # --- High-pass filter for ICA stability ---
        raw.filter(l_freq=1.0, h_freq=None)

        ica = mne.preprocessing.ICA(
            n_components=min(15, len(labels)),
            random_state=97
        )
        ica.fit(raw)

        # --- Safe EOG/ECG artifact detection ---
        eog_inds, ecg_inds = [], []
        try:
            eog_inds, _ = ica.find_bads_eog(raw)
        except RuntimeError:
            print("⚠️ No EOG channel found — skipping EOG artifact detection.")
        except ValueError:
            print("⚠️ EOG detection skipped (invalid channel type).")

        try:
            ecg_inds, _ = ica.find_bads_ecg(raw)
        except (RuntimeError, ValueError):
            print("⚠️ No ECG channel found — skipping ECG artifact detection.")

        ica.exclude = list(set(eog_inds + ecg_inds))
        report["artifact_ics"] = ica.exclude

        raw_clean = ica.apply(raw.copy())
        X_clean = raw_clean.get_data().T
    else:
        X_clean = X

    # Step 4: Save results
    save_path = ""
    if save:
        tstamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"EEG_cleaned_{tstamp}.mat")
        sio.savemat(save_path, {
            "X_clean": X_clean,
            "labels_clean": labels,
            "report": report,
            "Fs": Fs
        })

    return X_clean, labels, report, save_path
