# EEG Preprocessing Pipeline (Python + MNE)

This repository contains a **fully automated EEG preprocessing pipeline** built in Python, designed for both synthetic and real EEG datasets.  
It demonstrates how to clean EEG data by removing line noise, detecting noisy channels, and performing **ICA-based artifact rejection** using the [MNE-Python](https://mne.tools/stable/index.html) library.

---

## Features

- **Automatic line-noise removal** (60 Hz notch filter, configurable)
- **Noisy-channel detection** using z-score of channel standard deviation
- **ICA-based artifact removal** (blink, muscle, heart, and line-noise components)
- **Automatic EEG reconstruction** after IC removal
- **Before/after visualization** with aligned time axes and power spectra
- **Automatic saving** of cleaned EEG data (`.mat` file)

---

## Folder Structure
```
Preprocessing_Python/
│
├── generate_synthetic_EEG.py     # Creates test EEG data with artifacts
├── detect_noisy_channels.py      # Identifies noisy EEG channels
├── remove_line_noise.py          # Applies 60 Hz notch filter (configurable)
├── pipeline_preprocessing.py     # Core preprocessing pipeline using MNE
├── demo_preprocessing.py         # Example script (main entry point)
│
└── cleaned_data/                 # Automatically generated cleaned EEG files
```

---

## Configuration Options

The pipeline can be customized through parameters in `pipeline_preprocessing.py`.

```python
# Example configuration
X_clean, labels_clean, report, savePath = pipeline_preprocessing(
    X, Fs, labels,
    remove_line=True,   # Enable or disable 60 Hz notch filter
    use_ica=True,       # Enable ICA-based artifact removal
    save=True,          # Automatically save results to /cleaned_data/
)
```

---

## Usage

To run the preprocessing demo:

```bash
cd Preprocessing
python demo_preprocessing.py
```

The script will:
1. Generate synthetic EEG (if no input provided)
2. Run the full preprocessing pipeline
3. Display before/after EEG and spectra
4. Save the cleaned EEG data to `cleaned_data/`

---

## Requirements

- Python 3.9 or later  
- [MNE-Python](https://mne.tools/stable/index.html)  
- NumPy, SciPy, Matplotlib

Install dependencies:
```bash
pip install mne numpy scipy matplotlib
```

---

## Author
**Simin Jamshidi**  
Departments of Electrical & Computer Engineering (ECE) and Neurology  
University of Iowa

---

## Citation
If you use this pipeline in your research, please cite:

> Jamshidi, S., et al. (Year).  
> *EEG-Based Mortality and Cognitive Decline Prediction in Parkinson’s Disease using the LEAPD Method.*  
> Departments of Electrical & Computer Engineering (ECE) and Neurology, University of Iowa.  
> (Preprint or Journal Reference — to be updated)
