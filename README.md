# LEAPD Pipeline (Python + MNE Preprocessing)

**Author:** Simin Jamshidi (University of Iowa)  
**License:** MIT (with Citation Request)

---

## Overview
This repository provides a **Python reimplementation** of the original MATLAB **LEAPD** pipeline for EEG-based **classification** and **correlation** analysis.  
It faithfully reproduces the algorithmic behavior of the MATLAB version — including preprocessing, filtering, LPC feature extraction, hyperplane-based LEAPD index computation, and statistical evaluation.

The pipeline now includes a dedicated **EEG Preprocessing module (MNE-based)**, enabling artifact removal, noisy-channel detection, and ICA cleaning before LEAPD analysis.

---

## Key Features
- **Preprocessing (MNE):** automated line-noise removal, noisy-channel detection, and ICA-based artifact rejection  
- **Modes:** `classification` (binary) and `correlation` (continuous target)
- **Automatic parameter search** over filter bands and LPC orders
- **Single- and multi-channel** evaluation (1–10 channels)
- **LOOCV** and **out-of-sample** testing
- **Polarity alignment** for correlation analysis
- **Comprehensive metrics:** ACC, AUC, SEN, SPC, PPV, NPV, OR, LR⁺, ρ, p-value
- **Parallelization** with `joblib` for fast execution
- **Dual LPC backends:** Burg (exact MATLAB match) or Librosa (approximate)

---

## Repository Structure
```
LEAPD_Pipeline_Python
│
├── Preprocessing/                  # EEG cleaning module (MNE-based)
│   ├── demo_preprocessing.py
│   ├── pipeline_preprocessing.py
│   ├── generate_synthetic_EEG.py
│   ├── detect_noisy_channels.py
│   ├── remove_line_noise.py
│   ├── README.md
│   └── cleaned_data/
│
├── main_train.py                   # LEAPD training script
├── main_test.py                    # LEAPD testing script
├── demo_leapd.ipynb                # Jupyter demo (end-to-end)
│
├── utils/
│   ├── data_io.py
│   ├── signal.py
│   ├── geom.py
│   ├── metrics.py
│   ├── combos.py
│   ├── plot_utils.py
│   └── __init__.py
│
├── figures/
├── results/
│   ├── train_results/
│   └── test_results/
│
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Preprocessing Module (Optional)
The `Preprocessing/` folder provides a self-contained EEG artifact-cleaning pipeline using **MNE-Python**.

### Features:
- 60 Hz notch filtering (and harmonics)
- Automatic noisy-channel detection
- ICA-based EOG/ECG artifact removal
- Before/after time-domain and spectral plots
- Automatic `.mat` export for downstream LEAPD analysis

### Run Demo:
```bash
cd Preprocessing
python demo_preprocessing.py
```

Output:
- Cleaned EEG saved to `Preprocessing/cleaned_data/EEG_cleaned_<timestamp>.mat`
- Summary report of removed ICs, bad channels, and filters applied

You can directly feed this `.mat` file into your LEAPD training pipeline as:
```bash
python main_train.py --data_train Preprocessing/cleaned_data/EEG_cleaned_xxxxx.mat
```

---

## Quick Start
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Training
python main_train.py \
  --mode correlation \
  --data_train data/EEG_train.mat \
  --labels_file data/ClinicalLabels.xlsx \
  --lpc_backend burg \
  --n_jobs -1

# Testing
python main_test.py \
  --mode correlation \
  --data_test data/EEG_test.mat \
  --trained_model results/train_results/BestParamsAll.npz \
  --labels_file data/ClinicalLabels_Test.xlsx \
  --lpc_backend burg \
  --n_jobs -1
```

---

## Demo Notebook
Open **`demo_leapd.ipynb`** to:
1. Generate synthetic EEG-like data  
2. Run the complete LEAPD workflow  
3. Visualize ROC curves and channel heatmaps  

Outputs are automatically saved in the `/figures/` directory.

---

## Python–MATLAB Equivalence
This Python implementation numerically matches the MATLAB version:
- **Burg LPC** (`compute_lpc_burg`) ≈ MATLAB `arburg`
- **Zero-phase Butterworth** and notch filters via `sosfiltfilt`
- **SVD-based hyperplane distances** identical to LEAPD’s formulation
- **Performance metrics** within numerical tolerance (~1e-14 differences)

---

## Optional CLI Arguments
| Flag | Description | Default |
|------|--------------|----------|
| `--lpc_backend` | LPC computation backend (`burg` or `librosa`) | `burg` |
| `--n_jobs` | Number of CPU cores to use | `-1` (all cores) |
| `--is_norm_proj` | Use normalized projection distance | `0` |
| `--combo_sizes` | Channel combination sizes to test | `1 2 ... 10` |

---

## Requirements
```bash
pip install mne numpy scipy matplotlib joblib librosa pandas
```

---

## Author
**Simin Jamshidi**  
Departments of **Electrical & Computer Engineering (ECE)** and **Neurology**  
**University of Iowa**

Supervisors:  
**Prof. Soura Dasgupta** and **Dr. Nandakumar Narayanan**

---

## Citation
If you use this pipeline in academic or research work, please cite:

> Jamshidi, S., et al. (Year).  
> *EEG-Based Mortality and Cognitive Decline Prediction in Parkinson’s Disease using the LEAPD Method.*  
> Departments of Electrical & Computer Engineering (ECE) and Neurology, University of Iowa.  
> (Preprint or Journal Reference — to be updated)

---

## License
Released under the **MIT License (with Citation Request)**.  
See the [LICENSE](./LICENSE) file for details.
