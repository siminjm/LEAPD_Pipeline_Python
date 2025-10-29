# LEAPD Pipeline (Python)

**Author:** Simin Jamshidi (University of Iowa)  
**License:** MIT  

---

## Overview
This repository provides a **Python reimplementation** of the original MATLAB LEAPD pipeline for EEG-based **classification** and **correlation** analysis.  
It fully reproduces the algorithmic behavior of the MATLAB version — including filtering, LPC feature extraction, hyperplane-based LEAPD index computation, and statistical evaluation.

---

## Key Features
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
LEAPD_Pipeline_Python/
├── main_train.py
├── main_test.py
├── demo_leapd.ipynb
├── src/
│   └── leapd/
│       ├── data_io.py
│       ├── signal.py
│       ├── geom.py
│       ├── metrics.py
│       ├── combos.py
│       ├── plot_utils.py
│       └── __init__.py
├── figures/
├── results/
│   ├── train_results/
│   └── test_results/
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Quick Start
```bash
# Create environment
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Training
python main_train.py   --mode correlation   --data_train data/EEG_train.mat   --labels_file data/ClinicalLabels.xlsx   --lpc_backend burg   --n_jobs -1

# Testing
python main_test.py   --mode correlation   --data_test data/EEG_test.mat   --trained_model results/train_results/BestParamsAll.npz   --labels_file data/ClinicalLabels_Test.xlsx   --lpc_backend burg   --n_jobs -1
```

---

## Demo Notebook
Open `demo_leapd.ipynb` to:
1. Generate synthetic EEG-like data  
2. Run the full LEAPD process end-to-end  
3. Plot ROC and channel heatmap examples  

Outputs are saved in `/figures`.

---

## Python Version Equivalence
This Python version replicates all key MATLAB computations:
- **Burg LPC** via `compute_lpc_burg` matches MATLAB’s `arburg`
- **Filtering** uses `sosfiltfilt` (zero-phase Butterworth + notch)
- **SVD hyperplanes** and LEAPD ratio are mathematically identical
- **Results** match within numerical tolerance (~1e-14 differences)

---

## Optional Arguments
| Flag | Description | Default |
|------|--------------|----------|
| `--lpc_backend` | LPC computation backend (`burg` or `librosa`) | `burg` |
| `--n_jobs` | Number of parallel CPU jobs | `-1` (all cores) |
| `--is_norm_proj` | Use normalized projection mode | `0` |
| `--combo_sizes` | Range of channel combinations | `1 2 ... 10` |

---

## License
Distributed under the [MIT License](LICENSE).
