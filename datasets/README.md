# Datasets for Intelligent E-Nose Benchmark

This directory contains scripts and utilities for downloading, preprocessing, and loading E-nose datasets.

## 📦 Supported Datasets

### 1. UCSD/UCI Gas Sensor Array Drift Dataset
- **Description**: Long-term drift dataset spanning 36 months
- **Gases**: 6 gases (Ammonia, Ethanol, Ethylene, Methane, Methanol, Toluene)
- **Sensors**: 16 metal oxide sensors (MOS)
- **Samples**: 13,910 measurements
- **Batches**: 10 batches over 36 months
- **Download**: `bash download_ucsd.sh`

### 2. CQU E-Nose Drift Dataset
- **Description**: Cross-board and temporal drift dataset
- **Gases**: 6 gases
- **Sensors**: 10×8 MOS array (80 sensors total)
- **Samples**: 1,604 measurements
- **Batches**: 3 batches
- **Download**: `bash download_cqu.sh`

### 3. Additional Datasets
- **Beef E-Nose**: Beef spoilage detection with humidity variations
- **Wine Spoilage**: Short-term drift in wine quality monitoring
- **Air Quality**: Environmental monitoring with MOX sensors
- **UCI Low-Concentration**: ppb-level gas detection challenge

## 📥 Download Instructions

### UCSD Dataset
```bash
cd datasets
bash download_ucsd.sh
```

The script will:
1. Download the dataset from UCI Machine Learning Repository
2. Extract files to `data/ucsd/`
3. Verify data integrity

### CQU Dataset
```bash
cd datasets
bash download_cqu.sh
```

## 🔧 Dataset Loaders

Use the unified `dataset_loaders.py` interface:

```python
from datasets.dataset_loaders import load_ucsd, load_cqu

# Load UCSD dataset
X_train, y_train, X_test, y_test = load_ucsd(
    source_batches=[1, 2],
    target_batches=[3, 4, 5],
    feature_type='steady_state',
    normalize=True
)

# Load CQU dataset
X_train, y_train, X_test, y_test = load_cqu(
    source_batch=1,
    target_batch=2,
    feature_type='transient',
    normalize=True
)
```

## 📊 Feature Types

### Standard Feature Protocol (UCSD)
The released UCSD scripts use the standard feature extraction protocol (eight statistical features per sensor, 16 x 8 = 128 dimensions) as established by Vergara et al. The default `feature_type='standard'` applies this protocol. A legacy 16-dimensional variant using normalized steady-state readings directly is available via `feature_type='steady_state'` for rapid prototyping.

### Steady-State Features
- Maximum response
- Minimum response
- Post-equilibrium mean
- Simple and robust

### Transient Features
- Slope (response/recovery)
- Integral area
- Response time
- Recovery time
- Rich discriminatory information

### Transform-Domain Features
- Fourier transform coefficients
- Wavelet transform coefficients
- Multi-scale characteristics

## 🔄 Data Preprocessing

The loader includes built-in preprocessing:

```python
from datasets.dataset_loaders import preprocess_data

X_processed = preprocess_data(
    X_raw,
    method='standard',  # 'standard', 'minmax', 'robust'
    handle_missing=True,
    remove_outliers=True,
    outlier_threshold=3.0
)
```

## 📁 Directory Structure

After downloading, the structure should be:

```
datasets/
├── README.md              # This file
├── download_ucsd.sh       # UCSD download script
├── download_cqu.sh        # CQU download script
├── dataset_loaders.py     # Unified loading interface
└── data/                  # Downloaded data (not in git)
    ├── ucsd/
    │   ├── Datos3096.txt
    │   └── drift_data.csv
    └── cqu/
        ├── batch1.csv
        ├── batch2.csv
        └── batch3.csv
```

## 📝 Citation

When using these datasets, please cite the original authors:

**UCSD Dataset:**
```bibtex
@article{vergara2012chemical,
  title={Chemical gas sensor drift compensation using classifier ensembles},
  author={Vergara, Alexander and Yevseyev, Igor and Llobet, Eduard},
  journal={Sensors and Actuators B: Chemical},
  volume={166},
  pages={320--329},
  year={2012}
}
```

**CQU Dataset:**
```bibtex
@article{zhang2019drift,
  title={Drift compensation for gas sensor array using a modified domain adaptation method},
  author={Zhang, Jie and Cheng, Meng and Liu, Jun and others},
  journal={IEEE Sensors Journal},
  volume={19},
  number={15},
  pages={6285--6293},
  year={2019}
}
```

## ⚠️ License and Usage

Please respect the licenses of the original datasets. Check each dataset's terms before use.
