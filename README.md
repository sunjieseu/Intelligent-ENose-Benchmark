# Intelligent E-Nose Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📖 Overview

**Intelligent-ENose-Benchmark** is a comprehensive, open-source framework for building robust intelligent electronic nose (E-nose) systems under **data scarcity** and **sensor drift** challenges. This project provides:

- ✅ **Unified Information Fusion Perspective**: Systematic integration of Transfer Learning (TL), Few-Shot Learning (FSL), and Adaptive Drift Compensation (ADC)
- ✅ **Lifecycle-Aware Hierarchical Taxonomy**: Three synergistic levels (Feature-Level → Knowledge-Level → Decision-Level)
- ✅ **Fair Comparison Benchmark**: Standardized evaluation protocols with consistent metrics (FWT/BWT, N-way K-shot accuracy)
- ✅ **Open-Source Repository**: Complete implementations of typical intelligent algorithms and evaluation protocols
- ✅ **Public Dataset Support**: Scripts for downloading and preprocessing major E-nose datasets (UCSD, CQU, etc.)

This project accompanies the review paper: **"Building Robust Intelligent E-Nose Systems Under Data Scarcity and Sensor Drift: A Comprehensive Review"** (Nature-style format).

## 🏗️ Framework Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Intelligent E-Nose System Lifecycle             │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Feature-Level Fusion                               │
│  ├─ Multi-sensor signal processing                           │
│  ├─ Steady-state, transient, transform-domain features       │
│  └─ PCA, Attention, Concatenation strategies                 │
├─────────────────────────────────────────────────────────────┤
│  Phase 2: Knowledge-Level Fusion (TL + FSL)                  │
│  ├─ Transfer Learning: Domain adaptation (DANN, MMD, TCA)   │
│  ├─ Self-Supervised Pre-training: Contrastive learning       │
│  └─ Few-Shot Learning: Prototypical Networks, MAML           │
├─────────────────────────────────────────────────────────────┤
│  Phase 3: Decision-Level Fusion (ADC)                        │
│  ├─ Data-Level: OSC, Domain normalization                   │
│  ├─ Model-Level: Online learning, Active learning            │
│  └─ Ensemble-Level: Classifier replacement, Dynamic weighting│
└─────────────────────────────────────────────────────────────┘
```

## 📂 Project Structure

```
Intelligent-ENose-Benchmark/
│
├── README.md                  # Project homepage documentation
├── requirements.txt           # Python dependencies
├── setup.py                   # Installation script
│
├── datasets/                  # Dataset management
│   ├── README.md             # Dataset documentation
│   ├── download_ucsd.sh      # Download UCSD drift dataset
│   ├── download_cqu.sh       # Download CQU drift dataset
│   └── dataset_loaders.py    # Unified data loading interface
│
├── models/                    # Algorithm implementations
│   ├── __init__.py
│   ├── feature_fusion.py     # Level 1: Feature-level fusion
│   ├── transfer_learning.py  # Level 2: Transfer learning (DANN, TCA, etc.)
│   ├── few_shot.py           # Level 2: Few-shot learning (ProtoNets, MAML)
│   └── drift_compensation.py # Level 3: Online drift compensation
│
├── benchmarks/                # Performance evaluation scripts
│   ├── train_baseline.py     # Train static baseline models
│   ├── eval_drift.py         # Drift scenario evaluation (FWT/BWT)
│   └── eval_fewshot.py       # N-way K-shot evaluation
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── metrics.py            # Evaluation metrics
│   ├── data_utils.py         # Data preprocessing
│   └── config.py             # Configuration management
│
├── configs/                   # Configuration files
│   ├── ucisd_baseline.yaml
│   ├── ucisd_drift.yaml
│   └── ucisd_fewshot.yaml
│
├── notebooks/                 # Jupyter notebooks for tutorials
│   ├── 01_quickstart.ipynb
│   ├── 02_transfer_learning_demo.ipynb
│   └── 03_fewshot_learning_demo.ipynb
│
├── results/                   # Experimental results (auto-generated)
│   └── .gitkeep
│
└── tests/                     # Unit tests
    ├── test_datasets.py
    ├── test_models.py
    └── test_benchmarks.py
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/Intelligent-ENose-Benchmark-B022.git
cd Intelligent-ENose-Benchmark

# Create virtual environment (recommended)
conda create -n enose python=3.9
conda activate enose

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Download Datasets

The benchmark expects all raw data under `data/` (excluded from git). Three
UCI-hosted datasets are used; download each into its own subdirectory:

```bash
# 1) UCSD Gas Sensor Array Drift Dataset (UCI ID 224)
#    16 MOS sensors, 6 gases, 13,910 samples, 10 batches over 36 months.
#    -> data/ucsd/Dataset/batch1.dat ... batch10.dat
mkdir -p data/ucsd
curl -L -o data/ucsd/drift.zip \
  "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip"
unzip -o data/ucsd/drift.zip -d data/ucsd/
rm -f data/ucsd/drift.zip

# 2) Gas Sensor Array Drift Dataset at Different Concentrations (UCI ID 270)
#    Same UCSD measurements, but each row also carries the gas concentration
#    (format: "class;concentration feature1:v1 ..."). Used by the unified
#    benchmark via `--dataset 270`.
#    -> data/ucsd270/batch1.dat ... batch10.dat
mkdir -p data/ucsd270
curl -L -o data/ucsd270/drift270.zip \
  "https://archive.ics.uci.edu/static/public/270/gas+sensor+array+drift+dataset+at+different+concentrations.zip"
unzip -o data/ucsd270/drift270.zip -d data/ucsd270/
rm -f data/ucsd270/drift270.zip

# 3) GSALC - Gas Sensor Array Low-Concentration (UCI ID 1081, CQU array)
#    10 MOS sensors, 6 gases at 50/100/200 ppb, 90 samples, 9000 response
#    points per sample. Used for the cross-array validation
#    (benchmarks/eval_gsalc.py).
#    -> data/gsalc/gsalc.csv
mkdir -p data/gsalc
curl -L -o data/gsalc/gsalc.zip \
  "https://archive.ics.uci.edu/static/public/1081/gas+sensor+array+low-concentration.zip"
unzip -o data/gsalc/gsalc.zip -d data/gsalc/
rm -f data/gsalc/gsalc.zip
```

Alternative one-liner with the `ucimlrepo` Python package:

```python
from ucimlrepo import fetch_ucirepo
for ds_id in (224, 270, 1081):
    fetch_ucirepo(id=ds_id)   # fetches metadata + data from UCI
```

Notes:

- On Windows (PowerShell), replace `curl -L -o` with
  `Invoke-WebRequest -OutFile` or use `wget`, and unpack the `.zip` files
  with `Expand-Archive`.
- `datasets/download_ucsd.sh` automates download 1; `datasets/download_cqu.sh`
  contains placeholder URLs and is kept only as a template - use the direct
  UCI links above instead.

### Run Baseline Experiments

```bash
# Train a static baseline model
python benchmarks/train_baseline.py --config configs/ucisd_baseline.yaml

# Evaluate drift compensation performance
python benchmarks/eval_drift.py --config configs/ucisd_drift.yaml

# Evaluate few-shot learning (N-way K-shot)
python benchmarks/eval_fewshot.py --config configs/ucisd_fewshot.yaml
```

## 📊 Supported Datasets

| Dataset | Target Gases | Sensors | Samples | Duration | Use Case |
|---------|-------------|---------|---------|----------|----------|
| **UCSD/UCI Drift** | 6 gases | 16 MOS | 13,910 | 36 months | Long-term drift benchmark |
| **CQU E-Nose Drift** | 6 gases | 10×8 MOS | 1,604 | 3 batches | Board + time drift |
| **Beef E-Nose** | Beef spoilage | 11 MOS | 2,220 | Variable | Humidity/background shift |
| **Wine Spoilage** | Wine spoilage | 6 MOS | ~300 | Short-term | Short-term drift |
| **Air Quality** | CO, NOx, O₃ | 5 MOX + T/RH | 9,358 | 1 year | Env. drift/noise |
| **UCI Low-Concentration** | Multi-gas (ppb) | 10 MOS | 13,910 | 36 days | Low-conc. challenge |

## 🧪 Implemented Algorithms

### Feature-Level Fusion (Level 1)
- **Concatenation**: Direct feature stacking
- **PCA**: Principal Component Analysis
- **Attention**: Learnable sensor-level weighting

### Transfer Learning (Level 2)
- **TCA**: Transfer Component Analysis (MMD-based)
- **JDA**: Joint Distribution Adaptation
- **DANN**: Domain-Adversarial Neural Network
- **SAELM**: Stacked Autoencoder + ELM
- **DAST**: Domain Adaptation with Subspace + Manifold

### Few-Shot Learning (Level 2)
- **Prototypical Networks**: Metric-based few-shot classification
- **Relation Networks**: Learnable distance metric
- **MAML**: Model-Agnostic Meta-Learning
- **PSCN**: Wavelet Scattering + ELM
- **MDCN**: Multi-View Drift-Compensated Network

### Adaptive Drift Compensation (Level 3)
- **OSC**: Orthogonal Signal Correction
- **Active Learning**: Uncertainty-based sample selection
- **CRE**: Classifier Replacement Ensemble
- **WWH-SSO**: Weighted Weighted Histogram - Sequential Semi-supervised
- **TTA**: Test-Time Adaptation (Entropy minimization)

## 📈 Evaluation Metrics

### Standard Classification
- Accuracy, Precision, Recall, F1-Score (Macro/Micro)

### Long-Term Drift
- **Average Accuracy**: Mean accuracy across all batches
- **Final Batch Accuracy**: Performance on latest batch
- **Backward Transfer (BWT)**: Forgetting measurement
- **Forward Transfer (FWT)**: Knowledge transfer effectiveness
- **Computational/Memory Cost**: Deployment feasibility

### Few-Shot Learning
- **N-way K-shot Accuracy**: Standard episodic evaluation
- **Macro F1-Score**: Class-imbalanced fairness
- **Confidence Interval / Std**: Robustness indicator

## 🔬 Usage Examples

### Example 1: Transfer Learning with DANN

```python
from models.transfer_learning import DANN
from datasets.dataset_loaders import load_ucsd

# Load data
X_source, y_source, X_target, y_target = load_ucsd(source_batch=1, target_batch=5)

# Initialize DANN
model = DANN(
    input_dim=128,
    hidden_dim=128,
    num_classes=6,
    alpha=1.0  # Trade-off parameter
)

# Train
model.fit(X_source, y_source, X_target, epochs=100, batch_size=32)

# Evaluate
accuracy = model.evaluate(X_target, y_target)
print(f"Target domain accuracy: {accuracy:.4f}")
```

### Example 2: Few-Shot Learning with Prototypical Networks

```python
from models.few_shot import PrototypicalNetwork
from utils.data_utils import create_fewshot_task

# Create N-way K-shot task
support_set, query_set = create_fewshot_task(
    dataset='ucsd',
    n_way=5,
    k_shot=3,
    n_query=10
)

# Initialize model
proto_net = PrototypicalNetwork(
    encoder_dim=[128, 64],
    distance='euclidean'
)

# Train with episodic training
proto_net.train(support_set, query_set, episodes=1000)

# Test
accuracy = proto_net.test(query_set)
print(f"5-way 3-shot accuracy: {accuracy:.4f}")
```

### Example 3: Online Drift Compensation

```python
from models.drift_compensation import ClassifierReplacementEnsemble
from utils.metrics import compute_bwt, compute_fwd

# Initialize ensemble
ensemble = ClassifierReplacementEnsemble(
    base_model='resnet1d',
    threshold=0.05,  # Performance drop threshold
    update_interval=30  # Days between checks
)

# Online deployment
for batch in data_stream:
    # Detect drift
    if ensemble.detect_drift(batch.X):
        # Trigger adaptation
        if ensemble.performance_drop > threshold:
            ensemble.update(batch.X, batch.y)
    
    # Predict
    predictions = ensemble.predict(batch.X)
    
    # Log metrics
    bwt = compute_bwt(predictions, batch.y)
    fwd = compute_fwd(predictions, batch.y)
```

## 📝 Configuration Files

Configuration files use YAML format for easy experimentation:

```yaml
# configs/ucisd_baseline.yaml
dataset:
  name: 'ucsd'
  source_batches: [1, 2]
  target_batches: [3, 4, 5]
  feature_type: 'steady_state'

model:
  name: 'resnet1d'
  input_dim: 16
  hidden_dim: 128
  num_classes: 6
  dropout: 0.5

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
  optimizer: 'adam'
  
evaluation:
  metrics: ['accuracy', 'f1_macro', 'bwt', 'fwd']
  save_results: true
  output_dir: 'results/baseline'
```

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=models --cov=datasets tests/
```

## 📚 Citation

If you use this benchmark in your research, please cite our review paper:

```bibtex
@article{anonymous2026intelligent,
  title={Continual Learning for Robotic and Edge Chemical Sensing: A Critical Review and Lifecycle Benchmark Framework},
  author={Anonymous Authors},
  journal={Under double-blind review},
  year={2026}
}
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide
- Add unit tests for new features
- Update documentation
- Use type hints where possible

## 📧 Contact

- **Contact**: Anonymous during double-blind review
- **Institution**: Withheld during double-blind review
- **Repository**: Anonymous review link

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Acknowledgements are withheld during double-blind review.

---

**Star this repository** if you find it helpful for your research! ⭐
