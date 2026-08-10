# Intelligent E-Nose Benchmark - Project Summary

## 📋 Overview

This is a complete, submission-ready GitHub repository for the **Intelligent E-Nose Benchmark** framework, as described in the review paper:

> **"Building Robust Intelligent E-Nose Systems Under Data Scarcity and Sensor Drift: A Comprehensive Review"**

The repository provides a unified, open-source platform for:
- ✅ Benchmarking transfer learning, few-shot learning, and drift compensation algorithms
- ✅ Standardized evaluation protocols with consistent metrics
- ✅ Complete implementations of typical intelligent algorithms
- ✅ Public dataset support (UCSD, CQU, etc.)
- ✅ Lifecycle-aware hierarchical taxonomy implementation

## 🎯 Key Features

### 1. Unified Information Fusion Perspective
The framework systematically integrates three critical paradigms:
- **Transfer Learning (TL)**: Cross-domain knowledge integration
- **Few-Shot Learning (FSL)**: Rapid task adaptation with limited data
- **Adaptive Drift Compensation (ADC)**: Long-term temporal stability

### 2. Three-Level Hierarchical Taxonomy
```
Level 1: Feature-Level Fusion
  ├─ Concatenation
  ├─ PCA
  └─ Attention

Level 2: Knowledge-Level Fusion (TL + FSL)
  ├─ Transfer Learning: TCA, JDA, DANN
  ├─ Few-Shot Learning: Prototypical Networks, MAML, Relation Networks
  └─ Self-Supervised Pre-training

Level 3: Decision-Level Fusion (ADC)
  ├─ Data-Level: OSC
  ├─ Model-Level: Active Learning
  └─ Ensemble-Level: CRE, TTA
```

### 3. Comprehensive Benchmarking
- **Static baseline** training and evaluation
- **Drift scenario** evaluation (FWT/BWT metrics)
- **Few-shot learning** evaluation (N-way K-shot protocol)
- **Fair comparison** across all implemented methods

## 📁 Repository Structure

```
Intelligent-ENose-Benchmark/
│
├── README.md                     # Comprehensive project documentation
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── setup.py                      # Installation script
│
├── datasets/
│   ├── README.md                 # Dataset documentation
│   ├── download_ucsd.sh          # UCSD dataset download script
│   ├── download_cqu.sh           # CQU dataset download script
│   └── dataset_loaders.py        # Unified data loading interface
│
├── models/
│   ├── __init__.py
│   ├── feature_fusion.py         # Level 1: Feature fusion methods
│   ├── transfer_learning.py      # Level 2: Transfer learning (TCA, JDA, DANN)
│   ├── few_shot.py               # Level 2: Few-shot learning (ProtoNets, MAML)
│   └── drift_compensation.py     # Level 3: Drift compensation (OSC, CRE, TTA)
│
├── benchmarks/
│   ├── train_baseline.py         # Train static baseline models
│   ├── eval_drift.py             # Drift scenario evaluation
│   └── eval_fewshot.py           # N-way K-shot evaluation
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                # Evaluation metrics (FWT, BWT, etc.)
│   └── data_utils.py             # Data preprocessing utilities
│
├── configs/
│   ├── ucsd_baseline.yaml        # Baseline training config
│   ├── ucsd_drift.yaml           # Drift evaluation config
│   └── ucsd_fewshot.yaml         # Few-shot evaluation config
│
├── notebooks/
│   └── 01_quickstart.ipynb       # Quickstart tutorial
│
├── tests/
│   ├── test_datasets.py          # Dataset loader tests
│   └── test_models.py            # Model tests
│
└── results/                      # Auto-generated results directory
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://anonymous.4open.science/r/Intelligent-ENose-Benchmark.git
cd Intelligent-ENose-Benchmark

# Create and activate virtual environment
conda create -n enose python=3.9
conda activate enose

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Download Datasets

```bash
# Download UCSD Gas Sensor Drift Dataset
bash datasets/download_ucsd.sh

# Download CQU E-Nose Drift Dataset
bash datasets/download_cqu.sh
```

### Run Benchmarks

```bash
# 1. Train baseline model
python benchmarks/train_baseline.py --config configs/ucsd_baseline.yaml

# 2. Evaluate drift compensation
python benchmarks/eval_drift.py --config configs/ucsd_drift.yaml

# 3. Evaluate few-shot learning
python benchmarks/eval_fewshot.py --config configs/ucsd_fewshot.yaml
```

## 🧪 Implemented Algorithms

### Feature-Level Fusion (Level 1)
| Method | Description | File |
|--------|-------------|------|
| **Concatenation** | Direct feature stacking | `models/feature_fusion.py` |
| **PCA** | Principal Component Analysis | `models/feature_fusion.py` |
| **Attention** | Learnable sensor weighting | `models/feature_fusion.py` |

### Transfer Learning (Level 2)
| Method | Description | File |
|--------|-------------|------|
| **TCA** | Transfer Component Analysis (MMD-based) | `models/transfer_learning.py` |
| **JDA** | Joint Distribution Adaptation | `models/transfer_learning.py` |
| **DANN** | Domain-Adversarial Neural Network | `models/transfer_learning.py` |

### Few-Shot Learning (Level 2)
| Method | Description | File |
|--------|-------------|------|
| **Prototypical Networks** | Metric-based few-shot classification | `models/few_shot.py` |
| **MAML** | Model-Agnostic Meta-Learning | `models/few_shot.py` |
| **Relation Networks** | Learnable distance metric | `models/few_shot.py` |

### Adaptive Drift Compensation (Level 3)
| Method | Description | File |
|--------|-------------|------|
| **OSC** | Orthogonal Signal Correction | `models/drift_compensation.py` |
| **CRE** | Classifier Replacement Ensemble | `models/drift_compensation.py` |
| **TTA** | Test-Time Adaptation | `models/drift_compensation.py` |
| **Active Learning** | Uncertainty-based sample selection | `models/drift_compensation.py` |

## 📊 Supported Datasets

| Dataset | Gases | Sensors | Samples | Duration | Download Script |
|---------|-------|---------|---------|----------|-----------------|
| **UCSD/UCI Drift** | 6 | 16 MOS | 13,910 | 36 months | `datasets/download_ucsd.sh` |
| **CQU E-Nose Drift** | 6 | 10×8 MOS | 1,604 | 3 batches | `datasets/download_cqu.sh` |
| **Beef E-Nose** | Spoilage | 11 MOS | 2,220 | Variable | Manual |
| **Wine Spoilage** | Spoilage | 6 MOS | ~300 | Short-term | Manual |
| **Air Quality** | CO, NOx, O₃ | 5 MOX | 9,358 | 1 year | Manual |

## 📈 Evaluation Metrics

### Standard Classification
- Accuracy, Precision, Recall, F1-Score (Macro/Micro)
- Per-class metrics with confusion matrix analysis

### Long-Term Drift
- **Average Accuracy**: Mean across all batches
- **Final Batch Accuracy**: Performance on latest batch
- **Backward Transfer (BWT)**: Forgetting measurement
- **Forward Transfer (FWT)**: Knowledge transfer effectiveness
- **Performance Degradation**: Drop from first to last batch

### Few-Shot Learning
- **N-way K-shot Accuracy**: Episodic evaluation
- **Macro F1-Score**: Class-imbalanced fairness
- **95% Confidence Interval**: Robustness indicator
- **Min/Max Accuracy**: Performance range

## 🔬 Usage Examples

### Example 1: Transfer Learning with DANN

```python
from models.transfer_learning import DANN
from datasets.dataset_loaders import load_ucsd

# Load data
X_source, y_source, X_target, y_target = load_ucsd(
    source_batches=[1, 2],
    target_batches=[3, 4, 5]
)

# Initialize and train DANN
dann = DANN(
    input_dim=16,
    hidden_dim=128,
    num_classes=6,
    alpha=1.0
)

dann.fit(X_source, y_source, X_target, epochs=100, batch_size=32)

# Evaluate
accuracy = dann.evaluate(X_target, y_target)
print(f"Target domain accuracy: {accuracy:.4f}")
```

### Example 2: Few-Shot Learning

```python
from models.few_shot import PrototypicalNetwork, create_fewshot_episode

# Create 5-way 3-shot episode
support_set, query_set = create_fewshot_episode(
    X_target, y_target,
    n_way=5, k_shot=3, n_query=10
)

# Initialize and train ProtoNet
proto_net = PrototypicalNetwork(
    input_dim=16, hidden_dim=64, embedding_dim=32
)

# Training loop (see benchmarks/eval_fewshot.py for full example)
```

### Example 3: Drift Compensation

```python
from models.drift_compensation import ClassifierReplacementEnsemble

# Initialize ensemble
cre = ClassifierReplacementEnsemble(
    ensemble_size=5,
    threshold=0.05
)

# Initialize and update
cre.fit_initial(X_batch1, y_batch1)

# Online deployment
for batch in data_stream:
    if cre.detect_drift(batch.X, batch.y):
        cre.update(batch.X, batch.y)
    predictions = cre.predict(batch.X)
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest tests/ -v

# Run specific test module
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=models --cov=datasets tests/
```

## 📚 Citation

If you use this benchmark in your research, please cite:

```bibtex
@article{anonymous2026intelligent,
  title={Continual Learning for Robotic and Edge Chemical Sensing: A Critical Review and Lifecycle Benchmark Framework},
  author={Anonymous Authors},
  journal={Under double-blind review},
  year={2026}
}
```

## 🤝 Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📧 Contact

- **Contact**: Anonymous during double-blind review
- **Institution**: Withheld during double-blind review
- **Repository**: Anonymous review link

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Acknowledgements are withheld during double-blind review.

