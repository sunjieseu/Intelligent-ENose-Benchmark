# Contributing to Intelligent E-Nose Benchmark

We welcome contributions from the community! This document provides guidelines for contributing to the project.

## 🎯 Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Intelligent-ENose-Benchmark.git
   cd Intelligent-ENose-Benchmark
   ```
3. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/amazing-feature
   ```

## 📝 Code Style

We follow these coding standards:

### Python Style Guide
- Follow **PEP 8** style guide
- Use **type hints** where possible
- Write **docstrings** for all public functions and classes
- Maximum line length: **88 characters** (Black formatter style)

### Example Function Format
```python
def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute classification accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score in [0, 1]
    """
    return np.mean(y_true == y_pred)
```

### Testing
- Add **unit tests** for new features
- Tests should be in `tests/` directory
- Use **pytest** framework
- Aim for **high code coverage** (>80%)

## 🔄 Development Workflow

### 1. Install Development Dependencies
```bash
pip install -r requirements.txt
pip install -e .
pip install pytest pytest-cov black flake8
```

### 2. Run Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=models tests/

# Check code style
black --check .
flake8 .
```

### 3. Format Code
```bash
# Auto-format with Black
black .

# Sort imports with isort
isort .
```

## 📦 Types of Contributions

### 🐛 Bug Fixes
- Describe the bug in an issue
- Reference the issue in your PR
- Add a test case to prevent regression

### ✨ New Features
- **First, open an issue** to discuss the feature
- Implement the feature in a separate branch
- Add comprehensive documentation
- Include unit tests
- Update README if needed

### 📚 Documentation
- Fix typos or clarify existing docs
- Add tutorials or examples
- Improve code comments
- Translate documentation

### 🧪 Algorithms
To add a new algorithm:

1. **Create implementation** in appropriate file:
   - `models/feature_fusion.py` for fusion methods
   - `models/transfer_learning.py` for TL methods
   - `models/few_shot.py` for FSL methods
   - `models/drift_compensation.py` for ADC methods

2. **Follow the existing interface**:
   ```python
   class NewAlgorithm:
       def __init__(self, **kwargs):
           """Initialize algorithm."""
           pass
       
       def fit(self, X, y, **kwargs):
           """Train the model."""
           pass
       
       def predict(self, X):
           """Make predictions."""
           pass
       
       def evaluate(self, X, y):
           """Evaluate performance."""
           pass
   ```

3. **Add to `__init__.py`** exports
4. **Write tests** in `tests/test_models.py`
5. **Add documentation** and examples

### 📊 Datasets
To add dataset support:

1. **Add loader** in `datasets/dataset_loaders.py`
2. **Create download script** in `datasets/`
3. **Update documentation** in `datasets/README.md`
4. **Add example** usage

## 🎓 Pull Request Process

### Before Submitting
1. **Update tests**: Ensure all tests pass
2. **Update documentation**: README, docstrings, etc.
3. **Check code style**: Run Black and Flake8
4. **Squash commits**: Clean up commit history

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Algorithm implementation

## Testing
- [ ] Added/updated tests
- [ ] All tests passing
- [ ] Tested on sample data

## Documentation
- [ ] Updated docstrings
- [ ] Updated README
- [ ] Added examples

## Additional Notes
Any other relevant information
```

### Review Process
1. Maintainer reviews code
2. Automated checks run
3. Feedback may be provided
4. Changes may be requested
5. PR is merged when approved

## 🐛 Reporting Issues

### Bug Reports
Include:
- **Description**: What happened vs. what should happen
- **Steps to reproduce**: Exact steps to trigger bug
- **Expected behavior**: What you expected
- **Actual behavior**: What actually happened
- **Environment**: OS, Python version, package versions
- **Logs/tracebacks**: Error messages

### Feature Requests
Include:
- **Description**: Clear description of the feature
- **Motivation**: Why this feature is needed
- **Use case**: How it would be used
- **Alternatives**: Any alternative solutions considered

## 💡 Best Practices

### Code Quality
- Write **clean, readable code**
- Use **meaningful variable names**
- Add **comments** for complex logic
- Keep functions **focused and small**
- **DRY**: Don't Repeat Yourself

### Documentation
- Write **clear, concise** documentation
- Include **examples** for new features
- Use **markdown** formatting
- Add **links** to related resources

### Testing
- Test **edge cases** and error conditions
- Use **descriptive test names**
- Keep tests **fast and isolated**
- Mock **external dependencies**

## 📚 Resources

- [PEP 8 Style Guide](https://peps.python.org/pep-0008/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PyTest Documentation](https://docs.pytest.org/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/)

## 🎖️ Recognition

Contributors will be:
- Listed in the **README.md** acknowledgments
- Credited in **release notes**
- Added to **AUTHORS** file (if created)

## ❓ Questions?

- **Open an issue** for general questions
- **Contact maintainers**: withheld during double-blind review
- **Join discussions** in GitHub Discussions (if enabled)

---

Thank you for contributing to Intelligent E-Nose Benchmark! 🙏
