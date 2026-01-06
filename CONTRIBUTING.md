# Contributing to mlship

Thank you for your interest in contributing to mlship! This guide will help you get started.

---

## Development Setup

### 1. Clone the Repository

```bash
git clone https://github.com/sudhanvalabs/mlship
cd mlship
```

### 2. Install uv (Recommended)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Create Virtual Environment

```bash
uv venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate  # Windows
```

### 4. Install in Development Mode

```bash
# Install with all dependencies for development
uv pip install -e ".[dev,all]"
```

This installs:
- mlship in editable mode (`-e`)
- All framework support (sklearn, pytorch, tensorflow, huggingface)
- Development tools (pytest, black, ruff, mypy)

---

## Project Structure

```
mlship/
├── mlship/              # Source code
│   ├── __init__.py
│   ├── cli.py          # CLI commands (Click)
│   ├── server.py       # FastAPI app generator
│   ├── models.py       # Pydantic models
│   ├── errors.py       # Custom exceptions
│   ├── utils.py        # Helper functions
│   └── loaders/        # Model loaders
│       ├── base.py     # Base interface
│       ├── detector.py # Framework detection
│       ├── sklearn.py  # Scikit-learn
│       ├── pytorch.py  # PyTorch
│       ├── tensorflow.py # TensorFlow
│       └── huggingface.py # Hugging Face
├── tests/              # Test suite
│   ├── conftest.py     # Pytest fixtures
│   ├── test_cli.py     # CLI tests
│   └── test_loaders.py # Loader tests
├── examples/           # Example scripts
├── pyproject.toml      # Package configuration
└── README.md
```

---

## Running Tests

mlship has comprehensive integration tests that verify all model frameworks work correctly.

### Quick Test Run

**Option 1: Cross-platform Python script (recommended)**
```bash
# Works on Windows, macOS, and Linux
python run_tests.py
```

**Option 2: Bash script (macOS/Linux only)**
```bash
./run_tests.sh
```

Both scripts will:
- Check for missing dependencies
- Install any missing frameworks automatically
- Run all tests with coverage report

**Platform recommendations:**
- Windows: Use `python run_tests.py`
- macOS/Linux: Either script works, but `python run_tests.py` is more portable

### Run All Tests

```bash
pytest
```

### Run Integration Tests Only

These tests download models, test all API endpoints, and cleanup automatically:

```bash
pytest tests/test_integration.py -v
```

This tests:
- ✅ Sklearn models (created on-the-fly)
- ✅ PyTorch models (created on-the-fly)
- ✅ HuggingFace models (downloads DistilBERT ~256MB)
- ⏭️  TensorFlow models (skipped if not installed)

Each test:
1. Downloads/creates the model
2. Loads it with mlship
3. Tests all endpoints: `/health`, `/info`, `/predict`, `/docs`
4. Tests error handling
5. Cleans up (deletes downloaded models)

### Run with Coverage

```bash
pytest --cov=mlship --cov-report=html
```

View coverage report:
```bash
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### Run Specific Test

```bash
pytest tests/test_cli.py::test_version
```

### Run Tests for Specific Framework

```bash
pytest tests/test_integration.py::TestSklearnIntegration -v
pytest tests/test_integration.py::TestPyTorchIntegration -v
pytest tests/test_integration.py::TestHuggingFaceIntegration -v
```

### Pre-Push Checks

**Recommended:** Run all CI checks locally before pushing to catch issues early:

```bash
./pre_push.sh
```

This runs:
- ✅ Black formatting check
- ✅ Ruff linting
- ✅ MyPy type checking
- ✅ All tests

**Manual checks:**

```bash
# Quick check
pytest tests/test_integration.py --tb=short

# Full check with coverage
pytest --cov=mlship --cov-report=term-missing

# Format check only
black --check mlship/ tests/

# Lint check only
ruff check mlship/ tests/
```

### GitHub Actions (Automated Testing)

mlship uses GitHub Actions to automatically test on **every push** across:

**Platforms:**
- ✅ Ubuntu (Linux)
- ✅ Windows
- ✅ macOS

**Python versions:**
- ✅ Python 3.9
- ✅ Python 3.10
- ✅ Python 3.11

**Frameworks tested:**
- Individual tests for sklearn, pytorch, tensorflow, huggingface
- Full integration tests
- Code linting and formatting checks

The CI workflow is defined in `.github/workflows/test.yml` and runs automatically on every push to ensure cross-platform compatibility.

**Cost:** Free for public repositories (unlimited minutes)

---

## Code Quality

### Format Code

```bash
black mlship/ tests/
```

### Lint Code

```bash
ruff check mlship/ tests/
```

### Auto-fix Linting Issues

```bash
ruff check --fix mlship/ tests/
```

### Type Check

```bash
mypy mlship/
```

### Run All Checks

```bash
# Format
black mlship/ tests/

# Lint
ruff check --fix mlship/ tests/

# Type check
mypy mlship/

# Test
pytest
```

---

## Adding a New Model Framework

Want to add support for a new ML framework? Here's how:

### 1. Create Loader Class

Create `mlship/loaders/newframework.py`:

```python
from pathlib import Path
from typing import Any, Dict, List, Union
from mlship.errors import ModelLoadError, ValidationError
from mlship.loaders.base import ModelLoader

class NewFrameworkLoader(ModelLoader):
    """Loader for NewFramework models."""

    def load(self, model_path: Path) -> Any:
        """Load model from file."""
        try:
            # Load model using framework's library
            import newframework
            model = newframework.load(str(model_path))
            return model
        except ImportError:
            raise ModelLoadError(
                "NewFramework is not installed.\n\n"
                "Install with: uv pip install newframework"
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load model: {e}")

    def predict(self, model: Any, features: Union[List[float], List[List[float]]]) -> Dict[str, Any]:
        """Run prediction."""
        import numpy as np
        X = np.array(features)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        prediction = model.predict(X)[0]
        return {"prediction": float(prediction)}

    def get_metadata(self, model: Any) -> Dict[str, Any]:
        """Extract metadata."""
        return {
            "model_type": model.__class__.__name__,
            "framework": "newframework",
        }

    def validate_input(self, model: Any, features: Union[List[float], List[List[float]]]) -> None:
        """Validate input."""
        # Add validation logic
        pass
```

### 2. Update Detector

Edit `mlship/loaders/detector.py`:

```python
def detect_framework(model_path: Path) -> str:
    # ... existing code ...
    elif extension == ".newext":
        return "newframework"

def get_loader(framework: str) -> ModelLoader:
    # ... existing code ...
    elif framework == "newframework":
        from mlship.loaders.newframework import NewFrameworkLoader
        return NewFrameworkLoader()
```

### 3. Add Dependencies

Edit `pyproject.toml`:

```toml
[project.optional-dependencies]
newframework = ["newframework>=1.0"]
all = [
    # ... existing ...
    "newframework>=1.0",
]
```

### 4. Write Tests

Create `tests/test_newframework.py`:

```python
import pytest
from pathlib import Path
from mlship.loaders.newframework import NewFrameworkLoader

def test_load():
    """Test loading model."""
    loader = NewFrameworkLoader()
    # ... test implementation ...

def test_predict():
    """Test prediction."""
    # ... test implementation ...
```

### 5. Update Documentation

Add to `README.md` supported frameworks table.

---

## Testing Your Changes

### Manual Testing

```bash
# 1. Create a test model
python examples/sklearn_example.py

# 2. Serve it
mlship serve fraud_detector.pkl

# 3. Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}'
```

### Automated Testing

```bash
# Run tests
pytest

# Check coverage
pytest --cov=mlship --cov-report=term-missing
```

---

## Pull Request Process

1. **Fork the repository** on GitHub

2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Run quality checks**
   ```bash
   black mlship/ tests/
   ruff check --fix mlship/ tests/
   pytest
   ```

5. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add amazing feature"
   ```

   Use conventional commits:
   - `feat:` new feature
   - `fix:` bug fix
   - `docs:` documentation
   - `test:` tests
   - `refactor:` code refactoring

6. **Push to your fork**
   ```bash
   git push origin feature/amazing-feature
   ```

7. **Create Pull Request** on GitHub
   - Describe your changes
   - Reference any related issues
   - Ensure CI checks pass

---

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints
- Keep functions focused and small
- Write docstrings for public APIs

**Example:**
```python
def load_model(model_path: Path) -> Any:
    """
    Load ML model from file.

    Args:
        model_path: Path to model file

    Returns:
        Loaded model object

    Raises:
        ModelLoadError: If model cannot be loaded
    """
    # Implementation...
```

### Error Messages

Make error messages helpful:

```python
# ❌ Bad
raise ValueError("Invalid input")

# ✅ Good
raise ValidationError(
    f"Invalid input shape\n\n"
    f"Expected: {expected} features\n"
    f"Received: {received} features\n\n"
    f"Example correct input:\n"
    f'{{"features": [1.0, 2.0, 3.0]}}'
)
```

### CLI Output

Use colors and clear formatting:

```python
click.secho("✓ Success", fg="green")
click.secho("❌ Error", fg="red", err=True)
click.secho("⚠️  Warning", fg="yellow")
```

---

## Benchmarking

Run performance benchmarks:

```bash
python benchmark.py
```

See `PERFORMANCE.md` for current benchmarks.

---

## Release Process

(For maintainers)

1. **Update version** in `pyproject.toml`
   ```bash
   # Change version line: version = "0.1.2"
   git add pyproject.toml
   git commit -m "chore: bump version to 0.1.2"
   ```

2. **Clean and build package**
   ```bash
   rm -rf dist/ build/ *.egg-info
   python -m build
   ```

3. **Test installation locally**
   ```bash
   uv pip install dist/mlship-0.1.2-py3-none-any.whl
   mlship --version
   ```

4. **Test on TestPyPI first** ⚠️ **Important: Don't skip this!**
   ```bash
   twine upload --repository testpypi dist/*
   # Test install from TestPyPI
   pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mlship
   ```

5. **Publish to PyPI** (only after TestPyPI works)
   ```bash
   twine upload dist/*
   ```

6. **Create git tag and push**
   ```bash
   git tag v0.1.2
   git push origin main --tags
   ```

For detailed publishing instructions, see [PUBLISHING.md](PUBLISHING.md).

---

## Getting Help

- Read existing code - it's well-documented
- Check tests for usage examples
- Open a [Discussion](https://github.com/sudhanvalabs/mlship/discussions) for questions
- Open an [Issue](https://github.com/sudhanvalabs/mlship/issues) for bugs

---

## Code of Conduct

Be respectful and constructive. We're all here to build something useful together.

---

**Thank you for contributing to mlship!** 🚀
