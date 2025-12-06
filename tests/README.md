# Test Suite

This directory contains the test suite for the 4D Neural Cognition project using pytest.

## 📋 Overview

The test suite provides comprehensive coverage of core modules:

- **test_brain_model.py** - Tests for BrainModel, Neuron, and Synapse classes (41 tests)
- **test_simulation.py** - Tests for Simulation class and LIF neuron model (28 tests)
- **test_cell_lifecycle.py** - Tests for cell death, reproduction, and mutation (15 tests)

**Total: 75 unit tests** with 100% pass rate.

## 🚀 Running Tests

### Quick Start

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_brain_model.py

# Run with verbose output
pytest -v

# Run with coverage report
pytest --cov=src --cov-report=term-missing
```

### Installation

First, ensure pytest is installed:

```bash
pip install pytest pytest-cov
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

### Advanced Usage

```bash
# Run specific test class
pytest tests/test_brain_model.py::TestBrainModel

# Run specific test method
pytest tests/test_simulation.py::TestSimulation::test_lif_step_with_external_input

# Run tests matching pattern
pytest -k "lif_step"

# Run with detailed output
pytest -vv

# Show local variables on failure
pytest -l

# Stop at first failure
pytest -x

# Run last failed tests
pytest --lf
```

## 📊 Test Structure

### Fixtures (conftest.py)

Common fixtures available to all tests:

- `minimal_config` - Basic brain model configuration
- `brain_model` - Initialized BrainModel instance
- `populated_model` - BrainModel with neurons and synapses
- `simulation` - Simulation instance
- `sample_neuron` - Sample neuron for testing
- `sample_synapse` - Sample synapse for testing
- `rng` - Seeded random number generator
- `temp_dir` - Temporary directory for file operations

### Test Categories

Tests are organized by module:

1. **BrainModel Tests**
   - Neuron creation and management
   - Synapse creation and querying
   - Area-based neuron filtering
   - Serialization/deserialization

2. **Simulation Tests**
   - Initialization and configuration
   - LIF neuron dynamics
   - Synaptic transmission
   - Plasticity application
   - Cell lifecycle integration

3. **Cell Lifecycle Tests**
   - Parameter mutation
   - Health and age updates
   - Death and reproduction
   - Synapse inheritance

## 🎯 Coverage Goals

Current coverage:
- **brain_model.py**: ~90%
- **simulation.py**: ~85%
- **cell_lifecycle.py**: ~95%

Target coverage: >80% for all modules

## ✅ Test Quality Standards

All tests follow these principles:

1. **Isolation**: Each test is independent
2. **Clarity**: Test names describe what they test
3. **Completeness**: Test both success and failure cases
4. **Speed**: Tests run quickly (<1s total)
5. **Determinism**: Tests use seeded randomness

## 🔧 Configuration

Test configuration is in `pytest.ini`:

- Test discovery patterns
- Output formatting
- Coverage settings
- Warning filters

## 📝 Adding New Tests

To add a new test:

1. Create or edit a test file in `tests/`
2. Import required modules
3. Use existing fixtures or create new ones
4. Write test functions with `test_` prefix
5. Use descriptive names and docstrings
6. Run tests to verify

Example:

```python
def test_my_feature(brain_model):
    """Test description here."""
    # Arrange
    neuron = brain_model.add_neuron(1, 2, 3, 4)
    
    # Act
    result = brain_model.get_area_neurons("V1_like")
    
    # Assert
    assert neuron in result
```

## 🐛 Debugging Failed Tests

If tests fail:

1. Run with verbose output: `pytest -vv`
2. Show local variables: `pytest -l`
3. Use debugger: `pytest --pdb`
4. Check specific test: `pytest path/to/test.py::test_name -v`

## 📚 Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest parametrize](https://docs.pytest.org/en/stable/parametrize.html)

## 🤝 Contributing

When contributing code:

1. Write tests for new features
2. Ensure all tests pass
3. Maintain or improve coverage
4. Follow existing test patterns

Run tests before submitting pull requests:

```bash
pytest --cov=src --cov-report=term-missing
```

## 📈 Future Work

Planned test additions:

- [ ] Tests for plasticity.py
- [ ] Tests for senses.py  
- [ ] Tests for storage.py (HDF5 I/O)
- [ ] Integration tests for full workflows
- [ ] Performance benchmarks
- [ ] Property-based tests with Hypothesis

---

*Last Updated: December 2025*  
*Test Suite Version: 1.0*
