# Coding Standards

This document defines the coding standards and conventions for the 4D Neural Cognition project.

## General Principles

- **Clarity over cleverness**: Write code that is easy to understand
- **Consistency**: Follow established patterns in the codebase
- **Documentation**: Document public APIs and complex logic
- **Testing**: Write tests for new functionality
- **Security**: Validate inputs and handle errors properly

---

## Python Style

### PEP 8 Compliance

Follow [PEP 8](https://pep8.org/) with these specific guidelines:

- **Line Length**: 88 characters (Black default)
- **Indentation**: 4 spaces (no tabs)
- **Imports**: Organized in three groups (standard library, third-party, local)
- **Naming**: Follow PEP 8 conventions

### Code Formatting

Use [Black](https://github.com/psf/black) for automatic code formatting:

```bash
black src/ tests/
```

### Linting

Use [flake8](https://flake8.pycqa.org/) for style checking:

```bash
flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
```

---

## Naming Conventions

### Variables and Functions

```python
# Good: descriptive, lowercase with underscores
neuron_count = 100
spike_threshold = -55.0

def calculate_membrane_potential(neuron, current):
    """Calculate new membrane potential."""
    pass

# Bad: unclear, inconsistent
n = 100
th = -55.0

def calcMP(n, c):
    pass
```

### Classes

```python
# Good: PascalCase for classes
class BrainModel:
    pass

class NeuronNetwork:
    pass

# Bad: lowercase or underscores
class brain_model:
    pass
```

### Constants

```python
# Good: UPPERCASE with underscores
DEFAULT_LEARNING_RATE = 0.01
MAX_SYNAPSES_PER_NEURON = 1000

# Bad: lowercase
default_learning_rate = 0.01
```

### Private Members

```python
class Neuron:
    def __init__(self):
        self.membrane_potential = -70.0  # Public
        self._spike_history = []          # Protected (internal use)
        self.__id_counter = 0             # Private (name mangled)
```

---

## Type Hints

Use type hints for all public functions and methods:

```python
from typing import List, Dict, Optional, Tuple

def add_neuron(
    self,
    x: int,
    y: int,
    z: int,
    w: int,
    params: Optional[Dict[str, float]] = None
) -> Neuron:
    """Add a neuron to the model.
    
    Args:
        x, y, z, w: 4D coordinates
        params: Optional neuron parameters
        
    Returns:
        The created Neuron object
    """
    pass
```

---

## Documentation

### Docstrings

Use Google-style docstrings for all public functions, classes, and modules:

```python
def hebbian_update(
    synapse: Synapse,
    pre_active: bool,
    post_active: bool,
    model: BrainModel,
) -> None:
    """Apply Hebbian learning rule to update synapse weight.

    Implements the classic Hebbian learning principle:
    "Cells that fire together, wire together."
    
    This creates a correlation-based learning rule where:
    - Correlated activity strengthens connections (Long-Term Potentiation)
    - Uncorrelated activity weakens connections (Long-Term Depression)

    Args:
        synapse: The synapse to update.
        pre_active: Whether the presynaptic neuron spiked this step.
        post_active: Whether the postsynaptic neuron spiked this step.
        model: The brain model containing plasticity configuration.
        
    Raises:
        ValueError: If synapse is invalid.
        
    Example:
        >>> hebbian_update(syn, True, True, model)
        # Weight increased due to correlated activity
    """
    pass
```

### Module Docstrings

Every module should have a docstring explaining its purpose:

```python
"""Hebbian plasticity rules for synapse learning.

This module implements learning rules that modify synaptic weights
based on neural activity patterns. The primary rule is Hebbian learning,
which strengthens connections between co-active neurons.
"""
```

### Inline Comments

Add comments for complex logic:

```python
# Calculate spike time difference (Δt)
# Positive: pre fired before post (causal relationship)
# Negative: post fired before pre (acausal relationship)
delta_t = post_spike_time - pre_spike_time

if delta_t > 0:
    # Causal timing: Pre→Post suggests synapse contributed to post firing
    # Apply Long-Term Potentiation (LTP) with exponential decay
    delta_w = a_plus * math.exp(-delta_t / tau_plus)
```

---

## Error Handling

### Input Validation

Always validate inputs at public API boundaries:

```python
def feed_sense_input(
    model: BrainModel,
    sense_name: str,
    input_matrix: np.ndarray,
    z_layer: int = 0,
) -> None:
    """Feed sensory input to neurons."""
    # Validate input type
    if not isinstance(input_matrix, np.ndarray):
        raise TypeError(
            f"input_matrix must be a numpy array, got {type(input_matrix).__name__}"
        )
    
    # Validate input is 2D
    if input_matrix.ndim != 2:
        raise ValueError(
            f"input_matrix must be 2D, got shape {input_matrix.shape}"
        )
    
    # Validate sense exists
    if sense_name not in model.get_senses():
        raise ValueError(f"Unknown sense: '{sense_name}'")
```

### Error Messages

Provide clear, actionable error messages:

```python
# Good: specific and helpful
raise ValueError(
    f"z_layer {z_layer} out of range for sense '{sense_name}'. "
    f"Valid range: [0, {z_depth - 1}]"
)

# Bad: vague
raise ValueError("Invalid z_layer")
```

### Exception Types

Use appropriate exception types:

- `ValueError`: Invalid values or arguments
- `TypeError`: Wrong type
- `KeyError`: Missing dictionary key
- `IndexError`: Invalid index
- `FileNotFoundError`: File operations
- `RuntimeError`: General runtime errors

---

## Security Best Practices

### Input Sanitization

Always sanitize user inputs:

```python
def validate_filepath(filepath: str, allowed_dir: Path) -> Path:
    """Validate and sanitize file paths."""
    try:
        file_path = Path(filepath).resolve()
        allowed_path = allowed_dir.resolve()
        
        # Check if file is within allowed directory
        if not str(file_path).startswith(str(allowed_path)):
            raise ValueError("Access denied: Path outside allowed directory")
        
        return file_path
    except Exception as e:
        raise ValueError(f"Invalid file path: {str(e)}")
```

### Environment Variables

Use environment variables for sensitive data:

```python
# Good: use environment variable
secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-key-change-in-production')

# Bad: hardcoded secrets
secret_key = 'my-secret-key-12345'
```

### Size Limits

Prevent memory exhaustion with size limits:

```python
# Limit input size to prevent DoS
if len(input_data) > MAX_INPUT_SIZE:
    raise ValueError(f"Input too large (max {MAX_INPUT_SIZE} bytes)")
```

---

## Testing

### Test Organization

```python
# tests/test_brain_model.py
import pytest
from src.brain_model import BrainModel, Neuron

class TestNeuron:
    """Tests for Neuron class."""
    
    def test_neuron_creation(self):
        """Test that neurons are created with correct attributes."""
        neuron = Neuron(0, x=5, y=5, z=0, w=0)
        assert neuron.x == 5
        assert neuron.y == 5
    
    def test_neuron_spike_threshold(self):
        """Test that neurons spike when threshold is reached."""
        neuron = Neuron(0, x=0, y=0, z=0, w=0, params={'threshold': -55.0})
        neuron.membrane_potential = -54.0
        # Assert spike behavior
```

### Test Naming

- Test files: `test_<module>.py`
- Test classes: `Test<ClassName>`
- Test functions: `test_<feature>_<scenario>`

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture
def basic_model():
    """Create a basic brain model for testing."""
    config = {
        'lattice_shape': [10, 10, 5, 2],
        'neuron_model': {'threshold': -55.0}
    }
    return BrainModel(config=config)
```

---

## Logging

### Log Levels

Use appropriate log levels:

```python
import logging

logger = logging.getLogger(__name__)

# DEBUG: Detailed diagnostic information
logger.debug(f"Processing neuron {neuron_id} at position ({x}, {y}, {z}, {w})")

# INFO: General informational messages
logger.info(f"Model initialized: {lattice_shape}")

# WARNING: Warning messages (recoverable issues)
logger.warning(f"Input dimension mismatch: expected {expected}, got {actual}")

# ERROR: Error messages (failures)
logger.error(f"Failed to load model: {str(e)}")

# CRITICAL: Critical errors (system failure)
logger.critical(f"Unrecoverable error in simulation: {str(e)}")
```

### Structured Logging

Include context in log messages:

```python
# Good: structured with context
logger.info(
    f"Simulation step {step} completed: "
    f"{spike_count} spikes, {active_neurons} active neurons"
)

# Bad: vague
logger.info("Step done")
```

---

## Git Commit Messages

### Format

```
type(scope): subject

body (optional)

footer (optional)
```

### Types

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code formatting (no functional changes)
- `refactor`: Code restructuring (no functional changes)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `security`: Security fixes

### Examples

```
feat(plasticity): add STDP learning rule

Implement spike-timing-dependent plasticity with exponential
decay windows for both potentiation and depression.

Closes #42

---

fix(senses): validate input dimensions

Add proper validation for sensory input arrays to prevent
dimension mismatches and improve error messages.

Fixes #38

---

docs(api): add docstrings to storage module

Complete API documentation for save/load functions with
examples and parameter descriptions.
```

---

## Code Review Guidelines

### What to Look For

- [ ] Code follows style guidelines
- [ ] Functions have docstrings
- [ ] Inputs are validated
- [ ] Errors have clear messages
- [ ] No hardcoded secrets or credentials
- [ ] Tests are included for new functionality
- [ ] Documentation is updated
- [ ] No unnecessary complexity
- [ ] Variable names are clear and descriptive
- [ ] Comments explain "why", not "what"

### Giving Feedback

- Be constructive and specific
- Suggest alternatives, don't just criticize
- Acknowledge good code
- Focus on the code, not the person

---

## Performance Considerations

### Premature Optimization

> "Premature optimization is the root of all evil" - Donald Knuth

- Write clear, correct code first
- Profile before optimizing
- Focus on algorithmic improvements over micro-optimizations
- Document performance-critical sections

### NumPy Best Practices

```python
# Good: vectorized operations
membrane_potentials = np.array([n.membrane_potential for n in neurons])
membrane_potentials += input_currents
membrane_potentials *= decay_factor

# Bad: loops (slower)
for i, neuron in enumerate(neurons):
    neuron.membrane_potential += input_currents[i]
    neuron.membrane_potential *= decay_factor
```

---

## Anti-Patterns to Avoid

### Magic Numbers

```python
# Bad: unexplained constants
if voltage > -55:
    spike = True

# Good: named constants
SPIKE_THRESHOLD = -55.0  # mV
if voltage > SPIKE_THRESHOLD:
    spike = True
```

### Deep Nesting

```python
# Bad: deeply nested
if model:
    if neurons:
        for neuron in neurons:
            if neuron.active:
                if neuron.membrane_potential > threshold:
                    # Do something

# Good: early returns and flat structure
if not model or not neurons:
    return

for neuron in neurons:
    if not neuron.active:
        continue
    
    if neuron.membrane_potential > threshold:
        # Do something
```

### Mutable Default Arguments

```python
# Bad: mutable default
def add_neurons(model, neurons=[]):
    neurons.append(new_neuron)  # Modifies shared list!

# Good: None as default
def add_neurons(model, neurons=None):
    if neurons is None:
        neurons = []
    neurons.append(new_neuron)
```

---

## Resources

### Python
- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

### Tools
- [Black](https://github.com/psf/black) - Code formatter
- [flake8](https://flake8.pycqa.org/) - Linter
- [mypy](http://mypy-lang.org/) - Type checker
- [pytest](https://docs.pytest.org/) - Testing framework

### Security
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security Best Practices](https://python.readthedocs.io/en/stable/library/security_warnings.html)

---

*Last Updated: December 2025*  
*Version: 1.0*
