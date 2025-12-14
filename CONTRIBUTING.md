# Contributing to 4D Neural Cognition

First off, thank you for considering contributing to 4D Neural Cognition! It's people like you that make this project a great tool for the computational neuroscience and AI community.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Commit Guidelines](#commit-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check the [ISSUES.md](ISSUES.md) file and existing GitHub issues to avoid duplicates. When creating a bug report, include:

- **Clear descriptive title**
- **Detailed description** of the issue
- **Steps to reproduce** the behavior
- **Expected behavior**
- **Actual behavior**
- **Screenshots** if applicable
- **Environment details**:
  - OS and version
  - Python version
  - Package versions (`pip freeze`)
  - Browser (for UI issues)

### Suggesting Features

Feature suggestions are welcome! Before submitting:

1. Check [TODO.md](TODO.md) to see if it's already planned
2. Search existing GitHub issues
3. Consider if it aligns with project [VISION.md](VISION.md)

When suggesting a feature:

- **Use clear descriptive title**
- **Explain the motivation** - what problem does it solve?
- **Describe the solution** you'd like
- **Describe alternatives** you've considered
- **Provide examples** of how it would be used

### Contributing Code

We love code contributions! Here are some areas where you can help:

- **Bug fixes** - See [ISSUES.md](ISSUES.md)
- **New features** - See [TODO.md](TODO.md)
- **Performance improvements**
- **Documentation**
- **Tests**
- **Examples**

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic knowledge of NumPy and neural networks
- Familiarity with Flask (for web interface work)

### Setting Up Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/4D-Neural-Cognition.git
   cd 4D-Neural-Cognition
   ```

3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
   ```

4. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If exists
   ```

6. **Install development tools**:
   ```bash
   pip install pytest pylint black flake8 mypy
   ```

7. **Verify installation**:
   ```bash
   python example.py
   ```

### Understanding the Codebase

- Read [VISION.md](VISION.md) to understand project goals
- Review [README.md](README.md) for usage examples
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for technical details (if exists)
- Explore the code structure:
  ```
  src/
  ├── brain_model.py      # Core data structures
  ├── simulation.py       # Main simulation loop
  ├── cell_lifecycle.py   # Neuron death/reproduction
  ├── plasticity.py       # Learning rules
  ├── senses.py          # Sensory input
  └── storage.py         # Data persistence
  ```

## Development Workflow

### Creating a Branch

Always create a feature branch from `main`:

```bash
git checkout main
git pull upstream main
git checkout -b feature/your-feature-name
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding tests
- `perf/` - Performance improvements

Examples:
- `feature/add-stdp-plasticity`
- `fix/memory-leak-in-simulation`
- `docs/update-api-documentation`

### Making Changes

1. **Write code** following our [coding standards](#coding-standards)

2. **Test your changes**:
   ```bash
   pytest tests/
   python example.py  # Manual testing
   ```

3. **Check code quality**:
   ```bash
   black .              # Format code
   flake8 .            # Check style
   pylint src/         # Lint code
   mypy src/           # Type checking
   ```

4. **Update documentation** as needed

5. **Commit changes** following [commit guidelines](#commit-guidelines)

### Keeping Your Branch Updated

Regularly sync with upstream:

```bash
git fetch upstream
git rebase upstream/main
```

Resolve any conflicts that arise.

## Coding Standards

### Python Style

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces
- **Quotes**: Double quotes for strings
- **Imports**: Organized by standard library, third-party, local

### Code Formatting

Use **Black** for automatic formatting:

```bash
black src/ tests/ *.py
```

### Naming Conventions

- **Classes**: PascalCase (`BrainModel`, `Neuron`)
- **Functions**: snake_case (`initialize_neurons`, `feed_sense_input`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_AGE`, `DEFAULT_LEARNING_RATE`)
- **Private**: Prefix with underscore (`_internal_function`)

### Documentation

#### Docstrings

Use Google-style docstrings:

```python
def initialize_neurons(
    area_names: list[str] = None,
    density: float = 1.0
) -> None:
    """Initialize neurons in specified brain areas.

    Args:
        area_names: List of area names to initialize. If None, initializes
            all areas defined in configuration.
        density: Fraction of positions to fill with neurons, from 0 to 1.
            A density of 1.0 fills all positions, 0.5 fills half randomly.

    Raises:
        ValueError: If density is not in range [0, 1].
        KeyError: If area_name not found in configuration.

    Example:
        >>> sim = Simulation(model)
        >>> sim.initialize_neurons(["V1_like"], density=0.1)
    """
```

#### Comments

- Explain **why**, not **what**
- Use for complex algorithms
- Keep comments up-to-date

```python
# Good
# Use exponential decay to prevent runaway excitation
weight *= 0.99

# Bad
# Multiply weight by 0.99
weight *= 0.99
```

### Type Hints

Add type hints to all public functions:

```python
def add_synapse(
    self,
    pre_id: int,
    post_id: int,
    weight: float = 0.1
) -> Synapse:
    """Add a synaptic connection."""
    ...
```

### Error Handling

- Use exceptions for error conditions
- Provide informative error messages
- Don't catch exceptions silently

```python
# Good
if density < 0 or density > 1:
    raise ValueError(f"Density must be in [0, 1], got {density}")

# Bad
if density < 0 or density > 1:
    print("Invalid density")
    return
```

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

#### Type

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **perf**: Performance improvements
- **test**: Adding/updating tests
- **chore**: Build process, dependencies, etc.

#### Scope

The module or area affected (optional):
- `simulation`
- `plasticity`
- `web-ui`
- `storage`
- `docs`

#### Subject

- Use imperative mood ("add" not "added")
- Don't capitalize first letter
- No period at the end
- Max 50 characters

#### Body

- Explain what and why (not how)
- Wrap at 72 characters
- Separate from subject with blank line

#### Footer

- Reference issues: `Closes #123`, `Fixes #456`
- Breaking changes: `BREAKING CHANGE: description`

### Examples

```
feat(plasticity): add STDP learning rule

Implement spike-timing-dependent plasticity as alternative
to Hebbian learning. Includes exponential time window and
both potentiation and depression.

Closes #42
```

```
fix(simulation): prevent memory leak in long runs

Store only recent spike history instead of all spikes.
Limits memory growth while maintaining functionality.

Fixes #78
```

```
docs(readme): update installation instructions

Add troubleshooting section and Windows-specific steps.
```

## Pull Request Process

### Before Submitting

1. ✅ **Tests pass**: Run full test suite
2. ✅ **Code formatted**: Run Black
3. ✅ **Linting passes**: No errors from flake8/pylint
4. ✅ **Documentation updated**: README, docstrings, etc.
5. ✅ **Commits clean**: Well-formatted commit messages
6. ✅ **Branch updated**: Rebased on latest main

### Creating Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open PR** on GitHub

3. **Fill out template** with:
   - Description of changes
   - Motivation and context
   - Related issues
   - Testing performed
   - Screenshots (if UI changes)

4. **Request review** from maintainers

### PR Title Format

Same as commit messages:
```
feat(simulation): add GPU acceleration support
```

### PR Description Template

```markdown
## Description
Brief description of changes

## Motivation
Why is this change needed?

## Changes Made
- List of specific changes
- Another change

## Testing
- [ ] Unit tests added/updated
- [ ] Manual testing performed
- [ ] All tests pass

## Related Issues
Closes #123

## Screenshots
(if applicable)

## Checklist
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Maintainer reviews** your PR
2. **Address feedback** by pushing new commits
3. **Once approved**, maintainer will merge
4. **Celebrate** 🎉 Your contribution is now part of the project!

### After Merge

1. **Delete your branch**:
   ```bash
   git branch -d feature/your-feature-name
   git push origin --delete feature/your-feature-name
   ```

2. **Update main**:
   ```bash
   git checkout main
   git pull upstream main
   ```

## Testing Guidelines

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_simulation.py

# Specific test
pytest tests/test_simulation.py::test_initialize_neurons

# With coverage
pytest --cov=src tests/
```

### Writing Tests

- **Test file naming**: `test_<module>.py`
- **Test function naming**: `test_<functionality>`
- **One assertion per test** (when possible)
- **Use fixtures** for common setup
- **Use parametrize** for multiple inputs

Example:

```python
import pytest
from src.simulation import Simulation
from src.brain_model import BrainModel


@pytest.fixture
def basic_model():
    """Create a basic brain model for testing."""
    config = {
        "lattice_shape": [10, 10, 10, 10],
        "neuron_model": {...},
        ...
    }
    return BrainModel(config=config)


def test_initialize_neurons(basic_model):
    """Test neuron initialization in simulation."""
    sim = Simulation(basic_model, seed=42)
    sim.initialize_neurons(["V1_like"], density=0.5)
    
    # Check neurons were created
    assert len(sim.model.neurons) > 0
    
    # Check density is approximately correct
    # (with some tolerance for randomness)
    expected = volume * 0.5
    assert abs(len(sim.model.neurons) - expected) < expected * 0.2


@pytest.mark.parametrize("density", [0.0, 0.5, 1.0])
def test_initialize_neurons_density(basic_model, density):
    """Test different neuron densities."""
    sim = Simulation(basic_model, seed=42)
    sim.initialize_neurons(["V1_like"], density=density)
    
    if density == 0.0:
        assert len(sim.model.neurons) == 0
    else:
        assert len(sim.model.neurons) > 0
```

## Documentation

### What to Document

- **Public APIs**: All public functions/classes
- **Complex algorithms**: Non-obvious logic
- **Configuration options**: All config parameters
- **Examples**: How to use features
- **Architecture**: High-level design decisions

### Where to Document

- **Code**: Docstrings and comments
- **README.md**: Quick start, overview
- **Docs folder**: Detailed guides, tutorials
- **VISION.md**: Project direction
- **TODO.md**: Planned features
- **ISSUES.md**: Known problems

### Documentation Style

- **Clear and concise**
- **Use examples**
- **Assume basic knowledge** (Python, neural networks)
- **Link to related docs**
- **Keep up-to-date**

## Questions?

- Check [README.md](README.md)
- Check [ISSUES.md](ISSUES.md)
- Search existing GitHub issues
- Open new issue with question tag
- Contact maintainers

## Collaborative Research

Interested in research collaboration? See our [Collaborative Research Framework](docs/COLLABORATIVE_RESEARCH.md) for:
- Research partnerships
- Joint publications
- Data sharing protocols
- Authorship agreements
- Community research projects

## Recognition

Contributors are recognized in:
- README.md contributors section
- GitHub contributors page
- Release notes
- Co-authorship on relevant publications (see [Collaborative Research](docs/COLLABORATIVE_RESEARCH.md))

Thank you for contributing! 🎉

---

## Contact

**Project Maintainer**: Thomas Heisig  
**Email**: t_heisig@gmx.de  
**Location**: Ganderkesee, Germany  

**GitHub Repository**: https://github.com/Thomas-Heisig/4D-Neural-Cognition  
**Discussions**: https://github.com/Thomas-Heisig/4D-Neural-Cognition/discussions

---

*Last Updated: December 2025*
