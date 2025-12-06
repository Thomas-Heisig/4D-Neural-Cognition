# Developer Guide

Welcome to the 4D Neural Cognition Developer Guide! This guide helps contributors understand the codebase and development practices.

## 📖 Table of Contents

### Getting Started
1. **[Development Setup](development-setup.md)** - Set up your development environment
2. **[Contributing Guidelines](../../CONTRIBUTING.md)** - How to contribute
3. **[Code of Conduct](../../CODE_OF_CONDUCT.md)** - Community standards

### Architecture & Design
4. **[Architecture Overview](../ARCHITECTURE.md)** - System design and components
5. **[Design Patterns](design-patterns.md)** - Patterns used in the codebase
6. **[Code Organization](code-organization.md)** - How the code is structured
7. **[API Design](api-design.md)** - API design principles

### Development Guides
8. **[Coding Standards](coding-standards.md)** - Style and conventions
9. **[Testing Guide](testing-guide.md)** - Writing and running tests
10. **[Documentation Guide](documentation-guide.md)** - Writing documentation
11. **[Git Workflow](git-workflow.md)** - Branch and commit practices
12. **[CI/CD Setup](CI_CD_SETUP.md)** - Continuous integration and deployment guide

### Extension Guides
12. **[Adding Neuron Models](adding-neuron-models.md)** - Implement new neuron types
13. **[Adding Plasticity Rules](adding-plasticity-rules.md)** - Implement learning algorithms
14. **[Adding Senses](adding-senses.md)** - Add new sensory modalities
15. **[Adding Tasks](adding-tasks.md)** - Create benchmark tasks

### Advanced Topics
17. **[Performance Optimization](performance-optimization.md)** - Profiling and optimization
18. **[Debugging Techniques](debugging-techniques.md)** - Debug strategies
19. **[Security Considerations](../../SECURITY.md)** - Security best practices
20. **[CI/CD Setup](CI_CD_SETUP.md)** - Continuous integration setup guide

---

## 🚀 Quick Start for Developers

### 1. Fork and Clone

```bash
# Fork on GitHub, then:
git clone https://github.com/YOUR_USERNAME/4D-Neural-Cognition.git
cd 4D-Neural-Cognition
git remote add upstream https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
```

### 2. Set Up Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pylint black flake8 mypy
```

### 3. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

### 4. Make Changes

- Write code following [coding standards](coding-standards.md)
- Add tests for new functionality
- Update documentation

### 5. Test Your Changes

```bash
# Format code
black src/ tests/

# Check style
flake8 src/ tests/

# Run tests (when available)
pytest tests/

# Manual testing
python example.py
python app.py
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat: add your feature"
git push origin feature/your-feature-name
```

### 7. Create Pull Request

Open a PR on GitHub with a clear description of changes.

---

## 📁 Codebase Structure

```
4D-Neural-Cognition/
├── src/                    # Core source code
│   ├── brain_model.py     # Data structures (Neuron, Synapse, BrainModel)
│   ├── simulation.py      # Main simulation loop
│   ├── cell_lifecycle.py  # Aging, death, reproduction
│   ├── plasticity.py      # Learning rules
│   ├── senses.py          # Sensory input processing
│   ├── storage.py         # Data persistence
│   ├── tasks.py           # Task and environment framework
│   ├── evaluation.py      # Benchmark and comparison tools
│   └── knowledge_db.py    # Knowledge database system
│
├── app.py                 # Flask web application
├── example.py             # CLI example script
│
├── templates/             # HTML templates
│   └── index.html         # Web interface
│
├── static/                # Static web assets
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── docs/                  # Documentation
│   ├── user-guide/        # User documentation
│   ├── developer-guide/   # Developer documentation
│   ├── api/               # API reference
│   ├── tutorials/         # Tutorials
│   └── ARCHITECTURE.md    # Architecture documentation
│
├── examples/              # Example scripts
│   ├── simple_test.py
│   └── benchmark_example.py
│
└── tests/                 # Test suite (to be added)
```

---

## 🔧 Development Workflow

### Daily Development

```bash
# Update from upstream
git fetch upstream
git rebase upstream/main

# Make changes
# ... edit files ...

# Test locally
black .
flake8 .
python example.py

# Commit
git add .
git commit -m "type: description"

# Push
git push origin your-branch
```

### Before Submitting PR

- [ ] Code formatted with Black
- [ ] Passes flake8 checks
- [ ] Tests pass (if any)
- [ ] Documentation updated
- [ ] Commits are clean and well-formatted
- [ ] Branch is up-to-date with main

---

## 🎯 Contribution Areas

### Good First Issues

Perfect for new contributors:
- Documentation improvements
- Adding code comments
- Writing tests
- Fixing typos
- Adding examples

Check [good first issues](https://github.com/Thomas-Heisig/4D-Neural-Cognition/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

### Core Features

For experienced contributors:
- Implementing new neuron models
- Adding plasticity rules
- Performance optimization
- GPU acceleration
- Distributed computing

See [TODO.md](../../TODO.md) for planned features.

### Documentation

Always welcome:
- Tutorials and guides
- API documentation
- Architecture diagrams
- Code examples
- Translation (German)

### Testing

Critically needed:
- Unit tests for all modules
- Integration tests
- Performance benchmarks
- Test fixtures

---

## 📋 Development Standards

### Python Style

- **PEP 8** compliance
- **Black** for formatting (line length: 88)
- **Type hints** for all public functions
- **Docstrings** (Google style) for all public APIs

### Git Commit Messages

Format: `type(scope): subject`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `perf`: Performance
- `test`: Tests
- `chore`: Maintenance

Example: `feat(plasticity): add STDP learning rule`

### Code Review

All PRs require:
- Clean code following standards
- Tests for new functionality
- Updated documentation
- Approval from maintainer

---

## 🧪 Testing Guidelines

### Writing Tests

```python
import pytest
from src.brain_model import BrainModel

def test_neuron_creation():
    """Test that neurons are created correctly."""
    model = BrainModel(config={'lattice_shape': [10, 10, 10, 10]})
    neuron = model.add_neuron(5, 5, 5, 0)
    assert neuron.id is not None
    assert neuron.x == 5
    assert neuron.y == 5
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_brain_model.py

# With coverage
pytest --cov=src tests/
```

---

## 📚 API Documentation

### Writing Docstrings

Use Google-style docstrings:

```python
def add_synapse(self, pre_id: int, post_id: int, weight: float = 0.1) -> Synapse:
    """Add a synaptic connection between neurons.
    
    Args:
        pre_id: ID of pre-synaptic neuron
        post_id: ID of post-synaptic neuron
        weight: Synaptic strength (default: 0.1)
    
    Returns:
        The created Synapse object
    
    Raises:
        KeyError: If neuron IDs don't exist
        ValueError: If weight is invalid
    
    Example:
        >>> synapse = model.add_synapse(0, 5, weight=0.5)
        >>> print(synapse.weight)
        0.5
    """
```

---

## 🔍 Code Review Checklist

When reviewing PRs, check:

- [ ] Code follows style guidelines
- [ ] Tests are included and pass
- [ ] Documentation is updated
- [ ] No breaking changes (or documented)
- [ ] Commit messages are clear
- [ ] No merge conflicts
- [ ] Performance impact considered
- [ ] Security implications reviewed

---

## 🚨 Common Issues

### Import Errors

Ensure you're running from project root:
```bash
cd 4D-Neural-Cognition
python -m src.brain_model  # Not: python src/brain_model.py
```

### Linting Failures

```bash
# Auto-fix with Black
black .

# Check what flake8 wants
flake8 . --show-source
```

### Git Conflicts

```bash
# Update and rebase
git fetch upstream
git rebase upstream/main

# Fix conflicts manually
# Then:
git add .
git rebase --continue
```

---

## 🎓 Learning Resources

### Python
- [PEP 8 Style Guide](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Type Hints](https://docs.python.org/3/library/typing.html)

### NumPy
- [NumPy Documentation](https://numpy.org/doc/)
- [NumPy for MATLAB Users](https://numpy.org/doc/stable/user/numpy-for-matlab-users.html)

### Flask
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Tutorial](https://flask.palletsprojects.com/tutorial/)

### Testing
- [pytest Documentation](https://docs.pytest.org/)
- [Python Testing Best Practices](https://docs.python-guide.org/writing/tests/)

### Git
- [Pro Git Book](https://git-scm.com/book/)
- [Conventional Commits](https://www.conventionalcommits.org/)

---

## 📞 Developer Support

### Questions?

- Check [SUPPORT.md](../../SUPPORT.md)
- Ask in [GitHub Discussions](https://github.com/Thomas-Heisig/4D-Neural-Cognition/discussions)
- Review existing code and comments

### Found a Bug?

- Check [ISSUES.md](../../ISSUES.md)
- Search existing issues
- Create detailed bug report

### Want to Contribute?

- Read [CONTRIBUTING.md](../../CONTRIBUTING.md)
- Check [TODO.md](../../TODO.md) for tasks
- Look for "good first issue" labels

---

## 🏆 Recognition

Contributors are recognized in:
- README contributors section
- GitHub contributors page
- CHANGELOG release notes

Thank you for contributing!

---

*Last Updated: December 2025*  
*Developer Guide Version: 1.0*
