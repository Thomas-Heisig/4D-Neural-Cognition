# CI/CD Setup Guide

This guide provides recommendations for setting up Continuous Integration and Continuous Deployment for the 4D Neural Cognition project.

## Overview

CI/CD helps ensure code quality by automatically running tests, checks, and deployments when code changes are pushed to the repository.

## Recommended Setup

### GitHub Actions (Recommended)

GitHub Actions is recommended because:
- Native integration with GitHub
- Free for public repositories
- Easy to configure
- Wide community support

### Suggested Workflows

#### 1. Test Workflow

Create `.github/workflows/tests.yml`:

```yaml
name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest -v --cov=src --cov-report=xml --cov-report=term
    
    - name: Upload coverage
      if: matrix.python-version == '3.12'
      uses: codecov/codecov-action@v3
      with:
        files: ./coverage.xml
        flags: unittests
```

#### 2. Code Quality Workflow

Create `.github/workflows/quality.yml`:

```yaml
name: Code Quality

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  quality:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.12'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pylint flake8 black mypy
        pip install -r requirements.txt
    
    - name: Check formatting with black
      run: black --check src/ tests/ app.py example.py
    
    - name: Lint with flake8
      run: flake8 src/ tests/ app.py example.py --max-line-length=100
    
    - name: Lint with pylint
      run: pylint src/ --rcfile=.pylintrc || true
    
    - name: Type check with mypy
      run: mypy src/ --ignore-missing-imports || true
```

#### 3. Documentation Workflow

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [ main ]

jobs:
  docs:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Check documentation links
      uses: gaurav-nelson/github-action-markdown-link-check@v1
      with:
        use-quiet-mode: 'yes'
        config-file: '.github/markdown-link-check-config.json'
```

## Configuration Files

### pytest.ini

Already exists in the project. Ensure it includes:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    -v
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-report=term-missing
```

### .pylintrc

Create for consistent linting:

```ini
[MASTER]
ignore=CVS,.git,__pycache__

[MESSAGES CONTROL]
disable=
    C0111,  # missing-docstring (we have many)
    R0913,  # too-many-arguments
    R0914,  # too-many-locals

[FORMAT]
max-line-length=100
```

### .flake8

Create for flake8 configuration:

```ini
[flake8]
max-line-length = 100
exclude = 
    .git,
    __pycache__,
    .pytest_cache,
    venv,
    build,
    dist
ignore = E203, W503
```

## Badge Integration

Add badges to README.md:

```markdown
[![Tests](https://github.com/Thomas-Heisig/4D-Neural-Cognition/workflows/Tests/badge.svg)](https://github.com/Thomas-Heisig/4D-Neural-Cognition/actions/workflows/tests.yml)
[![Code Quality](https://github.com/Thomas-Heisig/4D-Neural-Cognition/workflows/Code%20Quality/badge.svg)](https://github.com/Thomas-Heisig/4D-Neural-Cognition/actions/workflows/quality.yml)
[![codecov](https://codecov.io/gh/Thomas-Heisig/4D-Neural-Cognition/branch/main/graph/badge.svg)](https://codecov.io/gh/Thomas-Heisig/4D-Neural-Cognition)
```

## Code Coverage

### Setup Codecov

1. Go to [codecov.io](https://codecov.io)
2. Connect your GitHub repository
3. Add the badge to README.md
4. Coverage reports will be automatically uploaded by the test workflow

## Pre-commit Hooks

For local development, set up pre-commit hooks:

```bash
pip install pre-commit
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black
        language_version: python3.12
  
  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']
  
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

Install hooks:

```bash
pre-commit install
```

## Deployment

For web application deployment, consider:

1. **Heroku**: Easy deployment with Procfile
2. **Docker**: Containerized deployment
3. **AWS/GCP/Azure**: Cloud hosting

Example Procfile for Heroku:

```
web: python app.py
```

## Security Scanning

Add security scanning:

```yaml
name: Security

on:
  push:
    branches: [ main ]
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  security:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o bandit-report.json || true
    
    - name: Run Safety check
      run: |
        pip install safety
        safety check --json || true
```

## Implementation Steps

1. Create `.github/workflows/` directory
2. Add workflow files one at a time
3. Test each workflow by pushing to a feature branch
4. Monitor Actions tab in GitHub repository
5. Fix any issues that arise
6. Add badges to README once workflows are stable
7. Set up branch protection rules to require passing tests

## Best Practices

- **Run tests locally** before pushing
- **Keep workflows fast** (< 5 minutes if possible)
- **Use matrix testing** for multiple Python versions
- **Cache dependencies** to speed up workflows
- **Fail fast** to get quick feedback
- **Monitor workflow runs** regularly

## Troubleshooting

### Tests fail in CI but pass locally

- Check Python version differences
- Verify all dependencies are in requirements.txt
- Check for OS-specific issues (Windows vs Linux)
- Ensure test data/fixtures are included in repository

### Workflows are slow

- Use caching for pip packages
- Run fewer Python versions
- Parallelize independent jobs
- Use faster runners if needed

## Further Reading

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [pytest Documentation](https://docs.pytest.org/)
- [Codecov Documentation](https://docs.codecov.com/)

---

*Last Updated: December 2025*  
*Status: Recommended setup, not yet implemented*
