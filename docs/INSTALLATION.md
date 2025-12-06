# Installation Guide

Complete installation instructions for the 4D Neural Cognition system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Quick Start](#quick-start)
3. [Detailed Installation](#detailed-installation)
4. [Platform-Specific Instructions](#platform-specific-instructions)
5. [Troubleshooting](#troubleshooting)
6. [Development Installation](#development-installation)
7. [Docker Installation](#docker-installation)
8. [Verification](#verification)

---

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **Python**: 3.8 or higher
- **RAM**: 4 GB
- **Disk Space**: 500 MB (1 GB recommended for large models)
- **Browser**: Chrome, Firefox, Edge, or Safari (latest version)

### Recommended Requirements

- **Operating System**: Linux (Ubuntu 20.04+) or macOS
- **Python**: 3.10 or higher
- **RAM**: 8 GB or more
- **Disk Space**: 2 GB or more
- **CPU**: Multi-core processor (4+ cores)
- **Browser**: Chrome or Firefox (latest)

### Optional Requirements

- **GPU**: For future GPU acceleration (currently CPU-only)
- **Git**: For development and version control

---

## Quick Start

For users who want to get started quickly:

```bash
# 1. Clone repository
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run example
python example.py

# 5. Or start web interface
python app.py
# Open browser to http://localhost:5000
```

---

## Detailed Installation

### Step 1: Install Python

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip
```

#### macOS

Using Homebrew:
```bash
brew install python@3.10
```

Or download from [python.org](https://www.python.org/downloads/)

#### Windows

Download installer from [python.org](https://www.python.org/downloads/) and run it.

**Important**: Check "Add Python to PATH" during installation.

### Step 2: Verify Python Installation

```bash
python3 --version
# Should output: Python 3.8.x or higher
```

### Step 3: Clone Repository

#### Using Git

```bash
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition
```

#### Using Download

1. Go to https://github.com/Thomas-Heisig/4D-Neural-Cognition
2. Click "Code" → "Download ZIP"
3. Extract ZIP file
4. Navigate to extracted folder

### Step 4: Create Virtual Environment

**Why?** Virtual environments isolate project dependencies.

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
# On Linux/macOS:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

You should see `(venv)` prefix in your terminal.

### Step 5: Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Dependencies installed**:
- `numpy` - Numerical computing
- `h5py` - HDF5 file format support
- `flask` - Web framework
- `flask-cors` - CORS support
- `flask-socketio` - WebSocket support
- `python-socketio` - Socket.IO client

### Step 6: Verify Installation

```bash
python example.py
```

You should see output like:
```
============================================================
4D Neural Cognition - Example Simulation
============================================================

1. Loading configuration from: brain_base_model.json
   Lattice shape: (20, 20, 20, 20)
   Total possible neurons: 160000
   Senses: ['vision', 'audition', ...]
   ...
```

---

## Platform-Specific Instructions

### Linux

#### Ubuntu/Debian

```bash
# Install system dependencies
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip build-essential

# Clone and install
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python example.py
```

#### Fedora/RHEL/CentOS

```bash
# Install system dependencies
sudo dnf install python3.10 python3-pip gcc

# Continue with standard installation
```

### macOS

#### Using Homebrew

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.10

# Clone and install
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run
python example.py
```

#### Apple Silicon (M1/M2)

Some packages may need ARM-specific installation:

```bash
# Use ARM-native Python
arch -arm64 brew install python@3.10

# Install with native arch
arch -arm64 pip install -r requirements.txt
```

### Windows

#### Using Command Prompt

```cmd
REM Clone repository
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition

REM Create virtual environment
python -m venv venv
venv\Scripts\activate.bat

REM Install dependencies
pip install -r requirements.txt

REM Run example
python example.py
```

#### Using PowerShell

```powershell
# Enable script execution (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Clone and install
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run
python example.py
```

#### Using WSL2 (Recommended for Windows)

```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Follow Linux instructions
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python example.py
```

---

## Troubleshooting

### Common Issues

#### Issue: "python3: command not found"

**Solution**:
```bash
# Try just 'python' instead of 'python3'
python --version

# Or install Python
# Ubuntu/Debian:
sudo apt install python3

# macOS:
brew install python

# Windows: Download from python.org
```

#### Issue: "pip: command not found"

**Solution**:
```bash
# Ubuntu/Debian:
sudo apt install python3-pip

# macOS:
python3 -m ensurepip

# Windows: Reinstall Python with pip
```

#### Issue: "Permission denied" errors

**Solution**:
```bash
# Don't use sudo with pip in virtual environment
# Instead, ensure virtual environment is activated
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

#### Issue: h5py installation fails

**Solution**:
```bash
# Ubuntu/Debian - install HDF5 libraries
sudo apt install libhdf5-dev

# macOS
brew install hdf5

# Then retry
pip install h5py
```

#### Issue: Flask app won't start

**Solution**:
```bash
# Check if port 5000 is in use
# Linux/macOS:
lsof -i :5000

# Windows:
netstat -ano | findstr :5000

# Use different port:
export FLASK_PORT=8080
python app.py
```

#### Issue: "ModuleNotFoundError: No module named 'src'"

**Solution**:
```bash
# Ensure you're in the project root directory
cd 4D-Neural-Cognition

# Ensure virtual environment is activated
source venv/bin/activate
```

#### Issue: Web interface doesn't load

**Solution**:
1. Check console for errors
2. Try different browser
3. Clear browser cache
4. Check firewall settings
5. Try `http://127.0.0.1:5000` instead of `localhost`

### Platform-Specific Issues

#### macOS: SSL Certificate Error

```bash
# Install certificates
/Applications/Python\ 3.10/Install\ Certificates.command
```

#### Windows: Long path errors

```powershell
# Enable long paths
reg add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1
```

#### Linux: Display issues with web interface

```bash
# If running over SSH, enable X11 forwarding
ssh -X user@host

# Or just use local browser to access server IP
```

---

## Development Installation

For contributors and developers:

### Additional Dependencies

```bash
# Activate virtual environment first
source venv/bin/activate

# Install development tools
pip install pytest pylint black flake8 mypy

# Optional: Install documentation tools
pip install sphinx sphinx-rtd-theme
```

### Git Hooks

```bash
# Set up pre-commit hooks
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
black src/ tests/
flake8 src/ tests/
EOF
chmod +x .git/hooks/pre-commit
```

### Running Tests

```bash
# Once tests are available
pytest tests/
pytest --cov=src tests/  # With coverage
```

---

## Docker Installation

### Using Docker

```bash
# Build image
docker build -t 4d-neural-cognition .

# Run container
docker run -p 5000:5000 4d-neural-cognition

# Access at http://localhost:5000
```

### Using Docker Compose

```bash
# Start services
docker-compose up

# Stop services
docker-compose down
```

**Note**: Docker support is planned but not yet implemented.

---

## Verification

### Verify Installation

Run these commands to ensure everything is working:

```bash
# 1. Check Python version
python3 --version
# Should be 3.8 or higher

# 2. Check dependencies
pip list | grep numpy
pip list | grep h5py
pip list | grep flask

# 3. Run example
python example.py
# Should complete without errors

# 4. Start web interface
python app.py &
# Open browser to http://localhost:5000
# Should see web interface

# 5. Run basic test
python -c "from src.brain_model import BrainModel; m = BrainModel(config_path='brain_base_model.json'); print('OK')"
# Should print: OK
```

### Verify Web Interface

1. Start app: `python app.py`
2. Open browser: `http://localhost:5000`
3. Click "Initialize Model"
4. Should see: "Model initialized successfully"
5. View heatmaps - should render without errors

### Performance Benchmark

```bash
# Run benchmark (if available)
python -m timeit -s "from src.simulation import Simulation; from src.brain_model import BrainModel; m = BrainModel(config_path='brain_base_model.json'); s = Simulation(m)" "s.step()"
```

---

## Next Steps

After successful installation:

1. **Read Documentation**:
   - [README.md](../README.md) - Overview and quick start
   - [VISION.md](../VISION.md) - Project vision and goals
   - [API.md](API.md) - Complete API reference

2. **Try Examples**:
   ```bash
   python example.py
   python app.py
   ```

3. **Explore Code**:
   - Check `src/` directory
   - Read source code comments
   - Experiment with parameters

4. **Join Community**:
   - Star the repository
   - Report issues
   - Contribute improvements

5. **Build Something**:
   - Create your own simulations
   - Add new features
   - Share your work

---

## Getting Help

If you encounter issues not covered here:

1. **Check Documentation**: Read [ISSUES.md](../ISSUES.md) for known issues
2. **Search Issues**: Check GitHub issues for similar problems
3. **Ask Community**: Open a new GitHub issue with:
   - Your OS and Python version
   - Complete error message
   - Steps to reproduce
   - What you've tried

4. **Contact Maintainers**: For sensitive issues, contact project maintainers directly

---

*Last Updated: December 2025*  
*Version: 1.0*
