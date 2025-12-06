# Documentation Index

Welcome to the 4D Neural Cognition documentation! This index helps you find the information you need.

## 📚 Getting Started

New to the project? Start here:

1. **[README.md](../README.md)** - Project overview, quick start, and basic usage
2. **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation instructions
3. **[VISION.md](../VISION.md)** - Understand the project's goals and direction
4. **Example Scripts** - Try `python example.py` or `python app.py`

## 📖 Core Documentation

### Project Information

- **[VISION.md](../VISION.md)** - Project vision, goals, roadmap, and philosophy
  - What are we building and why?
  - Long-term and short-term goals
  - Current state and future plans
  - Research questions and use cases

- **[TODO.md](../TODO.md)** - Planned features and task tracking
  - Prioritized task lists
  - Feature roadmap
  - Release planning
  - How to pick a task to work on

- **[ISSUES.md](../ISSUES.md)** - Known bugs, limitations, and technical debt
  - Active bugs and their workarounds
  - Performance limitations
  - Platform compatibility issues
  - Security considerations

- **[CHANGELOG.md](../CHANGELOG.md)** - Version history and release notes
  - What changed in each version
  - New features, bug fixes, breaking changes
  - Migration guides

### Development & Community

- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - How to contribute to the project
  - Development setup
  - Coding standards
  - Commit guidelines
  - Pull request process
  - Testing guidelines

- **[CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md)** - Community guidelines
  - Expected behavior
  - Unacceptable behavior
  - Enforcement policies
  - Contact information

- **[LICENSE](../LICENSE)** - MIT License terms

## 🔧 Technical Documentation

### API & Usage

- **[API.md](API.md)** - Complete API reference
  - Brain Model API
  - Simulation API
  - Senses API
  - Storage API
  - Plasticity API
  - Cell Lifecycle API
  - Web API (REST & WebSocket)
  - Code examples for all functions

### Architecture

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture details
  - System design and components
  - Data flow diagrams
  - Design patterns used
  - Storage architecture
  - Web interface architecture
  - Performance considerations
  - Extension points

### Installation

- **[INSTALLATION.md](INSTALLATION.md)** - Detailed setup guide
  - System requirements
  - Platform-specific instructions (Linux, macOS, Windows)
  - Troubleshooting common issues
  - Development installation
  - Docker installation (planned)
  - Verification steps

## 📂 Source Code Documentation

### Core Modules

Located in `src/` directory:

- **`brain_model.py`** - Core data structures
  - `Neuron` class - Individual neuron representation
  - `Synapse` class - Synaptic connections
  - `BrainModel` class - Main container for neural network

- **`simulation.py`** - Main simulation engine
  - `Simulation` class - Orchestrates simulation steps
  - Neuron dynamics (LIF model)
  - Spike detection and propagation
  - Statistics collection

- **`cell_lifecycle.py`** - Neuron lifecycle management
  - Aging and health decay
  - Cell death conditions
  - Reproduction with mutations
  - Generational tracking

- **`plasticity.py`** - Learning algorithms
  - Hebbian learning rule
  - Weight decay
  - Extensible for STDP, BCM, etc.

- **`senses.py`** - Sensory input processing
  - Input mapping to brain areas
  - Multiple sensory modalities
  - Digital sense for text/data

- **`storage.py`** - Data persistence
  - JSON format (human-readable)
  - HDF5 format (efficient, compressed)
  - Save/load functionality

### Web Application

- **`app.py`** - Flask web server
  - REST API endpoints
  - WebSocket event handlers
  - Session management
  - Real-time updates

- **`templates/index.html`** - Frontend HTML structure
- **`static/js/app.js`** - Frontend JavaScript logic
- **`static/css/style.css`** - UI styling

### Configuration

- **`brain_base_model.json`** - Default model configuration
  - Lattice dimensions
  - Neuron model parameters
  - Brain area definitions
  - Sensory system configuration

## 🎯 Usage Examples

### Command Line

```bash
# Run example simulation
python example.py
```

See `example.py` for annotated code demonstrating:
- Model initialization
- Neuron creation
- Synapse creation
- Sensory input
- Simulation execution
- State persistence

### Web Interface

```bash
# Start web application
python app.py
# Open http://localhost:5000
```

Features:
- Interactive model configuration
- Real-time visualization
- Training controls
- System monitoring

### Programmatic Usage

See [API.md](API.md) for complete examples of:
- Creating custom simulations
- Implementing new learning rules
- Adding custom sensory inputs
- Monitoring and analysis
- Batch processing

## 🔍 Finding Information

### By Task

- **Installing the software** → [INSTALLATION.md](INSTALLATION.md)
- **Understanding the vision** → [VISION.md](../VISION.md)
- **Learning the API** → [API.md](API.md)
- **Contributing code** → [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Reporting bugs** → [ISSUES.md](../ISSUES.md)
- **Planning features** → [TODO.md](../TODO.md)
- **Understanding design** → [ARCHITECTURE.md](ARCHITECTURE.md)

### By Role

**Researchers**:
1. [VISION.md](../VISION.md) - Research questions
2. [API.md](API.md) - How to use the system
3. [ARCHITECTURE.md](ARCHITECTURE.md) - How it works
4. [example.py](../example.py) - Working examples

**Developers**:
1. [CONTRIBUTING.md](../CONTRIBUTING.md) - Development workflow
2. [ARCHITECTURE.md](ARCHITECTURE.md) - System design
3. [API.md](API.md) - API reference
4. [TODO.md](../TODO.md) - What to work on

**Users**:
1. [README.md](../README.md) - Quick start
2. [INSTALLATION.md](INSTALLATION.md) - Setup guide
3. [API.md](API.md) - Usage examples
4. Web interface tutorial (in-app)

**Contributors**:
1. [CODE_OF_CONDUCT.md](../CODE_OF_CONDUCT.md) - Community rules
2. [CONTRIBUTING.md](../CONTRIBUTING.md) - How to contribute
3. [TODO.md](../TODO.md) - What needs doing
4. [ISSUES.md](../ISSUES.md) - Known problems

## 📱 Quick Reference

### Common Commands

```bash
# Installation
pip install -r requirements.txt

# Run example
python example.py

# Start web interface
python app.py

# Development tools
pytest tests/           # Run tests (when available)
black src/             # Format code
flake8 src/            # Check style
```

### Important Links

- **Repository**: https://github.com/Thomas-Heisig/4D-Neural-Cognition
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

### Key Concepts

- **4D Lattice**: Neurons organized in (x, y, z, w) coordinates
- **LIF Model**: Leaky Integrate-and-Fire neuron dynamics
- **Hebbian Learning**: "Fire together, wire together"
- **Cell Lifecycle**: Aging, death, reproduction
- **Digital Sense**: Novel sensory modality for data

## 🆘 Getting Help

1. **Check documentation** - Use this index to find relevant docs
2. **Search issues** - Someone may have had the same problem
3. **Read FAQ** - See [README.md](../README.md) FAQ section
4. **Ask the community** - Open a GitHub issue or discussion
5. **Contact maintainers** - For sensitive issues

## 📝 Documentation Standards

Our documentation follows these principles:

- **Clear and concise** - Easy to understand
- **Well-organized** - Easy to navigate
- **Up-to-date** - Reflects current state
- **Example-rich** - Show, don't just tell
- **Accessible** - For various skill levels
- **International** - English primary, multilingual support

### Documentation Quality

Each document includes:
- Clear purpose statement
- Table of contents (for long docs)
- Examples and code snippets
- Last updated date
- Version information

## 🔄 Keeping Updated

Documentation is actively maintained. Check:

- **[CHANGELOG.md](../CHANGELOG.md)** for version updates
- **[TODO.md](../TODO.md)** for upcoming changes
- **[ISSUES.md](../ISSUES.md)** for current known issues
- **Git commits** for latest changes

## 🌟 Documentation Roadmap

Planned documentation improvements:

- [ ] Video tutorials
- [ ] Interactive Jupyter notebooks
- [ ] API reference with type hints
- [ ] Architecture diagrams (enhanced)
- [ ] Performance tuning guide
- [ ] Advanced usage patterns
- [ ] Research paper (planned)
- [ ] Multi-language translations

See [TODO.md](../TODO.md) for complete roadmap.

---

## 📧 Feedback

Help us improve documentation:

- **Found an error?** Open an issue
- **Something unclear?** Ask a question
- **Have suggestions?** Submit a PR
- **Want more examples?** Request them

We appreciate all feedback!

---

*Last Updated: December 2025*  
*Documentation Version: 1.0*

---

**Navigation**: [↑ Top](#documentation-index) | [README](../README.md) | [Contributing](../CONTRIBUTING.md)
