# Documentation Index

Welcome to the 4D Neural Cognition documentation! This comprehensive index helps you navigate all project documentation.

## 📚 Getting Started

New to the project? Start here:

1. **[README.md](../README.md)** - Project overview, quick start, and basic usage
2. **[Installation Guide](user-guide/INSTALLATION.md)** - Detailed installation instructions
3. **[Quick Start Tutorial](tutorials/QUICK_START_EVALUATION.md)** - Get up and running in 5 minutes
4. **[VISION.md](../VISION.md)** - Understand the project's goals and direction
5. **Example Scripts** - Try `python example.py` or `python app.py`

## 📖 Documentation Structure

Our documentation is organized according to international standards (ISO/IEC/IEEE 26512:2018):

### 👤 User Documentation
**[User Guide](user-guide/)** - Complete guide for end users
- **[Installation Guide](user-guide/INSTALLATION.md)** - Platform-specific installation
- **[FAQ](user-guide/FAQ.md)** - Frequently Asked Questions
- **[Glossary](user-guide/GLOSSARY.md)** - Terminology and definitions
- **[Tasks & Evaluation](user-guide/TASKS_AND_EVALUATION.md)** - Benchmark framework

### 👨‍💻 Developer Documentation
**[Developer Guide](developer-guide/)** - Guide for contributors
- **[Development Setup](developer-guide/README.md)** - Get started with development
- **[Architecture](ARCHITECTURE.md)** - System design and components
- **[Contributing Guidelines](../CONTRIBUTING.md)** - How to contribute
- **[Code of Conduct](../CODE_OF_CONDUCT.md)** - Community standards

### 🔧 API Documentation
**[API Reference](api/)** - Technical API documentation
- **[API.md](api/API.md)** - Complete API reference
- Function signatures and parameters
- Code examples and usage patterns
- Return values and exceptions

### 📖 Tutorials
**[Tutorials](tutorials/)** - Step-by-step learning guides
- **[Quick Start](tutorials/QUICK_START_EVALUATION.md)** - 5-minute start
- **[Getting Started](tutorials/GETTING_STARTED.md)** - Complete beginner guide
- **[Basic Simulation](tutorials/BASIC_SIMULATION.md)** - Running simulations
- **[Sensory Input](tutorials/SENSORY_INPUT.md)** - Providing input to networks
- **[Plasticity](tutorials/PLASTICITY.md)** - Learning and adaptation
- **[Custom Neuron Models](tutorials/CUSTOM_NEURON_MODELS.md)** - Using LIF and Izhikevich models
- **[Performance Optimization](tutorials/PERFORMANCE_OPTIMIZATION.md)** - Speed and efficiency guide
- Working examples and demonstrations
- Best practices and patterns

## 📂 Core Documentation

### Project Information

- **[VISION.md](../VISION.md)** - Project vision, goals, and roadmap
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

- **[IMPLEMENTATION_SUMMARY.md](../IMPLEMENTATION_SUMMARY.md)** - Tasks & Evaluation implementation
  - Implementation details
  - Technical decisions
  - Testing results

### Community & Support

- **[SUPPORT.md](../SUPPORT.md)** - How to get help
  - Community support channels
  - Reporting bugs and requesting features
  - Response times and expectations

- **[SECURITY.md](../SECURITY.md)** - Security policy
  - Reporting vulnerabilities
  - Security best practices
  - Supported versions

- **[CONTRIBUTING.md](../CONTRIBUTING.md)** - How to contribute
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

## 🔧 Technical Reference

### API & Usage

- **[API Reference](api/API.md)** - Complete API documentation
  - Brain Model API (Neuron, Synapse, BrainModel)
  - Simulation API (Simulation, callbacks)
  - Senses API (sensory input processing)
  - Storage API (JSON/HDF5 persistence)
  - Plasticity API (learning rules)
  - Cell Lifecycle API (aging, death, reproduction)
  - Tasks API (Environment, Task)
  - Evaluation API (BenchmarkConfig, BenchmarkSuite)
  - Knowledge Database API (KnowledgeDatabase, KnowledgeBasedTrainer)
  - Web API (REST & WebSocket)
  - Code examples for all functions

### Architecture & Design

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture
  - System design and components
  - Data flow diagrams
  - Design patterns used
  - Storage architecture (JSON/HDF5)
  - Web interface architecture (Flask/Socket.IO)
  - Performance considerations
  - Extension points and plugins
  - Future architecture plans

## 📂 Source Code Documentation

### Core Modules

Located in `src/` directory (2,600+ lines):

- **`brain_model.py`** (240 lines) - Core data structures
  - `Neuron` class - Individual neuron (4D coordinates, LIF parameters, lifecycle)
  - `Synapse` class - Synaptic connections (weight, delay, plasticity)
  - `BrainModel` class - Main container for neural network

- **`simulation.py`** (246 lines) - Main simulation engine
  - `Simulation` class - Orchestrates simulation steps
  - Neuron dynamics (Leaky Integrate-and-Fire model)
  - Spike detection and propagation
  - Statistics collection and callbacks

- **`cell_lifecycle.py`** (142 lines) - Neuron lifecycle management
  - Aging and health decay
  - Cell death conditions
  - Reproduction with mutations
  - Generational tracking and evolution

- **`plasticity.py`** (107 lines) - Learning algorithms
  - Hebbian learning rule ("fire together, wire together")
  - Weight decay and clipping
  - Extensible for STDP, BCM, and other rules

- **`senses.py`** (202 lines) - Sensory input processing
  - Input mapping to brain areas
  - Seven sensory modalities (vision, audition, etc.)
  - Digital sense for abstract data/text

- **`storage.py`** (175 lines) - Data persistence
  - JSON format (human-readable, small models)
  - HDF5 format (efficient, compressed, large models)
  - Save/load with full state preservation

- **`tasks.py`** (549 lines) - Task & environment framework
  - Environment and Task base classes
  - PatternClassificationTask (vision patterns)
  - TemporalSequenceTask (sequence learning)
  - Standardized metrics (accuracy, reward, reaction time)

- **`evaluation.py`** (399 lines) - Benchmark and comparison
  - BenchmarkConfig (reproducible configurations)
  - BenchmarkSuite (task collections)
  - ConfigurationComparator (side-by-side comparison)
  - Result tracking and JSON output

- **`knowledge_db.py`** (530 lines) - Knowledge database system
  - KnowledgeDatabase (SQLite storage)
  - KnowledgeBasedTrainer (pre-training and fallback)
  - Sample data population
  - Batch training support

### Web Application

- **`app.py`** - Flask web server
  - REST API endpoints (initialize, step, train, save, load)
  - WebSocket event handlers (real-time updates)
  - Session management
  - Real-time statistics broadcasting

- **`templates/index.html`** - Frontend HTML structure
  - Control panel, heatmap viewer, terminal, chat, logger
  
- **`static/js/app.js`** - Frontend JavaScript
  - Socket.IO client, heatmap rendering, UI interactions
  
- **`static/css/style.css`** - UI styling
  - Dark theme, responsive layout

### Configuration

- **`brain_base_model.json`** - Default model configuration
  - Lattice dimensions [x, y, z, w]
  - Neuron model parameters (LIF: tau_m, v_rest, v_threshold, etc.)
  - Brain area definitions (coordinates and purposes)
  - Sensory system configuration (input mappings)
  - Plasticity parameters (learning_rate, weight_bounds)
  - Cell lifecycle parameters (max_age, health_decay, mutation_rate)

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
- [x] Performance tuning guide (Dec 2025)
- [x] Custom neuron models guide (Dec 2025)
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
