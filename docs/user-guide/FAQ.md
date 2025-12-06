# Frequently Asked Questions (FAQ)

## Table of Contents

- [General Questions](#general-questions)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Core Concepts](#core-concepts)
- [Technical Questions](#technical-questions)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## General Questions

### What is 4D Neural Cognition?

4D Neural Cognition is a brain simulation system that implements neural networks in four-dimensional space (x, y, z, w). It combines biological principles (like the Leaky Integrate-and-Fire neuron model, Hebbian plasticity, and cell lifecycle) with digital computing to explore novel computational paradigms.

### Why 4D instead of 3D?

The fourth dimension (w) provides additional organizational structure that can represent:
- Different modalities or processing streams
- Temporal depth or time slices
- Abstract organizational principles
- Hierarchical layers or representations

It's a flexible dimension that can be adapted to various use cases.

### Is this biologically accurate?

No, it's biologically *inspired*, not a detailed biological simulation. We use simplified models:
- **LIF neurons**: Simplified from real ion channel dynamics
- **Hebbian plasticity**: Simplified learning rule
- **Cell lifecycle**: Abstracted aging and reproduction

The focus is on computational exploration rather than biological accuracy.

### What can I use this for?

4D Neural Cognition is designed for:
- **Research**: Testing hypotheses about neural organization
- **Education**: Teaching computational neuroscience concepts
- **Applications**: Multi-dimensional pattern recognition, time-series forecasting
- **Experimentation**: Exploring alternative neural architectures

### What makes this different from other neural network frameworks?

Key differences:
- **4D spatial structure**: Novel organizational dimension
- **Cell lifecycle**: Neurons age, die, and reproduce
- **Digital sense**: Processing abstract data patterns
- **Biological inspiration**: Combines biological and digital principles
- **Real-time visualization**: Interactive web interface

---

## Installation

### What are the system requirements?

**Minimum**:
- Python 3.8+
- 4 GB RAM
- 500 MB disk space

**Recommended**:
- Python 3.10+
- 8+ GB RAM
- Multi-core CPU

See [INSTALLATION.md](../INSTALLATION.md) for details.

### Which Python version should I use?

Python 3.8 is the minimum, but we recommend Python 3.10 or 3.11 for best performance and compatibility.

### Do I need a GPU?

No, the current version runs on CPU only. GPU acceleration is planned for future releases (see [TODO.md](../../TODO.md)).

### Can I run this on Windows?

Yes, but we primarily test on Linux and macOS. Windows users may encounter minor issues. We recommend using WSL2 (Windows Subsystem for Linux) for the best experience.

### Installation fails with error "command not found"

Make sure Python and pip are properly installed and in your PATH:

```bash
python3 --version
pip3 --version
```

If not found, reinstall Python and ensure "Add to PATH" is selected.

### Dependencies won't install

Try upgrading pip first:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If specific packages fail, install them individually to see detailed error messages.

---

## Getting Started

### How do I run my first simulation?

```bash
# Command line
python example.py

# Web interface (recommended for beginners)
python app.py
# Open http://localhost:5000
```

See [README.md](../../README.md) for more details.

### What's the difference between example.py and app.py?

- **example.py**: Command-line demonstration of programmatic API
- **app.py**: Web-based interface with visualization and controls

Use `app.py` if you want a graphical interface, `example.py` if you want to write code.

### How do I modify the configuration?

Edit `brain_base_model.json` to change:
- Lattice dimensions
- Neuron parameters
- Brain areas
- Sensory systems
- Learning rates

See [API.md](../API.md) for configuration options.

### What sensory inputs are available?

Seven senses:
1. **Vision** (V1_like)
2. **Audition** (A1_like)
3. **Somatosensory** (S1_like)
4. **Taste** (Taste_like)
5. **Smell** (Olfactory_like)
6. **Vestibular** (balance/orientation)
7. **Digital** (novel: for abstract data)

### How do I feed input to the system?

```python
from src.senses import feed_sense_input
import numpy as np

# Vision input (2D array)
vision_data = np.random.rand(20, 20) * 10
feed_sense_input(model, 'vision', vision_data)

# Digital input (text)
from src.senses import create_digital_sense_input
digital_data = create_digital_sense_input("Hello, World!")
feed_sense_input(model, 'digital', digital_data)
```

---

## Core Concepts

### What is the "4th dimension" (w) used for?

The w-coordinate is flexible and can represent:
- Different modalities (vision, audio, etc. in different w-slices)
- Temporal depth (past states vs current state)
- Processing hierarchy (low-level vs high-level)
- Abstract organizational structure

It's up to you how to use it!

### How does the Leaky Integrate-and-Fire model work?

Simplified:
1. Neuron has membrane potential (voltage)
2. Inputs add charge to membrane
3. Charge "leaks" over time
4. When threshold reached, neuron spikes
5. After spike, reset to rest potential

See [ARCHITECTURE.md](../ARCHITECTURE.md) for equations.

### What is Hebbian plasticity?

"Cells that fire together, wire together"

- When pre-synaptic and post-synaptic neurons are both active
- The synapse between them strengthens
- This is the basic learning mechanism
- Future versions will add more sophisticated rules (STDP)

### How does the cell lifecycle work?

Neurons:
1. **Age** with each simulation step
2. **Health** decays over time
3. **Die** when health depletes or max age reached
4. **Reproduce** when active and healthy
5. **Offspring** inherit mutated properties

This creates evolutionary pressure for successful patterns.

### What are brain areas?

Brain areas are regions of the 4D lattice designated for specific functions:
- **V1_like**: Visual processing
- **A1_like**: Auditory processing
- **Digital_sensor**: Digital data processing
- etc.

Each area has defined coordinates and sensory mappings.

---

## Technical Questions

### How do I save and load models?

```python
from src.storage import save_to_hdf5, load_from_hdf5

# Save
save_to_hdf5(model, "my_model.h5")

# Load
model = load_from_hdf5("my_model.h5")
```

HDF5 is recommended for large models (compressed). JSON is available for small models (human-readable).

### Can I create custom neuron models?

Yes! Extend the neuron update logic in `simulation.py`. Future versions will have a plugin system. See [Developer Guide](../developer-guide/) for details.

### Can I add custom plasticity rules?

Yes! Add new functions to `plasticity.py`. Currently Hebbian is implemented; STDP and others are planned. See [CONTRIBUTING.md](../../CONTRIBUTING.md).

### How do I monitor the simulation?

Use callbacks:

```python
def monitor(sim, step):
    if step % 100 == 0:
        print(f"Step {step}: {len(sim.spike_history)} spikes")

sim.add_callback(monitor)
```

Or use the web interface for real-time visualization.

### Can I run experiments in batch?

Yes! Use the benchmark framework:

```python
from src.evaluation import BenchmarkSuite, BenchmarkConfig

config = BenchmarkConfig(...)
suite = BenchmarkSuite(...)
results = suite.run(config)
```

See [TASKS_AND_EVALUATION.md](../TASKS_AND_EVALUATION.md).

### Is there a Python API reference?

Yes, see [API.md](../API.md) for complete API documentation.

---

## Performance

### How large can models scale?

Current testing: ~50,000 neurons on a standard laptop.

Limitations:
- Memory: ~500 bytes per neuron
- CPU: O(n²) for synapse updates
- Single-threaded (Python GIL)

See [ISSUES.md](../../ISSUES.md) for performance limitations.

### Why is my simulation slow?

Common causes:
1. **Too many neurons**: Reduce density or lattice size
2. **Too many synapses**: Reduce connection probability
3. **Large inputs**: Reduce input resolution
4. **Long runs**: Use checkpointing

### How can I speed up simulations?

Optimization strategies:
1. Reduce neuron density (e.g., 0.1 instead of 1.0)
2. Lower connection probability (e.g., 0.01)
3. Smaller lattice dimensions
4. Disable cell lifecycle temporarily
5. Use HDF5 storage (faster than JSON)

Future: GPU acceleration (see [TODO.md](../../TODO.md))

### Does it support parallel processing?

Not currently. Python's GIL prevents true parallelization. Future versions may use:
- Multiprocessing
- Distributed computing
- GPU acceleration

### How much RAM do I need?

Rough estimate: **500 bytes × number of neurons**

Examples:
- 10,000 neurons = 5 MB
- 50,000 neurons = 25 MB
- 100,000 neurons = 50 MB

Plus overhead for synapses, Python objects, etc.

---

## Troubleshooting

### "ModuleNotFoundError" when running scripts

Make sure you're in the correct directory and have installed dependencies:

```bash
cd 4D-Neural-Cognition
pip install -r requirements.txt
python example.py
```

### Web interface won't start

Check if port 5000 is already in use:

```bash
# Linux/Mac
lsof -i :5000

# Windows
netstat -ano | findstr :5000
```

Use a different port:

```python
# In app.py, change:
socketio.run(app, port=5001)
```

### "KeyError" when feeding sensory input

Check that:
1. Sense name matches configuration (e.g., 'vision', not 'visual')
2. Input dimensions match expected size
3. Brain area exists for that sense

### Simulation crashes after many steps

Known issue: Memory leak in long runs. Workarounds:
- Restart periodically
- Use checkpointing
- See [ISSUES.md](../../ISSUES.md)

### "NaN" values in weights

Caused by weight overflow. Solutions:
- Lower learning rate (< 0.01)
- Check weight clipping settings
- See [ISSUES.md](../../ISSUES.md)

### Web interface freezes

Caused by too many neurons. Solutions:
- Reduce density (< 0.5)
- Disable visualization
- Use smaller lattice
- See [ISSUES.md](../../ISSUES.md)

---

## Contributing

### How can I contribute?

Many ways:
- Report bugs
- Suggest features
- Improve documentation
- Submit code
- Share examples

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### I found a bug. What should I do?

1. Check [ISSUES.md](../../ISSUES.md) if it's known
2. Search existing GitHub issues
3. Create a bug report with reproduction steps

### I have a feature idea. What's next?

1. Check [TODO.md](../../TODO.md) if it's planned
2. Search existing feature requests
3. Create a feature request on GitHub

### How do I submit code changes?

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests (if applicable)
5. Submit a pull request

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed workflow.

### What's a "good first issue"?

Issues labeled "good first issue" are:
- Well-defined
- Limited scope
- Don't require deep system knowledge
- Good for new contributors

Check the [issue tracker](https://github.com/Thomas-Heisig/4D-Neural-Cognition/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

---

## Still Have Questions?

- Check [SUPPORT.md](../../SUPPORT.md) for how to get help
- Search [GitHub Discussions](https://github.com/Thomas-Heisig/4D-Neural-Cognition/discussions)
- Open a new discussion in [Q&A](https://github.com/Thomas-Heisig/4D-Neural-Cognition/discussions/categories/q-a)

---

*Last Updated: December 2025*  
*Can't find your question? Open a [Discussion](https://github.com/Thomas-Heisig/4D-Neural-Cognition/discussions)!*
