# 4D Neural Cognition

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Status: Active](https://img.shields.io/badge/Status-Active-success.svg)]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation](https://img.shields.io/badge/docs-comprehensive-brightgreen.svg)](docs/INDEX.md)

> *A revolutionary brain simulation system bridging biological neuroscience with digital computing through four-dimensional neural networks.*

This project implements a 4D brain system that combines biological principles with digital extensions. It simulates neurons in a four-dimensional lattice that can age, die, and reproduce with inheritance of mutated properties. Different brain areas process specific senses – including a novel digital sense for system data and patterns.

**[English](#english-documentation) | [Deutsch](#deutsche-dokumentation)**

---

## 🌟 Key Features

- **4D Neuron Lattice**: Neurons in an (x, y, z, w) coordinate system
- **Leaky Integrate-and-Fire Model**: Biophysically inspired neurons with membrane potential
- **Cell Lifecycle**: Aging, death, and reproduction with inherited mutations
- **Brain Areas & Senses**: Vision, Audition, Somatosensory, Taste, Smell, Vestibular, Digital
- **Hebbian Plasticity**: "Cells that fire together, wire together" learning rule
- **Tasks & Evaluation**: Comprehensive benchmark framework for measuring network performance
- **Knowledge Database**: Pre-training and continued learning from stored knowledge
- **Configuration Comparison**: Objectively compare different network configurations
- **Efficient Storage**: JSON for configuration, HDF5 for efficient data persistence with compression
- **Modern Web Interface**: Browser-based interface with real-time visualization and logging

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)
- [License](#-license)
- [Citation](#-citation)

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition

# Install dependencies
pip install -r requirements.txt

# Run example simulation
python example.py

# Or start web interface
python app.py
# Open browser to http://localhost:5000
```

---

## 📚 Documentation

> **📑 Documentation Hub**: See [DOCUMENTATION.md](DOCUMENTATION.md) for complete overview  
> **📖 Full Index**: See [docs/INDEX.md](docs/INDEX.md) for detailed navigation

### 🚀 Getting Started
- **[User Guide](docs/user-guide/)** - Complete user documentation
  - **[Installation Guide](docs/user-guide/INSTALLATION.md)** - Detailed setup for all platforms
  - **[Quick Start Tutorial](docs/tutorials/QUICK_START_EVALUATION.md)** - Get up and running in 5 minutes
  - **[FAQ](docs/user-guide/FAQ.md)** - Frequently Asked Questions
  - **[Glossary](docs/user-guide/GLOSSARY.md)** - Terminology and definitions

### 📖 Core Documentation

- **[VISION.md](VISION.md)** - Project vision, goals, and roadmap
- **[TODO.md](TODO.md)** - Planned features and task tracking
- **[ISSUES.md](ISSUES.md)** - Known issues and limitations
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes

### 🤝 Community

- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to the project
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community guidelines
- **[SUPPORT.md](SUPPORT.md)** - How to get help
- **[SECURITY.md](SECURITY.md)** - Security policy

### 🔧 Technical Documentation

- **[API Reference](docs/api/API.md)** - Complete API documentation
- **[Architecture](docs/ARCHITECTURE.md)** - Technical architecture details
- **[Tasks & Evaluation](docs/user-guide/TASKS_AND_EVALUATION.md)** - Benchmark framework guide
- **[Developer Guide](docs/developer-guide/)** - Contributing and development

---

## 💻 Installation

### Simple Installation

```bash
pip install -r requirements.txt
```

For detailed platform-specific instructions, see [Installation Guide](docs/user-guide/INSTALLATION.md).

### Requirements

- Python 3.8+
- NumPy >= 1.20.0
- h5py >= 3.0.0
- Flask >= 2.0.0
- flask-cors >= 3.0.0
- flask-socketio >= 5.0.0
- python-socketio >= 5.0.0

---

## 🎯 Usage

### Web Interface (Recommended)

Start the web application for a user-friendly graphical interface:

```bash
python app.py
```

Then open a browser and navigate to `http://localhost:5000`.

**Features**:
- 🎮 **Model Control**: Initialize and configure models
- 🔥 **Heatmap Visualization**: Real-time display of neural activity
- 💻 **Terminal**: Input/output for sensory data
- 💬 **Chat Interface**: Interactive commands
- 📋 **Logging**: Complete event logging
- ⚡ **Training**: Start/stop/step controls

### Command Line

```bash
python example.py
```

### Programmatic Usage

```python
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.senses import feed_sense_input, create_digital_sense_input
import numpy as np

# Load model
model = BrainModel(config_path='brain_base_model.json')

# Initialize simulation
sim = Simulation(model, seed=42)

# Create neurons in areas
sim.initialize_neurons(area_names=['V1_like', 'Digital_sensor'], density=0.1)

# Create synaptic connections
sim.initialize_random_synapses(connection_probability=0.01)

# Prepare sensory input
vision_input = np.random.rand(20, 20) * 10
digital_input = create_digital_sense_input("Hello, World!")

# Run simulation
for step in range(100):
    if step % 10 == 0:
        feed_sense_input(model, 'vision', vision_input)
        feed_sense_input(model, 'digital', digital_input)
    stats = sim.step()
    print(f"Step {step}: {len(stats['spikes'])} spikes")
```

For complete API documentation, see [API Reference](docs/api/API.md).

## Features

- **4D Neuronengitter**: Neuronen in einem (x, y, z, w) Koordinatensystem
- **Leaky Integrate-and-Fire Modell**: Biophysikalisch inspirierte Neuronen mit Membranpotential
- **Zell-Lebenszyklus**: Alterung, Tod und Reproduktion mit Vererbung mutierter Eigenschaften
- **Hirnareale & Sinne**: Vision, Audition, Somatosensorik, Geschmack, Geruch, Vestibulär, Digital
- **Hebbsche Plastizität**: "Cells that fire together, wire together" Lernregel
- **Speicherung**: JSON für Konfiguration, HDF5 für effiziente Datenspeicherung (mit Kompression)
- **Web-Frontend**: Modernes Browser-Interface mit Echtzeit-Visualisierung und Logging

## Installation

```bash
pip install -r requirements.txt
```



## Verwendung

### Web-Frontend (empfohlen)

Starten Sie die Web-Anwendung für eine benutzerfreundliche grafische Oberfläche:

```bash
python app.py
```

Öffnen Sie dann einen Browser und navigieren Sie zu `http://localhost:5000`.

Das Frontend bietet:
- 🎮 **Modell-Steuerung**: Initialisierung und Konfiguration
- 🔥 **Heatmap-Visualisierung**: Echtzeit-Darstellung von Input-, Hidden- und Output-Layern
- 💻 **Terminal**: Input/Output für sensorische Daten
- 💬 **Chat-Interface**: Interaktive Befehle und Operationen
- 📋 **Logging**: Vollständige Protokollierung aller Systemereignisse
- ⚡ **Training**: Start/Stop-Kontrolle für Simulationsläufe

### Kommandozeilen-Beispiel

```bash
python example.py
```

### Programmatische Nutzung

```python
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.senses import feed_sense_input, create_digital_sense_input
import numpy as np

# Modell laden
model = BrainModel(config_path='brain_base_model.json')

# Simulation initialisieren
sim = Simulation(model, seed=42)

# Neuronen in Arealen erstellen
sim.initialize_neurons(area_names=['V1_like', 'Digital_sensor'], density=0.1)

# Synaptische Verbindungen erstellen
sim.initialize_random_synapses(connection_probability=0.01)

# Sensorische Eingabe vorbereiten
vision_input = np.random.rand(20, 20) * 10
digital_input = create_digital_sense_input("Hello, World!")

# Simulation ausführen
for step in range(100):
    if step % 10 == 0:
        feed_sense_input(model, 'vision', vision_input)
        feed_sense_input(model, 'digital', digital_input)
    stats = sim.step()
    print(f"Step {step}: {len(stats['spikes'])} spikes")
```

## 📁 Project Structure

```
4D-Neural-Cognition/
├── 📄 README.md              # This file
├── 📄 VISION.md              # Project vision and roadmap
├── 📄 TODO.md                # Planned features
├── 📄 ISSUES.md              # Known issues
├── 📄 CHANGELOG.md           # Version history
├── 📄 CONTRIBUTING.md        # Contribution guidelines
├── 📄 CODE_OF_CONDUCT.md     # Community guidelines
├── 📄 LICENSE                # MIT License
│
├── 📁 docs/                  # Technical documentation
│   ├── API.md               # API reference
│   ├── ARCHITECTURE.md      # Architecture details
│   └── INSTALLATION.md      # Installation guide
│
├── 📄 brain_base_model.json  # Base model configuration
├── 📄 example.py             # CLI example script
├── 📄 app.py                 # Flask web application
├── 📄 requirements.txt       # Python dependencies
│
├── 📁 templates/
│   └── index.html           # Web interface HTML
│
├── 📁 static/
│   ├── css/
│   │   └── style.css        # UI styling
│   └── js/
│       └── app.js           # Frontend JavaScript
│
└── 📁 src/                   # Core source code
    ├── __init__.py          # Package initialization
    ├── brain_model.py       # Neuron & synapse structures
    ├── simulation.py        # Main simulation loop
    ├── cell_lifecycle.py    # Cell death & reproduction
    ├── plasticity.py        # Hebbian learning rules
    ├── senses.py            # Sensory input processing
    └── storage.py           # HDF5/JSON persistence
```

---

## ⚙️ Configuration

The `brain_base_model.json` contains:

- **lattice_shape**: Size of 4D lattice [x, y, z, w]
- **neuron_model**: LIF parameters (tau_m, v_rest, v_reset, v_threshold)
- **cell_lifecycle**: Lifecycle parameters (max_age, health_decay, mutation rates)
- **plasticity**: Learning parameters (learning_rate, weight_bounds)
- **senses**: Sense configuration with areas and input sizes
- **areas**: Coordinate ranges for each brain area

---

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Frontend**: Vanilla JavaScript with Socket.IO
- **Styling**: Modern CSS with dark theme
- **Visualization**: HTML5 Canvas for heatmaps
- **Data Storage**: HDF5 with compression
- **Real-time Communication**: WebSocket (Flask-SocketIO)
- **Scientific Computing**: NumPy

---

## 🤝 Contributing

We welcome contributions from the community! Please read our [Contributing Guidelines](CONTRIBUTING.md) to get started.

### Ways to Contribute

- 🐛 **Report bugs** - See [ISSUES.md](ISSUES.md)
- ✨ **Suggest features** - See [TODO.md](TODO.md)
- 📝 **Improve documentation**
- 🔧 **Submit pull requests**
- ⭐ **Star the repository**

### Development Setup

```bash
# Clone repository
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pylint black flake8 mypy

# Run tests (when available)
pytest tests/
```

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Thomas Heisig and Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{4d_neural_cognition,
  author = {Heisig, Thomas and Contributors},
  title = {4D Neural Cognition: A Four-Dimensional Brain Simulation System},
  year = {2025},
  url = {https://github.com/Thomas-Heisig/4D-Neural-Cognition},
  version = {1.0.0}
}
```

---

## 🌐 Resources

- **GitHub Repository**: https://github.com/Thomas-Heisig/4D-Neural-Cognition
- **Documentation**: [docs/](docs/)
- **Issue Tracker**: GitHub Issues
- **Discussions**: GitHub Discussions

---

## ❓ FAQ

**Q: What is the "4th dimension" (w) used for?**  
A: The w-coordinate can represent different modalities, time slices, or abstract organizational principles. It's flexible and can be adapted to your use case.

**Q: Can I run this on GPU?**  
A: Not yet, but GPU acceleration is planned. See [TODO.md](TODO.md) for roadmap.

**Q: How large can models scale?**  
A: Currently tested up to ~50,000 neurons on a standard laptop. Larger models require optimization. See [ISSUES.md](ISSUES.md) for performance limitations.

**Q: Is this biologically accurate?**  
A: It's biologically *inspired*, not a detailed biological simulation. The LIF model and Hebbian plasticity are simplified abstractions.

---

## 🙏 Acknowledgments

- Inspired by computational neuroscience research
- Built with open-source technologies
- Thanks to all contributors

---

## 📞 Contact

For questions, suggestions, or collaborations:
- Open an issue on GitHub
- Check [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines

---

<details>
<summary><h2 id="deutsche-dokumentation">📝 Deutsche Dokumentation</h2></summary>

## Überblick

Dieses Modell implementiert ein 4D-Hirnsystem, das biologische Prinzipien mit digitalen Erweiterungen verbindet. Es simuliert Neuronen in einem vierdimensionalen Gitter, die altern, sterben und sich mit Vererbung mutierter Eigenschaften reproduzieren können.

## Features

- **4D Neuronengitter**: Neuronen in einem (x, y, z, w) Koordinatensystem
- **Leaky Integrate-and-Fire Modell**: Biophysikalisch inspirierte Neuronen
- **Zell-Lebenszyklus**: Alterung, Tod und Reproduktion
- **Hirnareale & Sinne**: Vision, Audition, Somatosensorik, Geschmack, Geruch, Vestibulär, Digital
- **Hebbsche Plastizität**: "Cells that fire together, wire together"
- **Web-Frontend**: Modernes Browser-Interface mit Echtzeit-Visualisierung

## Installation

```bash
pip install -r requirements.txt
```

## Verwendung

```bash
# Web-Interface starten
python app.py

# Kommandozeilen-Beispiel
python example.py
```

Weitere Details finden Sie in der [englischen Dokumentation](#english-documentation) oben.

</details>

---

*Last Updated: December 2025*  
*Version: 1.0.0*
