# Getting Started with 4D Neural Cognition

Welcome to the 4D Neural Cognition project! This tutorial will guide you through your first steps with the system.

## Table of Contents

1. [Installation](#installation)
2. [Understanding the Basics](#understanding-the-basics)
3. [Your First Simulation](#your-first-simulation)
4. [Using the Web Interface](#using-the-web-interface)
5. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB RAM minimum (8GB recommended)
- Modern web browser (Chrome, Firefox, or Edge)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Thomas-Heisig/4D-Neural-Cognition.git
cd 4D-Neural-Cognition
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- numpy - Numerical computing
- h5py - HDF5 file format support
- flask - Web application framework
- pytest - Testing framework

### Step 3: Verify Installation

Run the test suite to ensure everything is working:

```bash
pytest -v
```

You should see all tests passing (186 tests).

---

## Understanding the Basics

### What is 4D Neural Cognition?

The 4D Neural Cognition system simulates a brain-like network in four-dimensional space:
- **x, y, z**: 3D spatial coordinates (like our familiar 3D space)
- **w**: The fourth dimension (representing cognitive hierarchy/abstraction levels)

### Core Concepts

#### 1. Brain Model

The `BrainModel` defines the structure of your neural network:
- **Lattice**: A 4D grid where neurons can exist
- **Areas**: Functional regions (like V1 for vision, A1 for audio)
- **Senses**: Input channels (vision, audio, digital, etc.)

#### 2. Neurons

Individual processing units that:
- Receive inputs from other neurons via synapses
- Fire spikes when their membrane potential reaches a threshold
- Use the Leaky Integrate-and-Fire (LIF) model

#### 3. Synapses

Connections between neurons with:
- **Weight**: Strength of the connection
- **Delay**: Time for signal to propagate
- **Plasticity**: Can strengthen or weaken over time (learning)

#### 4. Simulation

The `Simulation` class:
- Steps through time
- Updates all neurons
- Applies learning rules
- Tracks spike history

---

## Your First Simulation

Let's create and run a simple simulation step by step.

### Example 1: Basic Command-Line Simulation

Create a file called `my_first_sim.py`:

```python
#!/usr/bin/env python3
"""My first 4D neural cognition simulation."""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from simulation import Simulation

def main():
    print("=== My First Simulation ===\n")
    
    # Step 1: Load the brain model configuration
    print("1. Loading brain model...")
    model = BrainModel(config_path="brain_base_model.json")
    print(f"   Lattice shape: {model.lattice_shape}")
    print(f"   Areas available: {[a['name'] for a in model.get_areas()]}")
    
    # Step 2: Create simulation
    print("\n2. Creating simulation...")
    sim = Simulation(model, seed=42)  # Seed for reproducibility
    
    # Step 3: Initialize some neurons
    print("\n3. Initializing neurons in V1_like area...")
    sim.initialize_neurons(
        areas=["V1_like"]
        density=0.05  # 5% of positions filled
    )
    print(f"   Created {len(sim.model.get_neurons())} neurons")
    
    # Step 4: Create connections between neurons
    print("\n4. Creating synaptic connections...")
    sim.initialize_random_synapses(
        connection_prob=0.1,  # 10% connection probability
        weight_mean=0.5
        weight_std=0.1
    )
    synapses = sim.model.get_synapses()
    print(f"   Created {len(synapses)} synapses")
    
    # Step 5: Run simulation
    print("\n5. Running simulation for 100 time steps...")
    for step in range(100):
        # Add some external input to stimulate the network
        if step < 50:
            # Give random input to first 10 neurons
            external_input = {}
            for i, (nid, _) in enumerate(list(sim.model.get_neurons().items())[:10]):
                external_input[nid] = 5.0  # Strong input
        else:
            external_input = {}
        
        # Run one simulation step
        spiked_neurons = sim.step(external_input=external_input)
        
        # Print progress
        if step % 20 == 0:
            print(f"   Step {step}: {len(spiked_neurons)} neurons spiked")
    
    # Step 6: Check results
    print("\n6. Simulation complete!")
    total_spikes = sum(len(spikes) for spikes in sim.spike_history.values())
    print(f"   Total spikes: {total_spikes}")
    print(f"   Average firing rate: {total_spikes / len(sim.model.get_neurons()) / 100:.2f} spikes/neuron/step")
    
    print("\n=== Done! ===")

if __name__ == "__main__":
    main()
```

Run it:

```bash
python my_first_sim.py
```

### Example 2: Simulation with Sensory Input

Create `sensory_example.py`:

```python
#!/usr/bin/env python3
"""Example with sensory input."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input

def main():
    print("=== Sensory Input Example ===\n")
    
    # Load model and create simulation
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    
    # Initialize neurons in visual area
    sim.initialize_neurons(areas=["V1_like"], density=0.1)
    sim.initialize_random_synapses(connection_prob=0.1)
    
    print(f"Neurons: {len(sim.model.get_neurons())}")
    print(f"Synapses: {len(sim.model.get_synapses())}")
    
    # Create a simple 10x10 visual pattern (a vertical line)
    visual_input = np.zeros((10, 10))
    visual_input[:, 5] = 1.0  # Vertical line in middle
    
    print("\nVisual input (10x10):")
    print("1 = active, 0 = inactive")
    for row in visual_input:
        print(''.join(['█' if x > 0 else '·' for x in row]))
    
    # Feed the visual input
    print("\nFeeding visual input to V1_like area...")
    feed_sense_input(
        sim.model
        sense_name="vision"
        input_data=visual_input
        z_layer=0,  # Which z-layer to stimulate
          # Strength of input
    )
    
    # Run simulation
    print("\nRunning 50 steps...")
    spike_counts = []
    for step in range(50):
        spiked = sim.step()
        spike_counts.append(len(spiked))
    
    print(f"\nTotal spikes: {sum(spike_counts)}")
    print(f"Average spikes per step: {np.mean(spike_counts):.1f}")
    print(f"Max spikes in one step: {max(spike_counts)}")
    
    print("\n=== Done! ===")

if __name__ == "__main__":
    main()
```

---

## Using the Web Interface

The web interface provides a visual way to interact with simulations.

### Step 1: Start the Web Server

```bash
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

### Step 2: Open in Browser

Navigate to: `http://localhost:5000`

### Step 3: Initialize the Model

1. The default configuration is loaded automatically
2. Click **"Initialize Model"** to create the brain structure
3. You'll see a confirmation message

### Step 4: Create Neurons

1. In the **Neurons** section:
   - Select areas (e.g., "V1_like", "A1_like")
   - Set density (e.g., 0.1 = 10%)
   - Click **"Initialize Neurons"**

2. Check the status display to see how many neurons were created

### Step 5: Create Synapses

1. In the **Synapses** section:
   - Set connection probability (e.g., 0.1 = 10%)
   - Set weight parameters
   - Click **"Initialize Synapses"**

### Step 6: Run Training

1. In the **Training** section:
   - Set number of steps (e.g., 100)
   - Enable/disable learning
   - Click **"Start Training"**

2. Watch the real-time heatmap showing neural activity

### Step 7: Provide Sensory Input

1. In the **Sensory Input** section:
   - Select sense type (vision, audio, etc.)
   - Provide input data
   - Click **"Feed Input"**

For vision:
```json
{
  "sense": "vision"
  "data": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
}
```

For digital/text:
```json
{
  "sense": "digital"
  "text": "Hello neural network!"
}
```

### Step 8: Save Your Work

1. In the **Storage** section:
   - Enter a filename
   - Choose format (JSON or HDF5)
   - Click **"Save Model"**

2. Later, you can load it back:
   - Enter the same filename
   - Click **"Load Model"**

---

## Next Steps

### Tutorials to Explore Next

1. **[Basic Simulation Tutorial](BASIC_SIMULATION.md)** - Learn simulation details
2. **[Sensory Input Tutorial](SENSORY_INPUT.md)** - Master the sensory system
3. **[Plasticity Tutorial](PLASTICITY.md)** - Understand learning mechanisms
4. **[Quick Start Evaluation](QUICK_START_EVALUATION.md)** - Run benchmarks

### Key Documentation

- **[API Documentation](../api/API.md)** - Complete API reference
- **[Architecture Guide](../ARCHITECTURE.md)** - System design details
- **[User Guide](../user-guide/README.md)** - Comprehensive user manual
- **[FAQ](../user-guide/FAQ.md)** - Common questions answered

### Example Scripts

Check out the `examples/` directory for more complex demonstrations:
- Pattern recognition
- Temporal sequence learning
- Multi-modal integration

### Experiment Ideas

Try these experiments to deepen your understanding:

1. **Network Size**: How does performance change with more neurons?
2. **Connection Density**: What happens with very sparse or very dense connections?
3. **Learning Rates**: How do different plasticity parameters affect learning?
4. **Sensory Patterns**: Can the network learn to recognize specific patterns?
5. **Multi-Area**: How do different brain areas interact?

---

## Troubleshooting

### Installation Issues

**Problem**: `ModuleNotFoundError`
- **Solution**: Make sure you installed requirements: `pip install -r requirements.txt`

**Problem**: Tests failing
- **Solution**: Check Python version (need 3.8+): `python --version`

### Simulation Issues

**Problem**: Memory errors with large simulations
- **Solution**: Reduce neuron density or use smaller lattice size

**Problem**: Simulation is slow
- **Solution**: Reduce number of neurons, synapses, or use shorter simulation runs

**Problem**: No spikes observed
- **Solution**: Increase external input intensity or reduce neuron threshold

### Web Interface Issues

**Problem**: Page won't load
- **Solution**: Check Flask is running on correct port, check for firewall issues

**Problem**: Heatmap not showing
- **Solution**: Model too large (>10,000 neurons), reduce density

---

## Getting Help

- **Documentation**: Check [docs/](../) directory
- **Examples**: See [examples/](../../examples/) directory  
- **Issues**: Report bugs on GitHub Issues
- **Support**: See [SUPPORT.md](../../SUPPORT.md) for contact options

---

## Summary

You've learned how to:
- ✅ Install the system
- ✅ Understand basic concepts (model, neurons, synapses)
- ✅ Run command-line simulations
- ✅ Use the web interface
- ✅ Provide sensory input
- ✅ Save and load models

**Next**: Try the [Basic Simulation Tutorial](BASIC_SIMULATION.md) to dive deeper!

---

*Last Updated: December 2025*  
*Part of the 4D Neural Cognition Documentation*
