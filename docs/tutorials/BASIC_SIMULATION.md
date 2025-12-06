# Basic Simulation Tutorial

This tutorial covers the fundamentals of creating and running simulations with the 4D Neural Cognition system.

## Table of Contents

1. [Simulation Lifecycle](#simulation-lifecycle)
2. [Creating a Simulation](#creating-a-simulation)
3. [Initializing Neurons](#initializing-neurons)
4. [Creating Connections](#creating-connections)
5. [Running Simulations](#running-simulations)
6. [Understanding Results](#understanding-results)
7. [Advanced Topics](#advanced-topics)

---

## Simulation Lifecycle

A typical simulation follows these steps:

```
1. Load Configuration → 2. Create Model → 3. Initialize Simulation
                                                    ↓
6. Analyze Results ← 5. Run Steps ← 4. Set Up Network (neurons + synapses)
```

---

## Creating a Simulation

### Loading the Brain Model

```python
from brain_model import BrainModel
from simulation import Simulation

# Load from configuration file
model = BrainModel(config_path="brain_base_model.json")

# Create simulation with random seed for reproducibility
sim = Simulation(model, seed=42)
```

### Understanding the Configuration

The `brain_base_model.json` file defines:

- **Lattice dimensions**: 4D space size (x, y, z, w)
- **Brain areas**: Functional regions like V1, A1, etc.
- **Senses**: Input channels and their mappings
- **Neuron parameters**: LIF model settings
- **Plasticity rules**: Learning parameters

You can inspect these:

```python
print(f"Lattice shape: {model.lattice_shape}")
print(f"Areas: {[a['name'] for a in model.get_areas()]}")
print(f"Senses: {list(model.get_senses().keys())}")
```

---

## Initializing Neurons

### Method 1: Initialize All Areas

```python
# Create neurons in all defined areas
sim.initialize_neurons(
    areas=None,  # None means all areas
    density=0.1   # 10% of positions filled
)
```

### Method 2: Initialize Specific Areas

```python
# Create neurons only in visual and audio areas
sim.initialize_neurons(
    areas=["V1_like", "A1_like"]
    density=0.15  # 15% density
)
```

### Method 3: Different Densities for Different Areas

```python
# Initialize vision with high density
sim.initialize_neurons(areas=["V1_like"], density=0.2)

# Initialize other areas with lower density
sim.initialize_neurons(
    areas=["A1_like", "Digital_sensor"]
    density=0.05
)
```

### Understanding Density

- **Density = 0.1**: 10% of lattice positions have neurons
- **Density = 1.0**: Every position has a neuron (very dense!)
- **Recommended**: 0.05 to 0.2 for most experiments

### Checking Neuron Count

```python
neurons = sim.model.get_neurons()
print(f"Total neurons: {len(neurons)}")

# Count by area
from collections import Counter
area_counts = Counter(n['area'] for n in neurons.values())
for area, count in area_counts.items():
    print(f"  {area}: {count} neurons")
```

---

## Creating Connections

### Random Connections

```python
# Create random synapses between neurons
sim.initialize_random_synapses(
    connection_prob=0.1,   # 10% connection probability
    weight_mean=0.5,       # Average synaptic weight
    weight_std=0.1,        # Standard deviation of weights
    delay_mean=1,          # Average delay (ms)
    delay_std=0.5          # Delay variation
)
```

### Understanding Connection Parameters

**Connection Probability**:
- `0.01`: Very sparse (1% of possible connections)
- `0.1`: Moderate (10% - good starting point)
- `0.5`: Dense (50% - computationally expensive)

**Synaptic Weight**:
- Positive values: Excitatory connections
- Higher values: Stronger influence
- Range typically 0 to 1

**Synaptic Delay**:
- In milliseconds
- Represents signal propagation time
- Adds realism and dynamics

### Checking Connection Count

```python
synapses = sim.model.get_synapses()
print(f"Total synapses: {len(synapses)}")

# Average connections per neuron
avg_out = len(synapses) / len(neurons)
print(f"Average outgoing connections: {avg_out:.1f}")
```

### Manual Connection Creation

For more control, you can create connections manually:

```python
# Get neuron IDs
neurons = sim.model.get_neurons()
neuron_ids = list(neurons.keys())

# Create specific connection
if len(neuron_ids) >= 2:
    sim.model.add_synapse(
        pre_id=neuron_ids[0]
        post_id=neuron_ids[1]
        weight=0.8
        delay=1
    )
```

---

## Running Simulations

### Basic Step-by-Step

```python
# Run one time step
spiked_neurons = sim.step()
print(f"Neurons that fired: {spiked_neurons}")
```

### Running Multiple Steps

```python
# Run 100 time steps
num_steps = 100
for step in range(num_steps):
    spiked = sim.step()
    
    # Print progress every 10 steps
    if step % 10 == 0:
        print(f"Step {step}: {len(spiked)} spikes")
```

### With External Input

```python
# Stimulate specific neurons
external_input = {
    neuron_id_1: 5.0,  # Strong input
    neuron_id_2: 2.0,  # Moderate input
    neuron_id_3: 1.0   # Weak input
}

spiked = sim.step(external_input=external_input)
```

### Batch Running

```python
# Run many steps at once
sim.run(
    steps=1000
    verbose=True  # Print progress
)
```

### With Learning

```python
# Enable plasticity during simulation
for step in range(100):
    spiked = sim.step()
    
    # Apply learning every 5 steps
    if step % 5 == 0:
        sim.apply_plasticity()
```

---

## Understanding Results

### Tracking Spikes

```python
# Spike history: dict[neuron_id, list[time_steps]]
spike_history = sim.spike_history

# Analyze firing rates
firing_rates = {}
for neuron_id, spike_times in spike_history.items():
    rate = len(spike_times) / sim.current_time
    firing_rates[neuron_id] = rate

# Find most active neurons
import heapq
top_5 = heapq.nlargest(5, firing_rates.items(), key=lambda x: x[1])
print("Most active neurons:")
for nid, rate in top_5:
    print(f"  Neuron {nid}: {rate:.2f} spikes/step")
```

### Population Activity

```python
import numpy as np

# Get spike counts per time step
steps = range(sim.current_time)
spikes_per_step = []

for t in steps:
    count = sum(1 for spikes in spike_history.values() if t in spikes)
    spikes_per_step.append(count)

# Statistics
print(f"Average population activity: {np.mean(spikes_per_step):.1f} spikes/step")
print(f"Peak activity: {max(spikes_per_step)} spikes")
print(f"Quiet steps: {sum(1 for s in spikes_per_step if s == 0)}")
```

### Neuron States

```python
# Get current state of a neuron
neuron = sim.model.get_neurons()[neuron_id]

print(f"Neuron {neuron_id}:")
print(f"  Membrane potential: {neuron['V']:.2f}")
print(f"  Threshold: {neuron['V_th']:.2f}")
print(f"  Refractory: {neuron['refrac_counter']}")
print(f"  Area: {neuron['area']}")
```

### Network Statistics

```python
# Overall network metrics
neurons = sim.model.get_neurons()
synapses = sim.model.get_synapses()

print("Network Statistics:")
print(f"  Neurons: {len(neurons)}")
print(f"  Synapses: {len(synapses)}")
print(f"  Connectivity: {len(synapses) / (len(neurons)**2) * 100:.2f}%")
print(f"  Simulation time: {sim.current_time} steps")

# Weight distribution
weights = [s['weight'] for s in synapses.values()]
import numpy as np
print(f"  Mean weight: {np.mean(weights):.3f}")
print(f"  Weight std: {np.std(weights):.3f}")
print(f"  Weight range: [{min(weights):.3f}, {max(weights):.3f}]")
```

---

## Advanced Topics

### Custom Callbacks

Execute custom code during simulation:

```python
def my_callback(sim, step):
    """Called after each simulation step."""
    if step % 100 == 0:
        print(f"Checkpoint at step {step}")
        # Could save state, log metrics, etc.

sim.add_callback(my_callback)
sim.run(steps=500)
```

### Cell Lifecycle

Enable neuron death and reproduction:

```python
# Enable cell lifecycle with specific parameters
sim.run(
    steps=1000
    apply_cell_lifecycle=True
)

# Check how many neurons died/were born
# (Neurons are replaced when they die)
```

### Reproducibility

Always use seeds for reproducible experiments:

```python
# Same seed = same results
sim1 = Simulation(model, seed=42)
sim2 = Simulation(model, seed=42)

# Different seeds = different results
sim3 = Simulation(model, seed=123)
```

### Performance Monitoring

```python
import time

# Measure simulation speed
start = time.time()
sim.run(steps=1000)
elapsed = time.time() - start

steps_per_sec = 1000 / elapsed
print(f"Performance: {steps_per_sec:.1f} steps/second")
```

### Memory Management

For long simulations, periodically clean up:

```python
# Spike history grows over time
# For very long runs, you might want to save and clear it

# Save spike history
import json
with open('spike_history.json', 'w') as f:
    # Convert to serializable format
    history = {k: list(v) for k, v in sim.spike_history.items()}
    json.dump(history, f)

# Clear history (keeps only recent)
sim.spike_history = {
    nid: [t for t in times if t > sim.current_time - 100]
    for nid, times in sim.spike_history.items()
}
```

---

## Example: Complete Simulation Script

Here's a full example putting it all together:

```python
#!/usr/bin/env python3
"""Complete simulation example."""

import sys
import os
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from simulation import Simulation

def analyze_results(sim):
    """Analyze and print simulation results."""
    neurons = sim.model.get_neurons()
    synapses = sim.model.get_synapses()
    
    # Calculate firing rates
    total_spikes = sum(len(spikes) for spikes in sim.spike_history.values())
    avg_rate = total_spikes / len(neurons) / sim.current_time
    
    # Find active neurons
    active = sum(1 for spikes in sim.spike_history.values() if len(spikes) > 0)
    
    print("\n" + "="*50)
    print("SIMULATION RESULTS")
    print("="*50)
    print(f"Duration: {sim.current_time} steps")
    print(f"Neurons: {len(neurons)}")
    print(f"Synapses: {len(synapses)}")
    print(f"Total spikes: {total_spikes}")
    print(f"Active neurons: {active} ({active/len(neurons)*100:.1f}%)")
    print(f"Avg firing rate: {avg_rate:.3f} spikes/neuron/step")
    print("="*50)

def main():
    # Setup
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    
    # Initialize network
    print("Setting up network...")
    sim.initialize_neurons(areas=["V1_like", "A1_like"], density=0.1)
    sim.initialize_random_synapses(connection_prob=0.1)
    
    print(f"Created {len(sim.model.get_neurons())} neurons")
    print(f"Created {len(sim.model.get_synapses())} synapses")
    
    # Run with initial stimulation
    print("\nRunning simulation...")
    for step in range(200):
        # Stimulate first 50 steps
        if step < 50:
            # Give input to random neurons
            neurons = list(sim.model.get_neurons().keys())
            external = {
                nid: np.random.uniform(2, 5)
                for nid in np.random.choice(neurons, size=min(10, len(neurons)), replace=False)
            }
        else:
            external = {}
        
        spiked = sim.step(external_input=external)
        
        # Apply learning every 10 steps
        if step % 10 == 0 and step > 0:
            sim.apply_plasticity()
        
        if step % 50 == 0:
            print(f"  Step {step}...")
    
    # Analyze
    analyze_results(sim)

if __name__ == "__main__":
    main()
```

---

## Next Steps

- **[Sensory Input Tutorial](SENSORY_INPUT.md)** - Learn to provide input
- **[Plasticity Tutorial](PLASTICITY.md)** - Master learning mechanisms
- **[API Documentation](../api/API.md)** - Full API reference

---

*Last Updated: December 2025*  
*Part of the 4D Neural Cognition Documentation*
