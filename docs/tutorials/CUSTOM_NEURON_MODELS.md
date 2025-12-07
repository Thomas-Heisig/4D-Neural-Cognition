# Custom Neuron Models Tutorial

This tutorial explains how to use and customize different neuron models in the 4D Neural Cognition framework.

## Table of Contents

1. [Overview](#overview)
2. [Available Neuron Models](#available-neuron-models)
3. [Using LIF Model](#using-lif-model)
4. [Using Izhikevich Model](#using-izhikevich-model)
5. [Customizing Parameters](#customizing-parameters)
6. [Creating Mixed Networks](#creating-mixed-networks)
7. [Advanced: Implementing Custom Models](#advanced-implementing-custom-models)

---

## Overview

The 4D Neural Cognition framework supports multiple neuron models, each with different dynamics and biological realism:

- **Leaky Integrate-and-Fire (LIF)**: Simple, computationally efficient
- **Izhikevich**: More biologically realistic, supports various firing patterns

Both models support excitatory and inhibitory neurons, enabling creation of balanced networks.

---

## Available Neuron Models

### Leaky Integrate-and-Fire (LIF)

The LIF model is the default and simplest neuron model. It features:
- Linear membrane potential dynamics
- Simple threshold-based spiking
- Efficient computation
- Good for large-scale simulations

### Izhikevich Model

The Izhikevich model offers more biological realism:
- Quadratic membrane dynamics
- Recovery variable for adaptation
- Multiple neuron types (regular spiking, fast spiking, bursting)
- Reproduces various cortical neuron behaviors

---

## Using LIF Model

### Basic LIF Neuron

```python
from src.brain_model import Neuron, BrainModel
from src.simulation import Simulation

# Create brain model
model = BrainModel(config_path="brain_base_model.json")

# Add LIF neuron (default model type)
neuron = Neuron(
    id=0,
    x=0, y=0, z=0, w=0,
    neuron_type="excitatory",  # or "inhibitory"
    model_type="lif"  # This is the default
)
model.add_neuron(neuron)
```

### Customizing LIF Parameters

```python
# Create neuron with custom LIF parameters
neuron = Neuron(
    id=0,
    x=0, y=0, z=0, w=0,
    model_type="lif",
    params={
        "v_rest": -70.0,        # Resting potential (mV)
        "v_threshold": -55.0,   # Spike threshold (mV)
        "v_reset": -70.0,       # Reset potential (mV)
        "tau_membrane": 20.0,   # Membrane time constant (ms)
    }
)
```

### LIF Parameter Guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `v_rest` | -65.0 | Resting membrane potential |
| `v_threshold` | -50.0 | Spike threshold |
| `v_reset` | -65.0 | Post-spike reset voltage |
| `tau_membrane` | 10.0 | Membrane time constant |

---

## Using Izhikevich Model

### Basic Izhikevich Neuron

```python
from src.brain_model import Neuron

# Create Izhikevich neuron
neuron = Neuron(
    id=0,
    x=0, y=0, z=0, w=0,
    model_type="izhikevich",
    neuron_type="regular_spiking",  # See types below
    u_recovery=0.0  # Initial recovery variable
)
```

### Neuron Types

The Izhikevich model supports multiple neuron types with pre-configured parameters:

#### Regular Spiking (RS)
Typical excitatory cortical neurons:
```python
neuron = Neuron(
    id=0, x=0, y=0, z=0, w=0,
    model_type="izhikevich",
    neuron_type="regular_spiking"
)
```
- Parameter set: a=0.02, b=0.2, c=-65, d=8
- Behavior: Adapts over time, regular firing

#### Fast Spiking (FS)
Inhibitory interneurons:
```python
neuron = Neuron(
    id=0, x=0, y=0, z=0, w=0,
    model_type="izhikevich",
    neuron_type="fast_spiking"
)
```
- Parameter set: a=0.1, b=0.2, c=-65, d=2
- Behavior: Rapid firing, minimal adaptation

#### Bursting Neurons
Generate bursts of spikes:
```python
neuron = Neuron(
    id=0, x=0, y=0, z=0, w=0,
    model_type="izhikevich",
    neuron_type="bursting"
)
```
- Parameter set: a=0.02, b=0.2, c=-55, d=4
- Behavior: Generates bursts of action potentials

### Customizing Izhikevich Parameters

```python
neuron = Neuron(
    id=0, x=0, y=0, z=0, w=0,
    model_type="izhikevich",
    params={
        "izh_a": 0.02,  # Recovery time scale
        "izh_b": 0.2,   # Sensitivity to u
        "izh_c": -65.0, # Reset voltage
        "izh_d": 8.0,   # Reset recovery
    }
)
```

### Izhikevich Parameter Guide

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `a` | Recovery time constant | 0.02 - 0.1 |
| `b` | Sensitivity of recovery | 0.2 - 0.25 |
| `c` | Reset voltage after spike | -65 to -50 |
| `d` | Recovery reset increment | 2 - 8 |

---

## Creating Mixed Networks

### Example: Balanced E-I Network with Mixed Models

```python
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.neuron_models import create_balanced_network_types

# Create model
model = BrainModel(config_path="brain_base_model.json")
sim = Simulation(model, seed=42)

# Create balanced network
num_neurons = 1000
neuron_types = create_balanced_network_types(
    num_neurons=num_neurons,
    inhibitory_fraction=0.2  # 20% inhibitory
)

# Initialize neurons with mixed types
sim.initialize_neurons(
    area_names=["cortex"],
    density=0.1
)

# Set neuron types and models
for i, neuron in enumerate(model.neurons.values()):
    if i < len(neuron_types):
        neuron.neuron_type = neuron_types[i]
        # Use Izhikevich for more realistic dynamics
        neuron.model_type = "izhikevich"

# Initialize synapses
sim.initialize_random_synapses(
    connection_probability=0.01,
    weight_mean=0.1,
    weight_std=0.05
)

# Run simulation
for _ in range(1000):
    sim.step()
```

### Checking Excitation-Inhibition Balance

```python
from src.neuron_models import calculate_excitation_inhibition_balance

# After running simulation, check E-I balance
excitatory_input = 50.0  # Example
inhibitory_input = 10.0  # Example

balance = calculate_excitation_inhibition_balance(
    excitatory_input, inhibitory_input
)

print(f"E-I Balance Ratio: {balance['balance_ratio']:.2f}")
print(f"Excitatory Fraction: {balance['e_fraction']:.2%}")
print(f"Inhibitory Fraction: {balance['i_fraction']:.2%}")
```

---

## Customizing Parameters

### Per-Neuron Customization

```python
# Create neurons with different parameters
excitatory = Neuron(
    id=0, x=0, y=0, z=0, w=0,
    model_type="izhikevich",
    neuron_type="regular_spiking",
    params={
        "izh_a": 0.02,
        "izh_d": 8.0
    }
)

inhibitory = Neuron(
    id=1, x=1, y=0, z=0, w=0,
    model_type="izhikevich",
    neuron_type="fast_spiking",
    params={
        "izh_a": 0.1,  # Faster recovery
        "izh_d": 2.0   # Less adaptation
    }
)
```

### Area-Specific Parameters

```python
# Different parameters for different brain areas
visual_params = {
    "v_threshold": -55.0,  # More sensitive
    "tau_membrane": 15.0
}

motor_params = {
    "v_threshold": -50.0,  # Less sensitive
    "tau_membrane": 10.0
}

# Apply when creating neurons in different areas
for neuron in model.get_neurons_in_area("vision"):
    neuron.params.update(visual_params)

for neuron in model.get_neurons_in_area("motor"):
    neuron.params.update(motor_params)
```

---

## Advanced: Implementing Custom Models

While LIF and Izhikevich models cover most use cases, you can implement custom neuron dynamics:

### Step 1: Create Update Function

```python
from typing import Tuple
from src.brain_model import Neuron

def update_custom_neuron(
    neuron: Neuron,
    synaptic_input: float,
    dt: float = 1.0
) -> Tuple[bool, float]:
    """Custom neuron update function.
    
    Args:
        neuron: Neuron to update
        synaptic_input: Total synaptic input
        dt: Time step
        
    Returns:
        (spiked, new_membrane_potential)
    """
    # Your custom dynamics here
    v = neuron.v_membrane
    
    # Example: Simple threshold with decay
    decay = 0.9
    new_v = v * decay + synaptic_input + neuron.external_input
    
    # Spike if above threshold
    threshold = neuron.params.get("threshold", -50.0)
    spiked = new_v >= threshold
    
    if spiked:
        new_v = neuron.params.get("v_reset", -65.0)
    
    return spiked, new_v
```

### Step 2: Integrate with Simulation

```python
from src.simulation import Simulation

# Modify the simulation step to use custom model
def custom_lif_step(sim, neuron_id, external_inputs):
    neuron = sim.model.neurons[neuron_id]
    
    # Calculate synaptic input
    synaptic_input = 0.0
    for synapse in sim.model.get_synapses_for_neuron(neuron_id, "post"):
        if synapse.pre_id in sim.spike_history.get(sim.time - synapse.delay, set()):
            synaptic_input += synapse.get_effective_weight()
    
    # Apply external input
    neuron.external_input = external_inputs.get(neuron_id, 0.0)
    
    # Use custom update function
    spiked, new_v = update_custom_neuron(neuron, synaptic_input)
    
    # Update neuron state
    neuron.v_membrane = new_v
    
    return spiked
```

---

## Best Practices

### 1. Choose the Right Model
- **Use LIF** for: Large networks, computational efficiency, simple dynamics
- **Use Izhikevich** for: Biological realism, diverse firing patterns, smaller networks

### 2. Balance Your Network
- Maintain ~20% inhibitory neurons for stability
- Use `create_balanced_network_types()` for automatic balancing
- Monitor E-I balance during simulation

### 3. Parameter Tuning
- Start with default parameters
- Adjust gradually based on behavior
- Document custom parameter sets
- Use consistent parameters within neuron types

### 4. Testing Custom Models
- Test with small networks first
- Verify spike timing and patterns
- Check for numerical stability
- Compare with known neuron behaviors

---

## Common Issues and Solutions

### Issue: Network Too Excitable
**Symptoms**: Runaway activity, all neurons firing
**Solutions**:
- Increase inhibitory fraction
- Reduce connection probability
- Lower synaptic weights
- Increase spike threshold

### Issue: Network Too Quiet
**Symptoms**: No or minimal activity
**Solutions**:
- Add external input
- Increase connection probability
- Increase synaptic weights
- Lower spike threshold

### Issue: Numerical Instability
**Symptoms**: NaN values, explosive growth
**Solutions**:
- Reduce time step size
- Add bounds checking
- Normalize weights
- Reduce parameter extremes

---

## Further Reading

- [BASIC_SIMULATION.md](BASIC_SIMULATION.md) - Simulation fundamentals
- [PLASTICITY.md](PLASTICITY.md) - Synaptic learning rules
- [SENSORY_INPUT.md](SENSORY_INPUT.md) - Providing input to networks

### Papers
- Izhikevich, E. M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569-1572.
- Gerstner, W., & Kistler, W. M. (2002). Spiking Neuron Models. Cambridge University Press.

---

*Last Updated: December 2025*
