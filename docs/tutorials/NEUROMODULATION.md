# Neuromodulation Tutorial

Learn how to use neuromodulation systems to create adaptive, biologically-inspired neural networks.

## Table of Contents

1. [What is Neuromodulation?](#what-is-neuromodulation)
2. [Dopamine System](#dopamine-system)
3. [Serotonin System](#serotonin-system)
4. [Norepinephrine System](#norepinephrine-system)
5. [Complete Neuromodulation System](#complete-neuromodulation-system)
6. [Integration with Learning](#integration-with-learning)
7. [Practical Examples](#practical-examples)

---

## What is Neuromodulation?

**Neuromodulation** refers to the regulation of neuronal activity through chemical messengers (neurotransmitters) that act over slower timescales than regular synaptic transmission. Unlike fast synaptic signals, neuromodulators broadly influence:

- Learning rates and plasticity
- Neural excitability and gain
- Behavioral states (arousal, mood, attention)

### Key Neuromodulators

| Neuromodulator | Primary Function | Computational Role |
|----------------|------------------|-------------------|
| **Dopamine** | Reward & Motivation | Reinforcement learning, plasticity modulation |
| **Serotonin** | Mood & Inhibition | Behavioral control, punishment processing |
| **Norepinephrine** | Arousal & Attention | Neural gain, uncertainty handling |

### Why Neuromodulation Matters

- **Adaptive Learning**: Adjust learning rates based on outcomes
- **State-Dependent Behavior**: Different brain states for exploration vs. exploitation
- **Biological Realism**: More brain-like than fixed-parameter networks
- **Robustness**: Better handling of uncertainty and changing environments

---

## Dopamine System

Dopamine signals **reward prediction errors** - the difference between expected and actual rewards.

### Basic Usage

```python
from src.neuromodulation import DopamineSystem

# Create dopamine system
dopamine = DopamineSystem()

# Process reward (unexpected reward increases dopamine)
prediction_error = dopamine.update(
    reward=1.0,           # Actual reward received
    expected_reward=0.0   # What was expected
)

print(f"Prediction error: {prediction_error}")  # 1.0 (positive surprise)
print(f"Dopamine level: {dopamine.state.level}")  # Increased

# Get learning rate multiplier
lr_multiplier = dopamine.get_learning_rate_multiplier()
print(f"Learning rate multiplier: {lr_multiplier}")  # > 1.0 (enhanced learning)
```

### Reward Prediction Error

The core concept is **Temporal Difference (TD) learning**:

```python
# Scenario 1: Unexpected reward (positive surprise)
dopamine.update(reward=1.0, expected_reward=0.0)
# → High dopamine, enhanced learning

# Scenario 2: Expected reward received (neutral)
dopamine.update(reward=1.0, expected_reward=1.0)
# → Baseline dopamine, normal learning

# Scenario 3: Expected reward not received (negative surprise)
dopamine.update(reward=0.0, expected_reward=1.0)
# → Low dopamine, reduced learning
```

### Modulating Plasticity

Dopamine directly affects learning:

```python
from src.plasticity import hebbian_update

# Without dopamine
base_delta_w = 0.01

# With high dopamine (after reward)
dopamine.state.level = 0.9
enhanced_delta_w = dopamine.modulate_plasticity(base_delta_w)
print(f"Enhanced: {enhanced_delta_w}")  # ~0.018 (1.8x stronger)

# With low dopamine (after punishment)
dopamine.state.level = 0.1
suppressed_delta_w = dopamine.modulate_plasticity(base_delta_w)
print(f"Suppressed: {suppressed_delta_w}")  # ~0.002 (0.2x weaker)
```

---

## Serotonin System

Serotonin regulates **behavioral inhibition** and processes aversive events.

### Basic Usage

```python
from src.neuromodulation import SerotoninSystem

# Create serotonin system
serotonin = SerotoninSystem()

# Process punishment or stress
serotonin.update(
    punishment=0.5,  # Aversive event
    stress=0.2       # Additional stress
)

print(f"Serotonin level: {serotonin.state.level}")  # Decreased

# Get inhibition factor (higher = more behavioral control)
inhibition = serotonin.get_inhibition_factor()
print(f"Inhibition factor: {inhibition}")
```

### Threshold Modulation

Serotonin makes neurons **harder to fire**, implementing behavioral inhibition:

```python
# Base firing threshold
base_threshold = 10.0

# Low serotonin → impulsive behavior (easier to fire)
serotonin.state.level = 0.2
low_threshold = serotonin.modulate_threshold(base_threshold)
print(f"Low serotonin threshold: {low_threshold}")  # ~10.1

# High serotonin → controlled behavior (harder to fire)
serotonin.state.level = 0.8
high_threshold = serotonin.modulate_threshold(base_threshold)
print(f"High serotonin threshold: {high_threshold}")  # ~11.0
```

### Punishment Processing

```python
# Agent receives negative feedback
serotonin.update(punishment=1.0, stress=0.0)

# This lowers serotonin, which:
# 1. Reduces behavioral inhibition (may try different strategy)
# 2. Signals aversive state
# 3. Complements dopamine's reward processing
```

---

## Norepinephrine System

Norepinephrine regulates **arousal**, **attention**, and responses to **uncertainty**.

### Basic Usage

```python
from src.neuromodulation import NorepinephrineSystem

# Create norepinephrine system
norepinephrine = NorepinephrineSystem()

# Process uncertainty or novelty
norepinephrine.update(
    uncertainty=0.5,  # Environmental uncertainty
    novelty=0.3       # Novel stimuli
)

print(f"Norepinephrine level: {norepinephrine.state.level}")  # Increased

# Get neural gain multiplier
gain = norepinephrine.get_gain_multiplier()
print(f"Gain multiplier: {gain}")  # > 1.0 (amplified responses)
```

### Input Amplification

Norepinephrine increases **neural gain** - the amplification of inputs:

```python
# Base synaptic input
base_input = 5.0

# Low NE → low arousal (weak amplification)
norepinephrine.state.level = 0.2
low_gain_input = norepinephrine.modulate_input(base_input)
print(f"Low arousal: {low_gain_input}")  # ~5.4

# High NE → high arousal (strong amplification)
norepinephrine.state.level = 0.9
high_gain_input = norepinephrine.modulate_input(base_input)
print(f"High arousal: {high_gain_input}")  # ~14.0
```

### Uncertainty and Attention

```python
# Uncertain environment → increase vigilance
norepinephrine.update(uncertainty=0.8, novelty=0.0)

# Novel stimulus → orient attention
norepinephrine.update(uncertainty=0.0, novelty=0.7)

# Both → maximal attention
norepinephrine.update(uncertainty=0.5, novelty=0.5)
```

---

## Complete Neuromodulation System

The `NeuromodulationSystem` integrates all three neuromodulators:

### Basic Setup

```python
from src.neuromodulation import NeuromodulationSystem

# Create complete system
neuromod = NeuromodulationSystem()

# Get current state
state = neuromod.get_state()
print(f"Dopamine: {state['dopamine']}")
print(f"Serotonin: {state['serotonin']}")
print(f"Norepinephrine: {state['norepinephrine']}")
```

### System Update

All neuromodulators naturally decay toward baseline:

```python
# Increase dopamine
neuromod.process_reward(reward=1.0, expected_reward=0.0)
print(f"Dopamine after reward: {neuromod.dopamine.state.level}")  # High

# Step the system (decay)
for _ in range(10):
    neuromod.step()

print(f"Dopamine after decay: {neuromod.dopamine.state.level}")  # Toward baseline
```

### Configuration

```python
from src.neuromodulation import create_neuromodulation_system

# Custom configuration
config = {
    "dopamine": {
        "baseline": 0.6,                # Higher baseline dopamine
        "decay_rate": 0.05,             # Slower decay
        "learning_rate_modulation": 3.0 # Stronger effect on learning
    },
    "serotonin": {
        "baseline": 0.4,                # Lower baseline (less inhibited)
        "inhibition_strength": 0.7      # Stronger inhibition when active
    },
    "norepinephrine": {
        "baseline": 0.5,
        "gain_modulation": 1.5          # Moderate gain control
    }
}

neuromod = create_neuromodulation_system(config)
```

---

## Integration with Learning

### Reward-Modulated Learning

```python
from src.brain_model import BrainModel, Synapse
from src.neuromodulation import NeuromodulationSystem

# Setup
model = BrainModel(config_path="brain_base_model.json")
neuromod = NeuromodulationSystem()

# Synapse to learn
synapse = Synapse(pre_id=0, post_id=1, weight=0.5)

# Base weight change from Hebbian learning
base_delta_w = 0.01

# Process outcome
reward = 1.0 if action_successful else -1.0
neuromod.process_reward(reward=reward, expected_reward=0.0)

# Modulate learning based on dopamine
modulated_delta_w = neuromod.modulate_learning(base_delta_w)

# Apply to synapse
synapse.weight += modulated_delta_w
```

### State-Dependent Processing

```python
from src.brain_model import Neuron

# Neuron to update
neuron = Neuron(id=0, x=0, y=0, z=0, w=0)
synaptic_input = 10.0
threshold = 15.0

# Process environmental state
neuromod.process_uncertainty(uncertainty=0.7, novelty=0.3)

# Modulate neuron update
modulated_input, modulated_threshold = neuromod.modulate_neuron_update(
    neuron=neuron,
    synaptic_input=synaptic_input,
    threshold=threshold
)

# High NE amplifies input
print(f"Original input: {synaptic_input}")      # 10.0
print(f"Modulated input: {modulated_input}")    # ~20.0 (amplified)

# Use modulated values in neuron update
# (neuron more likely to fire due to increased gain)
```

### Combined Effects

```python
# Scenario: Successful action in uncertain environment

# 1. High uncertainty → increase norepinephrine
neuromod.process_uncertainty(uncertainty=0.8, novelty=0.2)

# 2. Success → reward → increase dopamine
neuromod.process_reward(reward=1.0, expected_reward=0.3)

# 3. No punishment → serotonin stays normal
neuromod.process_punishment(punishment=0.0, stress=0.0)

# Result:
# - Enhanced learning (high dopamine)
# - Amplified inputs (high norepinephrine)
# - Normal inhibition (baseline serotonin)

# Apply all effects
base_delta_w = 0.01
final_delta_w = neuromod.modulate_learning(base_delta_w)
# Strong learning due to successful outcome in challenging situation
```

---

## Practical Examples

### Example 1: Reinforcement Learning

```python
from src.neuromodulation import NeuromodulationSystem
from src.brain_model import BrainModel
from src.simulation import Simulation

# Setup
model = BrainModel(config_path="brain_base_model.json")
sim = Simulation(model, seed=42)
neuromod = NeuromodulationSystem()

# Initialize network
sim.initialize_neurons(area_names=["V1_like"], density=0.1)
sim.initialize_random_synapses(connection_probability=0.1)

# Training loop
expected_reward = 0.0  # Start with no expectation
alpha = 0.1  # Learning rate for expected reward

for episode in range(100):
    # Reset state
    state = get_environment_state()
    
    # Run simulation
    for step in range(50):
        # Get action from network activity
        action = select_action_from_spikes(sim)
        
        # Take action, get reward
        reward = environment.step(action)
        
        # Update dopamine based on prediction error
        prediction_error = neuromod.process_reward(reward, expected_reward)
        
        # Update expected reward (simple TD)
        expected_reward += alpha * prediction_error
        
        # Apply reward-modulated learning
        for synapse in sim.model.synapses:
            # Compute base plasticity
            base_delta_w = compute_plasticity(synapse)
            
            # Modulate with dopamine
            modulated_delta_w = neuromod.modulate_learning(base_delta_w)
            
            # Update weight
            synapse.weight += modulated_delta_w
        
        # Simulation step with neuromodulation
        stats = sim.step()
        
        # Decay neuromodulators
        neuromod.step()
```

### Example 2: Attention and Uncertainty

```python
# Scenario: Object detection with varying uncertainty

for trial in range(trials):
    # Compute uncertainty from environment
    uncertainty = compute_environmental_uncertainty()
    
    # Update norepinephrine
    neuromod.process_uncertainty(uncertainty=uncertainty, novelty=0.0)
    
    # Get gain multiplier
    gain = neuromod.norepinephrine.get_gain_multiplier()
    
    # Process visual input with modulated gain
    for neuron_id, neuron in sim.model.neurons.items():
        synaptic_input = compute_input(neuron)
        
        # Apply gain modulation
        modulated_input = synaptic_input * gain
        neuron.external_input = modulated_input
    
    # Run simulation with modulated inputs
    sim.step()
    
    # High uncertainty → high gain → more sensitive to inputs
    # Low uncertainty → low gain → less sensitive (efficient processing)
```

### Example 3: Behavioral Inhibition

```python
# Scenario: Inhibitory control task

for trial in range(trials):
    # Present stimulus
    present_stimulus(sim)
    
    # Compute whether response should be inhibited
    should_inhibit = task_requires_inhibition()
    
    if should_inhibit:
        # Increase serotonin to enhance inhibition
        neuromod.serotonin.state.level = 0.8
    else:
        # Normal serotonin
        neuromod.serotonin.state.level = 0.5
    
    # Run simulation with modulated thresholds
    for neuron_id, neuron in sim.model.neurons.items():
        # Get neuron parameters
        params = neuron.params
        base_threshold = params.get("v_threshold", -50.0)
        
        # Modulate threshold based on serotonin
        modulated_threshold = neuromod.serotonin.modulate_threshold(base_threshold)
        
        # Update neuron (temporarily)
        params["v_threshold"] = modulated_threshold
    
    sim.step()
    
    # Measure response
    response = detect_response(sim)
    
    # High serotonin → higher threshold → less likely to respond
    # Implements "wait" signal
```

### Example 4: Full Integration

```python
#!/usr/bin/env python3
"""Complete neuromodulation example."""

from src.brain_model import BrainModel
from src.simulation import Simulation
from src.neuromodulation import create_neuromodulation_system

def main():
    # Load configuration with neuromodulation settings
    model = BrainModel(config_path="brain_base_model.json")
    neuromod_config = model.config.get("neuromodulation", {})
    
    # Create systems
    sim = Simulation(model, seed=42)
    neuromod = create_neuromodulation_system(neuromod_config)
    
    # Initialize network
    sim.initialize_neurons(area_names=["V1_like", "A1_like"], density=0.1)
    sim.initialize_random_synapses(connection_probability=0.1)
    
    print("=== Neuromodulated Simulation ===\n")
    
    # Training loop
    for step in range(1000):
        # Simulate environmental dynamics
        if step % 100 == 0:
            # Reward event
            reward = 1.0 if step > 500 else 0.5
            neuromod.process_reward(reward=reward, expected_reward=0.3)
            print(f"Step {step}: Reward received, DA={neuromod.dopamine.state.level:.2f}")
        
        if step % 150 == 50:
            # Punishment event
            neuromod.process_punishment(punishment=0.5, stress=0.2)
            print(f"Step {step}: Punishment, 5HT={neuromod.serotonin.state.level:.2f}")
        
        if step % 200 == 0:
            # Uncertainty increases
            neuromod.process_uncertainty(uncertainty=0.7, novelty=0.4)
            print(f"Step {step}: Uncertainty, NE={neuromod.norepinephrine.state.level:.2f}")
        
        # Apply neuromodulation to learning
        if step % 10 == 0:
            for synapse in sim.model.synapses:
                # Simplified plasticity with neuromodulation
                base_delta_w = 0.001
                modulated_delta_w = neuromod.modulate_learning(base_delta_w)
                synapse.weight += modulated_delta_w
        
        # Run simulation
        sim.step()
        
        # Decay neuromodulators
        neuromod.step()
        
        # Monitor
        if step % 200 == 0:
            state = neuromod.get_state()
            print(f"Step {step} State: DA={state['dopamine']:.2f}, "
                  f"5HT={state['serotonin']:.2f}, NE={state['norepinephrine']:.2f}\n")
    
    print("=== Simulation Complete ===")

if __name__ == "__main__":
    main()
```

---

## Tips and Best Practices

1. **Start with Defaults**: Use baseline configuration before customizing
2. **Monitor Levels**: Track neuromodulator levels during training
3. **Gradual Changes**: Avoid sudden jumps in neuromodulator levels
4. **Balanced Effects**: Don't make one system dominate all others
5. **Biological Inspiration**: Read research on computational neuromodulation
6. **Test Systematically**: Isolate effects of each neuromodulator

---

## References

- Dayan, P. (2012). Twenty-Five Lessons from Computational Neuromodulation
- Shine, J.M. (2021). Neuromodulatory influences on integration and segregation
- Computational Models of Neuromodulation (Frontiers)
- Improving adaptive learning with multi-neuromodulatory dynamics

---

## Next Steps

- **[Plasticity Tutorial](PLASTICITY.md)** - Advanced plasticity mechanisms
- **[API Documentation](../api/API.md)** - Full API reference
- **[Examples](../../examples/)** - More examples

---

*Last Updated: December 2025*  
*Part of the 4D Neural Cognition Documentation*
