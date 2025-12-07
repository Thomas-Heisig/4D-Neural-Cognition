# Plasticity Tutorial

Learn how to implement learning in your neural networks through synaptic plasticity mechanisms.

## Table of Contents

1. [What is Plasticity?](#what-is-plasticity)
2. [Hebbian Learning](#hebbian-learning)
3. [Weight Decay](#weight-decay)
4. [Learning Parameters](#learning-parameters)
5. [Training Strategies](#training-strategies)
6. [Monitoring Learning](#monitoring-learning)
7. [Advanced Topics](#advanced-topics)

---

## What is Plasticity?

**Synaptic plasticity** is the ability of synapses (connections) to strengthen or weaken over time based on neural activity.

### Key Concepts

- **Long-Term Potentiation (LTP)**: Strengthening of synapses
- **Long-Term Depression (LTD)**: Weakening of synapses
- **Hebbian Rule**: "Neurons that fire together, wire together"
- **Weight Decay**: Gradual return to baseline weights

### Why Plasticity Matters

Without plasticity, networks are static and cannot:
- Learn from experience
- Adapt to new inputs
- Form memories
- Recognize patterns

---

## Hebbian Learning

The fundamental learning rule: connections strengthen when pre- and post-synaptic neurons are active together.

### Basic Usage

```python
from simulation import Simulation
from plasticity import hebbian_update

# Apply Hebbian learning to all synapses
sim.apply_plasticity()
```

### Understanding Hebbian Update

The `hebbian_update` function modifies synapse weights based on:

1. **Pre-synaptic activity**: Did the sending neuron spike?
2. **Post-synaptic activity**: Did the receiving neuron spike?
3. **Learning rate**: How fast to change weights
4. **Current weight**: Weights are bounded

### Manual Hebbian Update

```python
from plasticity import hebbian_update

# Get a specific synapse
synapses = sim.model.get_synapses()
syn_id = list(synapses.keys())[0]
synapse = synapses[syn_id]

# Check if neurons were active
pre_active = synapse['pre'] in recently_spiked
post_active = synapse['post'] in recently_spiked

# Apply learning
new_weight = hebbian_update(
    weight=synapse['weight']
    pre_active=pre_active
    post_active=post_active
    learning_rate=0.01
    w_min=0.0
    w_max=1.0
)

# Update synapse
synapse['weight'] = new_weight
```

### Learning Rate Effects

```python
# Try different learning rates
learning_rates = [0.001, 0.01, 0.1]

for lr in learning_rates:
    print(f"\nLearning rate: {lr}")
    
    # Create fresh simulation
    sim = Simulation(model, seed=42)
    sim.initialize_neurons(areas=["V1_like"], density=0.1)
    sim.initialize_random_synapses(connection_prob=0.1)
    
    # Train with this learning rate
    for step in range(100):
        sim.step()
        if step % 10 == 0:
            # Apply plasticity with custom learning rate
            for syn in sim.model.get_synapses().values():
                # Custom plasticity call
                pass
    
    # Measure results
    weights = [s['weight'] for s in sim.model.get_synapses().values()]
    print(f"  Mean weight: {np.mean(weights):.3f}")
    print(f"  Std weight: {np.std(weights):.3f}")
```

---

## Weight Decay

Weight decay prevents weights from growing unbounded and adds stability.

### Basic Weight Decay

```python
from plasticity import weight_decay

# Apply decay to all synapses
for synapse in sim.model.get_synapses().values():
    synapse['weight'] = weight_decay(
        weight=synapse['weight']
        decay_rate=0.001  # 0.1% decay per step
    )
```

### Understanding Decay

- **Decay Rate**: Proportion of weight lost per step
- **Effect**: Weights gradually return to zero
- **Purpose**: Prevents runaway growth, enforces forgetting

### Decay Rate Effects

```python
# Small decay: slow forgetting
decay_rate = 0.0001  # 0.01% per step

# Medium decay: moderate forgetting  
decay_rate = 0.001   # 0.1% per step

# Large decay: rapid forgetting
decay_rate = 0.01    # 1% per step
```

### Implementing Both

```python
def apply_learning(sim, learning_rate=0.01, decay_rate=0.001):
    """Apply both Hebbian learning and weight decay."""
    
    # Get recent activity
    recent_spikes = set()
    for neuron_id, spike_times in sim.spike_history.items():
        if sim.current_time in spike_times:
            recent_spikes.add(neuron_id)
    
    # Update each synapse
    for synapse in sim.model.get_synapses().values():
        # Check activity
        pre_active = synapse['pre'] in recent_spikes
        post_active = synapse['post'] in recent_spikes
        
        # Apply Hebbian learning
        synapse['weight'] = hebbian_update(
            weight=synapse['weight']
            pre_active=pre_active
            post_active=post_active
            learning_rate=learning_rate
            w_min=0.0
            w_max=1.0
        )
        
        # Apply decay
        synapse['weight'] = weight_decay(
            weight=synapse['weight']
            decay_rate=decay_rate
        )

# Use in training loop
for step in range(1000):
    sim.step()
    if step % 5 == 0:
        apply_learning(sim, learning_rate=0.01, decay_rate=0.001)
```

---

## Learning Parameters

### Configuration in brain_base_model.json

```json
{
  "plasticity": {
    "learning_rate": 0.01
    "weight_decay": 0.001
    "w_min": 0.0
    "w_max": 1.0
    "ltd_ratio": 0.5
  }
}
```

### Parameter Guidelines

**Learning Rate** (typical: 0.001 - 0.1):
- Too low: Learning is very slow
- Too high: Unstable, oscillating weights
- Start with: 0.01

**Weight Decay** (typical: 0.0001 - 0.01):
- Too low: Weights saturate
- Too high: Network forgets too quickly
- Start with: 0.001

**LTD Ratio** (typical: 0.3 - 0.8):
- Ratio of depression to potentiation
- Lower = weaker LTD
- Higher = stronger LTD
- Start with: 0.5

### Finding Good Parameters

```python
def test_parameters(sim, lr, decay, steps=1000):
    """Test a parameter combination."""
    
    # Record initial weights
    initial_weights = [s['weight'] for s in sim.model.get_synapses().values()]
    
    # Train
    for step in range(steps):
        # Provide consistent input
        neurons = list(sim.model.get_neurons().keys())[:10]
        external = {nid: 5.0 for nid in neurons}
        
        sim.step(external_input=external)
        
        if step % 10 == 0:
            apply_learning(sim, learning_rate=lr, decay_rate=decay)
    
    # Record final weights
    final_weights = [s['weight'] for s in sim.model.get_synapses().values()]
    
    # Analyze
    weight_change = np.mean(final_weights) - np.mean(initial_weights)
    weight_variance = np.var(final_weights)
    
    return {
        'lr': lr
        'decay': decay
        'weight_change': weight_change
        'weight_variance': weight_variance
    }

# Grid search
results = []
for lr in [0.001, 0.01, 0.1]:
    for decay in [0.0001, 0.001, 0.01]:
        # Create fresh simulation for each test
        model = BrainModel(config_path="brain_base_model.json")
        sim = Simulation(model, seed=42)
        sim.initialize_neurons(areas=["V1_like"], density=0.1)
        sim.initialize_random_synapses(connection_prob=0.1)
        
        result = test_parameters(sim, lr, decay)
        results.append(result)
        print(f"LR={lr}, Decay={decay}: "
              f"Change={result['weight_change']:.3f}, "
              f"Var={result['weight_variance']:.3f}")
```

---

## Training Strategies

### Strategy 1: Continuous Learning

Apply plasticity every step:

```python
for step in range(1000):
    sim.step()
    sim.apply_plasticity()  # Every step
```

**Pros**: Immediate, fine-grained learning  
**Cons**: Computationally expensive

### Strategy 2: Periodic Learning

Apply plasticity every N steps:

```python
for step in range(1000):
    sim.step()
    
    if step % 10 == 0:  # Every 10 steps
        sim.apply_plasticity()
```

**Pros**: More efficient  
**Cons**: Coarser learning

### Strategy 3: Epoch-Based Learning

Learn after complete input presentations:

```python
def train_epoch(sim, inputs, steps_per_input=50):
    """Train for one epoch through all inputs."""
    
    for input_data in inputs:
        # Present input
        feed_sense_input(
            sim.model
            "vision"
            input_data
            
        )
        
        # Process
        for step in range(steps_per_input):
            sim.step()
        
        # Learn from this input
        sim.apply_plasticity()

# Create training set
training_inputs = [
    np.eye(10)
    np.rot90(np.eye(10))
    np.ones((10, 10)) * 0.5
]

# Train for multiple epochs
for epoch in range(10):
    print(f"Epoch {epoch+1}")
    train_epoch(sim, training_inputs)
```

### Strategy 4: Reinforcement-Style

Learn more from important events:

```python
def train_with_reward(sim, input_data, reward, steps=50):
    """Train with reward signal."""
    
    # Present input
    feed_sense_input(sim.model, "vision", input_data)
    
    # Process
    for step in range(steps):
        sim.step()
    
    # Apply plasticity scaled by reward
    if reward > 0:
        # Strong learning for positive outcomes
        for _ in range(int(reward * 5)):
            sim.apply_plasticity()
    elif reward < 0:
        # Weak or no learning for negative outcomes
        pass

# Use it
train_with_reward(sim, pattern1, reward=1.0)  # Good outcome
train_with_reward(sim, pattern2, reward=-1.0)  # Bad outcome
```

### Strategy 5: Curriculum Learning

Start simple, gradually increase difficulty:

```python
def curriculum_training(sim):
    """Train from simple to complex patterns."""
    
    # Phase 1: Simple patterns
    print("Phase 1: Simple patterns")
    simple_patterns = [
        np.eye(10),                   # Diagonal
        np.rot90(np.eye(10))          # Other diagonal
    ]
    
    for epoch in range(5):
        for pattern in simple_patterns:
            feed_sense_input(sim.model, "vision", pattern)
            sim.run(steps=30)
            sim.apply_plasticity()
    
    # Phase 2: Medium complexity
    print("Phase 2: Medium patterns")
    medium_patterns = [
        np.ones((10, 10)) * 0.5,      # Uniform
        np.random.rand(10, 10) * 0.3  # Weak noise
    ]
    
    for epoch in range(5):
        for pattern in medium_patterns:
            feed_sense_input(sim.model, "vision", pattern)
            sim.run(steps=30)
            sim.apply_plasticity()
    
    # Phase 3: Complex patterns
    print("Phase 3: Complex patterns")
    complex_patterns = [
        np.random.rand(10, 10)        # Full noise
    ]
    
    for epoch in range(5):
        for pattern in complex_patterns:
            feed_sense_input(sim.model, "vision", pattern)
            sim.run(steps=30)
            sim.apply_plasticity()

curriculum_training(sim)
```

---

## Monitoring Learning

### Weight Distribution Over Time

```python
import matplotlib.pyplot as plt

def monitor_learning(sim, num_steps=1000, check_interval=100):
    """Monitor weight changes during learning."""
    
    weight_history = []
    
    for step in range(num_steps):
        sim.step()
        
        if step % 10 == 0:
            sim.apply_plasticity()
        
        if step % check_interval == 0:
            weights = [s['weight'] for s in sim.model.get_synapses().values()]
            weight_history.append({
                'step': step
                'mean': np.mean(weights)
                'std': np.std(weights)
                'min': np.min(weights)
                'max': np.max(weights)
            })
            
            print(f"Step {step}: "
                  f"mean={np.mean(weights):.3f}, "
                  f"std={np.std(weights):.3f}")
    
    # Plot results
    steps = [h['step'] for h in weight_history]
    means = [h['mean'] for h in weight_history]
    stds = [h['std'] for h in weight_history]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, means, label='Mean weight')
    plt.fill_between(steps
                     [m - s for m, s in zip(means, stds)]
                     [m + s for m, s in zip(means, stds)]
                     alpha=0.3, label='±1 std')
    plt.xlabel('Step')
    plt.ylabel('Weight')
    plt.title('Weight Evolution During Learning')
    plt.legend()
    plt.savefig('weight_evolution.png')
    plt.close()

monitor_learning(sim)
```

### Activity-Dependent Changes

```python
def correlate_activity_and_learning(sim, num_steps=1000):
    """Correlate neural activity with weight changes."""
    
    initial_weights = {
        sid: s['weight']
        for sid, s in sim.model.get_synapses().items()
    }
    
    # Track activity
    synapse_coactivations = {sid: 0 for sid in initial_weights.keys()}
    
    for step in range(num_steps):
        spiked = sim.step()
        
        # Count co-activations
        for sid, syn in sim.model.get_synapses().items():
            if syn['pre'] in spiked and syn['post'] in spiked:
                synapse_coactivations[sid] += 1
        
        if step % 10 == 0:
            sim.apply_plasticity()
    
    # Analyze correlation
    final_weights = {
        sid: s['weight']
        for sid, s in sim.model.get_synapses().items()
    }
    
    weight_changes = [
        final_weights[sid] - initial_weights[sid]
        for sid in initial_weights.keys()
    ]
    
    coactivations = list(synapse_coactivations.values())
    
    # Correlation
    correlation = np.corrcoef(weight_changes, coactivations)[0, 1]
    print(f"Correlation between co-activation and weight change: {correlation:.3f}")

correlate_activity_and_learning(sim)
```

---

## Advanced Topics

### Homeostatic Plasticity

**Homeostatic plasticity** stabilizes network activity by globally scaling synaptic weights to maintain neurons near a target firing rate. This prevents runaway excitation or silencing after Hebbian changes.

```python
from src.plasticity import homeostatic_scaling

# Apply homeostatic scaling
homeostatic_scaling(
    neurons=sim.model.neurons,
    synapses=sim.model.synapses,
    target_rate=5.0,           # Target firing rate in Hz
    time_window=1000,          # Time window for rate calculation
    scaling_rate=0.01,         # Rate of adjustment
    model=sim.model            # For weight bounds
)

# Integrate with training
for step in range(1000):
    sim.step()
    
    # Regular plasticity
    if step % 10 == 0:
        sim.apply_plasticity()
    
    # Homeostatic scaling (slower timescale)
    if step % 100 == 0:
        homeostatic_scaling(
            neurons=sim.model.neurons,
            synapses=sim.model.synapses,
            model=sim.model
        )
```

**Key Features:**
- Multiplicative scaling of all incoming synapses
- Maintains target firing rate
- Prevents weight saturation
- Acts on slow timescales (100-1000 steps)

### Metaplasticity (BCM Theory)

**Metaplasticity** is "the plasticity of plasticity" - the threshold for inducing LTP/LTD itself changes based on recent activity.

```python
from src.plasticity import BCMThreshold, bcm_plasticity

# Create BCM threshold for each postsynaptic neuron
bcm_thresholds = {
    neuron_id: BCMThreshold(theta=0.5, target_rate=5.0, tau=1000.0)
    for neuron_id in sim.model.neurons.keys()
}

# Training with BCM plasticity
for step in range(1000):
    # Get neuron activities
    activities = sim.step()
    
    # Helper function to estimate firing rate from recent activity
    def estimate_rate(neuron_id, recent_activities, window=100):
        """Estimate firing rate from spike count in recent window."""
        spike_count = sum(1 for act in recent_activities[-window:] if neuron_id in act)
        return spike_count / window * 1000.0  # Convert to Hz
    
    # Update each synapse with BCM rule
    for synapse in sim.model.synapses:
        post_id = synapse.post_id
        
        # Estimate firing rates
        pre_rate = estimate_rate(synapse.pre_id, activities)
        post_rate = estimate_rate(post_id, activities)
        
        # Apply BCM plasticity
        bcm_plasticity(
            synapse=synapse,
            pre_rate=pre_rate,
            post_rate=post_rate,
            bcm_threshold=bcm_thresholds[post_id],
            learning_rate=0.01,
            model=sim.model
        )
        
        # Update threshold based on postsynaptic activity
        bcm_thresholds[post_id].update(postsynaptic_rate=post_rate, dt=1.0)
```

**BCM Learning Rule:**
- When `post_rate > threshold`: **LTP** (strengthening)
- When `post_rate < threshold`: **LTD** (weakening)
- Threshold adapts to maintain target activity level

### Short-Term Plasticity

**Short-term plasticity** (STP) modifies synaptic efficacy on timescales of milliseconds to seconds, implementing facilitation and depression.

```python
from src.plasticity import (
    ShortTermPlasticityState,
    apply_short_term_plasticity,
    create_facilitating_synapse,
    create_depressing_synapse
)

# Create STP states for each synapse
stp_states = {}

for synapse in sim.model.synapses:
    # Choose facilitation or depression based on synapse type
    if is_temporal_integration_synapse(synapse):
        # Facilitating (builds up with repeated activation)
        stp_states[synapse] = create_facilitating_synapse()
    else:
        # Depressing (adapts/habituates)
        stp_states[synapse] = create_depressing_synapse()

# Simulation with short-term plasticity
for step in range(1000):
    # Get spikes from previous step
    spikes = set(sim.step())
    
    # Update each synapse
    for synapse in sim.model.synapses:
        pre_spiked = synapse.pre_id in spikes
        stp_state = stp_states[synapse]
        
        # Get effective weight (modulated by STP)
        effective_weight = apply_short_term_plasticity(
            synapse=synapse,
            stp_state=stp_state,
            presynaptic_spike=pre_spiked,
            dt=1.0
        )
        
        # Use effective_weight for synaptic transmission
        # (This would be integrated into neuron update)
```

**STP Types:**
- **Facilitation**: Synapses strengthen with repeated use (temporal integration)
- **Depression**: Synapses weaken with repeated use (novelty detection, adaptation)

### Combining All Mechanisms

```python
from src.plasticity import (
    hebbian_update,
    apply_weight_decay,
    homeostatic_scaling,
    BCMThreshold,
    bcm_plasticity,
    ShortTermPlasticityState,
    apply_short_term_plasticity
)

# Setup all plasticity mechanisms
bcm_thresholds = {nid: BCMThreshold() for nid in sim.model.neurons.keys()}
stp_states = {syn: ShortTermPlasticityState() for syn in sim.model.synapses}

# Multi-scale plasticity training
for step in range(10000):
    spikes = set(sim.step())
    
    # Short-term plasticity (every step)
    for synapse in sim.model.synapses:
        pre_spiked = synapse.pre_id in spikes
        stp_state = stp_states[synapse]
        effective_weight = apply_short_term_plasticity(
            synapse, stp_state, pre_spiked, dt=1.0
        )
    
    # Long-term plasticity (every 10 steps)
    if step % 10 == 0:
        for synapse in sim.model.synapses:
            pre_active = synapse.pre_id in spikes
            post_active = synapse.post_id in spikes
            
            # Hebbian learning
            hebbian_update(synapse, pre_active, post_active, sim.model)
            
            # Weight decay
            apply_weight_decay(synapse, sim.model)
    
    # Metaplasticity (every 50 steps)
    if step % 50 == 0:
        # Helper to estimate firing rates
        def estimate_firing_rate(neuron_id):
            # Simplified: use recent spike count
            return len([s for s in recent_spikes if s == neuron_id]) / 50.0
        
        for synapse in sim.model.synapses:
            post_id = synapse.post_id
            pre_rate = estimate_firing_rate(synapse.pre_id)
            post_rate = estimate_firing_rate(post_id)
            
            bcm_plasticity(
                synapse, pre_rate, post_rate,
                bcm_thresholds[post_id], 0.01, sim.model
            )
            bcm_thresholds[post_id].update(post_rate, dt=50.0)
    
    # Homeostatic scaling (every 1000 steps)
    if step % 1000 == 0:
        homeostatic_scaling(
            sim.model.neurons,
            sim.model.synapses,
            model=sim.model
        )

print("Multi-scale plasticity training complete!")
```

### Selective Plasticity

Only learn in specific conditions:

```python
def selective_learning(sim, learn_threshold=5):
    """Only apply plasticity when activity is high enough."""
    
    spiked = sim.step()
    
    # Only learn if many neurons active
    if len(spiked) >= learn_threshold:
        sim.apply_plasticity()
        return True
    return False

# Use it
learning_events = 0
for step in range(1000):
    if selective_learning(sim, learn_threshold=10):
        learning_events += 1

print(f"Learned on {learning_events} out of 1000 steps")
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Complete plasticity training example."""

import numpy as np
from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input

def train_pattern_recognition():
    """Train network to recognize patterns."""
    
    # Setup
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    
    sim.initialize_neurons(areas=["V1_like"], density=0.1)
    sim.initialize_random_synapses(connection_prob=0.1)
    
    # Training patterns
    patterns = {
        'vertical': np.repeat([[0,0,1,0,0]], 5, axis=0)
        'horizontal': np.repeat([[0,0,0,0,0], [0,0,0,0,0], [1,1,1,1,1]], [2,2,1], axis=0)
        'diagonal': np.eye(5)
    }
    
    print("=== Training Pattern Recognition ===\n")
    
    # Train for multiple epochs
    for epoch in range(10):
        print(f"Epoch {epoch+1}/10")
        
        for name, pattern in patterns.items():
            # Show pattern
            feed_sense_input(
                sim.model
                "vision"
                pattern
                
            )
            
            # Process and learn
            for step in range(30):
                sim.step()
                
                if step % 5 == 0:
                    sim.apply_plasticity()
        
        # Check weights after epoch
        weights = [s['weight'] for s in sim.model.get_synapses().values()]
        print(f"  Mean weight: {np.mean(weights):.3f}\n")
    
    print("=== Training Complete ===")

if __name__ == "__main__":
    train_pattern_recognition()
```

---

## Tips and Best Practices

1. **Start Conservative**: Use low learning rates initially
2. **Monitor Weights**: Track weight distribution regularly
3. **Balance Learning and Decay**: Find equilibrium
4. **Use Epochs**: Organize training into epochs
5. **Save Checkpoints**: Save model state periodically
6. **Reproducibility**: Always use seeds for experiments

---

## Next Steps

- **[Quick Start Evaluation](QUICK_START_EVALUATION.md)** - Benchmark performance
- **[API Documentation](../api/API.md)** - Full API reference
- **[Examples](../../examples/)** - Advanced training examples

---

*Last Updated: December 2025*  
*Part of the 4D Neural Cognition Documentation*
