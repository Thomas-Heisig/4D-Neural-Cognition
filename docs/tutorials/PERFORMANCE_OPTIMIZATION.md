# Performance Optimization Guide

This guide provides strategies and best practices for optimizing the performance of 4D Neural Cognition simulations.

## Table of Contents

1. [Performance Overview](#performance-overview)
2. [Profiling Your Simulation](#profiling-your-simulation)
3. [Network Size Optimization](#network-size-optimization)
4. [Memory Management](#memory-management)
5. [Computational Optimizations](#computational-optimizations)
6. [Storage and I/O](#storage-and-io)
7. [Simulation Parameters](#simulation-parameters)
8. [Best Practices](#best-practices)

---

## Performance Overview

### Key Performance Factors

The performance of your simulation depends on:

1. **Network Size**: Number of neurons and synapses
2. **Connectivity**: Synapse density and connection patterns
3. **Simulation Length**: Number of time steps
4. **Model Complexity**: LIF vs Izhikevich, plasticity rules
5. **I/O Operations**: Checkpointing, logging, visualization
6. **Memory Usage**: History tracking, state storage

### Performance Metrics

Monitor these metrics:
- **Steps per second**: Overall simulation speed
- **Memory usage**: RAM consumption over time
- **CPU utilization**: Processor usage
- **Storage size**: Checkpoint and log file sizes

---

## Profiling Your Simulation

### Basic Timing

```python
import time
from src.simulation import Simulation

# Time your simulation
start = time.time()

sim = Simulation(model, seed=42)
sim.initialize_neurons(area_names=["cortex"], density=0.1)
sim.initialize_random_synapses(connection_probability=0.01)

for _ in range(1000):
    sim.step()

elapsed = time.time() - start
steps_per_sec = 1000 / elapsed

print(f"Simulation time: {elapsed:.2f}s")
print(f"Steps per second: {steps_per_sec:.2f}")
```

### Memory Profiling

```python
import tracemalloc

# Start memory tracking
tracemalloc.start()

# Run simulation
sim = Simulation(model, seed=42)
# ... initialize and run ...

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
```

### Performance Testing

```python
from tests.test_performance import (
    test_simulation_scaling,
    test_spike_checking_performance
)

# Run performance benchmarks
test_simulation_scaling()
test_spike_checking_performance()
```

---

## Network Size Optimization

### Choose Appropriate Network Size

```python
# Small network (fast, for development)
small_config = {
    "num_neurons": 100,
    "connection_prob": 0.05,
    "area_density": 0.05
}

# Medium network (balanced)
medium_config = {
    "num_neurons": 1000,
    "connection_prob": 0.01,
    "area_density": 0.1
}

# Large network (slow, for production)
large_config = {
    "num_neurons": 10000,
    "connection_prob": 0.001,
    "area_density": 0.2
}
```

### Sparse Connectivity

Use sparse connectivity to reduce computational load:

```python
# Instead of dense connections
# DON'T: connection_probability=0.5  # Too dense!

# Use sparse connectivity
sim.initialize_random_synapses(
    connection_probability=0.01,  # 1% connectivity
    weight_mean=0.1,
    weight_std=0.05
)
```

**Rule of Thumb**: Connection probability should be inversely proportional to network size
- 100 neurons: 0.05-0.1 (5-10%)
- 1000 neurons: 0.01-0.02 (1-2%)
- 10000 neurons: 0.001-0.005 (0.1-0.5%)

### Area-Based Organization

Organize neurons into areas for better performance:

```python
# Localized connectivity (faster)
sim.initialize_neurons(
    area_names=["vision", "motor", "association"],
    density=0.1
)

# Connect within areas more densely
for area in ["vision", "motor", "association"]:
    neurons_in_area = model.get_neurons_in_area(area)
    # Connect neurons within same area
    # This is more cache-friendly
```

---

## Memory Management

### Limit History Tracking

The framework automatically limits history to prevent memory leaks:

```python
# History is automatically limited to last 100 steps
# Check current implementation:
print(f"Spike history length: {len(sim.spike_history)}")
# Should not exceed 100 steps

# Older entries are automatically cleaned up
```

### Checkpoint Management

Control checkpoint frequency and retention:

```python
# Configure auto-checkpointing
checkpoint_config = {
    "auto_checkpoint": True,
    "checkpoint_interval": 1000,  # Every 1000 steps
    "max_checkpoints": 3,  # Keep only last 3
}

# Manual checkpointing for control
if sim.time % 5000 == 0:  # Every 5000 steps
    sim.save_checkpoint(f"checkpoint_{sim.time}.h5")
```

### Reduce Logging

```python
# Minimal logging for performance
sim = Simulation(model, seed=42, verbose=False)

# Or log only periodically
for i in range(10000):
    verbose = (i % 1000 == 0)  # Log every 1000 steps
    sim.step(verbose=verbose)
```

---

## Computational Optimizations

### Choose Efficient Neuron Model

```python
# LIF is faster than Izhikevich
# Use LIF for large simulations
for neuron in model.neurons.values():
    neuron.model_type = "lif"  # Faster

# Use Izhikevich only when biological realism is critical
# for neuron in model.neurons.values():
#     neuron.model_type = "izhikevich"  # More accurate but slower
```

### Optimize Spike Checking

The framework uses optimized O(m) spike checking with set lookups:

```python
# This is already optimized in the framework
# Spike checking uses set lookup: O(1) per synapse
# Old O(n*m) approach has been replaced

# Make sure you're on the latest version
# Check spike_history implementation:
assert isinstance(sim.spike_history[sim.time], set)
```

### Batch Operations

```python
# Add neurons in batch
neurons = [
    Neuron(id=i, x=i%10, y=i//10, z=0, w=0)
    for i in range(1000)
]
for neuron in neurons:
    model.add_neuron(neuron)

# Better: Use high-level initialization
sim.initialize_neurons(area_names=["cortex"], density=0.1)
```

### Disable Unnecessary Features

```python
# Disable plasticity if not needed
sim = Simulation(model, seed=42)
# Don't call plasticity updates
# for _ in range(1000):
#     sim.step()
# plasticity.update_weights(...)  # Skip this

# Disable cell lifecycle if not needed
# Similar approach - just don't call lifecycle updates
```

---

## Storage and I/O

### Use Efficient Storage Format

```python
# HDF5 is more efficient than JSON for large models
from src.storage import save_to_hdf5, load_from_hdf5

# Save
save_to_hdf5(model, "model.h5", compression="gzip")

# Load
model = load_from_hdf5("model.h5")

# HDF5 benefits:
# - Compressed storage
# - Faster loading
# - Better for large arrays
```

### Limit Save Frequency

```python
# Don't save every step
for i in range(10000):
    sim.step()
    
    # Save only periodically
    if i % 1000 == 0:
        save_to_hdf5(model, f"snapshot_{i}.h5")

# Or save only at end
# ...
save_to_hdf5(model, "final_state.h5")
```

### Compression

```python
# Use compression for storage
save_to_hdf5(
    model,
    "model.h5",
    compression="gzip",  # or "lzf"
    compression_opts=9   # Max compression
)
```

---

## Simulation Parameters

### Time Step Size

```python
# Default dt=1.0 is good for most cases
sim.step(dt=1.0)

# Smaller dt for numerical stability (but slower)
# sim.step(dt=0.1)

# Larger dt for speed (but less accurate)
# sim.step(dt=2.0)  # Use with caution
```

### Refractory Period

```python
# Adjust refractory period
for neuron in model.neurons.values():
    neuron.params["refractory_period"] = 2  # steps
    # Shorter = more computation (more spikes)
    # Longer = less computation (fewer spikes)
```

### Plasticity Update Frequency

```python
# Update plasticity less frequently
for i in range(10000):
    sim.step()
    
    # Update plasticity every 10 steps instead of every step
    if i % 10 == 0:
        plasticity.update_weights(model, sim.time)
```

---

## Best Practices

### 1. Start Small, Scale Up

```python
# Development: Small network
dev_sim = create_simulation(num_neurons=100)
# Test algorithms, debug issues

# Testing: Medium network  
test_sim = create_simulation(num_neurons=1000)
# Performance testing, validation

# Production: Large network
prod_sim = create_simulation(num_neurons=10000)
# Final experiments
```

### 2. Use Seeded Random Numbers

```python
# Reproducible and efficient
sim = Simulation(model, seed=42)

# Same results every time
# No need to re-run failed experiments
```

### 3. Monitor Performance

```python
import time

times = []
for i in range(1000):
    start = time.time()
    sim.step()
    elapsed = time.time() - start
    times.append(elapsed)
    
    # Check for performance degradation
    if i > 0 and i % 100 == 0:
        avg_time = sum(times[-100:]) / 100
        print(f"Step {i}: Avg time = {avg_time*1000:.2f}ms")
        
        # Alert if getting slower
        if avg_time > 0.1:  # 100ms threshold
            print("WARNING: Simulation slowing down!")
```

### 4. Clean Up Resources

```python
# Close database connections
if hasattr(sim, 'knowledge_db'):
    sim.knowledge_db.close()

# Clear large data structures
sim.spike_history.clear()

# Delete unused objects
del model
del sim
```

### 5. Parallel-Ready Code

While the framework doesn't currently support parallelization, write code that could be parallelized:

```python
# Good: Operations on independent neurons
for neuron in model.neurons.values():
    neuron.age += 1  # Independent operation

# Avoid: Global state dependencies
# Don't make neurons depend on global counters
```

---

## Performance Benchmarks

### Expected Performance (reference hardware: modern CPU)

| Network Size | Connectivity | Steps/Second |
|--------------|-------------|--------------|
| 100 neurons | 1% | 1000+ |
| 1,000 neurons | 1% | 100-200 |
| 10,000 neurons | 1% | 10-20 |
| 100,000 neurons | 0.1% | 1-2 |

### Optimization Impact

| Optimization | Expected Speedup |
|--------------|------------------|
| LIF vs Izhikevich | 1.5-2x faster |
| Sparse connectivity | Linear with density |
| Reduced checkpointing | 10-50% faster |
| Disabled plasticity | 20-30% faster |
| HDF5 vs JSON | 2-5x faster I/O |

---

## Troubleshooting

### Simulation Too Slow

1. **Profile** to identify bottleneck
2. **Reduce** network size or connectivity
3. **Use** LIF instead of Izhikevich
4. **Disable** unnecessary features
5. **Checkpoint** less frequently

### Memory Issues

1. **Check** spike history size (should be ≤100)
2. **Reduce** checkpoint retention
3. **Use** smaller network
4. **Clear** old data periodically
5. **Monitor** memory growth

### Numerical Instability

1. **Reduce** time step size
2. **Add** bounds checking
3. **Normalize** weights
4. **Check** parameter ranges
5. **Use** validation functions

---

## Future Optimizations

Planned for future releases:

- **GPU Acceleration**: CUDA/OpenCL support
- **Parallel Computing**: Multi-core CPU parallelization
- **Sparse Matrices**: Efficient synapse representation
- **JIT Compilation**: Numba/JAX integration
- **Distributed**: Multi-machine simulations

---

## Further Reading

- [BASIC_SIMULATION.md](BASIC_SIMULATION.md) - Simulation basics
- [CUSTOM_NEURON_MODELS.md](CUSTOM_NEURON_MODELS.md) - Model selection
- Performance tests in `tests/test_performance.py`

---

*Last Updated: December 2025*
