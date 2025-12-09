# Performance Optimization Guide

This guide covers performance optimization features in 4D Neural Cognition, including GPU acceleration, parallel computing, and memory optimization.

## Table of Contents

- [Overview](#overview)
- [GPU Acceleration](#gpu-acceleration)
- [Parallel Computing](#parallel-computing)
- [Memory Optimization](#memory-optimization)
- [Benchmarking](#benchmarking)
- [Best Practices](#best-practices)

---

## Overview

4D Neural Cognition provides three main categories of performance optimizations:

1. **GPU Acceleration**: Use CUDA-enabled GPUs for vectorized neuron updates and matrix operations
2. **Parallel Computing**: Multi-core CPU parallelization with spatial partitioning
3. **Memory Optimization**: Efficient memory usage through sparse matrices, compression, and caching

These features can be used independently or combined for maximum performance.

---

## GPU Acceleration

### Requirements

GPU acceleration requires:
- NVIDIA GPU with CUDA support
- CuPy library (CUDA Python)

### Installation

```bash
# For CUDA 12.x
pip install cupy-cuda12x

# For CUDA 11.x
pip install cupy-cuda11x
```

### Basic Usage

```python
from src.gpu_acceleration import GPUAccelerator

# Initialize GPU accelerator
accelerator = GPUAccelerator(use_gpu=True, device_id=0)

# Check if GPU is available
if accelerator.is_gpu_available():
    print(f"GPU acceleration enabled")
else:
    print("Falling back to CPU")
```

### Vectorized Neuron Updates

GPU acceleration provides vectorized LIF neuron updates:

```python
import numpy as np

# Prepare neuron states
n_neurons = 10000
v_membrane = np.random.randn(n_neurons) * 10 - 65
synaptic_input = np.random.randn(n_neurons)
external_input = np.random.randn(n_neurons)
refractory_mask = np.random.rand(n_neurons) < 0.1

# LIF parameters
params = {
    'tau_m': 20.0,
    'v_rest': -65.0,
    'v_threshold': -50.0,
    'v_reset': -70.0
}

# Update all neurons on GPU
v_new, spikes = accelerator.vectorized_lif_update(
    v_membrane,
    synaptic_input,
    external_input,
    refractory_mask,
    params
)

print(f"Updated {n_neurons} neurons")
print(f"Spikes: {spikes.sum()}")
```

### Sparse Matrix Operations

Efficient synapse computation using GPU:

```python
# Create sparse connectivity matrix
from scipy.sparse import csr_matrix

n_neurons = 5000
connectivity_prob = 0.01
connections = np.random.rand(n_neurons, n_neurons) < connectivity_prob
weights = connections * np.random.randn(n_neurons, n_neurons) * 0.1
weight_matrix = csr_matrix(weights)

# Compute synaptic currents on GPU
spike_vector = np.random.rand(n_neurons) < 0.05
synaptic_currents = accelerator.sparse_synapse_matmul(
    spike_vector,
    weight_matrix
)

print(f"Computed synaptic currents for {n_neurons} neurons")
```

### Benchmarking GPU vs CPU

```python
# Benchmark LIF update performance
results = accelerator.benchmark_vs_cpu(
    operation='lif_update',
    array_sizes=[1000, 5000, 10000, 50000],
    n_iterations=100
)

print("\nGPU vs CPU Performance:")
for size, cpu_time, gpu_time, speedup in zip(
    results['array_sizes'],
    results['cpu_times'],
    results['gpu_times'],
    results['speedups']
):
    print(f"  {size:6d} neurons: {speedup:5.2f}x speedup "
          f"(CPU: {cpu_time*1000:.2f}ms, GPU: {gpu_time*1000:.2f}ms)")
```

### Memory Management

Monitor and manage GPU memory:

```python
# Check GPU memory usage
memory = accelerator.get_memory_usage()
print(f"GPU Memory: {memory['used']:.1f} / {memory['total']:.1f} MB "
      f"({memory['percent']:.1f}% used)")

# Clear GPU memory cache
accelerator.clear_memory()
```

---

## Parallel Computing

### Requirements

Parallel computing uses Python's multiprocessing and works on any multi-core CPU.

### Basic Usage

```python
from src.parallel_computing import ParallelSimulator
from src.brain_model import BrainModel

# Create brain model
config = {
    "lattice_shape": [50, 50, 20, 10],
    "neuron_model": {"params_default": {...}},
    ...
}
model = BrainModel(config=config)

# Initialize parallel simulator with 4 processes
sim = ParallelSimulator(
    model,
    n_processes=4,
    use_spatial_partitioning=True
)
```

### Spatial Partitioning

Neurons are automatically partitioned across processes based on spatial locality:

```python
from src.parallel_computing import ParallelSimulationEngine

# Create engine
engine = ParallelSimulationEngine(n_processes=4)

# Create spatial partitions
lattice_shape = [50, 50, 20, 10]
partitions = engine.create_spatial_partitions(lattice_shape)

print(f"Created {len(partitions)} partitions")
for p in partitions:
    print(f"  Partition {p.partition_id}: "
          f"x={p.x_range}, y={p.y_range}, z={p.z_range}, w={p.w_range}")
```

### Load Balancing

Check load balance across partitions:

```python
# Assign neurons to partitions
engine.assign_neurons_to_partitions(model.neurons, partitions)

# Get load balance statistics
stats = engine.get_load_balance_stats()
print(f"Neurons per partition: {stats['neurons_per_partition']}")
print(f"Mean: {stats['mean']:.1f}, Std: {stats['std']:.1f}")
print(f"Imbalance: {stats['imbalance']:.2%}")
```

### Parallel Neuron Update

```python
# Prepare synaptic inputs
synaptic_inputs = {nid: 0.0 for nid in model.neurons.keys()}

# Update neurons in parallel
results = sim.parallel_neuron_update(
    synaptic_inputs,
    current_step=0
)

# Results contain (v_membrane, spiked) for each neuron
for neuron_id, (v_mem, spiked) in results.items():
    if spiked:
        print(f"Neuron {neuron_id} spiked!")
```

### Scaling Benchmarks

Measure parallel scaling characteristics:

```python
from src.parallel_computing import benchmark_parallel_scaling

results = benchmark_parallel_scaling(
    model,
    n_steps=100,
    process_counts=[1, 2, 4, 8]
)

print("\nParallel Scaling:")
for n_proc, time, speedup, efficiency in zip(
    results['process_counts'],
    results['times'],
    results['speedups'],
    results['efficiencies']
):
    print(f"  {n_proc} processes: {speedup:.2f}x speedup, "
          f"{efficiency:.1%} efficiency ({time:.2f}s)")
```

---

## Memory Optimization

### Sparse Matrix Representation

Use sparse matrices for efficient synapse storage:

```python
from src.brain_model import BrainModel

# Enable sparse connectivity
model = BrainModel(
    config=config,
    use_sparse_connectivity=True
)

# Add neurons and synapses as usual
# Internally uses sparse matrix for O(num_synapses) memory
```

### Memory-Mapped Files

Store large models on disk with memory mapping:

```python
from src.memory_optimization import MemoryMappedModel

# Create memory-mapped model
filepath = "large_model.h5"
mm_model = MemoryMappedModel(filepath, mode='w')

# Store neurons with compression
mm_model.store_neurons(
    model.neurons,
    chunk_size=1000,
    compression='gzip'
)

# Load neurons (entire model or partial)
loaded_neurons = mm_model.load_neurons()  # All neurons
loaded_subset = mm_model.load_neurons(indices=[0, 1, 2])  # Subset

mm_model.close()
```

### Inactive Neuron Compression

Compress neurons that haven't spiked recently:

```python
from src.memory_optimization import InactiveNeuronCompressor

compressor = InactiveNeuronCompressor(inactivity_threshold=1000)

# Compress inactive neurons
active_neurons, bytes_saved = compressor.compress_inactive_neurons(
    model.neurons,
    spike_history,
    current_step
)

print(f"Compressed {len(model.neurons) - len(active_neurons)} neurons")
print(f"Saved {bytes_saved / 1024:.1f} KB")

# Decompress when needed
neuron = compressor.decompress_neuron(neuron_id)
```

### Cache Optimization

Reorder neurons for better spatial locality:

```python
from src.memory_optimization import CacheOptimizer

optimizer = CacheOptimizer()

# Reorder neurons using Z-order curve
model.neurons = optimizer.reorder_neurons_spatial_locality(model.neurons)

# Analyze access patterns
access_sequence = [0, 1, 2, 3, 50, 51, 52]  # Example access pattern
stats = optimizer.analyze_access_pattern(access_sequence)
print(f"Locality score: {stats['locality_score']:.2f}")
```

### Memory Profiling

Monitor memory usage:

```python
from src.memory_optimization import MemoryProfiler

profiler = MemoryProfiler()

# Take initial snapshot
profiler.take_snapshot('initial', model)

# ... modify model ...

# Take another snapshot
profiler.take_snapshot('after_changes', model)

# Compare snapshots
diff = profiler.compare_snapshots('initial', 'after_changes')
print(f"Memory change: {diff['total_diff_mb']:.2f} MB")

# Print full report
profiler.print_report()
```

### High-Level Optimization

Apply multiple optimizations at once:

```python
from src.memory_optimization import optimize_model_memory

results = optimize_model_memory(
    model,
    enable_compression=True,
    enable_cache_optimization=True
)

print(f"Initial memory: {results['initial_memory_mb']:.2f} MB")
print(f"Final memory: {results['final_memory_mb']:.2f} MB")
print(f"Bytes saved: {results['bytes_saved'] / 1024:.1f} KB")
print(f"Optimizations: {results['optimizations_applied']}")
```

---

## Benchmarking

### Performance Benchmarks

Compare different configurations:

```python
from src.profiling_tools import PerformanceProfiler

profiler = PerformanceProfiler()

# Profile simulation step
profiler.start('simulation_step')
stats = simulation.step()
profiler.stop('simulation_step')

# Profile multiple iterations
for i in range(100):
    profiler.start('neuron_update')
    # ... neuron updates ...
    profiler.stop('neuron_update')

# Get results
results = profiler.get_results()
for result in results:
    print(f"{result.name}: {result.avg_time*1000:.2f}ms avg "
          f"({result.n_calls} calls)")
```

### Time-Indexed Spike Lookup

Use O(1) spike lookup for better performance:

```python
from src.simulation import Simulation

# Enable time-indexed spikes
sim = Simulation(
    model,
    seed=42,
    use_time_indexed_spikes=True  # O(1) spike lookup
)

# Simulation will automatically use optimized spike checking
```

---

## Best Practices

### When to Use GPU Acceleration

✅ **Use GPU when:**
- You have 10,000+ neurons
- Performing many simulation steps
- Using dense connectivity
- Available GPU memory is sufficient

❌ **Don't use GPU when:**
- Small models (<1000 neurons)
- Limited GPU memory
- Overhead exceeds benefits

### When to Use Parallel Computing

✅ **Use parallel processing when:**
- You have 4+ CPU cores
- Model has 1000+ neurons
- Neurons can be spatially partitioned
- Simulation is CPU-bound

❌ **Don't use parallel when:**
- Very small models
- Memory bandwidth limited
- Communication overhead is high

### Memory Optimization Tips

1. **Enable sparse connectivity** for large models with low connection probability
2. **Use time-indexed spikes** for models with many synapses
3. **Compress inactive neurons** in long-running simulations
4. **Use memory mapping** for models larger than available RAM
5. **Optimize cache locality** with spatial reordering

### Combined Optimizations

For best performance, combine multiple strategies:

```python
# Example: Large-scale simulation with all optimizations
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.gpu_acceleration import GPUAccelerator
from src.parallel_computing import ParallelSimulator
from src.memory_optimization import optimize_model_memory

# Create model with sparse connectivity
model = BrainModel(config=config, use_sparse_connectivity=True)

# Optimize memory
optimize_model_memory(model, enable_compression=True, enable_cache_optimization=True)

# Choose acceleration based on model size
n_neurons = len(model.neurons)

if n_neurons > 10000:
    # Large model: Use GPU if available
    accelerator = GPUAccelerator(use_gpu=True)
    if accelerator.is_gpu_available():
        print("Using GPU acceleration")
        # Integrate GPU updates with simulation
    else:
        # Fall back to parallel CPU
        sim = ParallelSimulator(model, n_processes=4)
elif n_neurons > 1000:
    # Medium model: Use parallel CPU
    sim = ParallelSimulator(model, n_processes=4)
else:
    # Small model: Standard simulation
    sim = Simulation(model, use_time_indexed_spikes=True)

# Run simulation
for step in range(1000):
    stats = sim.step() if hasattr(sim, 'step') else None
```

---

## Performance Tips

### General Tips

1. **Profile first**: Use profiling tools to identify bottlenecks
2. **Start simple**: Begin with standard simulation, add optimizations if needed
3. **Measure everything**: Benchmark before and after optimizations
4. **Consider tradeoffs**: More memory vs. faster computation

### Platform-Specific

**Linux**: Best performance, full feature support

**Windows**: GPU support may require WSL for CUDA

**macOS**: No CUDA support, focus on CPU parallelization

---

## Troubleshooting

### GPU Issues

**Problem**: CuPy not found
```
Solution: Install CuPy with: pip install cupy-cuda12x
```

**Problem**: Out of GPU memory
```
Solution: 
- Reduce model size
- Use batched processing
- Call accelerator.clear_memory() periodically
```

### Parallel Computing Issues

**Problem**: No speedup with parallel processing
```
Solution:
- Check if model is large enough (>1000 neurons)
- Verify load balancing with engine.get_load_balance_stats()
- Reduce number of processes
```

**Problem**: Deadlock or freeze
```
Solution:
- Ensure no shared state between processes
- Check for proper serialization of model data
```

### Memory Issues

**Problem**: Out of memory
```
Solution:
- Enable sparse connectivity
- Use memory-mapped files
- Compress inactive neurons
- Reduce model size
```

---

## References

- [CuPy Documentation](https://docs.cupy.dev/)
- [Python Multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
- [HDF5 and h5py](https://www.h5py.org/)
- [Sparse Matrices (SciPy)](https://docs.scipy.org/doc/scipy/reference/sparse.html)

---

*Last Updated: December 2025*
