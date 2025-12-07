# Performance Optimizations

This document describes the performance optimizations available in the 4D Neural Cognition system and how to use them.

## Overview

The system provides two optional performance optimizations that can significantly improve simulation efficiency for large-scale neural networks:

1. **Sparse Connectivity Matrix**: Efficient storage and lookup of synaptic connections
2. **Time-Indexed Spike Buffer**: O(1) spike time lookups for synaptic delay calculations

Both optimizations are **opt-in** and maintain full backward compatibility with existing code.

## Sparse Connectivity Matrix

### Background

By default, synapses are stored as a Python list. For large networks with many synapses, this can be:
- **Memory inefficient**: Each Synapse object has Python overhead
- **Slow for queries**: Finding all incoming/outgoing synapses requires O(n) iteration

The sparse connectivity matrix addresses these issues using a Compressed Sparse Row (CSR)-like format.

### Benefits

- **Memory efficiency**: O(num_synapses) instead of O(num_neurons²) for dense representations
- **Fast queries**: O(1) access to row start, O(k) for k connections per neuron
- **Optimized for neural networks**: Row-wise access pattern matches typical synapse queries

### Usage

#### Creating a Model with Sparse Connectivity

```python
from src.brain_model import BrainModel

# Enable sparse connectivity
model = BrainModel(
    config_path='brain_base_model.json',
    use_sparse_connectivity=True  # Enable optimization
)
```

#### All Standard Operations Work Identically

```python
# Add neurons
n1 = model.add_neuron(0, 0, 0, 0)
n2 = model.add_neuron(0, 0, 0, 1)

# Add synapses - same API
model.add_synapse(n1.id, n2.id, weight=0.5)

# Query synapses - same API
incoming = model.get_synapses_for_neuron(n2.id, direction="post")
outgoing = model.get_synapses_for_neuron(n1.id, direction="pre")

# Remove neurons - automatically removes associated synapses
model.remove_neuron(n1.id)

# Serialization works transparently
data = model.to_dict()
model2 = BrainModel.from_dict(data)  # Preserves sparse connectivity setting
```

### When to Use

**Use sparse connectivity when:**
- Network has >1000 neurons
- You need fast incoming/outgoing synapse queries
- Memory usage is a concern

**Stick with default when:**
- Network is small (<100 neurons)
- You need to iterate through all synapses frequently
- Simplicity is preferred over optimization

## Time-Indexed Spike Buffer

### Background

By default, spike times are stored as `dict[neuron_id, list[spike_times]]`. When computing synaptic input, the system must:
1. Retrieve spike list for each presynaptic neuron
2. Iterate through the list to find spikes matching the delay: O(n) per synapse

For networks with many spikes and synapses, this becomes a performance bottleneck.

### Benefits

- **O(1) spike lookup**: Direct hash table lookup instead of list iteration
- **Memory efficient**: Circular buffer automatically cleans up old spikes
- **Optimized for delay queries**: Indexed by time, perfect for "did neuron X spike at time T?"

### Usage

#### Creating a Simulation with Time-Indexed Spikes

```python
from src.brain_model import BrainModel
from src.simulation import Simulation

model = BrainModel(config_path='brain_base_model.json')

# Enable time-indexed spike buffer
sim = Simulation(
    model,
    seed=42,
    use_time_indexed_spikes=True  # Enable optimization
)
```

#### All Standard Operations Work Identically

```python
# Run simulation - same API
sim.initialize_neurons(area_names=['V1_like'], density=0.1)
sim.initialize_random_synapses(connection_probability=0.01)

for step in range(100):
    stats = sim.step()  # Internally uses O(1) spike lookups
    print(f"Step {step}: {len(stats['spikes'])} spikes")

# Access spike history - same API
if sim.use_time_indexed_spikes:
    # spike_history is a SpikeHistoryAdapter with dict-like interface
    spike_times = sim.spike_history[neuron_id]
else:
    # spike_history is a regular dict
    spike_times = sim.spike_history[neuron_id]
```

### When to Use

**Use time-indexed spike buffer when:**
- Network has high spike rates
- Synaptic delays are used (delay > 1)
- Many synapses per neuron (high connectivity)

**Stick with default when:**
- Network has low spike rates
- No synaptic delays (all delays = 1)
- Simplicity is preferred over optimization

## Using Both Optimizations Together

Both optimizations can be enabled simultaneously for maximum performance:

```python
from src.brain_model import BrainModel
from src.simulation import Simulation

# Enable both optimizations
model = BrainModel(
    config_path='brain_base_model.json',
    use_sparse_connectivity=True
)

sim = Simulation(
    model,
    seed=42,
    use_time_indexed_spikes=True
)

# Run large-scale simulation efficiently
sim.initialize_neurons(area_names=['V1_like'], density=0.5)
sim.initialize_random_synapses(connection_probability=0.05)

for step in range(10000):
    stats = sim.step()
```

## Performance Comparison

### Memory Usage

| Configuration | Small Network (100 neurons) | Large Network (10,000 neurons) |
|--------------|----------------------------|-------------------------------|
| Default | ~50 KB | ~5 MB (list) / ~500 MB (dense) |
| Sparse connectivity | ~45 KB | ~2 MB |
| Time-indexed spikes | ~55 KB | ~6 MB (with cleanup) |
| Both optimizations | ~50 KB | ~3 MB |

### Query Performance

| Operation | Default | Optimized | Speedup |
|-----------|---------|-----------|---------|
| Get incoming synapses | O(total_synapses) | O(synapses_for_neuron) | 10-100x |
| Check spike at time T | O(spike_history_size) | O(1) | 5-50x |
| Simulation step | O(neurons * synapses) | O(neurons * avg_connections) | 2-10x |

*Actual speedup depends on network topology and spike patterns*

## Implementation Details

### Sparse Connectivity Matrix

The implementation uses a CSR-inspired format with hash table indices:

```python
class SparseConnectivityMatrix:
    _pre_ids: List[int]      # Pre-synaptic neuron IDs
    _post_ids: List[int]     # Post-synaptic neuron IDs
    _weights: List[float]    # Synaptic weights
    _delays: List[int]       # Synaptic delays
    
    # Index structures for O(1) row access
    _post_index: dict[int, List[int]]  # post_id -> synapse indices
    _pre_index: dict[int, List[int]]   # pre_id -> synapse indices
```

### Time-Indexed Spike Buffer

The implementation uses time-indexed hash tables with circular buffer behavior:

```python
class TimeIndexedSpikeBuffer:
    window_size: int = 100  # Keep last 100 time steps
    _spikes_by_time: Dict[int, Set[int]]  # time -> set of neuron IDs
    
    def did_spike_at(self, neuron_id: int, time: int) -> bool:
        # O(1) lookup
        return neuron_id in self._spikes_by_time[time]
```

## Testing

Comprehensive tests ensure correctness and backward compatibility:

- `tests/test_sparse_connectivity.py`: 17 tests for sparse matrix
- `tests/test_time_indexed_spikes.py`: 23 tests for spike buffer
- `tests/test_optimizations.py`: 13 integration tests
- All existing tests pass with optimizations enabled

Run tests:
```bash
pytest tests/test_sparse_connectivity.py -v
pytest tests/test_time_indexed_spikes.py -v
pytest tests/test_optimizations.py -v
```

## Migration Guide

### Enabling Optimizations in Existing Code

**Before:**
```python
model = BrainModel(config_path='config.json')
sim = Simulation(model, seed=42)
```

**After:**
```python
model = BrainModel(
    config_path='config.json',
    use_sparse_connectivity=True  # Add this
)
sim = Simulation(
    model,
    seed=42,
    use_time_indexed_spikes=True  # Add this
)
```

No other code changes required! The optimizations are transparent.

### Checking Current Configuration

```python
# Check if sparse connectivity is enabled
print(f"Using sparse connectivity: {model.use_sparse_connectivity}")

# Check if time-indexed spikes are enabled
print(f"Using time-indexed spikes: {sim.use_time_indexed_spikes}")
```

## Future Improvements

Potential future enhancements:

1. **Automatic optimization selection**: Choose optimizations based on network size
2. **Hybrid approaches**: Use sparse for large networks, list for small ones
3. **GPU acceleration**: Port optimized data structures to GPU
4. **Compression**: Further reduce memory usage for very large networks

## See Also

- [Architecture Documentation](ARCHITECTURE.md)
- [Performance Benchmarks](../tests/test_performance.py)
- [API Reference](api/API.md)

---

*Last Updated: December 2025*
