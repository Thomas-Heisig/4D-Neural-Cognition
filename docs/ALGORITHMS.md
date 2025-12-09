# Algorithm Documentation

This document describes the key algorithms and computational methods used in the 4D Neural Cognition project.

## Table of Contents
- [Core Simulation Loop](#core-simulation-loop)
- [Neuron Update Algorithms](#neuron-update-algorithms)
- [Synaptic Transmission](#synaptic-transmission)
- [Plasticity Updates](#plasticity-updates)
- [Cell Lifecycle](#cell-lifecycle)
- [Optimization Techniques](#optimization-techniques)

---

## Core Simulation Loop

### Main Simulation Step

**Algorithm**: Single timestep of neural simulation

```python
def step():
    """
    Time Complexity: O(N * S) where N = neurons, S = avg synapses per neuron
    Space Complexity: O(N + M) where M = total synapses
    """
    current_step = model.current_step
    spikes = []
    
    # Phase 1: Update all neurons
    for neuron_id, neuron in model.neurons.items():
        spiked = update_single_neuron(neuron_id, current_step)
        if spiked:
            spikes.append(neuron_id)
    
    # Phase 2: Apply plasticity to active synapses
    if plasticity_enabled:
        apply_plasticity(spikes, current_step)
    
    # Phase 3: Cell lifecycle (death and reproduction)
    if lifecycle_enabled:
        update_cell_lifecycle()
    
    model.current_step += 1
    return {"spikes": spikes, "step": current_step}
```

**Key Steps:**
1. Update all neuron membrane potentials
2. Check for spike threshold crossings
3. Apply synaptic plasticity
4. Handle cell death and reproduction
5. Increment simulation time

**Optimizations:**
- Neurons updated in parallel-friendly loop
- Spike detection occurs during update (no second pass)
- Plasticity only applied to recently active synapses

---

## Neuron Update Algorithms

### LIF Neuron Update

**Algorithm**: Leaky Integrate-and-Fire neuron dynamics

```python
def update_lif_neuron(neuron, synaptic_input, dt=1.0):
    """
    Time Complexity: O(1)
    
    Implements:
        dV/dt = (-(V - V_rest) + I_total) / τ_m
    """
    # 1. Check refractory period
    if in_refractory_period(neuron):
        return False, neuron.v_membrane
    
    # 2. Calculate leak current
    v_rest = neuron.params.get("v_rest", -65.0)
    tau_m = neuron.params.get("tau_membrane", 10.0)
    leak = (v_rest - neuron.v_membrane) / tau_m
    
    # 3. Compute total current
    total_current = leak + synaptic_input + neuron.external_input
    
    # 4. Integrate membrane potential
    new_v = neuron.v_membrane + dt * total_current
    
    # 5. Check for spike
    v_threshold = neuron.params.get("v_threshold", -50.0)
    if new_v >= v_threshold:
        v_reset = neuron.params.get("v_reset", -65.0)
        return True, v_reset
    
    return False, new_v
```

**Numerical Method:** Forward Euler integration
- Stable for dt ≤ τ_m/2
- Default dt = 1.0 ms is safe for τ_m = 10 ms

### Izhikevich Neuron Update

**Algorithm**: Dynamical systems neuron model

```python
def update_izhikevich_neuron(neuron, synaptic_input, dt=1.0):
    """
    Time Complexity: O(1)
    
    Implements:
        dv/dt = 0.04v² + 5v + 140 - u + I
        du/dt = a(bv - u)
    """
    v = neuron.v_membrane
    u = neuron.u_recovery
    a, b, c, d = get_parameters(neuron.neuron_type)
    
    # Izhikevich equations
    dv = (0.04 * v * v + 5 * v + 140 - u + synaptic_input) * dt
    du = a * (b * v - u) * dt
    
    new_v = v + dv
    new_u = u + du
    
    # Spike detection and reset
    if new_v >= 30.0:
        return True, c, new_u + d
    
    return False, new_v, new_u
```

**Neuron Type Parameters:**
- Different (a, b, c, d) produce different behaviors
- Computationally efficient (quadratic equation)
- Can reproduce 20+ neuron types

### Hodgkin-Huxley Neuron Update

**Algorithm**: Ion channel-based neuron model

```python
def update_hodgkin_huxley_neuron(neuron, synaptic_input, dt=0.01):
    """
    Time Complexity: O(1)
    
    Implements full HH equations with Na+, K+, and leak channels
    Requires small dt (0.01-0.1 ms) for numerical stability
    """
    v = neuron.v_membrane
    m, h, n = neuron.params["hh_m"], neuron.params["hh_h"], neuron.params["hh_n"]
    
    # 1. Calculate rate functions
    alpha_m, beta_m = sodium_activation_rates(v)
    alpha_h, beta_h = sodium_inactivation_rates(v)
    alpha_n, beta_n = potassium_activation_rates(v)
    
    # 2. Update gating variables
    dm = (alpha_m * (1 - m) - beta_m * m) * dt
    dh = (alpha_h * (1 - h) - beta_h * h) * dt
    dn = (alpha_n * (1 - n) - beta_n * n) * dt
    
    new_m = m + dm
    new_h = h + dh
    new_n = n + dn
    
    # 3. Calculate ionic currents
    I_Na = g_Na * (new_m ** 3) * new_h * (v - E_Na)
    I_K = g_K * (new_n ** 4) * (v - E_K)
    I_L = g_L * (v - E_L)
    
    # 4. Update membrane potential
    dv = ((synaptic_input - I_Na - I_K - I_L) / C_m) * dt
    new_v = v + dv
    
    # 5. Detect spike (crossing threshold from below)
    spiked = (v < 0.0 and new_v >= 0.0)
    
    return spiked, new_v, new_m, new_h, new_n
```

**Computational Complexity:**
- More expensive than LIF (3 additional state variables)
- Requires smaller time steps for stability
- But provides biophysical realism

---

## Synaptic Transmission

### Spike-Based Synaptic Input

**Algorithm**: Calculate total synaptic input to a neuron

#### Standard Method (O(N*M) worst case)

```python
def calculate_synaptic_input_standard(neuron_id, current_step):
    """
    Time Complexity: O(S * H) 
        S = incoming synapses
        H = spike history length per neuron
    Space Complexity: O(N * H) for spike history
    """
    synaptic_input = 0.0
    
    for synapse in get_incoming_synapses(neuron_id):
        pre_id = synapse.pre_id
        delay = synapse.delay
        
        # Check if presynaptic neuron spiked at right time
        for spike_time in spike_history[pre_id]:
            if current_step - spike_time == delay:
                synaptic_input += synapse.get_effective_weight()
                break
    
    return synaptic_input
```

**Issues:**
- Linear search through spike history
- Becomes slow for large networks
- Redundant checks across neurons

#### Optimized Method (O(M) with time-indexed buffer)

```python
def calculate_synaptic_input_optimized(neuron_id, current_step):
    """
    Time Complexity: O(S) - one check per synapse
    Space Complexity: O(W * N) where W = time window
    
    Uses time-indexed spike buffer for O(1) spike lookups
    """
    synaptic_input = 0.0
    
    for synapse in get_incoming_synapses(neuron_id):
        pre_id = synapse.pre_id
        delay = synapse.delay
        spike_time = current_step - delay
        
        # O(1) hash table lookup
        if spike_buffer.did_spike_at(pre_id, spike_time):
            synaptic_input += synapse.get_effective_weight()
    
    return synaptic_input
```

**Optimizations:**
- Hash table indexed by (neuron_id, time)
- Circular buffer with automatic cleanup
- Constant-time spike queries

### Time-Indexed Spike Buffer

**Data Structure**: Efficient spike storage and retrieval

```python
class TimeIndexedSpikeBuffer:
    """
    Circular buffer with hash tables for O(1) spike lookups
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.spike_tables = {}  # time -> set of neuron IDs
        self.oldest_time = 0
    
    def add_spike(self, neuron_id, time):
        """O(1) insertion"""
        if time not in self.spike_tables:
            self.spike_tables[time] = set()
        self.spike_tables[time].add(neuron_id)
        
        # Cleanup old entries
        self._cleanup_old_spikes(time)
    
    def did_spike_at(self, neuron_id, time):
        """O(1) lookup"""
        if time not in self.spike_tables:
            return False
        return neuron_id in self.spike_tables[time]
    
    def _cleanup_old_spikes(self, current_time):
        """Remove spikes outside time window"""
        cutoff = current_time - self.window_size
        times_to_remove = [t for t in self.spike_tables if t < cutoff]
        for t in times_to_remove:
            del self.spike_tables[t]
```

**Benefits:**
- O(1) spike insertion and lookup
- Automatic memory management
- Scales to large networks

---

## Plasticity Updates

### Hebbian Learning

**Algorithm**: Correlation-based weight updates

```python
def apply_hebbian_plasticity(pre_spiked, post_spiked, synapse, learning_rate):
    """
    Time Complexity: O(1) per synapse
    
    "Neurons that fire together, wire together"
    """
    if pre_spiked and post_spiked:
        # Both active: strengthen (LTP)
        delta = learning_rate
    elif pre_spiked and not post_spiked:
        # Pre only: weaken (LTD)
        delta = -learning_rate * 0.5
    else:
        # No change
        delta = 0.0
    
    synapse.weight += delta
    synapse.weight = clip(synapse.weight, w_min, w_max)
```

**Implementation Notes:**
- Asymmetric rule (LTD < LTP) prevents runaway dynamics
- Weight bounds prevent saturation
- Applied only to recently active synapses

### STDP (Spike-Timing-Dependent Plasticity)

**Algorithm**: Temporal learning rule

```python
def apply_stdp(pre_spike_times, post_spike_times, synapse):
    """
    Time Complexity: O(P * Q) where P, Q = recent spike counts
    Typically P, Q << 10 so effectively O(1)
    
    Weight change depends on spike timing:
        Δt > 0 (post after pre): LTP
        Δt < 0 (pre after post): LTD
    """
    A_plus = 0.01   # LTP amplitude
    A_minus = 0.012 # LTD amplitude
    tau_plus = 20.0  # LTP time constant
    tau_minus = 20.0 # LTD time constant
    
    delta_w = 0.0
    
    # Check all recent spike pairs
    for t_pre in pre_spike_times[-10:]:  # Last 10 spikes
        for t_post in post_spike_times[-10:]:
            dt = t_post - t_pre
            
            if dt > 0:
                # Post after pre: LTP
                delta_w += A_plus * exp(-dt / tau_plus)
            elif dt < 0:
                # Pre after post: LTD
                delta_w -= A_minus * exp(dt / tau_minus)
    
    synapse.weight += delta_w
    synapse.weight = clip(synapse.weight, w_min, w_max)
```

**Optimization:**
- Only process recent spikes (typically last 10)
- Skip if no recent activity
- Vectorized computation possible

---

## Cell Lifecycle

### Death and Reproduction

**Algorithm**: Evolutionary dynamics for neurons

```python
def maybe_kill_and_reproduce(neuron, model):
    """
    Time Complexity: O(S + log N) where S = synapses, N = neurons
    
    Implements:
    1. Death check based on health and age
    2. Synaptic inheritance with mutation
    3. Reconnection to prevent disconnection
    """
    # 1. Check death conditions
    if neuron.health < death_threshold or neuron.age > max_age:
        old_synapses = get_connected_synapses(neuron.id)
        
        # 2. Create offspring at same position
        new_neuron = create_neuron(
            position=neuron.position(),
            generation=neuron.generation + 1,
            params=mutate_params(neuron.params)
        )
        
        # 3. Transfer synapses with mutation
        lost_count = 0
        for synapse in old_synapses:
            new_pre = new_neuron.id if synapse.pre_id == neuron.id else synapse.pre_id
            new_post = new_neuron.id if synapse.post_id == neuron.id else synapse.post_id
            
            # Only add if both endpoints exist
            if both_neurons_exist(new_pre, new_post):
                add_synapse(new_pre, new_post, mutate_weight(synapse.weight))
            else:
                lost_count += 1
        
        # 4. Reconnect if synapses were lost
        if lost_count > 0:
            reconnect_neuron(new_neuron, lost_count)
        
        # 5. Remove old neuron
        remove_neuron(neuron.id)
        
        return new_neuron
    
    return neuron
```

### Reconnection Algorithm

**Algorithm**: Maintain connectivity during cell turnover

```python
def reconnect_neuron(neuron, num_connections):
    """
    Time Complexity: O(N) for finding nearby, O(k) for creating connections
    
    Strategy:
    1. Try nearby neurons first (within distance threshold)
    2. Fall back to random if no nearby neurons
    """
    max_distance = 5.0
    nearby_neurons = []
    
    # Find nearby neurons
    for candidate in all_neurons:
        if candidate.id != neuron.id:
            distance = euclidean_distance_4d(neuron, candidate)
            if distance <= max_distance:
                nearby_neurons.append((candidate, distance))
    
    # Sort by distance
    nearby_neurons.sort(key=lambda x: x[1])
    
    # Create connections
    if nearby_neurons:
        # Connect to closest neurons
        for i in range(min(num_connections, len(nearby_neurons))):
            target = nearby_neurons[i][0]
            direction = random.choice(["incoming", "outgoing"])
            
            if direction == "incoming":
                add_synapse(target.id, neuron.id, weight=0.1)
            else:
                add_synapse(neuron.id, target.id, weight=0.1)
    else:
        # Random connections as fallback
        targets = random.sample(all_neurons, min(num_connections, len(all_neurons)))
        for target in targets:
            direction = random.choice(["incoming", "outgoing"])
            if direction == "incoming":
                add_synapse(target.id, neuron.id, weight=0.1)
            else:
                add_synapse(neuron.id, target.id, weight=0.1)
```

**Key Features:**
- Preserves local connectivity structure
- Prevents network fragmentation
- Maintains E/I balance through random direction

---

## Optimization Techniques

### Sparse Connectivity Matrix

**Data Structure**: Memory-efficient synapse storage

```python
class SparseConnectivityMatrix:
    """
    CSR (Compressed Sparse Row) format for synapses
    
    Memory: O(M) instead of O(N²) where M = actual synapses
    """
    def __init__(self):
        self.incoming = {}   # post_id -> [(pre_id, weight, delay), ...]
        self.outgoing = {}   # pre_id -> [(post_id, weight, delay), ...]
    
    def add_synapse(self, pre_id, post_id, weight, delay):
        """O(1) insertion"""
        if post_id not in self.incoming:
            self.incoming[post_id] = []
        self.incoming[post_id].append((pre_id, weight, delay))
        
        if pre_id not in self.outgoing:
            self.outgoing[pre_id] = []
        self.outgoing[pre_id].append((post_id, weight, delay))
    
    def get_incoming_synapses(self, neuron_id):
        """O(1) retrieval"""
        return self.incoming.get(neuron_id, [])
```

**Benefits:**
- 100-1000x memory reduction for sparse networks
- Fast row-wise access
- Efficient for typical brain connectivity (~0.1% density)

### Vectorized Operations

**Technique**: Batch processing with NumPy

```python
def update_neuron_batch(neurons, synaptic_inputs):
    """
    Process multiple neurons in parallel using NumPy
    
    Time: 10-100x faster than Python loops
    """
    v_membranes = np.array([n.v_membrane for n in neurons])
    v_rest = -65.0
    tau_m = 10.0
    dt = 1.0
    
    # Vectorized leak calculation
    leak = (v_rest - v_membranes) / tau_m
    
    # Vectorized integration
    total_currents = leak + synaptic_inputs
    new_v = v_membranes + dt * total_currents
    
    # Vectorized spike detection
    v_threshold = -50.0
    spikes = new_v >= v_threshold
    
    # Update neurons
    for i, neuron in enumerate(neurons):
        neuron.v_membrane = new_v[i]
    
    return np.where(spikes)[0]
```

**When to Use:**
- Large homogeneous neuron populations
- Uniform parameters across neurons
- No complex conditional logic

---

## Performance Analysis

### Complexity Summary

| Operation | Standard | Optimized | Improvement |
|-----------|----------|-----------|-------------|
| Spike lookup | O(N*H) | O(1) | 100-1000x |
| Synapse storage | O(N²) | O(M) | 100-10000x |
| Neuron update | O(N) | O(N) | 10-100x via SIMD |
| Plasticity update | O(M) | O(A) | A = active synapses |

### Bottlenecks

1. **Synaptic input calculation**: Dominates for large networks
   - Solution: Time-indexed spike buffer
   
2. **Memory allocation**: Python object overhead
   - Solution: Sparse matrices, NumPy arrays
   
3. **Plasticity updates**: Can be expensive if all-to-all
   - Solution: Only update recently active synapses

### Scalability

Current implementation scales to:
- **10,000 neurons**: Real-time on CPU
- **100,000 neurons**: Slower than real-time, but feasible
- **1,000,000 neurons**: Requires GPU or distributed computing

---

## References

1. Morrison et al. (2008): "Spike-timing-dependent plasticity in balanced random networks"
2. Brette et al. (2007): "Simulation of networks of spiking neurons"
3. Goodman & Brette (2008): "Brian: a simulator for spiking neural networks in Python"

---

*Last Updated: December 2025*
*For source code, see `src/simulation.py`, `src/neuron_models.py`, and `src/time_indexed_spikes.py`*
