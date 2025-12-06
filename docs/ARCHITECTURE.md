# Architecture Documentation

## Overview

The 4D Neural Cognition system is designed as a modular, extensible platform for simulating brain-like neural networks in four-dimensional space. This document describes the architectural design, key components, and design decisions.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Core Components](#core-components)
3. [Data Flow](#data-flow)
4. [Design Patterns](#design-patterns)
5. [Storage Architecture](#storage-architecture)
6. [Web Interface Architecture](#web-interface-architecture)
7. [Extension Points](#extension-points)

## High-Level Architecture

### System Layers

```
┌─────────────────────────────────────────────────────────┐
│                 Presentation Layer                       │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Web UI (Flask + JavaScript + Socket.IO)          │ │
│  │  - Visualization  - Controls  - Monitoring         │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/WebSocket
┌────────────────────▼────────────────────────────────────┐
│                Application Layer                         │
│  ┌────────────────────────────────────────────────────┐ │
│  │  Simulation Engine                                 │ │
│  │  - Orchestration  - State Management  - Callbacks │ │
│  └────────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│                 Domain Layer                             │
│  ┌──────────────┬──────────────┬─────────────────────┐ │
│  │ Brain Model  │  Plasticity  │  Cell Lifecycle     │ │
│  │ - Neurons    │  - Hebbian   │  - Aging/Death      │ │
│  │ - Synapses   │  - Updates   │  - Reproduction     │ │
│  └──────────────┴──────────────┴─────────────────────┘ │
│  ┌──────────────┬──────────────────────────────────┐   │
│  │ Senses       │  Storage                         │   │
│  │ - Input Map  │  - JSON/HDF5  - Serialization   │   │
│  └──────────────┴──────────────────────────────────┘   │
└──────────────────────────────────────────────────────────┘
```

### Component Interaction

```
┌──────────┐         ┌──────────────┐         ┌─────────┐
│   User   │◄───────►│  Web Server  │◄───────►│  Flask  │
│ (Browser)│         │  (Frontend)  │         │  Routes │
└──────────┘         └──────────────┘         └────┬────┘
                                                    │
                                             ┌──────▼─────┐
                                             │ Simulation │
                                             │   Engine   │
                                             └──────┬─────┘
                                                    │
                     ┌──────────────┬───────────────┼────────────┬──────────┐
                     │              │               │            │          │
                ┌────▼────┐   ┌─────▼─────┐  ┌─────▼─────┐  ┌──▼───┐  ┌──▼────┐
                │  Brain  │   │   Cell    │  │Plasticity │  │Senses│  │Storage│
                │  Model  │   │ Lifecycle │  │           │  │      │  │       │
                └─────────┘   └───────────┘  └───────────┘  └──────┘  └───────┘
```

## Core Components

### 1. Brain Model (`brain_model.py`)

**Purpose**: Core data structures for neurons and synapses.

**Key Classes**:

```python
@dataclass
class Neuron:
    """Represents a single neuron in 4D space."""
    id: int
    x, y, z, w: int          # 4D coordinates
    generation: int          # Generational tracking
    parent_id: int           # Inheritance tracking
    health: float            # Cell lifecycle
    age: int                 # Age in steps
    v_membrane: float        # Membrane potential
    external_input: float    # Input current
    last_spike_time: int     # For refractory period
    params: dict             # LIF parameters

@dataclass
class Synapse:
    """Synaptic connection between neurons."""
    pre_id: int              # Pre-synaptic neuron
    post_id: int             # Post-synaptic neuron
    weight: float            # Synaptic strength
    delay: int               # Transmission delay
    plasticity_tag: float    # For learning rules

class BrainModel:
    """Main container for neural network."""
    def __init__(self, config_path: str)
    def add_neuron(self, x, y, z, w) -> Neuron
    def add_synapse(self, pre_id, post_id, weight) -> Synapse
    def get_neuron(self, neuron_id) -> Neuron
    def remove_neuron(self, neuron_id)
```

**Design Decisions**:
- **Dataclasses** for clean, type-safe data structures
- **Dictionary storage** for neurons (O(1) lookup by ID)
- **List storage** for synapses (allows iteration)
- **4D coordinates** stored as separate integers (not tuple) for efficiency
- **Config-driven** initialization for flexibility

### 2. Simulation Engine (`simulation.py`)

**Purpose**: Orchestrates the simulation loop and neuron updates.

**Key Class**:

```python
class Simulation:
    """Main simulation orchestrator."""
    def __init__(self, model: BrainModel, seed: int)
    def initialize_neurons(self, area_names, density)
    def initialize_random_synapses(self, probability, weight_mean)
    def step() -> dict  # Main simulation step
    def add_callback(self, callback: Callable)
```

**Simulation Loop** (`step()` method):

```
1. Process external inputs
   └─ Feed sensory input to neurons

2. Update neuron states (LIF dynamics)
   ├─ Calculate membrane potential
   ├─ Check for spike threshold
   ├─ Apply refractory period
   └─ Reset if spiked

3. Propagate spikes through synapses
   ├─ Find post-synaptic neurons
   ├─ Apply synaptic delays
   └─ Add synaptic currents

4. Apply plasticity
   ├─ Update synaptic weights (Hebbian)
   └─ Apply weight decay

5. Cell lifecycle
   ├─ Age neurons
   ├─ Update health
   ├─ Kill unhealthy neurons
   └─ Reproduce successful neurons

6. Collect statistics
   └─ Spike counts, active neurons, etc.

7. Execute callbacks
   └─ User-defined monitoring functions
```

**Design Decisions**:
- **Single-threaded** execution (Python GIL limitation)
- **Discrete time** steps (not continuous)
- **Callback system** for extensibility
- **Statistics collection** for monitoring
- **Random seed** support for reproducibility

### 3. Cell Lifecycle (`cell_lifecycle.py`)

**Purpose**: Manages neuron aging, death, and reproduction.

**Key Functions**:

```python
def update_health_and_age(model: BrainModel, lifecycle_config: dict)
    """Age neurons and decay health."""

def maybe_kill_and_reproduce(
    model: BrainModel,
    lifecycle_config: dict,
    spike_counts: dict,
    rng: np.random.Generator
)
    """Kill unhealthy neurons, reproduce active ones."""
```

**Lifecycle Algorithm**:

```
For each neuron:
    1. Increment age
    2. Decay health (health -= health_decay_per_step)
    3. If health <= 0 or age > max_age:
        - Mark for death
    4. If recently spiked and health > threshold:
        - Candidate for reproduction
    5. Remove dead neurons
    6. Create offspring with mutations:
        - Mutate LIF parameters (Gaussian noise)
        - Mutate synaptic weights (Gaussian noise)
        - Inherit position (nearby location)
```

**Design Decisions**:
- **Health-based death** (not just age)
- **Activity-based reproduction** (spiking neurons)
- **Gaussian mutations** for parameter evolution
- **Generational tracking** for analysis

### 4. Plasticity (`plasticity.py`)

**Purpose**: Implements synaptic learning rules.

**Key Functions**:

```python
def hebbian_update(
    model: BrainModel,
    pre_spikes: list,
    post_spikes: list,
    learning_rate: float,
    weight_bounds: tuple
)
    """Hebbian learning: pre and post co-activation."""

def apply_weight_decay(
    model: BrainModel,
    decay_rate: float
)
    """Prevent runaway weight growth."""
```

**Hebbian Algorithm**:

```
If pre-synaptic neuron spiked AND post-synaptic neuron spiked:
    weight += learning_rate * pre_activity * post_activity
    weight = clip(weight, weight_min, weight_max)
```

**Design Decisions**:
- **Rate-based** Hebbian (not spike-timing)
- **Weight clipping** to prevent overflow
- **Decay** to prevent runaway potentiation
- **Extensible** design for adding STDP, etc.

### 5. Senses (`senses.py`)

**Purpose**: Maps external inputs to neuron activations.

**Key Functions**:

```python
def feed_sense_input(
    model: BrainModel,
    sense_name: str,
    input_data: np.ndarray
)
    """Map sensory input to neurons in specific area."""

def create_digital_sense_input(text: str) -> np.ndarray
    """Convert text to neural input pattern."""
```

**Input Mapping**:

```
1. Look up sense configuration
2. Get target brain area and w-slice
3. Reshape input_data to match area dimensions
4. For each position in input:
    - Find neuron at (x, y, z, w)
    - Set external_input proportional to input value
```

**Design Decisions**:
- **Direct mapping** (not learned receptive fields)
- **2D inputs** mapped to 3D areas (via w-coordinate)
- **Digital sense** uses character encoding
- **Flexible input sizes** via configuration

### 6. Storage (`storage.py`)

**Purpose**: Persistence of model state.

**Key Functions**:

```python
def save_to_json(model: BrainModel, filepath: str)
    """Human-readable format."""

def load_from_json(filepath: str) -> BrainModel
    """Restore from JSON."""

def save_to_hdf5(model: BrainModel, filepath: str)
    """Efficient compressed format."""

def load_from_hdf5(filepath: str) -> BrainModel
    """Restore from HDF5."""
```

**Storage Format**:

**JSON Structure**:
```json
{
  "config": {...},
  "neurons": [
    {"id": 0, "x": 1, "y": 2, "z": 3, "w": 0, ...},
    ...
  ],
  "synapses": [
    {"pre_id": 0, "post_id": 5, "weight": 0.5, ...},
    ...
  ],
  "current_step": 1000
}
```

**HDF5 Structure**:
```
/config            (JSON string)
/neurons           (structured array)
  - id, x, y, z, w, generation, health, age, ...
/synapses          (structured array)
  - pre_id, post_id, weight, delay, plasticity_tag
/neuron_params     (group)
  /0               (dict as JSON)
  /1               (dict as JSON)
  ...
/metadata          (attributes)
  - current_step
  - timestamp
```

**Design Decisions**:
- **Dual format** support (JSON for readability, HDF5 for efficiency)
- **Compression** in HDF5 for large models
- **Structured arrays** for efficient storage
- **Config included** for self-contained files

## Data Flow

### Training Loop Data Flow

```
User Command (Start Training)
    │
    ▼
Web Interface (Socket.IO)
    │
    ▼
Flask Route Handler
    │
    ▼
Simulation.step()
    │
    ├──► Feed sensory input
    │    └──► Update neuron.external_input
    │
    ├──► Update neuron dynamics (LIF)
    │    └──► Calculate new v_membrane
    │
    ├──► Detect spikes
    │    └──► Record in spike_history
    │
    ├──► Propagate through synapses
    │    └──► Add currents to post-synaptic neurons
    │
    ├──► Apply plasticity
    │    └──► Modify synapse.weight
    │
    ├──► Cell lifecycle
    │    ├──► Age/kill neurons
    │    └──► Reproduce neurons
    │
    └──► Collect statistics
         └──► Return to caller
    │
    ▼
Flask emits statistics (Socket.IO)
    │
    ▼
Web Interface updates visualization
    │
    ▼
User sees updated heatmap
```

### Sensory Input Flow

```
External Data (Image, Sound, Text)
    │
    ▼
Preprocessing (reshape, normalize)
    │
    ▼
feed_sense_input(sense_name, input_data)
    │
    ├──► Look up sense config
    │    └──► Get target area & w-slice
    │
    ├──► Map input positions to neuron positions
    │
    └──► Set neuron.external_input
    │
    ▼
Simulation.step() processes inputs
    │
    ▼
Neurons integrate inputs into membrane potential
```

## Design Patterns

### 1. Configuration Pattern

**All configurable parameters** are centralized in JSON config:
- Enables easy experimentation
- Version control for experiments
- No recompilation needed

### 2. Callback Pattern

**Simulation allows callbacks** for monitoring:
```python
def monitor(sim, step):
    if step % 100 == 0:
        print(f"Step {step}: {len(sim.spike_history)} spikes")

sim.add_callback(monitor)
```

### 3. Dataclass Pattern

**Use dataclasses** for data structures:
- Type safety
- Clean initialization
- Automatic methods (__repr__, __eq__)

### 4. Strategy Pattern

**Plasticity rules** can be swapped:
- Current: Hebbian
- Future: STDP, BCM, etc.
- Same interface, different implementations

### 5. Factory Pattern

**Neuron creation** abstracted:
```python
model.add_neuron(x, y, z, w)  # Factory handles ID assignment
```

## Storage Architecture

### Trade-offs: JSON vs HDF5

| Feature | JSON | HDF5 |
|---------|------|------|
| Human readable | ✅ Yes | ❌ Binary |
| File size | Large | Small (compressed) |
| Load speed | Slow | Fast |
| Random access | ❌ Load all | ✅ Partial read |
| Tool support | Universal | Requires h5py |
| Use case | Small models, debugging | Large models, production |

### Storage Lifecycle

```
Create Model
    │
    ▼
Configure (JSON)
    │
    ▼
Run Simulation
    │
    ▼
Checkpoint (HDF5)
    │
    ▼
Continue/Restart
    │
    ▼
Final Save (both formats)
```

## Web Interface Architecture

### Backend (Flask)

**Routes**:
- `GET /` - Serve HTML
- `POST /api/initialize` - Create model
- `POST /api/step` - Single simulation step
- `POST /api/train` - Multi-step training
- `POST /api/save` - Save model
- `POST /api/load` - Load model
- `POST /api/feed_input` - Sensory input

**WebSocket Events** (Socket.IO):
- `connect` - Client connected
- `disconnect` - Client disconnected
- `log_message` - Server → Client logging
- `training_update` - Server → Client progress
- `heatmap_data` - Server → Client visualization

### Frontend (JavaScript)

**Components**:
- **Control Panel**: Initialize, configure, save/load
- **Heatmap Viewer**: Canvas-based visualization
- **Terminal**: Input/output for sensory data
- **Chat Interface**: Command execution
- **Logger**: Event log display

**Communication**:
```
User Action
    │
    ▼
JavaScript Event Handler
    │
    ├──► HTTP Request (for state changes)
    │    └──► Wait for response
    │
    └──► WebSocket Message (for streaming)
         └──► Update UI on event
```

## Extension Points

### Adding New Neuron Models

1. Define new neuron type in `brain_model.py`
2. Add update logic in `simulation.py:step()`
3. Add config parameters
4. Update documentation

### Adding New Plasticity Rules

1. Implement rule in `plasticity.py`
2. Add config option for rule selection
3. Call from `simulation.py:step()`
4. Add tests and documentation

### Adding New Senses

1. Add sense config in JSON
2. Define brain area for sense
3. Implement preprocessing in `senses.py`
4. Add UI controls for input

### Adding New Visualizations

1. Add route in `app.py`
2. Implement data collection
3. Add frontend component
4. Connect via Socket.IO for real-time

## Performance Considerations

### Bottlenecks

1. **Synapse updates**: O(n²) complexity
   - Solution: Sparse matrices, GPU acceleration

2. **Python overhead**: Object allocation
   - Solution: NumPy arrays, Cython, C++ backend

3. **Memory**: Each neuron ~500 bytes
   - Solution: Memory mapping, compression

### Optimization Strategies

1. **Vectorization**: Use NumPy operations
2. **Caching**: Cache frequently accessed data
3. **Lazy evaluation**: Compute only when needed
4. **Batch processing**: Update neurons in batches
5. **Pruning**: Remove inactive synapses

## Future Architecture

### Planned Improvements

1. **Microservices**: Separate simulation, storage, UI
2. **Message queue**: For distributed processing
3. **GPU backend**: CUDA kernels for neuron updates
4. **Database**: PostgreSQL for metadata, metrics
5. **API Gateway**: RESTful API with versioning

### Scalability Path

```
Current: Single-process, single-machine
    │
    ▼
Next: Multi-process, single-machine (multiprocessing)
    │
    ▼
Future: Multi-machine (MPI, Dask)
    │
    ▼
Ultimate: Cloud-native, elastic scaling
```

---

*Last Updated: December 2025*  
*Version: 1.0*
