# API Documentation

Complete API reference for the 4D Neural Cognition system.

## Table of Contents

1. [Brain Model API](#brain-model-api)
2. [Simulation API](#simulation-api)
3. [Senses API](#senses-api)
4. [Storage API](#storage-api)
5. [Plasticity API](#plasticity-api)
6. [Cell Lifecycle API](#cell-lifecycle-api)
7. [Web API](#web-api)

---

## Brain Model API

### BrainModel

Main class for managing the neural network structure.

#### `__init__(config_path=None, config=None)`

Initialize a brain model from configuration.

**Parameters**:
- `config_path` (str, optional): Path to JSON configuration file
- `config` (dict, optional): Configuration dictionary

**Returns**: `BrainModel` instance

**Raises**:
- `ValueError`: If neither `config_path` nor `config` provided

**Example**:
```python
from src.brain_model import BrainModel

# From file
model = BrainModel(config_path='brain_base_model.json')

# From dict
config = {
    "lattice_shape": [20, 20, 20, 20],
    "neuron_model": {...},
    ...
}
model = BrainModel(config=config)
```

#### `add_neuron(x, y, z, w, **kwargs)`

Add a neuron at specified 4D coordinates.

**Parameters**:
- `x`, `y`, `z`, `w` (int): 4D coordinates
- `**kwargs`: Optional neuron parameters (generation, parent_id, health, etc.)

**Returns**: `Neuron` instance

**Raises**:
- `ValueError`: If coordinates out of bounds

**Example**:
```python
neuron = model.add_neuron(10, 10, 10, 0)
print(f"Created neuron {neuron.id} at {neuron.position()}")
```

#### `add_synapse(pre_id, post_id, weight=0.1, delay=1)`

Add a synaptic connection between neurons.

**Parameters**:
- `pre_id` (int): Pre-synaptic neuron ID
- `post_id` (int): Post-synaptic neuron ID
- `weight` (float, default=0.1): Synaptic weight
- `delay` (int, default=1): Transmission delay in steps

**Returns**: `Synapse` instance

**Raises**:
- `KeyError`: If neuron IDs don't exist

**Example**:
```python
synapse = model.add_synapse(0, 5, weight=0.5, delay=2)
print(f"Connected neuron {synapse.pre_id} → {synapse.post_id}")
```

#### `get_neuron(neuron_id)`

Get neuron by ID.

**Parameters**:
- `neuron_id` (int): Neuron ID

**Returns**: `Neuron` instance or `None`

**Example**:
```python
neuron = model.get_neuron(42)
if neuron:
    print(f"Membrane potential: {neuron.v_membrane}")
```

#### `remove_neuron(neuron_id)`

Remove a neuron and all its synapses.

**Parameters**:
- `neuron_id` (int): Neuron ID to remove

**Returns**: None

**Example**:
```python
model.remove_neuron(42)
```

#### `get_neuron_model_params()`

Get default neuron model parameters.

**Returns**: dict with parameters (tau_m, v_rest, v_reset, v_threshold, etc.)

**Example**:
```python
params = model.get_neuron_model_params()
print(f"Threshold: {params['v_threshold']}")
```

#### `get_senses()`

Get sensory configuration.

**Returns**: dict mapping sense names to configurations

**Example**:
```python
senses = model.get_senses()
for sense_name, config in senses.items():
    print(f"{sense_name}: area={config['areal']}")
```

#### `get_areas()`

Get brain area definitions.

**Returns**: list of area dictionaries

**Example**:
```python
areas = model.get_areas()
for area in areas:
    print(f"{area['name']}: {area['coord_ranges']}")
```

---

## Simulation API

### Simulation

Main simulation orchestrator.

#### `__init__(model, seed=None)`

Initialize simulation.

**Parameters**:
- `model` (BrainModel): Brain model to simulate
- `seed` (int, optional): Random seed for reproducibility

**Returns**: `Simulation` instance

**Example**:
```python
from src.simulation import Simulation

sim = Simulation(model, seed=42)
```

#### `initialize_neurons(area_names=None, density=1.0)`

Create neurons in specified brain areas.

**Parameters**:
- `area_names` (list[str], optional): Area names to initialize. If None, all areas.
- `density` (float, default=1.0): Fraction of positions to fill (0-1)

**Returns**: None

**Raises**:
- `ValueError`: If density not in [0, 1]

**Example**:
```python
# Initialize all areas at 50% density
sim.initialize_neurons(density=0.5)

# Initialize specific areas
sim.initialize_neurons(["V1_like", "A1_like"], density=0.3)
```

#### `initialize_random_synapses(connection_probability=0.01, weight_mean=0.1, weight_std=0.05)`

Create random synaptic connections.

**Parameters**:
- `connection_probability` (float, default=0.01): Probability of connection
- `weight_mean` (float, default=0.1): Mean initial weight
- `weight_std` (float, default=0.05): Std dev of weights

**Returns**: None

**Example**:
```python
sim.initialize_random_synapses(
    connection_probability=0.05,
    weight_mean=0.2,
    weight_std=0.1
)
```

#### `step()`

Execute one simulation step.

**Returns**: dict with statistics:
- `spikes`: List of neuron IDs that spiked
- `active_neurons`: Count of neurons with v > v_rest
- `total_neurons`: Total neuron count
- `total_synapses`: Total synapse count
- `avg_membrane_potential`: Average v_membrane
- `deaths`: Neurons that died this step
- `births`: Neurons created this step

**Example**:
```python
stats = sim.step()
print(f"Spikes: {len(stats['spikes'])}")
print(f"Active: {stats['active_neurons']}/{stats['total_neurons']}")
```

#### `add_callback(callback)`

Add callback function called each step.

**Parameters**:
- `callback` (Callable): Function taking (simulation, step) as arguments

**Returns**: None

**Example**:
```python
def log_spikes(sim, step):
    if step % 100 == 0:
        print(f"Step {step}: {len(sim.spike_history)} total spikes")

sim.add_callback(log_spikes)
```

---

## Senses API

### `feed_sense_input(model, sense_name, input_data)`

Feed sensory input to corresponding brain area.

**Parameters**:
- `model` (BrainModel): Brain model
- `sense_name` (str): Name of sense ('vision', 'audition', etc.)
- `input_data` (np.ndarray): Input data matching configured input_size

**Returns**: int (number of neurons activated)

**Raises**:
- `ValueError`: If sense_name not found or input shape mismatch

**Example**:
```python
from src.senses import feed_sense_input
import numpy as np

# Vision input (20x20 image)
vision_input = np.random.rand(20, 20) * 10
feed_sense_input(model, 'vision', vision_input)

# Audition input (20x20 spectrogram)
audio_input = np.random.rand(20, 20) * 5
feed_sense_input(model, 'audition', audio_input)
```

### `create_digital_sense_input(text)`

Convert text to neural input pattern.

**Parameters**:
- `text` (str): Text to encode

**Returns**: np.ndarray of shape (20, 20)

**Example**:
```python
from src.senses import create_digital_sense_input, feed_sense_input

digital_input = create_digital_sense_input("Hello, World!")
feed_sense_input(model, 'digital', digital_input)
```

---

## Storage API

### `save_to_json(model, filepath)`

Save model to JSON format (human-readable).

**Parameters**:
- `model` (BrainModel): Model to save
- `filepath` (str): Output file path

**Returns**: None

**Example**:
```python
from src.storage import save_to_json

save_to_json(model, 'my_model.json')
```

### `load_from_json(filepath)`

Load model from JSON file.

**Parameters**:
- `filepath` (str): Input file path

**Returns**: `BrainModel` instance

**Example**:
```python
from src.storage import load_from_json

model = load_from_json('my_model.json')
```

### `save_to_hdf5(model, filepath)`

Save model to HDF5 format (efficient, compressed).

**Parameters**:
- `model` (BrainModel): Model to save
- `filepath` (str): Output file path

**Returns**: None

**Example**:
```python
from src.storage import save_to_hdf5

save_to_hdf5(model, 'my_model.h5')
```

### `load_from_hdf5(filepath)`

Load model from HDF5 file.

**Parameters**:
- `filepath` (str): Input file path

**Returns**: `BrainModel` instance

**Example**:
```python
from src.storage import load_from_hdf5

model = load_from_hdf5('my_model.h5')
```

---

## Plasticity API

### `hebbian_update(model, pre_spike_ids, post_spike_ids, learning_rate, weight_bounds)`

Apply Hebbian learning rule.

**Parameters**:
- `model` (BrainModel): Brain model
- `pre_spike_ids` (list[int]): IDs of pre-synaptic neurons that spiked
- `post_spike_ids` (list[int]): IDs of post-synaptic neurons that spiked
- `learning_rate` (float): Learning rate
- `weight_bounds` (tuple): (min_weight, max_weight)

**Returns**: int (number of synapses updated)

**Example**:
```python
from src.plasticity import hebbian_update

# After detecting spikes
hebbian_update(
    model,
    pre_spike_ids=[0, 1, 2],
    post_spike_ids=[5, 6],
    learning_rate=0.001,
    weight_bounds=(-1.0, 1.0)
)
```

### `apply_weight_decay(model, decay_rate)`

Apply decay to all synaptic weights.

**Parameters**:
- `model` (BrainModel): Brain model
- `decay_rate` (float): Decay rate (multiplicative)

**Returns**: None

**Example**:
```python
from src.plasticity import apply_weight_decay

# Decay weights by 0.01%
apply_weight_decay(model, 0.9999)
```

---

## Cell Lifecycle API

### `update_health_and_age(model, lifecycle_config)`

Age neurons and decay health.

**Parameters**:
- `model` (BrainModel): Brain model
- `lifecycle_config` (dict): Lifecycle configuration

**Returns**: None

**Example**:
```python
from src.cell_lifecycle import update_health_and_age

lifecycle_config = model.get_lifecycle_config()
update_health_and_age(model, lifecycle_config)
```

### `maybe_kill_and_reproduce(model, lifecycle_config, spike_counts, rng)`

Kill unhealthy neurons and reproduce active ones.

**Parameters**:
- `model` (BrainModel): Brain model
- `lifecycle_config` (dict): Lifecycle configuration
- `spike_counts` (dict): Spike counts per neuron
- `rng` (np.random.Generator): Random number generator

**Returns**: tuple (num_deaths, num_births)

**Example**:
```python
from src.cell_lifecycle import maybe_kill_and_reproduce
import numpy as np

rng = np.random.default_rng(42)
lifecycle_config = model.get_lifecycle_config()
spike_counts = {0: 5, 1: 3, 2: 0}  # From simulation

deaths, births = maybe_kill_and_reproduce(
    model, lifecycle_config, spike_counts, rng
)
print(f"Deaths: {deaths}, Births: {births}")
```

---

## Web API

### REST Endpoints

#### `POST /api/initialize`

Initialize a new brain model.

**Request Body**:
```json
{
  "lattice_shape": [20, 20, 20, 20],
  "area_names": ["V1_like", "A1_like"],
  "neuron_density": 0.1,
  "connection_probability": 0.01
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model initialized",
  "neurons": 800,
  "synapses": 640
}
```

#### `POST /api/step`

Execute one simulation step.

**Request Body**:
```json
{
  "steps": 1
}
```

**Response**:
```json
{
  "status": "success",
  "stats": {
    "spikes": [0, 5, 12],
    "active_neurons": 150,
    "total_neurons": 800
  }
}
```

#### `POST /api/feed_input`

Feed sensory input.

**Request Body**:
```json
{
  "sense": "vision",
  "data": [[...], [...], ...]  // 2D array
}
```

**Response**:
```json
{
  "status": "success",
  "neurons_activated": 42
}
```

#### `POST /api/save`

Save model to file.

**Request Body**:
```json
{
  "filepath": "my_model.h5",
  "format": "hdf5"  // or "json"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model saved to my_model.h5"
}
```

#### `POST /api/load`

Load model from file.

**Request Body**:
```json
{
  "filepath": "my_model.h5",
  "format": "hdf5"
}
```

**Response**:
```json
{
  "status": "success",
  "message": "Model loaded from my_model.h5",
  "neurons": 800,
  "synapses": 640
}
```

### WebSocket Events

#### Server → Client

##### `log_message`

Server sends log messages.

**Payload**:
```json
{
  "level": "INFO",  // INFO, WARNING, ERROR, SUCCESS
  "message": "Model initialized successfully",
  "timestamp": "2025-12-06T10:30:00"
}
```

##### `training_update`

Training progress updates.

**Payload**:
```json
{
  "step": 500,
  "total_steps": 1000,
  "spikes": [0, 5, 12],
  "active_neurons": 150
}
```

##### `heatmap_data`

Heatmap visualization data.

**Payload**:
```json
{
  "input_layer": [[...], [...], ...],
  "hidden_layer": [[...], [...], ...],
  "output_layer": [[...], [...], ...]
}
```

---

## Data Types

### Neuron

```python
@dataclass
class Neuron:
    id: int                    # Unique identifier
    x: int                     # X coordinate
    y: int                     # Y coordinate
    z: int                     # Z coordinate
    w: int                     # W coordinate (4th dimension)
    generation: int            # Generational tracking
    parent_id: int             # Parent neuron ID (-1 if none)
    health: float              # Health (0-1)
    age: int                   # Age in steps
    v_membrane: float          # Membrane potential (mV)
    external_input: float      # External input current
    last_spike_time: int       # Last spike time (for refractory)
    params: dict               # LIF parameters
```

### Synapse

```python
@dataclass
class Synapse:
    pre_id: int                # Pre-synaptic neuron ID
    post_id: int               # Post-synaptic neuron ID
    weight: float              # Synaptic weight
    delay: int                 # Transmission delay (steps)
    plasticity_tag: float      # For learning rules
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Complete example using the API."""

import numpy as np
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.senses import feed_sense_input, create_digital_sense_input
from src.storage import save_to_hdf5, load_from_hdf5

# 1. Load configuration
model = BrainModel(config_path='brain_base_model.json')
print(f"Lattice: {model.lattice_shape}")

# 2. Initialize simulation
sim = Simulation(model, seed=42)

# 3. Create neurons
sim.initialize_neurons(
    area_names=['V1_like', 'Digital_sensor'],
    density=0.1
)
print(f"Neurons: {len(model.neurons)}")

# 4. Create synapses
sim.initialize_random_synapses(
    connection_probability=0.01,
    weight_mean=0.1,
    weight_std=0.05
)
print(f"Synapses: {len(model.synapses)}")

# 5. Add monitoring callback
def monitor(sim, step):
    if step % 100 == 0:
        stats = sim.step()
        print(f"Step {step}: {len(stats['spikes'])} spikes")

sim.add_callback(monitor)

# 6. Run simulation with input
for step in range(1000):
    # Feed input every 10 steps
    if step % 10 == 0:
        # Vision input
        vision_input = np.random.rand(20, 20) * 10
        feed_sense_input(model, 'vision', vision_input)
        
        # Digital input
        digital_input = create_digital_sense_input("Hello!")
        feed_sense_input(model, 'digital', digital_input)
    
    # Simulation step
    stats = sim.step()
    
    # Check for interesting events
    if len(stats['spikes']) > 100:
        print(f"High activity at step {step}!")

# 7. Save model
save_to_hdf5(model, 'trained_model.h5')
print("Model saved!")

# 8. Load and continue
loaded_model = load_from_hdf5('trained_model.h5')
sim2 = Simulation(loaded_model, seed=123)
# Continue simulation...
```

---

*Last Updated: December 2025*  
*API Version: 1.0*
