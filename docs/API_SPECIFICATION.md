# API Specification

## Overview

This document defines the formal programmatic API for the 4D Neural Cognition framework, enabling researchers to embed the simulator in their pipelines, run large-scale parameter sweeps, and develop new plasticity rules without modifying core code.

**Author**: Thomas Heisig  
**Contact**: t_heisig@gmx.de  
**Location**: Ganderkesee, Germany  
**Last Updated**: December 2025  
**API Version**: 1.0.0

---

## Table of Contents

- [API Design Principles](#api-design-principles)
- [Python Native API](#python-native-api)
- [RESTful API](#restful-api)
- [Configuration API](#configuration-api)
- [Plugin API](#plugin-api)
- [Data Export API](#data-export-api)
- [Versioning & Compatibility](#versioning--compatibility)

---

## API Design Principles

### Core Principles

1. **Simplicity**: Common tasks should be simple, complex tasks should be possible
2. **Consistency**: Similar operations use similar patterns
3. **Discoverability**: Clear naming, comprehensive documentation, helpful errors
4. **Extensibility**: Easy to add new components via plugins
5. **Type Safety**: Type hints throughout (Python 3.8+)
6. **Backward Compatibility**: Semantic versioning (MAJOR.MINOR.PATCH)

### Design Patterns

- **Builder Pattern**: For complex object construction (network configuration)
- **Factory Pattern**: For creating neurons, synapses, plasticity rules
- **Observer Pattern**: For monitoring simulation events
- **Strategy Pattern**: For interchangeable algorithms (backends, plasticity)

---

## Python Native API

### Core Classes

#### BrainModel

Primary interface for creating and managing neural networks.

```python
from src.brain_model import BrainModel

class BrainModel:
    """
    4D neural network model.
    
    Attributes:
        neurons (List[Neuron]): Network neurons
        synapses (SparseConnectivityMatrix): Network connections
        time (int): Current simulation timestep
        config (dict): Network configuration
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        neurons: Optional[List[Neuron]] = None
    ):
        """
        Initialize brain model.
        
        Args:
            config: Configuration dictionary or path to JSON file
            neurons: Pre-initialized neuron list (optional)
        
        Example:
            >>> model = BrainModel(config="config.json")
            >>> model = BrainModel(config={"neurons": 1000, "dt": 1.0})
        """
    
    def add_neuron(
        self,
        x: float, y: float, z: float, w: float,
        neuron_type: str = "LIF",
        **kwargs
    ) -> int:
        """
        Add neuron to network.
        
        Args:
            x, y, z, w: 4D spatial coordinates
            neuron_type: "LIF", "Izhikevich", "HodgkinHuxley"
            **kwargs: Type-specific parameters
        
        Returns:
            neuron_id: Unique neuron identifier
        
        Example:
            >>> neuron_id = model.add_neuron(0.5, 0.5, 0.5, 0.0, neuron_type="LIF")
        """
    
    def connect(
        self,
        pre_id: int,
        post_id: int,
        weight: float = 1.0,
        delay: float = 1.0,
        plasticity: Optional[str] = None
    ) -> None:
        """
        Create synapse between neurons.
        
        Args:
            pre_id: Presynaptic neuron ID
            post_id: Postsynaptic neuron ID
            weight: Initial synaptic weight
            delay: Synaptic delay (ms)
            plasticity: "STDP", "Hebbian", None
        
        Example:
            >>> model.connect(0, 1, weight=0.5, plasticity="STDP")
        """
    
    def step(self, external_input: Optional[np.ndarray] = None) -> None:
        """
        Advance simulation by one timestep.
        
        Args:
            external_input: Input currents for each neuron (optional)
        
        Example:
            >>> model.step()
            >>> model.step(external_input=np.random.randn(1000))
        """
    
    def run(
        self,
        duration: float,
        input_fn: Optional[Callable] = None,
        callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Simulation time (ms)
            input_fn: Function generating input at each timestep
            callback: Function called after each timestep
        
        Returns:
            results: Dictionary containing spike times, states, metrics
        
        Example:
            >>> results = model.run(1000.0)
            >>> results = model.run(1000.0, callback=lambda: print(model.time))
        """
    
    def save(self, filepath: str, format: str = "hdf5") -> None:
        """
        Save model state.
        
        Args:
            filepath: Output file path
            format: "hdf5", "json", "pickle"
        
        Example:
            >>> model.save("model.h5", format="hdf5")
        """
    
    @classmethod
    def load(cls, filepath: str) -> "BrainModel":
        """
        Load model from file.
        
        Args:
            filepath: Path to saved model
        
        Returns:
            model: Loaded BrainModel instance
        
        Example:
            >>> model = BrainModel.load("model.h5")
        """
    
    def get_spike_trains(self, neuron_ids: Optional[List[int]] = None) -> Dict[int, np.ndarray]:
        """
        Get spike times for specified neurons.
        
        Args:
            neuron_ids: List of neuron IDs (None = all neurons)
        
        Returns:
            spike_trains: Dictionary mapping neuron_id -> spike_times
        
        Example:
            >>> spikes = model.get_spike_trains([0, 1, 2])
        """
    
    def get_states(self) -> np.ndarray:
        """
        Get current state of all neurons.
        
        Returns:
            states: Array of shape (n_neurons, n_state_vars)
        
        Example:
            >>> voltages = model.get_states()[:, 0]  # Membrane voltages
        """
    
    def reset(self) -> None:
        """
        Reset simulation to initial state.
        
        Example:
            >>> model.reset()
        """
```

---

#### NetworkBuilder

Fluent interface for constructing networks.

```python
from src.network_builder import NetworkBuilder

class NetworkBuilder:
    """Fluent API for network construction."""
    
    def __init__(self):
        """Initialize builder."""
    
    def add_layer(
        self,
        name: str,
        n_neurons: int,
        w_coordinate: float,
        neuron_type: str = "LIF",
        **kwargs
    ) -> "NetworkBuilder":
        """
        Add a layer of neurons at specific w-coordinate.
        
        Args:
            name: Layer identifier
            n_neurons: Number of neurons in layer
            w_coordinate: W-dimension value for this layer
            neuron_type: Type of neurons
            **kwargs: Additional parameters
        
        Returns:
            self: For method chaining
        
        Example:
            >>> builder.add_layer("input", 784, w_coordinate=0.0)
        """
    
    def connect_layers(
        self,
        source: str,
        target: str,
        connectivity: float = 0.1,
        weight_dist: str = "uniform",
        **kwargs
    ) -> "NetworkBuilder":
        """
        Connect two layers.
        
        Args:
            source: Source layer name
            target: Target layer name
            connectivity: Connection probability (0-1)
            weight_dist: "uniform", "normal", "constant"
            **kwargs: Distribution parameters
        
        Returns:
            self: For method chaining
        
        Example:
            >>> builder.connect_layers("input", "hidden", connectivity=0.2)
        """
    
    def add_plasticity(
        self,
        layer: str,
        rule: str,
        **params
    ) -> "NetworkBuilder":
        """
        Add plasticity rule to layer.
        
        Args:
            layer: Layer name
            rule: "STDP", "Hebbian", "Oja"
            **params: Rule-specific parameters
        
        Returns:
            self: For method chaining
        
        Example:
            >>> builder.add_plasticity("hidden", "STDP", tau_plus=20.0)
        """
    
    def build(self) -> BrainModel:
        """
        Construct BrainModel from specification.
        
        Returns:
            model: Configured BrainModel instance
        
        Example:
            >>> model = (NetworkBuilder()
            ...     .add_layer("input", 100, 0.0)
            ...     .add_layer("hidden", 50, 0.5)
            ...     .connect_layers("input", "hidden")
            ...     .build())
        """
```

---

#### Task API

Interface for defining and running tasks.

```python
from src.tasks import Task, TaskRunner

class Task:
    """Base class for cognitive tasks."""
    
    def generate_input(self, trial: int) -> np.ndarray:
        """
        Generate input for given trial.
        
        Args:
            trial: Trial number
        
        Returns:
            input_array: Input currents
        """
    
    def evaluate(self, output: np.ndarray, trial: int) -> Dict[str, float]:
        """
        Evaluate network output.
        
        Args:
            output: Network activity
            trial: Trial number
        
        Returns:
            metrics: Performance metrics
        """

class TaskRunner:
    """Runs tasks on networks."""
    
    def __init__(self, model: BrainModel, task: Task):
        """
        Initialize task runner.
        
        Args:
            model: Neural network model
            task: Task to run
        """
    
    def run_trial(self, trial: int) -> Dict[str, Any]:
        """
        Run single trial.
        
        Args:
            trial: Trial number
        
        Returns:
            results: Trial results
        """
    
    def run_experiment(
        self,
        n_trials: int,
        learning: bool = True
    ) -> pd.DataFrame:
        """
        Run multiple trials.
        
        Args:
            n_trials: Number of trials
            learning: Enable plasticity
        
        Returns:
            results_df: DataFrame with trial-by-trial results
        
        Example:
            >>> runner = TaskRunner(model, SpatialNavTask())
            >>> results = runner.run_experiment(100, learning=True)
        """
```

---

### Neuron Models API

```python
from src.neuron_models import NeuronModel, LIF, Izhikevich

class NeuronModel(ABC):
    """Abstract base class for neuron models."""
    
    @abstractmethod
    def update(self, I: float, dt: float) -> Tuple[float, bool]:
        """
        Update neuron state.
        
        Args:
            I: Input current
            dt: Timestep
        
        Returns:
            voltage: Membrane voltage
            spiked: Whether neuron spiked
        """
    
    @abstractmethod
    def reset(self) -> None:
        """Reset neuron to initial state."""

# Usage
neuron = LIF(tau_m=10.0, V_rest=-65.0, V_threshold=-50.0)
for t in range(1000):
    V, spiked = neuron.update(I=5.0, dt=1.0)
```

---

### Plasticity API

```python
from src.plasticity import PlasticityRule, STDP

class PlasticityRule(ABC):
    """Abstract base class for plasticity rules."""
    
    @abstractmethod
    def update(
        self,
        pre_spike_time: float,
        post_spike_time: float,
        weight: float,
        dt: float
    ) -> float:
        """
        Update synaptic weight.
        
        Args:
            pre_spike_time: Presynaptic spike time
            post_spike_time: Postsynaptic spike time
            weight: Current weight
            dt: Time since last update
        
        Returns:
            new_weight: Updated weight
        """

# Usage
stdp = STDP(tau_plus=20.0, tau_minus=20.0, A_plus=0.01, A_minus=0.01)
new_weight = stdp.update(pre_spike=10.0, post_spike=15.0, weight=0.5, dt=1.0)
```

---

## RESTful API

For web-based access and language-agnostic integration.

### Base URL

```
http://localhost:5000/api/v1
```

### Authentication

```http
Authorization: Bearer <api_key>
```

### Endpoints

#### Create Model

```http
POST /models
Content-Type: application/json

{
  "name": "my_model",
  "config": {
    "neurons": 1000,
    "connectivity": 0.1,
    "neuron_type": "LIF"
  }
}

Response:
{
  "model_id": "abc123",
  "status": "created",
  "neurons": 1000,
  "synapses": 100000
}
```

#### Run Simulation

```http
POST /models/{model_id}/run
Content-Type: application/json

{
  "duration": 1000.0,
  "timestep": 1.0,
  "save_spikes": true
}

Response:
{
  "job_id": "job456",
  "status": "running",
  "estimated_time": 120.0
}
```

#### Get Results

```http
GET /models/{model_id}/results/{job_id}

Response:
{
  "status": "completed",
  "duration": 1000.0,
  "spike_count": 50000,
  "results_url": "/api/v1/results/job456/download"
}
```

#### List Models

```http
GET /models

Response:
{
  "models": [
    {
      "model_id": "abc123",
      "name": "my_model",
      "created": "2025-12-14T10:00:00Z",
      "neurons": 1000
    }
  ]
}
```

#### Delete Model

```http
DELETE /models/{model_id}

Response:
{
  "status": "deleted",
  "model_id": "abc123"
}
```

### Error Handling

```json
{
  "error": {
    "code": 400,
    "message": "Invalid configuration",
    "details": "Number of neurons must be positive"
  }
}
```

### Rate Limiting

- 100 requests per minute per API key
- Headers: `X-RateLimit-Limit`, `X-RateLimit-Remaining`

---

## Configuration API

### Configuration Schema

```python
from typing import TypedDict, Literal

class NeuronConfig(TypedDict):
    count: int
    type: Literal["LIF", "Izhikevich", "HodgkinHuxley"]
    parameters: Dict[str, float]

class ConnectivityConfig(TypedDict):
    probability: float
    weight_distribution: Literal["uniform", "normal", "constant"]
    weight_params: Dict[str, float]
    delay_distribution: Literal["constant", "uniform", "normal"]
    delay_params: Dict[str, float]

class PlasticityConfig(TypedDict):
    enabled: bool
    rule: Literal["STDP", "Hebbian", "Oja", "BCM"]
    parameters: Dict[str, float]

class SimulationConfig(TypedDict):
    timestep: float  # ms
    duration: float  # ms
    backend: Literal["numpy", "jax", "custom"]
    seed: int

class BrainConfig(TypedDict):
    neurons: NeuronConfig
    connectivity: ConnectivityConfig
    plasticity: PlasticityConfig
    simulation: SimulationConfig
```

### Configuration Validation

```python
from src.config import validate_config, ConfigError

try:
    validate_config(config_dict)
except ConfigError as e:
    print(f"Invalid configuration: {e}")
```

### Configuration Builder

```python
from src.config import ConfigBuilder

config = (ConfigBuilder()
    .set_neurons(count=1000, type="LIF")
    .set_connectivity(probability=0.1)
    .set_plasticity(rule="STDP", enabled=True)
    .set_simulation(timestep=1.0, duration=1000.0)
    .build())
```

---

## Plugin API

Extend framework with custom components.

### Neuron Plugin

```python
from src.plugin_system import NeuronPlugin

class MyNeuronModel(NeuronPlugin):
    """Custom neuron model."""
    
    name = "MyNeuron"
    version = "1.0.0"
    
    def __init__(self, **params):
        super().__init__()
        # Initialize parameters
    
    def update(self, I: float, dt: float) -> Tuple[float, bool]:
        # Custom update logic
        pass
    
    def reset(self) -> None:
        # Reset logic
        pass
    
    @classmethod
    def default_parameters(cls) -> Dict[str, float]:
        return {"tau": 10.0, "threshold": -50.0}

# Register plugin
from src.plugin_system import register_neuron_plugin
register_neuron_plugin(MyNeuronModel)

# Use plugin
model = BrainModel(config={"neuron_type": "MyNeuron"})
```

### Plasticity Plugin

```python
from src.plugin_system import PlasticityPlugin

class MyPlasticityRule(PlasticityPlugin):
    """Custom plasticity rule."""
    
    name = "MyPlasticity"
    version = "1.0.0"
    
    def update(self, pre, post, weight, dt):
        # Custom plasticity logic
        return new_weight

# Register and use
register_plasticity_plugin(MyPlasticityRule)
```

### Task Plugin

```python
from src.plugin_system import TaskPlugin

class MyTask(TaskPlugin):
    """Custom cognitive task."""
    
    name = "MyTask"
    version = "1.0.0"
    
    def generate_input(self, trial: int) -> np.ndarray:
        # Generate input
        pass
    
    def evaluate(self, output: np.ndarray, trial: int) -> Dict:
        # Evaluate performance
        pass
```

---

## Data Export API

### Export Formats

```python
from src.data_export import Exporter

class Exporter:
    """Export simulation data."""
    
    def to_hdf5(
        self,
        filepath: str,
        spike_trains: Dict,
        states: np.ndarray,
        metadata: Dict
    ) -> None:
        """
        Export to HDF5 format.
        
        Structure:
            /spikes/neuron_0001
            /spikes/neuron_0002
            /states/voltage
            /states/recovery
            /metadata
        """
    
    def to_neo(self) -> neo.Block:
        """Export to Neo format (neuroscience standard)."""
    
    def to_pandas(self) -> pd.DataFrame:
        """Export to Pandas DataFrame."""
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Export to NumPy arrays."""
    
    def to_json(self, filepath: str) -> None:
        """Export to JSON (small datasets only)."""
```

### Import Formats

```python
from src.data_export import Importer

class Importer:
    """Import external data."""
    
    def from_hdf5(self, filepath: str) -> BrainModel:
        """Load from HDF5."""
    
    def from_neo(self, block: neo.Block) -> BrainModel:
        """Load from Neo format."""
    
    def from_numpy(self, arrays: Dict) -> BrainModel:
        """Load from NumPy arrays."""
```

---

## Versioning & Compatibility

### Semantic Versioning

API follows semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes (incompatible API changes)
- **MINOR**: New features (backward-compatible)
- **PATCH**: Bug fixes (backward-compatible)

Current version: **1.0.0**

### Deprecation Policy

1. Features marked deprecated in version X.Y.0
2. Warnings issued for 2 minor versions (X.Y and X.(Y+1))
3. Removed in version X.(Y+2).0

Example:
```python
@deprecated(version="1.2.0", alternative="new_method")
def old_method(self):
    """This method is deprecated."""
    warnings.warn("Use new_method instead", DeprecationWarning)
```

### Compatibility Matrix

| API Version | Python Version | NumPy | JAX | Status |
|-------------|----------------|-------|-----|--------|
| 1.0.x | 3.8-3.12 | ≥1.20 | ≥0.4.0 | Current |
| 0.9.x | 3.8-3.11 | ≥1.18 | ≥0.3.0 | Deprecated |

### Migration Guides

Migration guides available in `docs/migration/`:
- `v0.9_to_v1.0.md`
- Future versions as released

---

## Usage Examples

### Example 1: Simple Network

```python
from src.brain_model import BrainModel
import numpy as np

# Create model
model = BrainModel(config={
    "neurons": 100,
    "connectivity": 0.1,
    "dt": 1.0
})

# Run simulation
results = model.run(
    duration=1000.0,
    input_fn=lambda: np.random.randn(100) * 5.0
)

# Analyze results
spikes = model.get_spike_trains()
print(f"Total spikes: {sum(len(s) for s in spikes.values())}")

# Save model
model.save("my_model.h5")
```

### Example 2: Parameter Sweep

```python
from src.experiment_management import ParameterSweep

sweep = ParameterSweep(
    base_config="config.json",
    parameters={
        "connectivity": [0.05, 0.1, 0.2],
        "learning_rate": [0.001, 0.01, 0.1]
    }
)

results = sweep.run(n_repeats=5)
sweep.plot_results()
sweep.save_results("sweep_results.csv")
```

### Example 3: Custom Task

```python
from src.tasks import Task, TaskRunner
import numpy as np

class PatternRecognition(Task):
    def generate_input(self, trial):
        # Generate random pattern
        return np.random.choice([0, 1], size=100)
    
    def evaluate(self, output, trial):
        # Evaluate based on output spikes
        accuracy = self.compute_accuracy(output)
        return {"accuracy": accuracy}

# Run task
model = BrainModel.load("trained_model.h5")
task = PatternRecognition()
runner = TaskRunner(model, task)
results = runner.run_experiment(n_trials=100)
```

---

## API Documentation Generation

Documentation auto-generated from docstrings:

```bash
# Generate API docs
python scripts/generate_api_docs.py

# Output: docs/api/generated/
```

### Interactive API Explorer

```bash
# Start interactive documentation server
python -m src.api_server --docs

# Navigate to: http://localhost:5000/api/docs
```

---

## Testing API

All API endpoints include comprehensive tests:

```bash
# Run API tests
pytest tests/test_api/ -v

# Test coverage
pytest tests/test_api/ --cov=src --cov-report=html
```

---

## Support & Feedback

### Reporting API Issues

Use GitHub issues with `[API]` tag:
- Bug reports
- Feature requests
- Documentation improvements

### API Discussion

GitHub Discussions, API category:
- Best practices
- Usage examples
- Design decisions

### Contact

**Maintainer**: Thomas Heisig  
**Email**: t_heisig@gmx.de  
**Location**: Ganderkesee, Germany

---

**API Version**: 1.0.0  
**Document Version**: 1.0  
**Last Updated**: December 2025  
**License**: MIT (see repository LICENSE file)
