# Integration Guide - New Features

This guide covers the newly integrated features in 4D Neural Cognition, including accelerated backends, plugin system, experiment management, biological components, and framework bridges.

## Table of Contents

1. [Accelerated Backends](#accelerated-backends)
2. [Plugin System](#plugin-system)
3. [Experiment Management](#experiment-management)
4. [Biological Components](#biological-components)
5. [Applications](#applications)
6. [Framework Bridges](#framework-bridges)
7. [Validation and Benchmarking](#validation-and-benchmarking)

---

## Accelerated Backends

The new backend system automatically selects the optimal computation backend based on network size and available hardware.

### Quick Start

```python
from src.backends import AcceleratedEngine

# Create engine (auto-detects available backends)
engine = AcceleratedEngine()

# Check available backends
print(engine.get_available_backends())
# Output: {'numpy': True, 'jax': True, 'graph': True, 'gpu_acceleration': True}

# Select backend automatically based on network size
num_neurons = 50000
num_synapses = 100000
backend = engine.select_backend(num_neurons, num_synapses)

print(f"Selected backend: {backend.name}")
```

### Backend Types

1. **NumPy Backend**: For small networks (<10K neurons)
   - Pure NumPy implementation
   - No external dependencies
   - Good for prototyping

2. **JAX Backend**: For large networks (>10K neurons)
   - Automatic GPU/TPU support
   - JIT compilation for speed
   - Requires: `pip install jax jaxlib`

3. **Graph Backend**: For sparse connectivity
   - Optimized for sparse networks
   - Graph-based operations
   - Best for <10% connectivity

### Manual Backend Selection

```python
# Force specific backend
backend = engine.select_backend(
    num_neurons=1000,
    num_synapses=5000,
    force_backend='jax'  # or 'numpy', 'graph'
)

# Or set preferred backend
engine = AcceleratedEngine(prefer_backend='jax')
```

---

## Plugin System

The plugin system allows extending the framework without modifying core code.

### Creating a Custom Neuron Model

```python
from src.plugin_system import NeuronModelBase, register_plugin

class AdaptiveExponentialIF(NeuronModelBase):
    """Adaptive Exponential Integrate-and-Fire neuron model."""
    
    @property
    def name(self) -> str:
        return "AdExIF"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def update(self, v_membrane, u_recovery, external_input, params, dt=1.0):
        # Implement AdEx dynamics
        delta_T = params.get('delta_T', 2.0)
        v_thresh = params.get('v_threshold', -50.0)
        
        # Exponential term
        exp_term = delta_T * np.exp((v_membrane - v_thresh) / delta_T)
        
        # Update equations
        dv = (-v_membrane + exp_term + external_input - u_recovery) * dt
        du = (params['a'] * (v_membrane - params['v_rest']) - u_recovery) / params['tau_w'] * dt
        
        new_v = v_membrane + dv
        new_u = u_recovery + du
        
        # Check spike
        did_spike = new_v >= params['v_peak']
        if did_spike:
            new_v = params['v_reset']
            new_u += params['b']
        
        return new_v, new_u, did_spike
    
    def get_default_params(self):
        return {
            'v_rest': -70.0,
            'v_reset': -60.0,
            'v_threshold': -50.0,
            'v_peak': 20.0,
            'delta_T': 2.0,
            'a': 0.01,
            'b': 0.1,
            'tau_w': 100.0
        }

# Register plugin
register_plugin('neuron_model', AdaptiveExponentialIF())
```

### Creating a Custom Task

```python
from src.plugin_system import TaskBase, register_plugin

class SpatialNavigationTask(TaskBase):
    """Custom spatial navigation task."""
    
    @property
    def name(self) -> str:
        return "spatial_navigation"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    def generate_trial(self, trial_num):
        # Generate spatial navigation trial
        start_pos = np.random.rand(2) * 10
        goal_pos = np.random.rand(2) * 10
        
        return {
            'start_position': start_pos,
            'goal_position': goal_pos,
            'trial_num': trial_num
        }
    
    def evaluate_response(self, response, trial_data):
        # Evaluate navigation performance
        final_pos = response.get('final_position', np.zeros(2))
        goal_pos = trial_data['goal_position']
        
        distance = np.linalg.norm(final_pos - goal_pos)
        success = distance < 0.5
        
        return {
            'distance_to_goal': distance,
            'success': float(success),
            'time_taken': response.get('time_taken', 0)
        }
    
    def get_num_trials(self):
        return 100

# Register plugin
register_plugin('task', SpatialNavigationTask())
```

### Discovering Plugins

```python
from src.plugin_system import get_plugin_registry

registry = get_plugin_registry()

# Discover plugins from directory
loaded = registry.discover_plugins('path/to/plugins')
print(f"Loaded plugins: {loaded}")

# List all plugins
plugins = registry.list_plugins()
print(f"Available plugins: {plugins}")

# Get specific plugin
plugin = registry.get_plugin('neuron_model', 'AdExIF')
```

---

## Experiment Management

Manage experiments with YAML/JSON configurations and parameter sweeps.

### Creating an Experiment

```python
from src.experiment_management import ExperimentConfig, ExperimentManager

# Create experiment config
config = ExperimentConfig(
    name="attention_study",
    description="Study attention mechanisms",
    network_config={
        'lattice_shape': [10, 10, 10, 5],
        'density': 0.2
    },
    variables=[
        {
            'name': 'learning_rate',
            'path': 'plasticity.learning_rate',
            'values': [0.001, 0.01, 0.1]
        }
    ],
    metrics=['accuracy', 'convergence_time'],
    seed=42
)

# Save config
config.to_yaml('experiments/configs/attention_study.yaml')

# Create experiment manager
manager = ExperimentManager('experiments')
exp_id = manager.create_experiment('attention_study', config)
```

### Running Experiments

```python
from src.experiment_management import ExperimentResult

# Run experiment
result = ExperimentResult(
    experiment_name='attention_study',
    run_id='run_001',
    config=config.to_dict(),
    metrics={
        'accuracy': 0.95,
        'convergence_time': 100
    },
    status='completed'
)

# Save result
manager.save_result(result)

# Load and compare results
results = manager.load_results('attention_study')
comparison = manager.compare_results(
    ['run_001', 'run_002'],
    metric='accuracy'
)
```

### Parameter Sweeps

```python
from src.experiment_management import ParameterSweep

# Create parameter sweep
base_config = {'network': {'tau_m': 10.0}}
sweep = ParameterSweep(base_config)

sweep.add_parameter('network.tau_m', [5.0, 10.0, 20.0])
sweep.add_parameter('network.learning_rate', [0.01, 0.1])

# Generate all combinations
configs = sweep.generate_configs()
print(f"Generated {len(configs)} configurations")

for config_id, config in configs:
    print(f"{config_id}: {config}")
```

---

## Biological Components

### Homeostatic Plasticity

Maintains stable activity levels in large networks.

```python
from src.homeostatic_plasticity import HomeostaticPlasticityManager

# Create manager
homeostasis = HomeostaticPlasticityManager(
    enable_synaptic_scaling=True,
    enable_intrinsic_excitability=True,
    target_rate=5.0  # Target firing rate in Hz
)

# Update during simulation
for neuron_id, did_spike in spikes.items():
    homeostasis.update_neuron(neuron_id, did_spike, current_time)

# Apply synaptic scaling
scaled_weights = homeostasis.apply_to_weights(
    weights=synapse_weights,
    post_neuron_ids=post_ids
)

# Get adjusted thresholds
threshold = homeostasis.get_threshold_adjustment(neuron_id)

# Get statistics
stats = homeostasis.get_statistics()
print(f"Mean firing rate: {stats['mean_firing_rate']:.2f} Hz")
```

### Short-Term Plasticity

Implements facilitation and depression for working memory.

```python
from src.shortterm_plasticity import ShortTermPlasticityManager

# Create manager
stp = ShortTermPlasticityManager(
    default_type='balanced',  # or 'facilitating', 'depressing'
    enable_stp=True
)

# Set specific synapse types
stp.set_synapse_type(synapse_id=123, stp_type='facilitating')

# Process spikes during simulation
spiking_synapses = np.array([1, 5, 10])  # Synapse IDs
strengths = stp.process_spikes(spiking_synapses, current_time)

# Apply to weights
modified_weights = stp.apply_to_weights(
    weights=base_weights,
    synapse_ids=all_synapse_ids,
    spiking_mask=spike_mask,
    current_time=current_time
)

# Working memory circuit
from src.shortterm_plasticity import WorkingMemoryCircuit

wm = WorkingMemoryCircuit(num_items=7, maintenance_threshold=0.5)

# Encode item
wm.encode_item(item_id=2, current_time=100)

# Maintain items
wm.maintain(current_time=110)

# Check capacity
capacity = wm.get_capacity()
print(f"Items in working memory: {capacity}")
```

### Attention Mechanisms

```python
from src.attention_mechanisms import AttentionManager

# Create attention manager
attention = AttentionManager(
    enable_bottom_up=True,
    enable_top_down=True,
    enable_wta=True
)

# Compute combined attention
attention_weights = attention.compute_combined_attention(
    neuron_ids=neuron_ids,
    activations=activations,
    positions=positions,
    features=features,
    bottom_up_weight=0.6,
    top_down_weight=0.4
)

# Apply attention to activations
modulated = attention.apply_attention_to_activations(
    activations=activations,
    attention_weights=attention_weights
)

# Select highly attended neurons
attended = attention.select_attended_neurons(
    neuron_ids=neuron_ids,
    attention_weights=attention_weights,
    threshold=0.7
)

# Spatial attention
from src.attention_mechanisms import SpatialAttention

spatial_attn = SpatialAttention(focus_radius=3.0)
spatial_attn.set_focus([5, 5, 5, 2])  # 4D coordinates

spatial_weights = spatial_attn.compute_spatial_attention(positions)
```

---

## Applications

### Neuro-Symbolic Integration

Bridge neural and symbolic reasoning.

```python
from src.neurosymbolic import SymbolicLayer, SymbolicRule

# Create symbolic layer
symbolic = SymbolicLayer(
    clustering_threshold=0.7,
    min_cluster_size=5
)

# Extract concepts from neural activity
neuron_activations = {
    neuron_id: activation_history
    for neuron_id, activation_history in history.items()
}

concepts = symbolic.extract_concepts(neuron_activations)
print(f"Extracted {len(concepts)} concepts")

# Add symbolic rules
rule = SymbolicRule(
    name="if_A_and_B_then_C",
    antecedents=["concept_0", "concept_1"],
    consequent="concept_2",
    confidence=0.9
)
symbolic.add_rule(rule)

# Perform reasoning
current_activations = {nid: v for nid, v in current_state.items()}
concept_activations = symbolic.get_concept_activations(current_activations)

# Apply rules
result_activations = symbolic.reason(
    neuron_activations=current_activations,
    rule_threshold=0.5,
    num_iterations=3
)

# Map back to neurons
neuron_boosts = symbolic.map_to_neurons(result_activations)

# Export/import knowledge
symbolic.export_knowledge('knowledge.json')
symbolic.import_knowledge('knowledge.json')
```

### Temporal Prediction

Use w-dimension for temporal processing.

```python
from src.temporal_prediction import TemporalPredictor, EchoStateNetwork

# Temporal predictor
predictor = TemporalPredictor(w_depth=10, leak_rate=0.3)

# Update temporal state
for w_layer in range(10):
    predictor.update_temporal_state(w_layer, input_signal)
    predictor.propagate_temporal_information()

# Predict sequence
predictions = predictor.predict_sequence(
    input_sequence=history,
    prediction_steps=10
)

# Echo State Network
esn = EchoStateNetwork(
    reservoir_size=100,
    input_size=10,
    output_size=10
)

# Collect training data
for input_val, target_val in training_data:
    esn.update(input_val, collect_for_training=True, target=target_val)

# Train
esn.train_readout(regularization=1e-6)

# Predict
prediction = esn.predict(test_input)
```

---

## Framework Bridges

Integrate with PyTorch and TensorFlow.

### PyTorch Integration

```python
from src.framework_bridges import BrainToPyTorchConverter, PyTorchBrainWrapper

# Convert brain model to PyTorch
pytorch_model = BrainToPyTorchConverter.to_pytorch_module(brain_model)

# Use as PyTorch module
import torch
input_tensor = torch.randn(1, num_neurons)
output = pytorch_model(input_tensor)

# Export weights
weights = BrainToPyTorchConverter.export_weights(brain_model)

# Create hybrid model
from src.framework_bridges import HybridModel

hybrid = HybridModel(brain_model, framework='pytorch')
model = hybrid.get_model()

# Train with PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

for inputs, targets in dataloader:
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

### TensorFlow Integration

```python
from src.framework_bridges import TensorFlowBrainWrapper

# Convert to TensorFlow/Keras
wrapper = TensorFlowBrainWrapper(brain_model)
layer = wrapper.get_layer()

# Build model
import tensorflow as tf

inputs = tf.keras.Input(shape=(num_neurons,))
x = layer(inputs)
x = tf.keras.layers.Dense(64, activation='relu')(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train
model.fit(train_data, train_labels, epochs=10)
```

### Model Export

```python
from src.framework_bridges import ModelExporter

# Export to ONNX
ModelExporter.export_to_onnx(
    brain_model,
    'model.onnx',
    input_shape=(1, num_neurons)
)

# Export to TensorFlow SavedModel
ModelExporter.export_to_saved_model(
    brain_model,
    'saved_model/'
)

# Export to JSON
ModelExporter.export_to_json(
    brain_model,
    'model.json'
)
```

---

## Validation and Benchmarking

### Biological Validation

```bash
# Run biological plausibility checks
python scripts/validate_biology.py brain_base_model.json
```

Output:
```
==============================================================
BIOLOGICAL PLAUSIBILITY VALIDATION
==============================================================

Checking Dale's Law...
  ✅ PASSED: All neurons follow Dale's Law
Checking connection probabilities...
  Connection probability: 0.0500
  ✅ PASSED: Connection probability is reasonable
...
```

### Performance Tracking

```bash
# Run performance benchmarks
python scripts/performance_tracker.py

# Quick benchmark
python scripts/performance_tracker.py --quick
```

Output:
```
==============================================================
PERFORMANCE BENCHMARK SUITE
==============================================================

Benchmarking: 100 neurons, 0.100 connectivity
  Neurons/sec: 15234
  Steps/sec: 152.3
  Memory: 0.05 MB
...
```

---

## Best Practices

1. **Backend Selection**
   - Use NumPy for prototyping (<1K neurons)
   - Use JAX for production (>10K neurons)
   - Use Graph backend for sparse networks

2. **Plugin Development**
   - Follow base class interfaces strictly
   - Implement validation methods
   - Provide comprehensive documentation

3. **Experiment Management**
   - Use YAML for configurations
   - Track all metrics consistently
   - Use seeds for reproducibility

4. **Biological Plausibility**
   - Run validation regularly
   - Follow Dale's Law
   - Maintain realistic E/I balance

5. **Performance**
   - Benchmark before and after changes
   - Monitor for regressions
   - Optimize hot paths first

---

## Troubleshooting

### JAX Backend Issues

```python
# Check JAX installation
from src.backends import AcceleratedEngine
engine = AcceleratedEngine()
info = engine.get_backend_info()
print(info)

# Install JAX with CPU support
# pip install jax jaxlib

# Install JAX with GPU support (CUDA 12)
# pip install jax[cuda12_pip]
```

### Plugin Loading Errors

```python
# Debug plugin loading
from src.plugin_system import get_plugin_registry
registry = get_plugin_registry()

try:
    loaded = registry.discover_plugins('plugins/')
    print(f"Loaded: {loaded}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

### Performance Issues

```bash
# Profile simulation
python -m cProfile -o profile.stats your_simulation.py

# Analyze profile
python -m pstats profile.stats
```

---

## Further Reading

- [API Reference](api/API.md)
- [Architecture](ARCHITECTURE.md)
- [Contributing](../CONTRIBUTING.md)
- [Examples](../examples/)

---

*Last Updated: December 2025*
