# Learning Systems Framework

## Overview

The Learning Systems Framework integrates both biological/psychological learning systems and machine learning approaches into the 4D Neural Cognition system. This bridges natural and artificial intelligence, providing a comprehensive framework for understanding and implementing different types of learning.

## Fundamental Difference

**Biological Learning**: Consciousness-capable, flexible, and context-dependent. Involves awareness, emotions, and real-world experiences.

**Machine Learning**: Statistical pattern recognition based on algorithms. Operates on data without consciousness or subjective experience.

## Architecture

### Core Components

1. **LearningSystem** (Abstract Base Class)
   - Common interface for all learning systems
   - Tracks learning history and metrics
   - Can be activated/deactivated

2. **LearningContext**
   - Encapsulates the current learning environment
   - Contains timestep, environment state, internal state, and metadata

3. **LearningResult**
   - Represents the outcome of a learning process
   - Includes success status, learning delta, updated parameters, and metrics

4. **LearningSystemManager**
   - Coordinates multiple learning systems
   - Manages activation/deactivation
   - Executes learning across active systems
   - Provides metrics and categorization

## Biological/Psychological Learning Systems

### 1. Associative Learning
**Description**: Linking stimuli, actions, and consequences through repeated co-occurrence.

**Use Cases**:
- Classical conditioning experiments
- Stimulus-response associations
- Linking events in memory

**Parameters**:
- `learning_rate`: Rate of association formation (default: 0.1)

**Example**:
```python
from src.learning_systems import AssociativeLearning, LearningContext

assoc = AssociativeLearning()
context = LearningContext(timestep=1)

data = {
    "stimulus_a": "bell",
    "stimulus_b": "food",
    "strength": 1.0
}

result = assoc.learn(context, data)
print(f"Association strength: {result.metrics['association_strength']}")
```

### 2. Non-Associative Learning
**Description**: Habituation and sensitization through repeated stimulus exposure.

**Use Cases**:
- Adapting to repeated stimuli
- Modeling sensory adaptation
- Response modulation

**Parameters**:
- `habituation_rate`: Rate of response decrease (default: 0.05)
- `sensitization_rate`: Rate of response increase (default: 0.05)

**Example**:
```python
non_assoc = NonAssociativeLearning()

# Habituation
data = {
    "stimulus": "loud_noise",
    "type": "habituation"
}
result = non_assoc.learn(context, data)

# Sensitization
data = {
    "stimulus": "pain",
    "type": "sensitization"
}
result = non_assoc.learn(context, data)
```

### 3. Operant Conditioning
**Description**: Learning through rewards and punishments.

**Use Cases**:
- Behavior modification
- Action-consequence learning
- Reinforcement-based training

**Parameters**:
- `learning_rate`: Rate of value updates (default: 0.1)

**Example**:
```python
operant = OperantConditioning()

# Positive reinforcement
data = {
    "behavior": "press_lever",
    "reward": 1.0
}
result = operant.learn(context, data)

# Punishment
data = {
    "behavior": "bad_action",
    "reward": -1.0
}
result = operant.learn(context, data)
```

## Machine Learning Systems

### 1. Supervised Learning
**Description**: Learning from labeled training data to map inputs to outputs.

**Use Cases**:
- Classification tasks
- Regression problems
- Pattern recognition with labels

**Parameters**:
- `learning_rate`: Gradient descent learning rate (default: 0.01)

**Example**:
```python
from src.learning_systems import SupervisedLearning

supervised = SupervisedLearning()

data = {
    "input": [1, 2, 3],
    "label": 1,
    "error": 0.3
}

result = supervised.learn(context, data)
print(f"Training samples: {result.metrics['samples']}")
```

### 2. Unsupervised Learning
**Description**: Discovering patterns and structure in unlabeled data through clustering.

**Use Cases**:
- Data exploration
- Pattern discovery
- Clustering similar items

**Parameters**:
- `num_clusters`: Number of clusters for pattern assignment (default: 5)

**Example**:
```python
unsupervised = UnsupervisedLearning()

data = {
    "input": [4.5, 2.3, 7.8]
}

result = unsupervised.learn(context, data)
print(f"Cluster assignment: {result.feedback}")
```

### 3. Reinforcement Learning
**Description**: Learning optimal actions through trial-and-error with reward signals.

**Use Cases**:
- Sequential decision making
- Policy optimization
- Agent training

**Parameters**:
- `learning_rate`: Q-value update rate (default: 0.1)
- `discount_factor`: Future reward discount (default: 0.9)

**Example**:
```python
rl = ReinforcementLearning()

data = {
    "state": "s1",
    "action": "a1",
    "reward": 1.0,
    "next_state": "s2"
}

result = rl.learn(context, data)
print(f"Q-value: {result.metrics['q_value']}")
```

### 4. Transfer Learning
**Description**: Transferring knowledge from source domains to new target domains.

**Use Cases**:
- Cross-domain adaptation
- Pre-trained model fine-tuning
- Knowledge reuse

**Example**:
```python
transfer = TransferLearning()

data = {
    "source_domain": "images",
    "target_domain": "sketches",
    "domain_similarity": 0.8
}

result = transfer.learn(context, data)
print(f"Transfer successful: {result.success}")
```

### 5. Meta-Learning
**Description**: Learning to learn - optimizing the learning process itself.

**Use Cases**:
- Algorithm selection
- Hyperparameter optimization
- Few-shot learning

**Example**:
```python
meta = MetaLearning()

data = {
    "strategy": "gradient_descent",
    "performance": 0.85
}

result = meta.learn(context, data)
print(f"Average performance: {result.metrics['avg_performance']}")
```

## Using the Learning System Manager

The `LearningSystemManager` coordinates multiple learning systems:

```python
from src.learning_systems import create_default_learning_systems

# Create manager with all default systems
manager = create_default_learning_systems()

# Activate specific systems
manager.activate_system("Associative Learning")
manager.activate_system("Supervised Learning")
manager.activate_system("Reinforcement Learning")

# Prepare learning context
context = LearningContext(timestep=1)

# Prepare data for each active system
data = {
    "Associative Learning": {
        "stimulus_a": "signal",
        "stimulus_b": "action",
        "strength": 1.0
    },
    "Supervised Learning": {
        "input": [1, 2, 3],
        "label": 1,
        "error": 0.2
    },
    "Reinforcement Learning": {
        "state": "s1",
        "action": "a1",
        "reward": 1.0,
        "next_state": "s2"
    }
}

# Execute learning across all active systems
results = manager.learn(context, data)

# Check results
for system_name, result in results.items():
    print(f"{system_name}: success={result.success}, delta={result.learning_delta}")

# Get metrics from all systems
all_metrics = manager.get_all_metrics()
print(f"All metrics: {all_metrics}")

# Get systems by category
bio_systems = manager.get_biological_systems()
ml_systems = manager.get_machine_systems()
print(f"Biological systems: {len(bio_systems)}")
print(f"Machine learning systems: {len(ml_systems)}")
```

## Integration with Brain Model

The learning systems can be integrated with the existing brain model simulation:

```python
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.learning_systems import (
    create_default_learning_systems,
    LearningContext,
)

# Initialize brain model
model = BrainModel(config_path='brain_base_model.json')
sim = Simulation(model, seed=42)

# Initialize neurons and synapses
sim.initialize_neurons(area_names=['V1_like'], density=0.1)
sim.initialize_random_synapses(connection_probability=0.01)

# Create learning system manager
learning_manager = create_default_learning_systems()
learning_manager.activate_system("Operant Conditioning")
learning_manager.activate_system("Reinforcement Learning")

# Simulation loop with learning
for step in range(100):
    # Run simulation step
    stats = sim.step()
    
    # Create learning context
    context = LearningContext(
        timestep=step,
        environment_state={"spikes": len(stats["spikes"])},
    )
    
    # Determine reward based on simulation activity
    reward = 1.0 if len(stats["spikes"]) > 10 else -0.1
    
    # Apply learning
    learning_data = {
        "Operant Conditioning": {
            "behavior": "neural_activity",
            "reward": reward
        },
        "Reinforcement Learning": {
            "state": f"step_{step}",
            "action": "activate",
            "reward": reward,
            "next_state": f"step_{step+1}"
        }
    }
    
    results = learning_manager.learn(context, learning_data)
    
    if step % 20 == 0:
        print(f"Step {step}: {len(stats['spikes'])} spikes, reward={reward:.2f}")
        for system_name, result in results.items():
            print(f"  {system_name}: delta={result.learning_delta:.3f}")
```

## Performance Metrics

Each learning system tracks the following metrics:

- **success_rate**: Proportion of successful learning episodes
- **average_learning_delta**: Average amount of learning per episode
- **total_learning_episodes**: Total number of learning episodes

System-specific metrics are also available in the `metrics` field of each `LearningResult`.

## Best Practices

1. **Choose the Right System**: Select learning systems appropriate for your task
   - Use biological systems for modeling natural learning processes
   - Use machine learning systems for data-driven pattern recognition

2. **Combine Systems**: Multiple learning systems can work together
   - Associative + Operant for complex conditioning
   - Supervised + Transfer for cross-domain tasks
   - Meta-learning to optimize other systems

3. **Monitor Metrics**: Track learning progress through metrics
   - Check success rates to ensure learning is occurring
   - Monitor learning deltas to detect convergence
   - Compare biological vs. machine learning performance

4. **Adjust Parameters**: Fine-tune learning rates and other parameters
   - Start with default values
   - Adjust based on observed performance
   - Use meta-learning for automatic tuning

5. **Context Matters**: Provide rich learning contexts
   - Include relevant environment state
   - Track internal state for complex scenarios
   - Use metadata for additional information

## Future Enhancements

Planned additions to the learning systems framework:

- Additional biological systems:
  - Explicit Learning (conscious knowledge acquisition)
  - Implicit Learning (unconscious skill learning)
  - Social Learning (observational learning)
  - Classical Conditioning (Pavlovian associations)
  - Insight Learning (sudden problem solving)
  - Emotional Learning (affective associations)
  - Motor Learning (movement skills)
  - Cognitive Learning (mental models)
  - Metacognitive Learning (learning about learning)
  - Situated Learning (context-based learning)
  - Exploratory Learning (discovery-based)

- Additional machine learning systems:
  - Semi-supervised Learning
  - Self-supervised Learning
  - Batch Learning
  - Online Learning (extended)
  - Ensemble Learning (extended)
  - Deep Learning
  - Federated Learning
  - Evolutionary Learning
  - Symbolic Learning
  - Bayesian Learning
  - Instance-based Learning

- Integration features:
  - Automatic system selection
  - Learning system composition
  - Cross-system knowledge transfer
  - Unified evaluation framework

## References

- Pavlov, I. P. (1927). *Conditioned Reflexes*
- Skinner, B. F. (1938). *The Behavior of Organisms*
- Hebb, D. O. (1949). *The Organization of Behavior*
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*
- Mitchell, T. M. (1997). *Machine Learning*

## See Also

- [API Documentation](api/API.md)
- [Examples](../examples/)
- [Architecture](ARCHITECTURE.md)
- [Tasks and Evaluation](user-guide/TASKS_AND_EVALUATION.md)
