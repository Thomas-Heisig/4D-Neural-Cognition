# Task System Guide

The Task System provides a standardized framework for defining tasks and environments to evaluate and train neural networks.

## Table of Contents

1. [Overview](#overview)
2. [Environment Interface](#environment-interface)
3. [Task Interface](#task-interface)
4. [Built-in Tasks](#built-in-tasks)
5. [Creating Custom Tasks](#creating-custom-tasks)
6. [Running Evaluations](#running-evaluations)
7. [Examples](#examples)

---

## Overview

### What is the Task System?

The Task System provides:
- **Standard interface**: Consistent API for all tasks (similar to OpenAI Gym)
- **Benchmarking**: Predefined tasks for evaluation
- **Metrics**: Standard performance measurements
- **Reproducibility**: Seeded environments for consistent results

### Key Components

- **Environment**: Manages task state and observations
- **Task**: Wraps environment with evaluation logic
- **TaskResult**: Standardized metrics (accuracy, reward, etc.)

---

## Environment Interface

### Basic Environment Structure

```python
from tasks import Environment
import numpy as np

class MyEnvironment(Environment):
    """Custom environment example."""
    
    def __init__(self, seed=None):
        super().__init__(seed)
        # Initialize your environment state
        
    def reset(self):
        """Reset to initial state."""
        observation = {
            'vision': np.zeros((10, 10)),
            'digital': np.zeros((5, 5))
        }
        info = {'episode': 0}
        return observation, info
        
    def step(self, action=None):
        """Execute one step."""
        # Update environment
        observation = self._get_observation()
        reward = self._calculate_reward()
        done = self.current_step >= 100
        info = self.get_info()
        
        self.current_step += 1
        self.episode_reward += reward
        
        return observation, reward, done, info
        
    def render(self):
        """Visualize environment."""
        # Return visualization or None
        return None
```

### Environment Methods

**`reset()`**
- Resets environment to initial state
- Returns: `(observation, info)`
- Called at start of each episode

**`step(action)`**
- Advances environment by one timestep
- Returns: `(observation, reward, done, info)`
- Core interaction method

**`render()`**
- Visualizes current state
- Returns: Array or None
- Optional visualization

**`get_info()`**
- Returns current environment metadata
- Returns: Dictionary with current step, reward, etc.

---

## Task Interface

### Basic Task Structure

```python
from tasks import Task, TaskResult

class MyTask(Task):
    """Custom task example."""
    
    def __init__(self, seed=None):
        env = MyEnvironment(seed)
        super().__init__(env, name="MyTask")
        
    def evaluate(self, simulation, num_episodes=10):
        """Evaluate simulation on this task."""
        results = []
        
        for episode in range(num_episodes):
            result = self._run_episode(simulation)
            results.append(result)
        
        # Aggregate results
        return self._aggregate_results(results)
        
    def _run_episode(self, simulation):
        """Run one episode."""
        obs, info = self.env.reset()
        done = False
        
        while not done:
            # Feed observation to network
            # Get response from network
            # Take step in environment
            obs, reward, done, info = self.env.step()
        
        return TaskResult(
            accuracy=1.0,
            reward=self.env.episode_reward
        )
```

---

## Built-in Tasks

### Pattern Classification Task

Classify visual patterns:

```python
from tasks import PatternClassificationTask
from brain_model import BrainModel
from simulation import Simulation

# Create task
task = PatternClassificationTask(
    pattern_size=10,
    num_patterns=5,
    seed=42
)

# Setup network
model = BrainModel(config_path="brain_base_model.json")
sim = Simulation(model, seed=42)
sim.initialize_neurons(areas=["V1_like"], density=0.1)
sim.initialize_random_synapses(connection_prob=0.1)

# Evaluate
result = task.evaluate(sim, num_episodes=20)

print(f"Accuracy: {result.accuracy:.2f}")
print(f"Avg Reward: {result.reward:.2f}")
print(f"Reaction Time: {result.reaction_time:.2f}")
```

### Temporal Sequence Task

Learn temporal patterns:

```python
from tasks import TemporalSequenceTask

# Create task
task = TemporalSequenceTask(
    sequence_length=5,
    num_sequences=3,
    seed=42
)

# Evaluate
result = task.evaluate(sim, num_episodes=20)

print(f"Sequence Accuracy: {result.accuracy:.2f}")
```

### Digital Sense Task

Process text/symbolic information:

```python
from tasks import DigitalSenseTask

# Create task with specific patterns
task = DigitalSenseTask(
    patterns=["hello", "world", "test"],
    seed=42
)

# Evaluate
result = task.evaluate(sim, num_episodes=15)

print(f"Recognition Accuracy: {result.accuracy:.2f}")
```

---

## Creating Custom Tasks

### Example: Simple Reaction Task

Test how quickly network responds to stimuli:

```python
from tasks import Environment, Task, TaskResult
import numpy as np
from senses import feed_sense_input

class ReactionEnvironment(Environment):
    """Environment for reaction time task."""
    
    def __init__(self, seed=None):
        super().__init__(seed)
        self.stimulus_time = None
        self.response_time = None
        
    def reset(self):
        """Reset for new trial."""
        self.current_step = 0
        self.stimulus_time = self.rng.integers(10, 30)
        self.response_time = None
        
        observation = {
            'vision': np.zeros((10, 10))
        }
        info = {'stimulus_time': self.stimulus_time}
        
        return observation, info
        
    def step(self, action=None):
        """Execute step."""
        self.current_step += 1
        
        # Show stimulus at predetermined time
        if self.current_step == self.stimulus_time:
            observation = {
                'vision': np.ones((10, 10))
            }
        else:
            observation = {
                'vision': np.zeros((10, 10))
            }
        
        # Check for response
        if action is not None and action > 0.5 and self.response_time is None:
            if self.current_step > self.stimulus_time:
                self.response_time = self.current_step - self.stimulus_time
                reward = 1.0 / self.response_time  # Faster = better
            else:
                reward = -1.0  # Early response penalty
        else:
            reward = 0.0
        
        done = self.current_step >= 100 or self.response_time is not None
        info = self.get_info()
        
        return observation, reward, done, info
        
    def render(self):
        """Visualize."""
        return None


class ReactionTask(Task):
    """Task for measuring reaction time."""
    
    def __init__(self, seed=None):
        env = ReactionEnvironment(seed)
        super().__init__(env, name="Reaction Task")
        
    def evaluate(self, simulation, num_episodes=20):
        """Evaluate reaction time."""
        reaction_times = []
        accuracies = []
        
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            done = False
            
            responded = False
            
            while not done:
                # Feed observation
                feed_sense_input(
                    simulation.model,
                    sense_name="vision",
                    input_data=obs['vision'],
                    intensity=5.0
                )
                
                # Run simulation
                spiked = simulation.step()
                
                # Check for response (high activity = response)
                action = len(spiked) / len(simulation.model.get_neurons())
                
                # Step environment
                obs, reward, done, info = self.env.step(action)
                
                if self.env.response_time is not None and not responded:
                    responded = True
                    reaction_times.append(self.env.response_time)
                    accuracies.append(1.0 if reward > 0 else 0.0)
            
            if not responded:
                # No response = failure
                accuracies.append(0.0)
        
        return TaskResult(
            accuracy=np.mean(accuracies),
            reward=np.mean([1.0/rt for rt in reaction_times]) if reaction_times else 0.0,
            reaction_time=np.mean(reaction_times) if reaction_times else float('inf'),
            stability=1.0 - np.std(reaction_times)/np.mean(reaction_times) if reaction_times else 0.0
        )
```

### Using the Custom Task

```python
# Create and run
reaction_task = ReactionTask(seed=42)

model = BrainModel(config_path="brain_base_model.json")
sim = Simulation(model, seed=42)
sim.initialize_neurons(areas=["V1_like"], density=0.1)
sim.initialize_random_synapses(connection_prob=0.1)

result = reaction_task.evaluate(sim, num_episodes=20)

print(f"Accuracy: {result.accuracy:.2%}")
print(f"Avg Reaction Time: {result.reaction_time:.1f} steps")
print(f"Stability: {result.stability:.2f}")
```

---

## Running Evaluations

### Single Task Evaluation

```python
from tasks import PatternClassificationTask

# Create task
task = PatternClassificationTask(seed=42)

# Setup network
model = BrainModel(config_path="brain_base_model.json")
sim = Simulation(model, seed=42)
sim.initialize_neurons(areas=["V1_like"], density=0.1)
sim.initialize_random_synapses(connection_prob=0.1)

# Run evaluation
print("Evaluating...")
result = task.evaluate(sim, num_episodes=50)

print("\nResults:")
print(f"  Accuracy: {result.accuracy:.2%}")
print(f"  Reward: {result.reward:.2f}")
print(f"  Reaction Time: {result.reaction_time:.2f}")
print(f"  Stability: {result.stability:.2f}")
```

### Multiple Task Evaluation

```python
from tasks import PatternClassificationTask, TemporalSequenceTask

# Create multiple tasks
tasks = {
    'Pattern': PatternClassificationTask(seed=42),
    'Sequence': TemporalSequenceTask(seed=42)
}

# Evaluate on all
results = {}
for name, task in tasks.items():
    print(f"\nEvaluating {name}...")
    result = task.evaluate(sim, num_episodes=20)
    results[name] = result
    
    print(f"  Accuracy: {result.accuracy:.2%}")
    print(f"  Reward: {result.reward:.2f}")

# Compare
print("\n=== Comparison ===")
for name, result in results.items():
    print(f"{name:15} | Acc: {result.accuracy:.2%} | Reward: {result.reward:.2f}")
```

### Configuration Comparison

```python
from evaluation import BenchmarkConfig

# Define configurations
configs = [
    BenchmarkConfig(
        name="Low Density",
        neuron_density=0.05,
        synapse_prob=0.1,
        learning_rate=0.01
    ),
    BenchmarkConfig(
        name="High Density",
        neuron_density=0.15,
        synapse_prob=0.15,
        learning_rate=0.01
    )
]

# Evaluate each configuration
for config in configs:
    print(f"\n{'='*50}")
    print(f"Configuration: {config.name}")
    print(f"{'='*50}")
    
    # Create model with config
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    sim.initialize_neurons(
        areas=["V1_like"],
        density=config.neuron_density
    )
    sim.initialize_random_synapses(
        connection_prob=config.synapse_prob
    )
    
    # Evaluate on task
    task = PatternClassificationTask(seed=42)
    result = task.evaluate(sim, num_episodes=20)
    
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"Reward: {result.reward:.2f}")
```

---

## Examples

### Example 1: Complete Task Evaluation

```python
#!/usr/bin/env python3
"""Complete task evaluation example."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from simulation import Simulation
from tasks import PatternClassificationTask, TemporalSequenceTask

def main():
    print("=== Neural Network Task Evaluation ===\n")
    
    # Setup network
    print("Setting up network...")
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    sim.initialize_neurons(areas=["V1_like", "Digital_sensor"], density=0.1)
    sim.initialize_random_synapses(connection_prob=0.1)
    
    print(f"  Neurons: {len(sim.model.get_neurons())}")
    print(f"  Synapses: {len(sim.model.get_synapses())}")
    
    # Define tasks
    tasks = [
        PatternClassificationTask(pattern_size=10, num_patterns=5, seed=42),
        TemporalSequenceTask(sequence_length=5, num_sequences=3, seed=42)
    ]
    
    # Evaluate each task
    for task in tasks:
        print(f"\n{'='*50}")
        print(f"Task: {task.name}")
        print(f"{'='*50}")
        
        result = task.evaluate(sim, num_episodes=20)
        
        print(f"Results:")
        print(f"  Accuracy: {result.accuracy*100:.1f}%")
        print(f"  Reward: {result.reward:.2f}")
        print(f"  Reaction Time: {result.reaction_time:.2f} steps")
        print(f"  Stability: {result.stability:.2f}")
    
    print("\n=== Evaluation Complete ===")

if __name__ == "__main__":
    main()
```

### Example 2: Task with Training

```python
#!/usr/bin/env python3
"""Train network on task then evaluate."""

from brain_model import BrainModel
from simulation import Simulation
from tasks import PatternClassificationTask
from senses import feed_sense_input

def train_on_task(sim, task, num_episodes=50):
    """Train network by running task episodes."""
    print("Training...")
    
    for episode in range(num_episodes):
        obs, info = task.env.reset()
        done = False
        
        while not done:
            # Feed observation
            for sense_name, data in obs.items():
                feed_sense_input(
                    sim.model,
                    sense_name=sense_name,
                    input_data=data,
                    intensity=5.0
                )
            
            # Process
            sim.step()
            
            # Learn
            if sim.current_time % 10 == 0:
                sim.apply_plasticity()
            
            # Get next observation
            obs, reward, done, info = task.env.step()
        
        if episode % 10 == 0:
            print(f"  Episode {episode}/{num_episodes}")

def main():
    # Setup
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    sim.initialize_neurons(areas=["V1_like"], density=0.1)
    sim.initialize_random_synapses(connection_prob=0.1)
    
    task = PatternClassificationTask(seed=42)
    
    # Evaluate before training
    print("Before Training:")
    result_before = task.evaluate(sim, num_episodes=10)
    print(f"  Accuracy: {result_before.accuracy:.2%}")
    
    # Train
    train_on_task(sim, task, num_episodes=50)
    
    # Evaluate after training
    print("\nAfter Training:")
    result_after = task.evaluate(sim, num_episodes=10)
    print(f"  Accuracy: {result_after.accuracy:.2%}")
    
    # Show improvement
    improvement = (result_after.accuracy - result_before.accuracy) * 100
    print(f"\nImprovement: {improvement:+.1f}%")

if __name__ == "__main__":
    main()
```

---

## Best Practices

1. **Use Seeds**: Always seed environments for reproducibility
2. **Multiple Episodes**: Run many episodes for reliable metrics
3. **Standard Metrics**: Use TaskResult for consistency
4. **Document Tasks**: Clearly describe task objectives
5. **Test Environments**: Test environment logic independently
6. **Incremental Complexity**: Start simple, add complexity gradually

---

## API Reference

### Environment Class
- `reset() -> (observation, info)`
- `step(action) -> (observation, reward, done, info)`
- `render() -> Optional[np.ndarray]`
- `get_info() -> Dict[str, Any]`

### Task Class
- `evaluate(simulation, num_episodes) -> TaskResult`

### TaskResult Dataclass
- `accuracy`: Classification/success rate
- `reward`: Cumulative reward
- `reaction_time`: Average response time
- `stability`: Consistency metric
- `additional_metrics`: Custom metrics dictionary

---

*Last Updated: December 2025*  
*Part of the 4D Neural Cognition Documentation*
