# Tasks & Evaluation Framework

## Overview

The Tasks & Evaluation framework provides a standardized way to measure and compare the performance of different 4D Neural Cognition configurations. It addresses the fundamental question: **"How do we know if the system works?"**

## Key Components

### 1. Environment Layer

The `Environment` class provides a standard interface similar to OpenAI Gym:

```python
from src.tasks import Environment

class MyEnvironment(Environment):
    def reset(self):
        """Reset environment to initial state."""
        return observation, info
        
    def step(self, action):
        """Execute one step."""
        return observation, reward, done, info
        
    def render(self):
        """Visualize environment state."""
        return visualization
```

**Key Features:**
- Standard `step()`, `reset()`, `render()` interface
- Observation dictionary mapping sense names to input arrays
- Reward signals for reinforcement learning
- Info dictionary for additional metadata
- Reproducible with seed parameter

### 2. Task API

Tasks wrap environments and define evaluation metrics:

```python
from src.tasks import Task, TaskResult

class MyTask(Task):
    def evaluate(self, simulation, num_episodes, max_steps):
        """Evaluate simulation on this task."""
        # Run task and collect metrics
        return TaskResult(
            accuracy=...,
            reward=...,
            reaction_time=...,
            stability=...
        )
        
    def get_metrics(self):
        """Describe available metrics."""
        return {
            'accuracy': 'Classification accuracy (0-1)',
            'reward': 'Average reward per episode'
        }
```

**Key Features:**
- Consistent evaluation interface
- Multiple metrics per task
- Configurable episode/step counts
- Result tracking with TaskResult

### 3. Benchmark Suite

The `BenchmarkSuite` class runs multiple tasks on a configuration:

```python
from src.evaluation import BenchmarkSuite, BenchmarkConfig

suite = BenchmarkSuite(name="My Suite", description="...")
suite.add_task(PatternClassificationTask(...))
suite.add_task(TemporalSequenceTask(...))

config = BenchmarkConfig(
    name="baseline",
    config_path="brain_base_model.json",
    seed=42,
    initialization_params={...}
)

results = suite.run(config, output_dir="./results")
```

**Key Features:**
- Multiple tasks in one suite
- Automated result collection
- JSON output for analysis
- Reproducibility tracking

### 4. Configuration Comparison

Compare multiple configurations objectively:

```python
from src.evaluation import run_configuration_comparison

configs = [config1, config2, config3]
report = run_configuration_comparison(
    configs=configs,
    suite=suite,
    output_dir="./results"
)

# Report shows best performer for each metric
```

**Key Features:**
- Side-by-side comparison
- Best performer identification
- Statistical summary
- Automated report generation

### 5. Knowledge Database

Store and access training data:

```python
from src.knowledge_db import KnowledgeDatabase, KnowledgeBasedTrainer

# Create/open database
db = KnowledgeDatabase("knowledge.db")

# Add training data
entry = KnowledgeEntry(
    category='pattern_recognition',
    data_type='vision',
    data=pattern_array,
    label=class_label,
    ...
)
db.add_entry(entry)

# Pre-train from database
trainer = KnowledgeBasedTrainer(simulation, db)
stats = trainer.pretrain(
    category='pattern_recognition',
    num_samples=100
)
```

**Key Features:**
- SQLite database for efficiency
- Category and data type filtering
- Batch operations
- Pre-training capabilities
- Fallback learning (access DB when network untrained)

## Standard Benchmark Tasks

### Pattern Classification Task

Tests basic sensory processing and pattern recognition.

```python
from src.tasks import PatternClassificationTask

task = PatternClassificationTask(
    num_classes=4,
    pattern_size=(20, 20),
    noise_level=0.1,
    seed=42
)
```

**Metrics:**
- **Accuracy**: Classification accuracy (0-1)
- **Reward**: Average reward per episode
- **Reaction Time**: Steps to first response
- **Stability**: Response time consistency

**What it tests:**
- Vision processing
- Pattern discrimination
- Noise robustness
- Response generation

### Temporal Sequence Task

Tests temporal processing and memory.

```python
from src.tasks import TemporalSequenceTask

task = TemporalSequenceTask(
    sequence_length=5,
    vocabulary_size=8,
    seed=42
)
```

**Metrics:**
- **Accuracy**: Sequence prediction accuracy
- **Reward**: Average reward per episode

**What it tests:**
- Temporal integration
- Sequence memory
- Prediction capability
- Digital sense processing

## Usage Examples

### Example 1: Single Benchmark Run

```python
from src.evaluation import BenchmarkConfig, create_standard_benchmark_suite

# Define configuration
config = BenchmarkConfig(
    name="baseline",
    description="Standard configuration",
    config_path="brain_base_model.json",
    seed=42,
    initialization_params={
        'area_names': ['V1_like', 'Digital_sensor'],
        'density': 0.1,
        'connection_probability': 0.01
    }
)

# Run standard benchmark suite
suite = create_standard_benchmark_suite()
results = suite.run(config, output_dir="./results")

# View results
for result in results:
    print(f"{result.task_name}: Accuracy={result.task_result.accuracy:.4f}")
```

### Example 2: Comparing Configurations

```python
from src.evaluation import run_configuration_comparison

configs = [
    BenchmarkConfig(name="sparse", ...),
    BenchmarkConfig(name="dense", ...),
    BenchmarkConfig(name="strong_weights", ...)
]

report = run_configuration_comparison(
    configs=configs,
    output_dir="./comparison_results"
)

# Report shows which config performs best on each task
```

### Example 3: Using Knowledge Database

```python
from src.knowledge_db import (
    KnowledgeDatabase,
    KnowledgeBasedTrainer,
    populate_sample_knowledge
)

# Create and populate database
populate_sample_knowledge("knowledge.db")

# Pre-train network
db = KnowledgeDatabase("knowledge.db")
trainer = KnowledgeBasedTrainer(simulation, db)

stats = trainer.pretrain(
    category='pattern_recognition',
    num_samples=50,
    steps_per_sample=30
)

print(f"Pre-trained with {stats['samples_processed']} samples")
print(f"Average activity: {stats['avg_activity']:.2f} spikes/step")
```

### Example 4: Fallback Learning

```python
# Train with database fallback for untrained networks
stats = trainer.train_with_fallback(
    current_data=my_pattern,
    data_type='vision',
    category='pattern_recognition',
    steps=50,
    use_database=True
)

if stats['used_database']:
    print(f"Network was untrained, used {stats['database_samples']} DB examples")
else:
    print("Network responded with sufficient activity")
```

## Reproducibility

All benchmark runs track:

- **Git commit hash** (if in git repo)
- **Configuration file** (embedded in results)
- **Random seeds** (for all random operations)
- **Hardware/backend** (CPU/GPU information)
- **Library versions** (NumPy, etc.)
- **Timestamp** (when run was executed)

This ensures all results are fully reproducible.

## Output Format

Results are saved as JSON files:

```json
{
  "suite_name": "Standard 4D Neural Cognition Benchmark",
  "config": {
    "name": "baseline",
    "seed": 42,
    ...
  },
  "results": [
    {
      "task_name": "PatternClassification-4class",
      "task_result": {
        "accuracy": 0.7500,
        "reward": 0.7500,
        "reaction_time": 12.5,
        "stability": 0.8234
      },
      "execution_time": 45.2,
      "timestamp": "2025-12-06 13:00:00"
    }
  ]
}
```

## Best Practices

### 1. Always Use Seeds

```python
# Good: Reproducible
config = BenchmarkConfig(..., seed=42)
task = PatternClassificationTask(..., seed=42)

# Bad: Non-reproducible
config = BenchmarkConfig(..., seed=None)
```

### 2. Save All Results

```python
# Always specify output_dir
results = suite.run(config, output_dir="./results")
```

### 3. Compare Apples to Apples

```python
# Use same suite for fair comparison
suite = create_standard_benchmark_suite()
for config in configs:
    suite.run(config, ...)
```

### 4. Document Configuration Changes

```python
config = BenchmarkConfig(
    name="dense_network",
    description="Increased density from 0.1 to 0.2",  # Clear description
    ...
)
```

### 5. Pre-train Before Evaluation

```python
# Pre-train network first
trainer.pretrain(category='pattern_recognition', num_samples=100)

# Then evaluate
task.evaluate(simulation, ...)
```

## Creating Custom Tasks

To create a custom task:

1. **Define Environment** (if needed):

```python
class MyEnvironment(Environment):
    def reset(self):
        # Reset logic
        return observation, info
        
    def step(self, action):
        # Step logic
        return observation, reward, done, info
```

2. **Define Task**:

```python
class MyTask(Task):
    def __init__(self, ...):
        super().__init__(seed)
        self.env = MyEnvironment(...)
        
    def get_name(self):
        return "MyTask"
        
    def get_description(self):
        return "Description of what this task tests"
        
    def evaluate(self, simulation, num_episodes, max_steps):
        # Evaluation logic
        return TaskResult(...)
        
    def get_metrics(self):
        return {
            'metric1': 'Description',
            'metric2': 'Description'
        }
```

3. **Add to Suite**:

```python
suite = BenchmarkSuite(...)
suite.add_task(MyTask(...))
```

## Future Enhancements

Planned improvements to the framework:

- [ ] Sensorimotor control tasks
- [ ] Multi-modal integration tasks
- [ ] Information theory metrics
- [ ] Learning curve visualization
- [ ] Automated hyperparameter optimization
- [ ] Distributed benchmarking
- [ ] Online leaderboards

## See Also

- [Example Script](../examples/benchmark_example.py)
- [API Reference](API.md)
- [Architecture](ARCHITECTURE.md)
- [TODO List](../TODO.md)

---

*Last Updated: December 2025*
