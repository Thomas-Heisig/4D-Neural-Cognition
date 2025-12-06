# Quick Start: Tasks & Evaluation

This guide will get you started with the Tasks & Evaluation framework in 5 minutes.

## 1. Run a Simple Benchmark (30 seconds)

```python
from src.evaluation import BenchmarkConfig, create_standard_benchmark_suite

# Define your configuration
config = BenchmarkConfig(
    name="my_first_test",
    description="My first benchmark test",
    config_path="brain_base_model.json",
    seed=42,
    initialization_params={
        'area_names': ['V1_like'],
        'density': 0.1,
        'connection_probability': 0.01
    }
)

# Run standard benchmarks
suite = create_standard_benchmark_suite()
results = suite.run(config, output_dir="./my_results")

# View results
for r in results:
    print(f"{r.task_name}: Accuracy={r.task_result.accuracy:.2%}")
```

**Output**: Accuracy scores for pattern classification and sequence tasks.

## 2. Compare Configurations (1 minute)

```python
from src.evaluation import run_configuration_comparison

configs = [
    BenchmarkConfig(name="sparse", ..., initialization_params={'density': 0.1}),
    BenchmarkConfig(name="dense", ..., initialization_params={'density': 0.2}),
]

report = run_configuration_comparison(configs, output_dir="./comparison")
```

**Output**: Which configuration performs best on each task.

## 3. Use Knowledge Database (2 minutes)

```python
from src.knowledge_db import populate_sample_knowledge, KnowledgeDatabase, KnowledgeBasedTrainer
from src.brain_model import BrainModel
from src.simulation import Simulation

# Setup database
populate_sample_knowledge("knowledge.db")
db = KnowledgeDatabase("knowledge.db")

# Create network
model = BrainModel(config_path="brain_base_model.json")
sim = Simulation(model, seed=42)
sim.initialize_neurons(area_names=['V1_like'], density=0.1)
sim.initialize_random_synapses(connection_probability=0.01)

# Pre-train
trainer = KnowledgeBasedTrainer(sim, db)
stats = trainer.pretrain(category='pattern_recognition', num_samples=50)

print(f"Pre-trained with {stats['samples_processed']} samples")
```

**Output**: Network pre-trained with 50 pattern examples from database.

## 4. Create Custom Task (5 minutes)

```python
from src.tasks import Task, TaskResult, Environment
import numpy as np

class MyTask(Task):
    def get_name(self):
        return "MyCustomTask"
    
    def get_description(self):
        return "Tests my specific use case"
    
    def evaluate(self, simulation, num_episodes=10, max_steps=100):
        # Your evaluation logic here
        accuracy = 0.0  # Calculate from your task
        return TaskResult(accuracy=accuracy, reward=0.0)
    
    def get_metrics(self):
        return {'accuracy': 'Task-specific accuracy'}

# Use it
task = MyTask(seed=42)
result = task.evaluate(my_simulation)
```

**Output**: Custom task integrated into benchmark framework.

## Common Use Cases

### Use Case 1: Test if Changes Improved Performance

```python
# Before changes
baseline = BenchmarkConfig(name="before", ...)
results_before = suite.run(baseline)

# After changes
improved = BenchmarkConfig(name="after", ...)
results_after = suite.run(improved)

# Compare
report = run_configuration_comparison([baseline, improved])
```

### Use Case 2: Find Best Configuration

```python
configs = [
    BenchmarkConfig(name="config_1", ...),
    BenchmarkConfig(name="config_2", ...),
    BenchmarkConfig(name="config_3", ...),
]

report = run_configuration_comparison(configs)
# Check report['best_performers'] for winners
```

### Use Case 3: Pre-train Before Testing

```python
# 1. Pre-train from database
trainer.pretrain(category='pattern_recognition', num_samples=100)

# 2. Then evaluate
results = task.evaluate(simulation)
```

### Use Case 4: Monitor Learning Progress

```python
# Evaluate at different training stages
results_0 = task.evaluate(sim)  # Untrained

trainer.pretrain(num_samples=50)
results_50 = task.evaluate(sim)  # After 50 samples

trainer.pretrain(num_samples=50)
results_100 = task.evaluate(sim)  # After 100 samples

# Compare accuracies
print(f"Untrained: {results_0.accuracy:.2%}")
print(f"50 samples: {results_50.accuracy:.2%}")
print(f"100 samples: {results_100.accuracy:.2%}")
```

## Key Concepts

### Environment
Provides observations and rewards, similar to OpenAI Gym.

### Task
Wraps an environment and defines how to evaluate it (metrics).

### BenchmarkConfig
Defines a complete configuration (network params, seeds, etc.) for reproducibility.

### BenchmarkSuite
Collection of tasks to run on a configuration.

### Knowledge Database
SQLite database storing training examples that networks can learn from.

## Tips for Success

1. **Always use seeds**: `seed=42` for reproducibility
2. **Start small**: Low density, few episodes during development
3. **Save results**: Always specify `output_dir`
4. **Compare fairly**: Use same suite for all configs
5. **Pre-train first**: For better baseline performance

## Next Steps

- Read [full documentation](TASKS_AND_EVALUATION.md)
- Try [example scripts](../examples/)
- Create your own custom tasks
- Add more training data to knowledge database

## Troubleshooting

**Problem**: Benchmark is too slow  
**Solution**: Reduce `density`, `connection_probability`, or `num_episodes`

**Problem**: Results not reproducible  
**Solution**: Set `seed` parameter in all configs and tasks

**Problem**: Low accuracy on all tasks  
**Solution**: Try pre-training from knowledge database first

**Problem**: Import errors  
**Solution**: Make sure you're in the correct directory and have installed requirements

## Quick Reference

```python
# Imports
from src.evaluation import BenchmarkConfig, BenchmarkSuite, create_standard_benchmark_suite
from src.tasks import PatternClassificationTask, TemporalSequenceTask
from src.knowledge_db import KnowledgeDatabase, KnowledgeBasedTrainer, populate_sample_knowledge

# Create config
config = BenchmarkConfig(name="...", config_path="...", seed=42, initialization_params={...})

# Run benchmark
suite = create_standard_benchmark_suite()
results = suite.run(config, output_dir="./results")

# Use database
populate_sample_knowledge("knowledge.db")
db = KnowledgeDatabase("knowledge.db")
trainer = KnowledgeBasedTrainer(simulation, db)
stats = trainer.pretrain(category='pattern_recognition', num_samples=100)

# Compare configs
from src.evaluation import run_configuration_comparison
report = run_configuration_comparison([config1, config2, config3])
```

---

*For more details, see [TASKS_AND_EVALUATION.md](TASKS_AND_EVALUATION.md)*
