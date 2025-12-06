# Examples

This directory contains example scripts demonstrating various features of the 4D Neural Cognition system.

## Available Examples

### simple_test.py

**Purpose**: Basic integration tests to verify the Tasks & Evaluation framework works correctly.

**What it tests**:
- Knowledge database creation and population
- Pre-training from database
- Simple benchmark run with pattern classification

**Usage**:
```bash
cd examples
python3 simple_test.py
```

**Expected output**: All tests should pass, demonstrating:
- Database contains training data
- Network can pre-train from database
- Benchmark suite runs successfully

### benchmark_example.py

**Purpose**: Comprehensive demonstration of the benchmark and evaluation system.

**Features demonstrated**:
1. Single benchmark run
2. Configuration comparison
3. Knowledge database usage
4. Custom task creation

**Usage**:
```bash
cd examples
python3 benchmark_example.py
```

**Note**: This is an interactive script that pauses between examples. Press Enter to continue through each example.

## Quick Start

To get started with the examples:

1. Install dependencies:
```bash
pip install -r ../requirements.txt
```

2. Run the simple test to verify everything works:
```bash
python3 simple_test.py
```

3. Explore the comprehensive benchmark examples:
```bash
python3 benchmark_example.py
```

## Understanding the Output

### Benchmark Results

When you run benchmarks, you'll see output like:

```
Running task: PatternClassification-4class
Description: Classify visual patterns into 4 classes...

Results:
  Accuracy: 0.7500
  Reward: 0.7500
  Reaction Time: 12.50
  Stability: 0.8234
  Execution Time: 45.20s
```

**Metrics explained**:
- **Accuracy**: Proportion of correct classifications (0-1)
- **Reward**: Average reward received per episode
- **Reaction Time**: Average simulation steps to first response
- **Stability**: Consistency of reaction times (1 = very stable)
- **Execution Time**: Wall-clock time to complete the task

### Configuration Comparison

When comparing configurations, you'll see:

```
BEST PERFORMERS BY TASK

Task: PatternClassification-4class
  accuracy              : baseline                    (0.7500)
  reward                : dense_network               (0.8000)
  reaction_time         : stronger_weights            (10.2000)
```

This shows which configuration performed best on each metric for each task.

## Creating Your Own Examples

To create a custom example:

1. **Import required modules**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from tasks import PatternClassificationTask
from evaluation import BenchmarkConfig, BenchmarkSuite
```

2. **Create a configuration**:
```python
config = BenchmarkConfig(
    name="my_config",
    description="My custom configuration",
    config_path="../brain_base_model.json",
    seed=42,
    initialization_params={
        'area_names': ['V1_like'],
        'density': 0.1,
        'connection_probability': 0.01
    }
)
```

3. **Create and run a benchmark**:
```python
suite = BenchmarkSuite(name="My Suite")
suite.add_task(PatternClassificationTask(num_classes=4, seed=42))
results = suite.run(config, output_dir="../results")
```

## Tips

### Performance

- Start with small networks (low density, few connections) for faster testing
- Increase network size once your configuration is working
- Use fewer episodes/steps during development

### Reproducibility

- Always set seeds for reproducibility
- Save all results to output directories
- Document your configuration changes

### Debugging

If a benchmark fails:
1. Check the error message
2. Verify your configuration file path is correct
3. Try with a simpler task first
4. Reduce network size to isolate the issue

## See Also

- [Tasks & Evaluation Documentation](../docs/TASKS_AND_EVALUATION.md)
- [API Reference](../docs/API.md)
- [Main README](../README.md)

---

*For questions or issues, please refer to the main documentation or open an issue on GitHub.*
