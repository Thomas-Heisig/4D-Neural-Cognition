# Model Comparison and Ablation Studies

This document describes tools for comparing neural network models and performing ablation studies.

## Overview

The `model_comparison.py` module provides statistical tools for:
- Comparing multiple model configurations
- Statistical significance testing
- Performance benchmarking
- Ablation studies (component importance analysis)

## Model Comparison

### Basic Usage

```python
from src.model_comparison import ModelComparator, ModelResult

# Create comparator
comparator = ModelComparator()

# Add model results
result1 = ModelResult(
    model_name="Baseline",
    performance_metrics={"accuracy": 0.85, "f1_score": 0.82},
    training_time=120.5,
    inference_time=0.05,
    memory_usage=512.0,
    configuration={"neurons": 1000, "layers": 3}
)

result2 = ModelResult(
    model_name="Enhanced",
    performance_metrics={"accuracy": 0.92, "f1_score": 0.89},
    training_time=180.2,
    inference_time=0.08,
    memory_usage=768.0,
    configuration={"neurons": 2000, "layers": 4}
)

comparator.add_result(result1)
comparator.add_result(result2)

# Compare on specific metric
comparison = comparator.compare_performance("accuracy")
print(f"Best model: {comparison['best_model']}")
print(f"Best value: {comparison['best_value']:.3f}")
```

### Statistical Testing

```python
# Perform pairwise statistical comparison
# Requires multiple runs per model

model1_runs = [0.85, 0.87, 0.84, 0.86, 0.88]  # 5 runs
model2_runs = [0.91, 0.93, 0.90, 0.92, 0.94]  # 5 runs

result = comparator.pairwise_comparison(
    model1_name="Baseline",
    model2_name="Enhanced",
    metric_name="accuracy",
    model1_runs=model1_runs,
    model2_runs=model2_runs,
    test_type="t-test"  # Options: 't-test', 'wilcoxon', 'mann-whitney'
)

print(f"P-value: {result['p_value']:.4f}")
print(f"Significant: {result['significant']}")
print(f"Effect size: {result['effect_size']}")
print(f"Winner: {result['winner']}")
```

### Benchmarking

```python
# Benchmark computational efficiency
benchmark = comparator.benchmark_models(
    include_training_time=True,
    include_inference_time=True,
    include_memory=True
)

print(f"Fastest training: {benchmark['fastest_training']}")
print(f"Fastest inference: {benchmark['fastest_inference']}")
print(f"Most memory efficient: {benchmark['most_memory_efficient']}")
```

### Generate Report

```python
# Generate comprehensive comparison report
report = comparator.generate_comparison_report()
print(report)
```

## Ablation Studies

Ablation studies systematically remove or modify components to assess their importance.

### Basic Ablation Study

```python
from src.model_comparison import AblationStudy

# Define baseline configuration
baseline_config = {
    "num_neurons": 1000,
    "use_inhibition": True,
    "use_stdp": True,
    "use_homeostasis": True,
    "learning_rate": 0.01
}

# Create ablation study
study = AblationStudy(baseline_config)

# Test removing inhibition
ablated_config = study.ablate_component(
    component_name="Inhibitory Neurons",
    component_key="use_inhibition",
    ablation_value=False
)

# ... train and evaluate with ablated config ...
baseline_performance = 0.92
ablated_performance = 0.78

study.add_ablation_result(
    component_name="Inhibitory Neurons",
    baseline_performance=baseline_performance,
    ablated_performance=ablated_performance,
    metric_name="accuracy"
)
```

### Multiple Ablations

```python
# Test multiple components
components_to_test = [
    ("Inhibitory Neurons", "use_inhibition", False),
    ("STDP Plasticity", "use_stdp", False),
    ("Homeostasis", "use_homeostasis", False),
    ("Learning Rate", "learning_rate", 0.0)
]

for name, key, value in components_to_test:
    ablated_config = study.ablate_component(name, key, value)
    
    # Train and evaluate model with ablated config
    perf = train_and_evaluate(ablated_config)
    
    study.add_ablation_result(name, baseline_performance, perf, "accuracy")

# Rank components by importance
ranking = study.rank_component_importance("accuracy")

print("Component Importance Ranking:")
for i, result in enumerate(ranking, 1):
    print(f"{i}. {result['component']}: "
          f"Impact = {result['impact']:+.3f} ({result['relative_impact_pct']:+.1f}%)")
```

### Generate Ablation Report

```python
# Generate detailed ablation report
report = study.generate_ablation_report()
print(report)
```

## Advanced Features

### Bootstrap Confidence Intervals

```python
from src.model_comparison import bootstrap_confidence_interval

# Calculate confidence interval for model performance
performances = [0.85, 0.87, 0.84, 0.86, 0.88, 0.85, 0.87]
lower, upper = bootstrap_confidence_interval(performances, confidence_level=0.95)

print(f"95% CI: [{lower:.3f}, {upper:.3f}]")
```

### Cross-Validation Comparison

```python
from src.model_comparison import cross_validation_comparison

# Compare models using CV results
model_performances = {
    "Baseline": [0.84, 0.86, 0.85, 0.87, 0.84],  # 5-fold CV
    "Enhanced": [0.91, 0.93, 0.92, 0.94, 0.91],
    "Advanced": [0.95, 0.96, 0.94, 0.97, 0.95]
}

cv_results = cross_validation_comparison(model_performances, cv_folds=5)

for model, stats in cv_results.items():
    if model != "ranking":
        print(f"{model}: Mean = {stats['mean']:.3f} ± {stats['std']:.3f}")

print(f"Ranking: {cv_results['ranking']}")
```

## Complete Example

```python
from src.model_comparison import ModelComparator, ModelResult, AblationStudy
from src.brain_model import BrainModel
from src.evaluation import evaluate_model

# Initialize comparator
comparator = ModelComparator()

# Test different configurations
configs = [
    {"name": "Small", "neurons": 500, "inhibition": False},
    {"name": "Medium", "neurons": 1000, "inhibition": True},
    {"name": "Large", "neurons": 2000, "inhibition": True}
]

for config in configs:
    # Create and train model
    model = BrainModel(lattice_size=(config["neurons"]//100, 10, 10, 1))
    # ... training code ...
    
    # Evaluate
    metrics = evaluate_model(model, test_data)
    
    # Record result
    result = ModelResult(
        model_name=config["name"],
        performance_metrics=metrics,
        training_time=training_time,
        inference_time=inference_time,
        memory_usage=memory_usage,
        configuration=config
    )
    comparator.add_result(result)

# Generate comparison report
print(comparator.generate_comparison_report())

# Perform ablation study on best model
best_config = configs[2]  # Assume Large is best
ablation = AblationStudy(best_config)

# Test component importance
# ... ablation experiments ...

print(ablation.generate_ablation_report())
```

## Best Practices

1. **Multiple Runs**: Always use multiple runs for statistical significance
2. **Same Conditions**: Keep evaluation conditions identical across models
3. **Appropriate Tests**: Choose statistical tests based on data distribution
4. **Document Everything**: Record all configurations and hyperparameters
5. **Systematic Ablation**: Test one component at a time

## See Also

- [Information Theory Metrics](INFORMATION_THEORY.md)
- [Reinforcement Learning](REINFORCEMENT_LEARNING.md)
- [Evaluation Framework](../tutorials/EVALUATION.md)
