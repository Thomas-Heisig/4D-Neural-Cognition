# Model Comparison Framework

This document describes the tools and methods for comparing different configurations of the 4D Neural Cognition model.

## Table of Contents
- [Overview](#overview)
- [Configuration Comparison](#configuration-comparison)
- [Statistical Analysis](#statistical-analysis)
- [Visualization Tools](#visualization-tools)
- [Best Practices](#best-practices)
- [Examples](#examples)

---

## Overview

The model comparison framework allows you to:
- Compare multiple model configurations objectively
- Identify best-performing configurations for specific tasks
- Perform statistical significance testing
- Visualize performance differences
- Track reproducibility across runs

### Key Components

1. **BenchmarkConfig**: Defines a model configuration
2. **BenchmarkSuite**: Collection of evaluation tasks
3. **ConfigurationComparator**: Compares multiple configurations
4. **Statistical Tests**: Validates significance of differences

---

## Configuration Comparison

### Basic Usage

```python
from src.evaluation import (
    BenchmarkConfig, 
    BenchmarkSuite,
    ConfigurationComparator,
    create_standard_benchmark_suite
)

# Define configurations to compare
configs = [
    BenchmarkConfig(
        name="baseline",
        description="Standard LIF neurons",
        config_path="configs/baseline.json",
        seed=42,
        initialization_params={"neuron_model": "lif"}
    ),
    BenchmarkConfig(
        name="izhikevich",
        description="Izhikevich neurons",
        config_path="configs/izhikevich.json",
        seed=42,
        initialization_params={"neuron_model": "izhikevich"}
    ),
    BenchmarkConfig(
        name="enhanced_plasticity",
        description="STDP with homeostasis",
        config_path="configs/stdp.json",
        seed=42,
        initialization_params={
            "plasticity": "stdp",
            "homeostatic": True
        }
    )
]

# Create benchmark suite
suite = create_standard_benchmark_suite()

# Run all configurations
comparator = ConfigurationComparator(output_dir="./comparison_results")

for config in configs:
    print(f"\nEvaluating {config.name}...")
    results = suite.run(config)
    comparator.add_results(config.name, results)

# Generate comparison report
report = comparator.compare()
comparator.print_summary(report)
```

### Comparison Metrics

The framework automatically tracks:

**Performance Metrics:**
- Accuracy (higher is better)
- Reward (higher is better)
- Stability (higher is better)
- Reaction time (lower is better)
- Execution time (lower is better)

**Statistical Metrics:**
- Mean performance across tasks
- Standard deviation (variability)
- Best/worst case performance
- Consistency score

### Report Structure

```python
{
    "timestamp": "2025-12-09 10:30:00",
    "num_configs": 3,
    "config_names": ["baseline", "izhikevich", "enhanced_plasticity"],
    "task_comparisons": {
        "PatternClassification": {
            "baseline": {
                "accuracy": 0.78,
                "reward": 0.65,
                "reaction_time": 45.2,
                "stability": 0.82
            },
            "izhikevich": {
                "accuracy": 0.81,
                "reward": 0.70,
                "reaction_time": 42.1,
                "stability": 0.85
            }
        }
    },
    "best_performers": {
        "PatternClassification": {
            "accuracy": {"config": "izhikevich", "value": 0.81},
            "reward": {"config": "izhikevich", "value": 0.70}
        }
    }
}
```

---

## Statistical Analysis

### Statistical Significance Testing

Use statistical tests to determine if performance differences are significant or due to chance.

```python
import numpy as np
from scipy import stats

def compare_configurations_statistical(results_a, results_b, metric='accuracy'):
    """
    Compare two configurations with statistical significance testing.
    
    Args:
        results_a: List of BenchmarkResult for configuration A
        results_b: List of BenchmarkResult for configuration B
        metric: Metric to compare ('accuracy', 'reward', etc.)
    
    Returns:
        Dictionary with test results
    """
    # Extract metric values
    values_a = [getattr(r.task_result, metric) for r in results_a]
    values_b = [getattr(r.task_result, metric) for r in results_b]
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(values_a, values_b)
    
    # Calculate effect size (Cohen's d)
    mean_diff = np.mean(values_a) - np.mean(values_b)
    pooled_std = np.sqrt((np.var(values_a) + np.var(values_b)) / 2)
    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
    
    return {
        'mean_a': np.mean(values_a),
        'mean_b': np.mean(values_b),
        'std_a': np.std(values_a),
        'std_b': np.std(values_b),
        't_statistic': t_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'cohens_d': cohens_d,
        'effect_size': 'large' if abs(cohens_d) > 0.8 else 
                      'medium' if abs(cohens_d) > 0.5 else 'small'
    }

# Example usage
stats_result = compare_configurations_statistical(
    results_baseline, 
    results_izhikevich,
    metric='accuracy'
)

print(f"Mean difference: {stats_result['mean_a'] - stats_result['mean_b']:.3f}")
print(f"P-value: {stats_result['p_value']:.4f}")
print(f"Significant: {stats_result['significant']}")
print(f"Effect size: {stats_result['effect_size']} (d={stats_result['cohens_d']:.2f})")
```

### Multiple Comparisons Correction

When comparing multiple configurations, apply correction to control false positive rate:

```python
from scipy.stats import false_discovery_control

def multiple_comparison_correction(p_values, alpha=0.05, method='fdr_bh'):
    """
    Apply multiple comparison correction.
    
    Args:
        p_values: List of p-values from comparisons
        alpha: Significance level
        method: 'bonferroni' or 'fdr_bh' (Benjamini-Hochberg)
    
    Returns:
        Adjusted significance decisions
    """
    if method == 'bonferroni':
        # Bonferroni correction (conservative)
        adjusted_alpha = alpha / len(p_values)
        return [p < adjusted_alpha for p in p_values]
    
    elif method == 'fdr_bh':
        # Benjamini-Hochberg (less conservative)
        rejected = false_discovery_control(p_values, alpha=alpha)
        return rejected
    
    else:
        raise ValueError(f"Unknown method: {method}")

# Example: Compare 5 configurations (10 pairwise comparisons)
p_values = [0.03, 0.01, 0.15, 0.002, 0.08, 0.12, 0.04, 0.20, 0.05, 0.09]

bonferroni_results = multiple_comparison_correction(p_values, method='bonferroni')
fdr_results = multiple_comparison_correction(p_values, method='fdr_bh')

print(f"Bonferroni: {sum(bonferroni_results)}/{len(p_values)} significant")
print(f"FDR: {sum(fdr_results)}/{len(p_values)} significant")
```

### Confidence Intervals

Calculate confidence intervals for performance metrics:

```python
def bootstrap_confidence_interval(results, metric='accuracy', n_bootstrap=1000, confidence=0.95):
    """
    Calculate bootstrap confidence interval for a metric.
    
    Args:
        results: List of BenchmarkResult objects
        metric: Metric to analyze
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level (e.g., 0.95 for 95%)
    
    Returns:
        Dictionary with mean and confidence interval
    """
    values = [getattr(r.task_result, metric) for r in results]
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentiles
    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_means, alpha * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return {
        'mean': np.mean(values),
        'ci_lower': lower,
        'ci_upper': upper,
        'confidence': confidence
    }

# Example usage
ci_result = bootstrap_confidence_interval(results_baseline, metric='accuracy')
print(f"Mean: {ci_result['mean']:.3f}")
print(f"95% CI: [{ci_result['ci_lower']:.3f}, {ci_result['ci_upper']:.3f}]")
```

---

## Visualization Tools

### Performance Comparison Plots

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_performance_comparison(comparator_report, metric='accuracy'):
    """
    Create bar plot comparing configurations across tasks.
    
    Args:
        comparator_report: Report from ConfigurationComparator
        metric: Metric to plot
    """
    task_comparisons = comparator_report['task_comparisons']
    config_names = comparator_report['config_names']
    
    # Organize data
    tasks = list(task_comparisons.keys())
    n_tasks = len(tasks)
    n_configs = len(config_names)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_tasks)
    width = 0.8 / n_configs
    
    # Plot bars for each configuration
    for i, config_name in enumerate(config_names):
        values = [task_comparisons[task][config_name][metric] 
                 for task in tasks]
        offset = (i - n_configs/2 + 0.5) * width
        ax.bar(x + offset, values, width, label=config_name)
    
    # Formatting
    ax.set_xlabel('Task')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} Comparison Across Tasks')
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{metric}.png', dpi=300)
    plt.close()

def plot_radar_chart(comparator_report, config_name):
    """
    Create radar chart showing configuration performance across metrics.
    
    Args:
        comparator_report: Report from ConfigurationComparator
        config_name: Configuration to visualize
    """
    import matplotlib.pyplot as plt
    from math import pi
    
    # Get data for one task
    task_name = list(comparator_report['task_comparisons'].keys())[0]
    data = comparator_report['task_comparisons'][task_name][config_name]
    
    # Metrics to plot
    metrics = ['accuracy', 'reward', 'stability']
    values = [data[m] for m in metrics]
    
    # Add first point at end to close the circle
    metrics = metrics + [metrics[0]]
    values = values + [values[0]]
    
    # Angles for each metric
    angles = [n / float(len(metrics)-1) * 2 * pi for n in range(len(metrics))]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics[:-1])
    ax.set_ylim(0, 1)
    ax.set_title(f'{config_name} Performance Profile', pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'radar_{config_name}.png', dpi=300)
    plt.close()

def plot_heatmap_comparison(comparator_report):
    """
    Create heatmap showing all configurations vs all tasks.
    
    Args:
        comparator_report: Report from ConfigurationComparator
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    task_comparisons = comparator_report['task_comparisons']
    config_names = comparator_report['config_names']
    tasks = list(task_comparisons.keys())
    
    # Create matrix
    matrix = np.zeros((len(config_names), len(tasks)))
    for i, config_name in enumerate(config_names):
        for j, task_name in enumerate(tasks):
            matrix[i, j] = task_comparisons[task_name][config_name]['accuracy']
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=tasks, yticklabels=config_names,
                vmin=0, vmax=1, cbar_kws={'label': 'Accuracy'})
    ax.set_title('Configuration Performance Heatmap')
    
    plt.tight_layout()
    plt.savefig('heatmap_comparison.png', dpi=300)
    plt.close()
```

---

## Best Practices

### Configuration Design

1. **Control Variables**: Change one aspect at a time for clear attribution
2. **Use Seeds**: Set random seeds for reproducibility
3. **Multiple Runs**: Run each configuration multiple times for statistics
4. **Document Changes**: Clearly describe what differs between configs

### Statistical Testing

1. **Pre-register Hypotheses**: Decide comparisons before running experiments
2. **Correction for Multiple Comparisons**: Use when testing many configurations
3. **Report Effect Sizes**: P-values alone can be misleading
4. **Check Assumptions**: Verify normality for parametric tests

### Reporting Results

1. **Include Variability**: Report mean ± std or confidence intervals
2. **Show Raw Data**: Include individual run results, not just averages
3. **Visualize**: Use plots to make comparisons clear
4. **Document**: Save configuration files and random seeds

---

## Examples

### Example 1: Neuron Model Comparison

Compare LIF vs Izhikevich vs Hodgkin-Huxley models:

```python
# Define configurations
configs = [
    BenchmarkConfig(name="lif", config_path="lif.json", seed=42,
                   initialization_params={"neuron_model": "lif"}),
    BenchmarkConfig(name="izhikevich", config_path="izhikevich.json", seed=42,
                   initialization_params={"neuron_model": "izhikevich"}),
    BenchmarkConfig(name="hodgkin_huxley", config_path="hh.json", seed=42,
                   initialization_params={"neuron_model": "hodgkin_huxley"})
]

# Run comparison (code as shown above)
# ...

# Statistical analysis
print("\nStatistical Comparison (LIF vs Izhikevich):")
stats = compare_configurations_statistical(
    results_lif, results_izhikevich, metric='accuracy'
)
print(f"  Δ mean: {stats['mean_a'] - stats['mean_b']:.3f}")
print(f"  p-value: {stats['p_value']:.4f} {'*' if stats['significant'] else ''}")
print(f"  Effect: {stats['effect_size']}")
```

### Example 2: Plasticity Rule Comparison

Compare Hebbian vs STDP vs Homeostatic plasticity:

```python
configs = [
    BenchmarkConfig(name="hebbian", config_path="hebbian.json", seed=42,
                   initialization_params={"plasticity": "hebbian"}),
    BenchmarkConfig(name="stdp", config_path="stdp.json", seed=42,
                   initialization_params={"plasticity": "stdp"}),
    BenchmarkConfig(name="homeostatic", config_path="homeostatic.json", seed=42,
                   initialization_params={"plasticity": "hebbian", 
                                        "homeostatic": True})
]

# Run and analyze...
```

### Example 3: Ablation Study

Systematically remove features to understand their contribution:

```python
configs = [
    BenchmarkConfig(name="full", config_path="full.json", seed=42,
                   initialization_params={"all_features": True}),
    BenchmarkConfig(name="no_stdp", config_path="no_stdp.json", seed=42,
                   initialization_params={"stdp": False}),
    BenchmarkConfig(name="no_homeostasis", config_path="no_homeo.json", seed=42,
                   initialization_params={"homeostatic": False}),
    BenchmarkConfig(name="no_neuromodulation", config_path="no_neuromod.json", seed=42,
                   initialization_params={"neuromodulation": False})
]

# This reveals which features contribute most to performance
```

---

## Future Enhancements

### Planned Features

1. **Automated Hyperparameter Search**
   - Grid search over parameter space
   - Random search for high-dimensional spaces
   - Bayesian optimization for efficient search

2. **Multi-Objective Optimization**
   - Pareto frontier identification
   - Trade-off analysis (speed vs accuracy)
   - Visualization of trade-offs

3. **Cross-Validation**
   - K-fold cross-validation for robustness
   - Stratified sampling for balanced tasks
   - Leave-one-out for small datasets

4. **Meta-Analysis**
   - Aggregate results across multiple studies
   - Publication bias detection
   - Effect size meta-analysis

---

## References

1. **Statistical Methods:**
   - Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Erlbaum.
   - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society*, 57(1), 289-300.

2. **Model Comparison:**
   - Busemeyer, J. R., & Diederich, A. (2010). *Cognitive modeling*. Sage.
   - Wagenmakers, E. J., & Farrell, S. (2004). AIC model selection using Akaike weights. *Psychonomic Bulletin & Review*, 11(1), 192-196.

3. **Experimental Design:**
   - Montgomery, D. C. (2017). *Design and analysis of experiments* (9th ed.). Wiley.

---

*Last Updated: December 2025*
*Maintained by: Project Contributors*

**Note**: This framework is continuously evolving. Contributions welcome!
