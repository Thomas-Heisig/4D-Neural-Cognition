# Information Theory Metrics

This document describes the information theory metrics available in the 4D Neural Cognition system.

## Overview

The `metrics.py` module provides comprehensive information theory tools for analyzing neural network behavior. These metrics help quantify information flow, uncertainty, and dependencies in neural systems.

## Available Metrics

### 1. Entropy

Shannon entropy measures uncertainty in a distribution.

```python
from src.metrics import calculate_entropy

spike_counts = [10, 20, 15, 5]
entropy = calculate_entropy(spike_counts)
print(f"Entropy: {entropy:.3f} bits")
```

### 2. Mutual Information

Measures how much knowing one variable reduces uncertainty about another.

```python
from src.metrics import calculate_mutual_information

stimulus_types = [0, 1, 0, 1, 0, 1]
neural_responses = [10, 25, 12, 28, 9, 26]

mi = calculate_mutual_information(stimulus_types, neural_responses)
print(f"Mutual Information: {mi:.3f} bits")
```

### 3. Conditional Entropy

Measures remaining uncertainty in Y given X.

```python
from src.metrics import calculate_conditional_entropy

x_values = [0, 1, 0, 1, 0]
y_values = [1, 1, 0, 1, 0]

h_y_given_x = calculate_conditional_entropy(x_values, y_values)
print(f"H(Y|X): {h_y_given_x:.3f} bits")
```

### 4. Transfer Entropy

Measures directed information transfer from source to target.

```python
from src.metrics import calculate_transfer_entropy

source_signal = [0, 1, 1, 0, 1, 1, 0, 0]
target_signal = [0, 0, 1, 1, 0, 1, 1, 0]

te = calculate_transfer_entropy(source_signal, target_signal, target_history=1)
print(f"Transfer Entropy (Source→Target): {te:.3f} bits")
```

### 5. Information Gain

Measures reduction in entropy from a split (useful for feature importance).

```python
from src.metrics import calculate_information_gain

prior_counts = [50, 50]  # Before split
posterior_counts = [[40, 10], [10, 40]]  # After split by feature

ig = calculate_information_gain(prior_counts, posterior_counts)
print(f"Information Gain: {ig:.3f} bits")
```

### 6. Joint Entropy

Measures uncertainty in joint distribution of two variables.

```python
from src.metrics import calculate_joint_entropy

x_values = [0, 1, 0, 1]
y_values = [0, 0, 1, 1]

h_xy = calculate_joint_entropy(x_values, y_values)
print(f"H(X,Y): {h_xy:.3f} bits")
```

## Neural Network Analysis

### Analyzing Information Flow

```python
from src.metrics import calculate_transfer_entropy, calculate_mutual_information

# Analyze information flow between neuron populations
population_1_activity = [...]  # Activity of population 1
population_2_activity = [...]  # Activity of population 2

# Check if population 1 influences population 2
te_1_to_2 = calculate_transfer_entropy(population_1_activity, population_2_activity)
te_2_to_1 = calculate_transfer_entropy(population_2_activity, population_1_activity)

if te_1_to_2 > te_2_to_1:
    print("Information flows primarily from Population 1 to Population 2")
else:
    print("Information flows primarily from Population 2 to Population 1")
```

### Measuring Stimulus Encoding

```python
from src.metrics import calculate_mutual_information

# Measure how well neurons encode stimulus
stimulus_labels = []
neuron_spike_counts = []

for trial in trials:
    stimulus_labels.append(trial.stimulus_type)
    neuron_spike_counts.append(trial.neuron_spikes)

mi = calculate_mutual_information(stimulus_labels, neuron_spike_counts)
print(f"Stimulus encoding: {mi:.3f} bits")
```

## Mathematical Background

### Entropy
H(X) = -Σ p(x) log₂ p(x)

### Mutual Information
I(X;Y) = H(X) + H(Y) - H(X,Y)

### Conditional Entropy
H(Y|X) = H(X,Y) - H(X)

### Transfer Entropy
TE(X→Y) = I(Y_future; X_past | Y_past)

## Best Practices

1. **Sufficient Sample Size**: Ensure enough data points for reliable estimates
2. **Discretization**: For continuous variables, discretize appropriately
3. **Bias Correction**: Consider bias for small samples
4. **Interpretation**: Higher entropy = more uncertainty/randomness

## See Also

- [Network Analysis Tools](../user-guide/NETWORK_ANALYSIS.md)
- [Model Comparison](MODEL_COMPARISON.md)
- [Metrics Module](../api/METRICS.md)
