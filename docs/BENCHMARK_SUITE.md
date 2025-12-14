# Benchmark Suite Documentation

## Overview

This document describes the comprehensive benchmark suite for the 4D Neural Cognition project, providing standardized tests for measuring network performance, comparing configurations, and validating scientific hypotheses.

**Author**: Thomas Heisig  
**Contact**: t_heisig@gmx.de  
**Location**: Ganderkesee, Germany  
**Last Updated**: December 2025

---

## Table of Contents

- [Introduction](#introduction)
- [Benchmark Categories](#benchmark-categories)
- [Standardized Datasets](#standardized-datasets)
- [Performance Metrics](#performance-metrics)
- [Comparison Framework](#comparison-framework)
- [Running Benchmarks](#running-benchmarks)
- [Contributing Benchmarks](#contributing-benchmarks)

---

## Introduction

### Purpose

The benchmark suite serves multiple purposes:

1. **Scientific Validation**: Test formal hypotheses (see `SCIENTIFIC_HYPOTHESES.md`)
2. **Performance Comparison**: Compare 4D networks against baselines
3. **Configuration Optimization**: Identify best hyperparameters
4. **Regression Testing**: Ensure updates don't degrade performance
5. **Community Standards**: Enable reproducible comparisons

### Design Principles

- **Reproducibility**: Fixed random seeds, versioned datasets
- **Standardization**: Common metrics, protocols, reporting formats
- **Extensibility**: Easy to add new benchmarks
- **Efficiency**: Fast execution for continuous integration
- **Comprehensiveness**: Cover all major capabilities

---

## Benchmark Categories

### 1. Cognitive Performance Benchmarks

#### 1.1 Spatial Reasoning

**Task**: Navigate a 3D maze to find hidden target

**Metrics**:
- Success rate (%)
- Average steps to target
- Sample efficiency (episodes to 80% success)

**Baseline Comparisons**:
- Random policy
- Q-learning with spatial memory
- Recurrent neural network (LSTM)

**Implementation**: `src/ki_benchmarks/spatial_reasoning.py`

**Dataset**: Generated mazes (10×10×10 grids, varying complexity)

---

#### 1.2 Temporal Pattern Memory

**Task**: Learn and recall complex temporal sequences

**Metrics**:
- Sequence recall accuracy (%)
- Maximum sequence length learned
- Generalization to novel sequences

**Baseline Comparisons**:
- LSTM (matched parameters)
- Transformer (small, matched FLOPs)
- Echo State Network

**Implementation**: `src/ki_benchmarks/temporal_memory.py`

**Datasets**:
- Sequential MNIST
- Copy task (Graves et al., 2014)
- Music sequences (MAESTRO subset)

---

#### 1.3 Cross-Modal Association

**Task**: Associate visual and auditory patterns

**Metrics**:
- Cross-modal retrieval accuracy (%)
- Modality transfer efficiency
- Zero-shot generalization

**Baseline Comparisons**:
- Late fusion (concatenation + MLP)
- Early fusion (shared encoder)
- Cross-attention mechanisms

**Implementation**: `src/ki_benchmarks/multimodal.py`

**Datasets**:
- Audio-Visual MNIST
- Speech-Image pairs
- Custom 4D multimodal dataset

---

### 2. Biological Plausibility Benchmarks

#### 2.1 Neural Dynamics Validation

**Task**: Validate emergent activity patterns against neuroscience data

**Metrics**:
- Traveling wave velocity (m/s)
- Oscillation frequencies (Hz)
- Branching parameter λ
- Correlation time constants (ms)

**Baseline Comparisons**:
- Published experimental data
- NEST simulator
- Brian2 simulator

**Implementation**: `src/ki_benchmarks/biological_validation.py`

**Reference Data**:
- MEA recordings (publicly available datasets)
- Literature values with confidence intervals

---

#### 2.2 Plasticity Dynamics

**Task**: Validate synaptic plasticity produces biological weight distributions

**Metrics**:
- Weight distribution shape (KS-test vs. log-normal)
- Stabilization time (iterations)
- Correlation with activity patterns

**Baseline Comparisons**:
- Loewenstein et al. (2011) experimental data
- Morrison et al. (2008) STDP models

**Implementation**: `src/ki_benchmarks/plasticity_validation.py`

**Analysis**:
- Statistical distribution tests
- Moment matching
- Temporal evolution tracking

---

### 3. Learning Efficiency Benchmarks

#### 3.1 Sample Efficiency

**Task**: Measure learning speed on supervised tasks

**Metrics**:
- Samples to 80% accuracy
- Samples to 95% accuracy
- Learning curve shape (AUC)

**Baseline Comparisons**:
- Standard MLP (matched parameters)
- Convolutional network
- Spiking neural network (BindsNET)

**Implementation**: `src/ki_benchmarks/sample_efficiency.py`

**Datasets**:
- MNIST (standard)
- Fashion-MNIST
- CIFAR-10 (downsampled)

---

#### 3.2 Continual Learning

**Task**: Sequential task learning without catastrophic forgetting

**Metrics**:
- Accuracy retention on task A after learning B (%)
- Forward transfer (B performance improvement)
- Backward transfer (A performance after B)

**Baseline Comparisons**:
- Standard network (no protection)
- Elastic Weight Consolidation (EWC)
- Progressive Neural Networks

**Implementation**: `src/ki_benchmarks/continual_learning.py`

**Datasets**:
- Split MNIST (5 tasks)
- Permuted MNIST (10 tasks)
- Sequential domains (MNIST → Fashion → CIFAR)

---

#### 3.3 Transfer Learning

**Task**: Measure knowledge transfer between related tasks

**Metrics**:
- Transfer efficiency (% reduction in training time)
- Fine-tuning convergence speed
- Zero-shot performance on related tasks

**Baseline Comparisons**:
- Random initialization
- ImageNet pre-training (for visual tasks)
- Task-specific training

**Implementation**: `src/ki_benchmarks/transfer_learning.py`

**Task Families**:
- Digit recognition variants
- Spatial reasoning in different environments
- Sequence prediction with shifted distributions

---

### 4. Scalability Benchmarks

#### 4.1 Computational Performance

**Task**: Measure simulation speed and resource usage

**Metrics**:
- Simulation time per timestep (ms)
- Memory usage (GB)
- Neurons per second throughput
- GPU utilization (%)

**Comparisons**:
- Different backends (NumPy, JAX, custom)
- CPU vs. GPU
- Different network sizes

**Implementation**: `src/ki_benchmarks/performance.py`

**Test Configurations**:
- Small: 1K neurons
- Medium: 10K neurons
- Large: 100K neurons
- Extra-large: 1M neurons

---

#### 4.2 Parallel Scaling

**Task**: Measure scaling efficiency with multiple cores/GPUs

**Metrics**:
- Speedup factor vs. core count
- Parallel efficiency (%)
- Communication overhead

**Implementation**: `src/ki_benchmarks/parallel_scaling.py`

**Test Scenarios**:
- 1, 2, 4, 8, 16 CPU cores
- 1, 2, 4 GPUs (if available)
- Distributed simulation (future)

---

### 5. Emergent Behavior Benchmarks

#### 5.1 Self-Organization

**Task**: Measure emergence of functional specialization

**Metrics**:
- Modularity index Q
- Functional clustering
- Receptive field analysis

**Analysis Methods**:
- Graph community detection
- Mutual information analysis
- Dimensionality reduction (t-SNE, UMAP)

**Implementation**: `src/ki_benchmarks/self_organization.py`

---

#### 5.2 Criticality Analysis

**Task**: Measure network operating regime

**Metrics**:
- Branching parameter λ
- Avalanche size distribution
- Power-law exponent
- Dynamic range

**Baseline Comparisons**:
- Theoretical critical state (λ=1.0)
- Subcritical (λ<1.0)
- Supercritical (λ>1.0)

**Implementation**: `src/ki_benchmarks/criticality.py`

---

## Standardized Datasets

### 4D-MNIST

**Description**: MNIST digits extended to 4D space

**Structure**:
- Spatial: (x, y) pixel coordinates
- z-dimension: Multiple views/rotations
- w-dimension: Temporal evolution or abstraction level

**Size**: 60,000 training, 10,000 test samples

**Format**: HDF5 with metadata

**Location**: `data/4D-MNIST/`

**Generation Script**: `scripts/generate_4d_mnist.py`

---

### Multimodal Audio-Visual Dataset

**Description**: Paired audio-visual samples for cross-modal learning

**Modalities**:
- Visual: 32×32 images
- Audio: Spectrograms (128 frequency bins × 32 time steps)
- Digital: 16-bit feature vectors

**Size**: 50,000 training, 10,000 validation, 10,000 test

**Format**: HDF5 with aligned modalities

**Location**: `data/multimodal-av/`

---

### Spatial Navigation Environments

**Description**: 3D maze environments for spatial reasoning

**Variants**:
- Simple: 5×5×5 grid, direct path
- Medium: 10×10×10 grid, obstacles
- Complex: 20×20×10 grid, multi-level

**Format**: JSON configuration files

**Location**: `data/spatial-nav/`

---

### Temporal Sequence Benchmarks

**Description**: Standardized sequence learning tasks

**Tasks**:
- Copy task: Memorize and reproduce sequences
- Repeat copy: Multiple repetitions
- Associative recall: Cued recall of sequences

**Format**: NumPy arrays

**Location**: `data/temporal-seq/`

---

## Performance Metrics

### Classification Metrics

```python
{
    "accuracy": float,           # Overall accuracy
    "precision": float,          # Micro-averaged precision
    "recall": float,             # Micro-averaged recall
    "f1_score": float,           # Micro-averaged F1
    "confusion_matrix": array,   # Full confusion matrix
    "per_class_metrics": dict    # Class-wise breakdown
}
```

### Regression Metrics

```python
{
    "mse": float,                # Mean squared error
    "rmse": float,               # Root mean squared error
    "mae": float,                # Mean absolute error
    "r2_score": float,           # R² coefficient
    "explained_variance": float  # Explained variance
}
```

### Learning Efficiency Metrics

```python
{
    "samples_to_threshold": int,     # Samples to reach accuracy threshold
    "convergence_speed": float,      # Learning rate (acc/sample)
    "sample_efficiency": float,      # Relative to baseline
    "learning_curve_auc": float,     # Area under learning curve
    "final_performance": float       # Final accuracy/error
}
```

### Biological Metrics

```python
{
    "branching_parameter": float,    # λ value
    "oscillation_frequencies": list, # Dominant frequencies (Hz)
    "wave_velocity": float,          # Propagation speed (m/s)
    "correlation_time": float,       # Auto-correlation decay (ms)
    "criticality_score": float       # Distance from critical state
}
```

### Computational Metrics

```python
{
    "simulation_time": float,        # Seconds per epoch
    "memory_usage": float,           # Peak memory (GB)
    "throughput": float,             # Neurons × timesteps / second
    "efficiency": float,             # vs. theoretical maximum
    "scalability": float             # Speedup with parallelization
}
```

---

## Comparison Framework

### Baseline Networks

#### Standard MLP
- Architecture: Matched parameter count
- Training: Adam optimizer, standard hyperparameters
- Purpose: Non-spiking baseline

#### LSTM Network
- Architecture: Matched state dimensions
- Training: Backpropagation through time
- Purpose: Temporal processing baseline

#### Spiking Neural Network
- Simulator: Brian2 or BindsNET
- Model: LIF neurons with STDP
- Purpose: Spiking network baseline

#### Convolutional Network
- Architecture: Standard CNN (for vision tasks)
- Training: Standard supervised learning
- Purpose: Spatial processing baseline

---

### Statistical Comparison

All comparisons include:

1. **Significance Testing**
   - Paired t-test (same data, different models)
   - Wilcoxon signed-rank test (non-parametric)
   - Effect size (Cohen's d)

2. **Confidence Intervals**
   - 95% CI via bootstrap (n=1000)
   - Standard error of the mean

3. **Multiple Comparison Correction**
   - Bonferroni correction
   - False discovery rate control

4. **Visualization**
   - Learning curves with confidence bands
   - Bar charts with error bars
   - Scatter plots for correlation

---

## Running Benchmarks

### Quick Start

```bash
# Run full benchmark suite
python examples/benchmark_example.py

# Run specific category
python examples/benchmark_example.py --category cognitive

# Run single benchmark
python examples/benchmark_example.py --benchmark spatial_reasoning

# Compare configurations
python examples/benchmark_example.py --compare config1.json config2.json
```

### Configuration

Benchmark parameters are specified in YAML:

```yaml
# benchmark_config.yaml
benchmark:
  name: "spatial_reasoning"
  iterations: 10
  random_seed: 42
  
network:
  neurons: 10000
  connectivity: 0.1
  learning_rate: 0.001
  
task:
  environment: "maze_10x10x10"
  episodes: 1000
  max_steps: 500
  
reporting:
  save_results: true
  output_dir: "results/"
  plot_learning_curves: true
```

### Batch Execution

```bash
# Run parameter sweep
python scripts/run_benchmark_sweep.py \
  --benchmark temporal_memory \
  --param learning_rate \
  --values 0.0001 0.001 0.01 0.1 \
  --repeats 5

# Compare against baselines
python scripts/compare_baselines.py \
  --task classification \
  --dataset 4D-MNIST \
  --baselines mlp lstm cnn
```

### Continuous Integration

Benchmarks integrated into CI/CD:

```yaml
# .github/workflows/benchmarks.yml
- name: Run Performance Benchmarks
  run: |
    python examples/benchmark_example.py --fast --save-results
    python scripts/check_performance_regression.py
```

---

## Results Tracking

### Storage Format

Results stored in structured JSON:

```json
{
  "benchmark": "spatial_reasoning",
  "version": "1.0.0",
  "timestamp": "2025-12-14T10:00:00Z",
  "configuration": { ... },
  "results": {
    "accuracy": 0.87,
    "sample_efficiency": 0.75,
    "convergence_speed": 0.0032
  },
  "baselines": {
    "random": 0.10,
    "lstm": 0.62
  },
  "statistics": {
    "mean": 0.87,
    "std": 0.03,
    "ci_lower": 0.84,
    "ci_upper": 0.90
  },
  "metadata": {
    "git_commit": "abc123...",
    "hardware": "Intel i9, 32GB RAM",
    "python_version": "3.10.5"
  }
}
```

### Results Database

- Location: `docs/benchmarks/results/`
- Format: JSON files per benchmark
- Tracking: Version-controlled in Git
- Visualization: Automated plots in `docs/benchmarks/plots/`

### Performance Dashboard

Web dashboard displays:
- Latest benchmark results
- Historical trends
- Comparison tables
- Interactive plots

Access: `http://localhost:5000/benchmarks` (when running web interface)

---

## Contributing Benchmarks

### Adding a New Benchmark

1. **Create Benchmark Class**

```python
# src/ki_benchmarks/my_benchmark.py

from src.ki_benchmarks.base import Benchmark

class MyBenchmark(Benchmark):
    """Description of benchmark."""
    
    def __init__(self, config):
        super().__init__(config)
        # Initialize benchmark-specific components
    
    def setup(self):
        """Prepare data, networks, baselines."""
        pass
    
    def run(self):
        """Execute benchmark."""
        pass
    
    def evaluate(self):
        """Compute metrics."""
        pass
    
    def report(self):
        """Generate report."""
        pass
```

2. **Register Benchmark**

```python
# src/ki_benchmarks/__init__.py

from .my_benchmark import MyBenchmark

BENCHMARK_REGISTRY = {
    # ... existing benchmarks ...
    "my_benchmark": MyBenchmark
}
```

3. **Add Tests**

```python
# tests/ki_benchmarks/test_my_benchmark.py

def test_my_benchmark():
    config = load_test_config()
    benchmark = MyBenchmark(config)
    results = benchmark.run()
    assert results["accuracy"] > 0.5
```

4. **Update Documentation**

- Add benchmark description to this document
- Include in `docs/benchmarks/README.md`
- Reference in `SCIENTIFIC_HYPOTHESES.md` if applicable

### Submission Process

1. Fork repository
2. Create feature branch
3. Implement benchmark with tests
4. Submit pull request with:
   - Benchmark code
   - Tests
   - Documentation
   - Example results

See `CONTRIBUTING.md` for detailed guidelines.

---

## Benchmark Suite Roadmap

### Current Status (v1.0)

- ✅ Basic framework implemented
- ✅ Cognitive benchmarks (3)
- ✅ Biological validation (2)
- ✅ Learning efficiency (3)
- ✅ Performance metrics
- ✅ Baseline comparisons

### Planned Additions (v1.1-1.3)

- [ ] Complete standardized datasets (4D-MNIST, multimodal)
- [ ] Additional cognitive tasks (reasoning, planning)
- [ ] Comparison with more simulators (Nengo, CARLsim)
- [ ] Automated report generation
- [ ] Public leaderboard

### Future Directions (v2.0+)

- [ ] Neuromorphic hardware benchmarks (Loihi, SpiNNaker)
- [ ] Large-scale benchmarks (1M+ neurons)
- [ ] Real-world application tasks
- [ ] Community-contributed benchmarks
- [ ] Integration with MLCommons

---

## References

### Benchmark Design

- Graves, A., et al. (2014). Neural Turing Machines. arXiv:1410.5401
- Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179-211.

### Biological Validation

- Beggs, J. M., & Plenz, D. (2003). Neuronal avalanches. *Journal of Neuroscience*, 23(35), 11167-11177.
- Morrison, A., et al. (2008). Spike-timing-dependent plasticity in balanced random networks. *Neural Computation*, 20(6), 1473-1514.

### Continual Learning

- Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting. *PNAS*, 114(13), 3521-3526.
- Zenke, F., et al. (2017). Continual learning through synaptic intelligence. *ICML*.

---

## Contact & Support

**Maintainer**: Thomas Heisig (t_heisig@gmx.de)  
**Location**: Ganderkesee, Germany  
**GitHub**: https://github.com/Thomas-Heisig/4D-Neural-Cognition  
**Discussions**: https://github.com/Thomas-Heisig/4D-Neural-Cognition/discussions

For benchmark-related questions:
- Open GitHub issue with `[Benchmark]` tag
- Start discussion in Benchmarking category
- Email maintainer for collaboration inquiries

---

**Document Version**: 1.0  
**Last Updated**: December 2025  
**License**: MIT (see repository LICENSE file)
