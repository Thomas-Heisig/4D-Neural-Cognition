"""Model comparison and statistical analysis tools.

This module provides tools for comparing different neural network configurations,
performing statistical significance testing, benchmarking, and ablation studies.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from scipy import stats
from dataclasses import dataclass
import time


@dataclass
class ModelResult:
    """Results from a model evaluation.
    
    Attributes:
        model_name: Name/identifier of the model.
        performance_metrics: Dictionary of metric names to values.
        training_time: Time taken to train (seconds).
        inference_time: Average inference time (seconds).
        memory_usage: Peak memory usage (MB).
        configuration: Model configuration dictionary.
    """
    model_name: str
    performance_metrics: Dict[str, float]
    training_time: float
    inference_time: float
    memory_usage: float
    configuration: Dict[str, Any]


class ModelComparator:
    """Compare multiple neural network models statistically."""
    
    def __init__(self):
        """Initialize model comparator."""
        self.results: List[ModelResult] = []
    
    def add_result(self, result: ModelResult) -> None:
        """Add a model result to the comparison.
        
        Args:
            result: ModelResult object with evaluation data.
        """
        self.results.append(result)
    
    def compare_performance(
        self,
        metric_name: str,
        significance_level: float = 0.05
    ) -> Dict[str, Any]:
        """Compare models on a specific performance metric.
        
        Args:
            metric_name: Name of metric to compare.
            significance_level: P-value threshold for significance.
            
        Returns:
            Dictionary with comparison results including rankings and statistics.
        """
        if not self.results:
            return {"error": "No results to compare"}
        
        # Extract metric values for each model
        model_metrics = {}
        for result in self.results:
            if metric_name in result.performance_metrics:
                model_metrics[result.model_name] = result.performance_metrics[metric_name]
        
        if len(model_metrics) < 2:
            return {"error": f"Need at least 2 models with metric '{metric_name}'"}
        
        # Rank models by metric
        ranked = sorted(model_metrics.items(), key=lambda x: x[1], reverse=True)
        
        # For statistical testing, we'd need multiple runs per model
        # Here we provide the framework for single-run comparison
        
        return {
            "metric": metric_name,
            "rankings": ranked,
            "best_model": ranked[0][0],
            "best_value": ranked[0][1],
            "worst_model": ranked[-1][0],
            "worst_value": ranked[-1][1],
            "value_range": ranked[0][1] - ranked[-1][1],
            "mean": float(np.mean(list(model_metrics.values()))),
            "std": float(np.std(list(model_metrics.values())))
        }
    
    def pairwise_comparison(
        self,
        model1_name: str,
        model2_name: str,
        metric_name: str,
        model1_runs: List[float],
        model2_runs: List[float],
        test_type: str = "t-test"
    ) -> Dict[str, Any]:
        """Perform statistical comparison between two models.
        
        Args:
            model1_name: Name of first model.
            model2_name: Name of second model.
            metric_name: Metric being compared.
            model1_runs: Multiple evaluation runs for model 1.
            model2_runs: Multiple evaluation runs for model 2.
            test_type: Statistical test ('t-test', 'wilcoxon', 'mann-whitney').
            
        Returns:
            Dictionary with test results including p-value and conclusion.
        """
        if len(model1_runs) < 2 or len(model2_runs) < 2:
            return {"error": "Need at least 2 runs per model for statistical testing"}
        
        # Perform statistical test
        if test_type == "t-test":
            statistic, p_value = stats.ttest_ind(model1_runs, model2_runs)
            test_name = "Independent T-Test"
        elif test_type == "wilcoxon":
            if len(model1_runs) != len(model2_runs):
                return {"error": "Wilcoxon test requires equal number of runs"}
            statistic, p_value = stats.wilcoxon(model1_runs, model2_runs)
            test_name = "Wilcoxon Signed-Rank Test"
        elif test_type == "mann-whitney":
            statistic, p_value = stats.mannwhitneyu(model1_runs, model2_runs)
            test_name = "Mann-Whitney U Test"
        else:
            return {"error": f"Unknown test type: {test_type}"}
        
        # Calculate effect size (Cohen's d)
        mean1 = np.mean(model1_runs)
        mean2 = np.mean(model2_runs)
        pooled_std = np.sqrt((np.var(model1_runs) + np.var(model2_runs)) / 2)
        
        if pooled_std > 0:
            cohens_d = (mean1 - mean2) / pooled_std
        else:
            cohens_d = 0.0
        
        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interpretation = "small"
        elif abs(cohens_d) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        return {
            "test": test_name,
            "metric": metric_name,
            "model1": {
                "name": model1_name,
                "mean": float(mean1),
                "std": float(np.std(model1_runs)),
                "n": len(model1_runs)
            },
            "model2": {
                "name": model2_name,
                "mean": float(mean2),
                "std": float(np.std(model2_runs)),
                "n": len(model2_runs)
            },
            "statistic": float(statistic),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "cohens_d": float(cohens_d),
            "effect_size": effect_interpretation,
            "winner": model1_name if mean1 > mean2 else model2_name
        }
    
    def benchmark_models(
        self,
        include_training_time: bool = True,
        include_inference_time: bool = True,
        include_memory: bool = True
    ) -> Dict[str, Any]:
        """Benchmark all models on computational efficiency.
        
        Args:
            include_training_time: Include training time in benchmark.
            include_inference_time: Include inference time in benchmark.
            include_memory: Include memory usage in benchmark.
            
        Returns:
            Dictionary with benchmark results.
        """
        if not self.results:
            return {"error": "No results to benchmark"}
        
        benchmark = {"models": {}}
        
        for result in self.results:
            model_benchmark = {}
            
            if include_training_time:
                model_benchmark["training_time"] = result.training_time
            
            if include_inference_time:
                model_benchmark["inference_time"] = result.inference_time
            
            if include_memory:
                model_benchmark["memory_usage"] = result.memory_usage
            
            benchmark["models"][result.model_name] = model_benchmark
        
        # Find fastest/most efficient models
        if include_training_time:
            fastest_training = min(self.results, key=lambda x: x.training_time)
            benchmark["fastest_training"] = fastest_training.model_name
        
        if include_inference_time:
            fastest_inference = min(self.results, key=lambda x: x.inference_time)
            benchmark["fastest_inference"] = fastest_inference.model_name
        
        if include_memory:
            most_efficient = min(self.results, key=lambda x: x.memory_usage)
            benchmark["most_memory_efficient"] = most_efficient.model_name
        
        return benchmark
    
    def generate_comparison_report(self) -> str:
        """Generate a formatted comparison report.
        
        Returns:
            Formatted string report.
        """
        if not self.results:
            return "No results to compare."
        
        report = ["=" * 60]
        report.append("MODEL COMPARISON REPORT")
        report.append("=" * 60)
        report.append(f"\nTotal models compared: {len(self.results)}\n")
        
        # Performance comparison
        report.append("Performance Metrics:")
        report.append("-" * 60)
        
        # Get all unique metrics
        all_metrics = set()
        for result in self.results:
            all_metrics.update(result.performance_metrics.keys())
        
        for metric in sorted(all_metrics):
            comparison = self.compare_performance(metric)
            if "error" not in comparison:
                report.append(f"\n{metric}:")
                report.append(f"  Best: {comparison['best_model']} ({comparison['best_value']:.4f})")
                report.append(f"  Mean: {comparison['mean']:.4f} ± {comparison['std']:.4f}")
        
        # Benchmark comparison
        report.append("\n" + "-" * 60)
        report.append("Computational Efficiency:")
        report.append("-" * 60)
        
        benchmark = self.benchmark_models()
        if "error" not in benchmark:
            if "fastest_training" in benchmark:
                report.append(f"  Fastest Training: {benchmark['fastest_training']}")
            if "fastest_inference" in benchmark:
                report.append(f"  Fastest Inference: {benchmark['fastest_inference']}")
            if "most_memory_efficient" in benchmark:
                report.append(f"  Most Memory Efficient: {benchmark['most_memory_efficient']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


class AblationStudy:
    """Framework for performing ablation studies on neural network components."""
    
    def __init__(self, baseline_config: Dict[str, Any]):
        """Initialize ablation study with baseline configuration.
        
        Args:
            baseline_config: Full configuration dictionary.
        """
        self.baseline_config = baseline_config.copy()
        self.ablation_results: List[Dict[str, Any]] = []
    
    def ablate_component(
        self,
        component_name: str,
        component_key: str,
        ablation_value: Any = None
    ) -> Dict[str, Any]:
        """Create an ablated configuration by removing or modifying a component.
        
        Args:
            component_name: Human-readable name for the component.
            component_key: Key in configuration dictionary.
            ablation_value: Value to set (None to remove component).
            
        Returns:
            Ablated configuration dictionary.
        """
        ablated_config = self.baseline_config.copy()
        
        if ablation_value is None:
            # Remove component
            if component_key in ablated_config:
                del ablated_config[component_key]
        else:
            # Modify component
            ablated_config[component_key] = ablation_value
        
        return {
            "name": component_name,
            "key": component_key,
            "ablation_value": ablation_value,
            "config": ablated_config
        }
    
    def add_ablation_result(
        self,
        component_name: str,
        baseline_performance: float,
        ablated_performance: float,
        metric_name: str
    ) -> None:
        """Record result of an ablation experiment.
        
        Args:
            component_name: Name of ablated component.
            baseline_performance: Performance with component.
            ablated_performance: Performance without component.
            metric_name: Name of performance metric.
        """
        impact = baseline_performance - ablated_performance
        relative_impact = (impact / baseline_performance * 100) if baseline_performance != 0 else 0.0
        
        self.ablation_results.append({
            "component": component_name,
            "metric": metric_name,
            "baseline": baseline_performance,
            "ablated": ablated_performance,
            "impact": impact,
            "relative_impact_pct": relative_impact
        })
    
    def rank_component_importance(self, metric_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Rank components by their importance (impact when removed).
        
        Args:
            metric_name: Optional metric to filter by.
            
        Returns:
            List of components ranked by importance.
        """
        filtered_results = self.ablation_results
        
        if metric_name:
            filtered_results = [r for r in self.ablation_results if r["metric"] == metric_name]
        
        # Sort by absolute impact (descending)
        ranked = sorted(filtered_results, key=lambda x: abs(x["impact"]), reverse=True)
        
        return ranked
    
    def generate_ablation_report(self) -> str:
        """Generate formatted ablation study report.
        
        Returns:
            Formatted string report.
        """
        if not self.ablation_results:
            return "No ablation results recorded."
        
        report = ["=" * 60]
        report.append("ABLATION STUDY REPORT")
        report.append("=" * 60)
        
        # Get all unique metrics
        metrics = set(r["metric"] for r in self.ablation_results)
        
        for metric in sorted(metrics):
            report.append(f"\nMetric: {metric}")
            report.append("-" * 60)
            
            ranked = self.rank_component_importance(metric)
            
            for i, result in enumerate(ranked, 1):
                report.append(f"{i}. {result['component']}")
                report.append(f"   Baseline: {result['baseline']:.4f}")
                report.append(f"   Ablated:  {result['ablated']:.4f}")
                report.append(f"   Impact:   {result['impact']:+.4f} ({result['relative_impact_pct']:+.1f}%)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def bootstrap_confidence_interval(
    data: List[float],
    confidence_level: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for data.
    
    Args:
        data: List of data points.
        confidence_level: Desired confidence level (e.g., 0.95 for 95%).
        n_bootstrap: Number of bootstrap samples.
        
    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(data) < 2:
        mean_val = np.mean(data) if data else 0.0
        return (mean_val, mean_val)
    
    data_array = np.array(data)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_array, size=len(data_array), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    # Calculate percentiles
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_means, lower_percentile)
    upper_bound = np.percentile(bootstrap_means, upper_percentile)
    
    return (float(lower_bound), float(upper_bound))


def cross_validation_comparison(
    model_performances: Dict[str, List[float]],
    cv_folds: int = 5
) -> Dict[str, Any]:
    """Compare models using cross-validation results.
    
    Args:
        model_performances: Dict mapping model names to list of fold performances.
        cv_folds: Number of cross-validation folds.
        
    Returns:
        Dictionary with comparison statistics.
    """
    results = {}
    
    for model_name, performances in model_performances.items():
        if len(performances) != cv_folds:
            results[model_name] = {"error": f"Expected {cv_folds} fold results"}
            continue
        
        mean_perf = np.mean(performances)
        std_perf = np.std(performances)
        ci_lower, ci_upper = bootstrap_confidence_interval(performances)
        
        results[model_name] = {
            "mean": float(mean_perf),
            "std": float(std_perf),
            "min": float(np.min(performances)),
            "max": float(np.max(performances)),
            "ci_95": (ci_lower, ci_upper)
        }
    
    # Rank models by mean performance
    valid_models = {k: v for k, v in results.items() if "error" not in v}
    if valid_models:
        ranked = sorted(valid_models.items(), key=lambda x: x[1]["mean"], reverse=True)
        results["ranking"] = [model_name for model_name, _ in ranked]
    
    return results
