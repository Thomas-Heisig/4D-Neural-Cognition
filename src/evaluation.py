"""Evaluation and benchmarking tools for 4D Neural Cognition.

This module provides tools for:
- Running benchmark suites
- Comparing different configurations
- Tracking reproducibility
- Generating evaluation reports
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .brain_model import BrainModel
    from .simulation import Simulation
    from .tasks import Task, TaskResult
except ImportError:
    from brain_model import BrainModel
    from simulation import Simulation
    from tasks import Task, TaskResult


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    name: str
    description: str
    config_path: str
    seed: int
    initialization_params: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkConfig":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    config_name: str
    task_name: str
    task_result: TaskResult
    execution_time: float
    seed: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["task_result"] = asdict(self.task_result)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary."""
        task_result = TaskResult(**data["task_result"])
        data_copy = data.copy()
        data_copy["task_result"] = task_result
        return cls(**data_copy)


class BenchmarkSuite:
    """Suite of benchmark tasks for evaluating configurations."""

    def __init__(self, name: str = "Default Suite", description: str = "") -> None:
        """Initialize benchmark suite.

        Args:
            name: Name of the benchmark suite
            description: Description of what this suite tests
        """
        self.name = name
        self.description = description
        self.tasks: List[Task] = []

    def add_task(self, task: Task) -> None:
        """Add a task to the suite.

        Args:
            task: Task to add
        """
        self.tasks.append(task)

    def run(self, config: BenchmarkConfig, output_dir: Optional[Path] = None) -> List[BenchmarkResult]:
        """Run all tasks in the suite with a given configuration.

        Args:
            config: Configuration to evaluate
            output_dir: Optional directory to save results

        Returns:
            List of benchmark results
        """
        results = []

        print(f"\n{'=' * 60}")
        print(f"Running Benchmark Suite: {self.name}")
        print(f"Configuration: {config.name}")
        print(f"{'=' * 60}\n")

        for task in self.tasks:
            print(f"Running task: {task.get_name()}")
            print(f"Description: {task.get_description()}")

            # Initialize model and simulation
            model = BrainModel(config_path=config.config_path)
            sim = Simulation(model, seed=config.seed)

            # Initialize neurons and synapses based on config
            init_params = config.initialization_params
            sim.initialize_neurons(area_names=init_params.get("area_names"), density=init_params.get("density", 0.1))
            sim.initialize_random_synapses(
                connection_probability=init_params.get("connection_probability", 0.01),
                weight_mean=init_params.get("weight_mean", 0.1),
                weight_std=init_params.get("weight_std", 0.05),
            )

            # Run task evaluation
            start_time = time.time()
            task_result = task.evaluate(sim)
            execution_time = time.time() - start_time

            # Create benchmark result
            result = BenchmarkResult(
                config_name=config.name,
                task_name=task.get_name(),
                task_result=task_result,
                execution_time=execution_time,
                seed=config.seed,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            )
            results.append(result)

            # Print results
            print("Results:")
            print(f"  Accuracy: {task_result.accuracy:.4f}")
            print(f"  Reward: {task_result.reward:.4f}")
            print(f"  Reaction Time: {task_result.reaction_time:.2f}")
            print(f"  Stability: {task_result.stability:.4f}")
            print(f"  Execution Time: {execution_time:.2f}s")
            print()

        # Save results if output directory specified
        if output_dir:
            self._save_results(results, output_dir, config)

        return results

    def _save_results(self, results: List[BenchmarkResult], output_dir: Path, config: BenchmarkConfig) -> None:
        """Save results to JSON file.

        Args:
            results: List of results to save
            output_dir: Directory to save to
            config: Configuration used
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{config.name}_{timestamp}.json"
        filepath = output_dir / filename

        # Prepare data
        data = {
            "suite_name": self.name,
            "suite_description": self.description,
            "config": config.to_dict(),
            "results": [r.to_dict() for r in results],
        }

        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {filepath}")


class ConfigurationComparator:
    """Compare multiple configurations on the same tasks."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initialize comparator.

        Args:
            output_dir: Optional directory to save comparison results
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.results: Dict[str, List[BenchmarkResult]] = {}

    def add_results(self, config_name: str, results: List[BenchmarkResult]) -> None:
        """Add results for a configuration.

        Args:
            config_name: Name of the configuration
            results: List of benchmark results
        """
        self.results[config_name] = results

    def compare(self) -> Dict[str, Any]:
        """Compare all configurations and generate report.

        Returns:
            Comparison report as dictionary
        """
        if not self.results:
            return {"error": "No results to compare"}

        # Organize results by task
        task_comparisons = {}

        for config_name, results in self.results.items():
            for result in results:
                task_name = result.task_name
                if task_name not in task_comparisons:
                    task_comparisons[task_name] = {}

                task_comparisons[task_name][config_name] = {
                    "accuracy": result.task_result.accuracy,
                    "reward": result.task_result.reward,
                    "reaction_time": result.task_result.reaction_time,
                    "stability": result.task_result.stability,
                    "execution_time": result.execution_time,
                }

        # Calculate best performing config per task and metric
        best_performers = {}

        for task_name, configs in task_comparisons.items():
            best_performers[task_name] = {}

            # Find best for each metric
            metrics = ["accuracy", "reward", "stability"]
            for metric in metrics:
                values = {cfg: data[metric] for cfg, data in configs.items()}
                if values:
                    best_config = max(values, key=values.get)
                    best_performers[task_name][metric] = {"config": best_config, "value": values[best_config]}

            # For reaction time and execution time, lower is better
            for metric in ["reaction_time", "execution_time"]:
                values = {cfg: data[metric] for cfg, data in configs.items()}
                if values:
                    best_config = min(values, key=values.get)
                    best_performers[task_name][metric] = {"config": best_config, "value": values[best_config]}

        # Create comparison report
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_configs": len(self.results),
            "config_names": list(self.results.keys()),
            "task_comparisons": task_comparisons,
            "best_performers": best_performers,
        }

        # Save report if output directory specified
        if self.output_dir:
            self._save_report(report)

        return report

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save comparison report to file.

        Args:
            report: Report dictionary to save
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"comparison_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\nComparison report saved to: {filepath}")

    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a human-readable summary of the comparison.

        Args:
            report: Comparison report
        """
        print("\n" + "=" * 80)
        print("CONFIGURATION COMPARISON SUMMARY")
        print("=" * 80)

        print(f"\nConfigurations compared: {report['num_configs']}")
        for name in report["config_names"]:
            print(f"  - {name}")

        print("\n" + "-" * 80)
        print("BEST PERFORMERS BY TASK")
        print("-" * 80)

        for task_name, metrics in report["best_performers"].items():
            print(f"\nTask: {task_name}")
            for metric_name, data in metrics.items():
                print(f"  {metric_name:20s}: {data['config']:30s} ({data['value']:.4f})")

        print("\n" + "=" * 80 + "\n")


def create_standard_benchmark_suite() -> BenchmarkSuite:
    """Create the standard benchmark suite.

    Returns:
        BenchmarkSuite with standard tasks
    """
    from .tasks import PatternClassificationTask, TemporalSequenceTask

    suite = BenchmarkSuite(
        name="Standard 4D Neural Cognition Benchmark",
        description="Standard suite of tasks for evaluating 4D neural network configurations",
    )

    # Add pattern classification tasks
    suite.add_task(PatternClassificationTask(num_classes=4, pattern_size=(20, 20), noise_level=0.1, seed=42))

    # Add temporal sequence task
    suite.add_task(TemporalSequenceTask(sequence_length=5, vocabulary_size=8, seed=42))

    return suite


def run_configuration_comparison(
    configs: List[BenchmarkConfig], suite: Optional[BenchmarkSuite] = None, output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Run a comparison of multiple configurations.

    Args:
        configs: List of configurations to compare
        suite: Benchmark suite to use (default: standard suite)
        output_dir: Optional directory to save results

    Returns:
        Comparison report
    """
    if suite is None:
        suite = create_standard_benchmark_suite()

    comparator = ConfigurationComparator(output_dir=output_dir)

    # Run each configuration
    for config in configs:
        results = suite.run(config, output_dir=output_dir)
        comparator.add_results(config.name, results)

    # Generate comparison
    report = comparator.compare()
    comparator.print_summary(report)

    return report
