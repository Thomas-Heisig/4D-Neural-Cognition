"""Unit tests for evaluation module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.evaluation import (
    BenchmarkConfig,
    BenchmarkResult,
    BenchmarkSuite,
    ConfigurationComparator,
    create_standard_benchmark_suite,
)
from src.tasks import TaskResult


class TestBenchmarkConfig:
    """Test BenchmarkConfig dataclass."""

    def test_benchmark_config_creation(self):
        """Test creating a benchmark config."""
        config = BenchmarkConfig(
            name="test_config",
            description="Test configuration",
            config_path="config.json",
            seed=42,
            initialization_params={"area_names": ["vision"], "density": 0.1},
        )

        assert config.name == "test_config"
        assert config.description == "Test configuration"
        assert config.config_path == "config.json"
        assert config.seed == 42
        assert config.initialization_params["area_names"] == ["vision"]

    def test_benchmark_config_to_dict(self):
        """Test converting config to dictionary."""
        config = BenchmarkConfig(
            name="test",
            description="desc",
            config_path="path.json",
            seed=42,
            initialization_params={},
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["name"] == "test"
        assert config_dict["seed"] == 42

    def test_benchmark_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "name": "test",
            "description": "desc",
            "config_path": "path.json",
            "seed": 42,
            "initialization_params": {"density": 0.1},
        }

        config = BenchmarkConfig.from_dict(data)

        assert config.name == "test"
        assert config.seed == 42
        assert config.initialization_params["density"] == 0.1

    def test_benchmark_config_roundtrip(self):
        """Test converting to dict and back."""
        original = BenchmarkConfig(
            name="test",
            description="desc",
            config_path="path.json",
            seed=123,
            initialization_params={"area_names": ["vision", "motor"]},
        )

        config_dict = original.to_dict()
        restored = BenchmarkConfig.from_dict(config_dict)

        assert restored.name == original.name
        assert restored.seed == original.seed
        assert restored.initialization_params == original.initialization_params


class TestBenchmarkResult:
    """Test BenchmarkResult dataclass."""

    def test_benchmark_result_creation(self):
        """Test creating a benchmark result."""
        task_result = TaskResult(
            accuracy=0.85, reward=10.5, reaction_time=5.2, stability=0.9, additional_metrics={}
        )

        result = BenchmarkResult(
            config_name="test_config",
            task_name="test_task",
            task_result=task_result,
            execution_time=1.5,
            seed=42,
            timestamp="2025-01-01 12:00:00",
        )

        assert result.config_name == "test_config"
        assert result.task_name == "test_task"
        assert result.task_result.accuracy == 0.85
        assert result.execution_time == 1.5

    def test_benchmark_result_to_dict(self):
        """Test converting result to dictionary."""
        task_result = TaskResult(
            accuracy=0.75, reward=8.0, reaction_time=3.0, stability=0.8, additional_metrics={}
        )

        result = BenchmarkResult(
            config_name="cfg",
            task_name="task",
            task_result=task_result,
            execution_time=2.0,
            seed=42,
            timestamp="2025-01-01 12:00:00",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["config_name"] == "cfg"
        assert result_dict["task_result"]["accuracy"] == 0.75

    def test_benchmark_result_from_dict(self):
        """Test creating result from dictionary."""
        data = {
            "config_name": "cfg",
            "task_name": "task",
            "task_result": {
                "accuracy": 0.9,
                "reward": 12.0,
                "reaction_time": 4.0,
                "stability": 0.95,
                "additional_metrics": {},
            },
            "execution_time": 1.8,
            "seed": 42,
            "timestamp": "2025-01-01 12:00:00",
        }

        result = BenchmarkResult.from_dict(data)

        assert result.config_name == "cfg"
        assert result.task_result.accuracy == 0.9

    def test_benchmark_result_roundtrip(self):
        """Test converting to dict and back."""
        task_result = TaskResult(
            accuracy=0.88, reward=11.0, reaction_time=6.0, stability=0.85, additional_metrics={"test": 1.0}
        )

        original = BenchmarkResult(
            config_name="test",
            task_name="task",
            task_result=task_result,
            execution_time=2.5,
            seed=123,
            timestamp="2025-01-01 12:00:00",
        )

        result_dict = original.to_dict()
        restored = BenchmarkResult.from_dict(result_dict)

        assert restored.config_name == original.config_name
        assert restored.task_result.accuracy == original.task_result.accuracy


class TestBenchmarkSuite:
    """Test BenchmarkSuite class."""

    def test_benchmark_suite_creation(self):
        """Test creating a benchmark suite."""
        suite = BenchmarkSuite(name="Test Suite", description="Test description")

        assert suite.name == "Test Suite"
        assert suite.description == "Test description"
        assert len(suite.tasks) == 0

    def test_add_task(self):
        """Test adding tasks to suite."""
        suite = BenchmarkSuite()

        # Create mock task
        mock_task = MagicMock()
        mock_task.get_name.return_value = "Task 1"

        suite.add_task(mock_task)

        assert len(suite.tasks) == 1
        assert suite.tasks[0] == mock_task

    def test_add_multiple_tasks(self):
        """Test adding multiple tasks."""
        suite = BenchmarkSuite()

        task1 = MagicMock()
        task2 = MagicMock()

        suite.add_task(task1)
        suite.add_task(task2)

        assert len(suite.tasks) == 2

    @patch("src.evaluation.BrainModel")
    @patch("src.evaluation.Simulation")
    def test_suite_run(self, mock_simulation, mock_brain_model):
        """Test running suite with a configuration."""
        suite = BenchmarkSuite(name="Test Suite")

        # Create mock task
        mock_task = MagicMock()
        mock_task.get_name.return_value = "Test Task"
        mock_task.get_description.return_value = "Test Description"
        mock_task.evaluate.return_value = TaskResult(
            accuracy=0.8, reward=10.0, reaction_time=5.0, stability=0.9, additional_metrics={}
        )
        suite.add_task(mock_task)

        # Create config
        config = BenchmarkConfig(
            name="test_config",
            description="test",
            config_path="config.json",
            seed=42,
            initialization_params={"area_names": ["vision"], "density": 0.1},
        )

        # Run suite
        results = suite.run(config)

        assert len(results) == 1
        assert results[0].config_name == "test_config"
        assert results[0].task_name == "Test Task"
        assert results[0].task_result.accuracy == 0.8

    @patch("src.evaluation.BrainModel")
    @patch("src.evaluation.Simulation")
    def test_suite_run_saves_results(self, mock_simulation, mock_brain_model):
        """Test running suite saves results when output_dir specified."""
        suite = BenchmarkSuite(name="Test Suite")

        # Create mock task
        mock_task = MagicMock()
        mock_task.get_name.return_value = "Test Task"
        mock_task.get_description.return_value = "Test Description"
        mock_task.evaluate.return_value = TaskResult(
            accuracy=0.8, reward=10.0, reaction_time=5.0, stability=0.9, additional_metrics={}
        )
        suite.add_task(mock_task)

        # Create config
        config = BenchmarkConfig(
            name="test_config",
            description="test",
            config_path="config.json",
            seed=42,
            initialization_params={"area_names": ["vision"]},
        )

        # Use temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = suite.run(config, output_dir=output_dir)

            # Check that file was created
            json_files = list(output_dir.glob("*.json"))
            assert len(json_files) >= 1

            # Verify content
            with open(json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)

            assert data["suite_name"] == "Test Suite"
            assert len(data["results"]) == 1


class TestConfigurationComparator:
    """Test ConfigurationComparator class."""

    def test_comparator_creation(self):
        """Test creating a comparator."""
        comparator = ConfigurationComparator()

        assert comparator.output_dir is None
        assert len(comparator.results) == 0

    def test_comparator_with_output_dir(self):
        """Test creating a comparator with output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ConfigurationComparator(output_dir=Path(tmpdir))

            assert comparator.output_dir == Path(tmpdir)

    def test_add_results(self):
        """Test adding results to comparator."""
        comparator = ConfigurationComparator()

        task_result = TaskResult(
            accuracy=0.8, reward=10.0, reaction_time=5.0, stability=0.9, additional_metrics={}
        )
        result = BenchmarkResult(
            config_name="cfg1",
            task_name="task1",
            task_result=task_result,
            execution_time=1.5,
            seed=42,
            timestamp="2025-01-01 12:00:00",
        )

        comparator.add_results("cfg1", [result])

        assert "cfg1" in comparator.results
        assert len(comparator.results["cfg1"]) == 1

    def test_compare_no_results(self):
        """Test comparing with no results."""
        comparator = ConfigurationComparator()

        report = comparator.compare()

        assert "error" in report
        assert report["error"] == "No results to compare"

    def test_compare_single_config(self):
        """Test comparing single configuration."""
        comparator = ConfigurationComparator()

        task_result = TaskResult(
            accuracy=0.8, reward=10.0, reaction_time=5.0, stability=0.9, additional_metrics={}
        )
        result = BenchmarkResult(
            config_name="cfg1",
            task_name="task1",
            task_result=task_result,
            execution_time=1.5,
            seed=42,
            timestamp="2025-01-01 12:00:00",
        )

        comparator.add_results("cfg1", [result])
        report = comparator.compare()

        assert "num_configs" in report
        assert report["num_configs"] == 1
        assert "cfg1" in report["config_names"]

    def test_compare_multiple_configs(self):
        """Test comparing multiple configurations."""
        comparator = ConfigurationComparator()

        # Add results for config 1
        task_result1 = TaskResult(
            accuracy=0.8, reward=10.0, reaction_time=5.0, stability=0.9, additional_metrics={}
        )
        result1 = BenchmarkResult(
            config_name="cfg1",
            task_name="task1",
            task_result=task_result1,
            execution_time=1.5,
            seed=42,
            timestamp="2025-01-01 12:00:00",
        )

        # Add results for config 2
        task_result2 = TaskResult(
            accuracy=0.9, reward=12.0, reaction_time=4.0, stability=0.95, additional_metrics={}
        )
        result2 = BenchmarkResult(
            config_name="cfg2",
            task_name="task1",
            task_result=task_result2,
            execution_time=1.2,
            seed=42,
            timestamp="2025-01-01 12:00:00",
        )

        comparator.add_results("cfg1", [result1])
        comparator.add_results("cfg2", [result2])

        report = comparator.compare()

        assert report["num_configs"] == 2
        assert "cfg1" in report["config_names"]
        assert "cfg2" in report["config_names"]

        # Check best performers
        assert "task1" in report["best_performers"]
        # cfg2 should be best for accuracy (0.9 > 0.8)
        assert report["best_performers"]["task1"]["accuracy"]["config"] == "cfg2"
        # cfg2 should be best for reaction time (4.0 < 5.0, lower is better)
        assert report["best_performers"]["task1"]["reaction_time"]["config"] == "cfg2"

    def test_compare_saves_report(self):
        """Test that compare saves report when output_dir specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            comparator = ConfigurationComparator(output_dir=Path(tmpdir))

            task_result = TaskResult(
                accuracy=0.8, reward=10.0, reaction_time=5.0, stability=0.9, additional_metrics={}
            )
            result = BenchmarkResult(
                config_name="cfg1",
                task_name="task1",
                task_result=task_result,
                execution_time=1.5,
                seed=42,
                timestamp="2025-01-01 12:00:00",
            )

            comparator.add_results("cfg1", [result])
            report = comparator.compare()

            # Check that report file was created
            json_files = list(Path(tmpdir).glob("comparison_*.json"))
            assert len(json_files) >= 1

    def test_print_summary(self, capsys):
        """Test printing comparison summary."""
        comparator = ConfigurationComparator()

        report = {
            "num_configs": 2,
            "config_names": ["cfg1", "cfg2"],
            "best_performers": {
                "task1": {
                    "accuracy": {"config": "cfg2", "value": 0.9},
                    "reward": {"config": "cfg2", "value": 12.0},
                }
            },
        }

        comparator.print_summary(report)

        captured = capsys.readouterr()
        assert "CONFIGURATION COMPARISON SUMMARY" in captured.out
        assert "cfg1" in captured.out
        assert "cfg2" in captured.out


class TestCreateStandardBenchmarkSuite:
    """Test creating standard benchmark suite."""

    def test_create_standard_suite(self):
        """Test creating standard benchmark suite."""
        suite = create_standard_benchmark_suite()

        assert suite.name == "Standard 4D Neural Cognition Benchmark"
        assert len(suite.tasks) > 0

    def test_standard_suite_has_tasks(self):
        """Test that standard suite contains tasks."""
        suite = create_standard_benchmark_suite()

        # Should have at least pattern classification and temporal sequence
        assert len(suite.tasks) >= 2

        # Tasks should have names
        task_names = [task.get_name() for task in suite.tasks]
        assert any("Pattern" in name or "Classification" in name for name in task_names)
        assert any("Temporal" in name or "Sequence" in name for name in task_names)
