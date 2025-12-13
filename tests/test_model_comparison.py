"""Tests for model comparison tools."""

import pytest
import numpy as np
from src.model_comparison import (
    ModelComparator,
    ModelResult,
    AblationStudy,
    bootstrap_confidence_interval,
    cross_validation_comparison
)


class TestModelResult:
    """Test ModelResult dataclass."""

    def test_create_result(self):
        """Test creating a model result."""
        result = ModelResult(
            model_name="Test Model",
            performance_metrics={"accuracy": 0.95},
            training_time=100.0,
            inference_time=0.5,
            memory_usage=512.0,
            configuration={"param": "value"}
        )
        
        assert result.model_name == "Test Model"
        assert result.performance_metrics["accuracy"] == 0.95
        assert result.training_time == 100.0


class TestModelComparator:
    """Test ModelComparator class."""

    def test_initialization(self):
        """Test comparator initialization."""
        comparator = ModelComparator()
        assert len(comparator.results) == 0

    def test_add_result(self):
        """Test adding results."""
        comparator = ModelComparator()
        result = ModelResult(
            model_name="Model1",
            performance_metrics={"acc": 0.9},
            training_time=50.0,
            inference_time=0.1,
            memory_usage=256.0,
            configuration={}
        )
        
        comparator.add_result(result)
        assert len(comparator.results) == 1

    def test_compare_performance(self):
        """Test performance comparison."""
        comparator = ModelComparator()
        
        # Add multiple results
        for i in range(3):
            result = ModelResult(
                model_name=f"Model{i}",
                performance_metrics={"accuracy": 0.8 + i * 0.05},
                training_time=100.0,
                inference_time=0.1,
                memory_usage=512.0,
                configuration={}
            )
            comparator.add_result(result)
        
        # Compare
        comparison = comparator.compare_performance("accuracy")
        
        assert comparison["best_model"] == "Model2"
        assert comparison["best_value"] == 0.9
        assert comparison["worst_model"] == "Model0"
        assert "mean" in comparison
        assert "std" in comparison

    def test_compare_empty(self):
        """Test comparison with no results."""
        comparator = ModelComparator()
        comparison = comparator.compare_performance("accuracy")
        
        assert "error" in comparison

    def test_compare_missing_metric(self):
        """Test comparison with missing metric."""
        comparator = ModelComparator()
        
        result = ModelResult(
            model_name="Model1",
            performance_metrics={"accuracy": 0.9},
            training_time=50.0,
            inference_time=0.1,
            memory_usage=256.0,
            configuration={}
        )
        comparator.add_result(result)
        
        comparison = comparator.compare_performance("f1_score")
        assert "error" in comparison

    def test_pairwise_comparison(self):
        """Test pairwise statistical comparison."""
        comparator = ModelComparator()
        
        # Create two clearly different models
        model1_runs = [0.7, 0.72, 0.71, 0.73, 0.70]
        model2_runs = [0.85, 0.87, 0.86, 0.88, 0.84]
        
        result = comparator.pairwise_comparison(
            model1_name="Model1",
            model2_name="Model2",
            metric_name="accuracy",
            model1_runs=model1_runs,
            model2_runs=model2_runs,
            test_type="t-test"
        )
        
        assert "p_value" in result
        assert "significant" in result
        assert "cohens_d" in result
        assert result["winner"] == "Model2"

    def test_pairwise_insufficient_data(self):
        """Test pairwise comparison with insufficient data."""
        comparator = ModelComparator()
        
        result = comparator.pairwise_comparison(
            "Model1", "Model2", "accuracy",
            [0.9], [0.8], "t-test"
        )
        
        assert "error" in result

    def test_pairwise_wilcoxon(self):
        """Test Wilcoxon test."""
        comparator = ModelComparator()
        
        # Use slightly different runs for proper Wilcoxon test
        runs1 = [0.8, 0.82, 0.81, 0.83, 0.79]
        runs2 = [0.81, 0.83, 0.82, 0.84, 0.80]
        
        result = comparator.pairwise_comparison(
            "Model1", "Model2", "accuracy",
            runs1, runs2, "wilcoxon"
        )
        
        assert "p_value" in result

    def test_pairwise_mann_whitney(self):
        """Test Mann-Whitney U test."""
        comparator = ModelComparator()
        
        model1_runs = [0.7, 0.71, 0.72]
        model2_runs = [0.8, 0.81]
        
        result = comparator.pairwise_comparison(
            "Model1", "Model2", "accuracy",
            model1_runs, model2_runs, "mann-whitney"
        )
        
        assert "p_value" in result

    def test_benchmark_models(self):
        """Test model benchmarking."""
        comparator = ModelComparator()
        
        # Add models with different speeds
        for i in range(3):
            result = ModelResult(
                model_name=f"Model{i}",
                performance_metrics={},
                training_time=100.0 + i * 50,
                inference_time=0.1 + i * 0.05,
                memory_usage=256.0 + i * 128,
                configuration={}
            )
            comparator.add_result(result)
        
        benchmark = comparator.benchmark_models()
        
        assert "fastest_training" in benchmark
        assert "fastest_inference" in benchmark
        assert "most_memory_efficient" in benchmark
        assert benchmark["fastest_training"] == "Model0"

    def test_generate_report(self):
        """Test report generation."""
        comparator = ModelComparator()
        
        result = ModelResult(
            model_name="TestModel",
            performance_metrics={"accuracy": 0.95},
            training_time=100.0,
            inference_time=0.1,
            memory_usage=512.0,
            configuration={}
        )
        comparator.add_result(result)
        
        report = comparator.generate_comparison_report()
        
        assert isinstance(report, str)
        assert "TestModel" in report
        # Accuracy should be in the report
        assert len(report) > 0


class TestAblationStudy:
    """Test AblationStudy class."""

    def test_initialization(self):
        """Test ablation study initialization."""
        config = {"param1": True, "param2": 10}
        study = AblationStudy(config)
        
        assert study.baseline_config == config

    def test_ablate_component_remove(self):
        """Test removing a component."""
        config = {"use_feature": True, "size": 100}
        study = AblationStudy(config)
        
        ablated = study.ablate_component(
            "Feature X",
            "use_feature",
            None
        )
        
        assert "use_feature" not in ablated["config"]
        assert ablated["name"] == "Feature X"

    def test_ablate_component_modify(self):
        """Test modifying a component."""
        config = {"use_feature": True, "size": 100}
        study = AblationStudy(config)
        
        ablated = study.ablate_component(
            "Feature X",
            "use_feature",
            False
        )
        
        assert ablated["config"]["use_feature"] is False

    def test_add_ablation_result(self):
        """Test adding ablation results."""
        study = AblationStudy({})
        
        study.add_ablation_result(
            component_name="Feature A",
            baseline_performance=0.9,
            ablated_performance=0.7,
            metric_name="accuracy"
        )
        
        assert len(study.ablation_results) == 1
        result = study.ablation_results[0]
        assert result["component"] == "Feature A"
        assert abs(result["impact"] - 0.2) < 0.001

    def test_rank_component_importance(self):
        """Test ranking components by importance."""
        study = AblationStudy({})
        
        # Add multiple ablation results
        study.add_ablation_result("Feature A", 0.9, 0.85, "accuracy")
        study.add_ablation_result("Feature B", 0.9, 0.6, "accuracy")
        study.add_ablation_result("Feature C", 0.9, 0.88, "accuracy")
        
        ranked = study.rank_component_importance("accuracy")
        
        # Feature B has biggest impact
        assert ranked[0]["component"] == "Feature B"
        assert ranked[1]["component"] == "Feature A"
        assert ranked[2]["component"] == "Feature C"

    def test_generate_ablation_report(self):
        """Test ablation report generation."""
        study = AblationStudy({})
        
        study.add_ablation_result("Feature A", 0.9, 0.7, "accuracy")
        
        report = study.generate_ablation_report()
        
        assert isinstance(report, str)
        assert "Feature A" in report
        assert "accuracy" in report


class TestBootstrapCI:
    """Test bootstrap confidence interval."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap CI."""
        data = [0.8, 0.82, 0.81, 0.83, 0.79, 0.84, 0.80, 0.82]
        
        lower, upper = bootstrap_confidence_interval(data, confidence_level=0.95)
        
        assert lower < upper
        assert lower >= min(data)
        assert upper <= max(data)

    def test_bootstrap_single_value(self):
        """Test bootstrap with single value."""
        data = [0.8]
        
        lower, upper = bootstrap_confidence_interval(data)
        
        assert lower == upper == 0.8

    def test_bootstrap_empty(self):
        """Test bootstrap with empty data."""
        lower, upper = bootstrap_confidence_interval([])
        
        assert lower == upper == 0.0

    def test_bootstrap_confidence_levels(self):
        """Test different confidence levels."""
        data = [0.8, 0.82, 0.81, 0.83, 0.79]
        
        lower_95, upper_95 = bootstrap_confidence_interval(data, 0.95)
        lower_99, upper_99 = bootstrap_confidence_interval(data, 0.99)
        
        # 99% CI should be wider
        assert (upper_99 - lower_99) >= (upper_95 - lower_95)


class TestCrossValidationComparison:
    """Test cross-validation comparison."""

    def test_basic_cv_comparison(self):
        """Test basic CV comparison."""
        model_performances = {
            "Model1": [0.8, 0.82, 0.81, 0.83, 0.79],
            "Model2": [0.85, 0.87, 0.86, 0.88, 0.84]
        }
        
        results = cross_validation_comparison(model_performances, cv_folds=5)
        
        assert "Model1" in results
        assert "Model2" in results
        assert "ranking" in results
        assert results["ranking"][0] == "Model2"

    def test_cv_with_statistics(self):
        """Test CV comparison statistics."""
        model_performances = {
            "Model1": [0.8, 0.82, 0.81, 0.83, 0.79]
        }
        
        results = cross_validation_comparison(model_performances, cv_folds=5)
        
        model_stats = results["Model1"]
        assert "mean" in model_stats
        assert "std" in model_stats
        assert "min" in model_stats
        assert "max" in model_stats
        assert "ci_95" in model_stats

    def test_cv_wrong_fold_count(self):
        """Test CV with wrong number of folds."""
        model_performances = {
            "Model1": [0.8, 0.82, 0.81]  # Only 3 values
        }
        
        results = cross_validation_comparison(model_performances, cv_folds=5)
        
        assert "error" in results["Model1"]


class TestIntegration:
    """Integration tests for model comparison."""

    def test_full_comparison_workflow(self):
        """Test complete comparison workflow."""
        comparator = ModelComparator()
        
        # Add multiple models
        for i in range(3):
            result = ModelResult(
                model_name=f"Model{i}",
                performance_metrics={
                    "accuracy": 0.8 + i * 0.05,
                    "f1": 0.75 + i * 0.05
                },
                training_time=100.0 - i * 10,
                inference_time=0.1 + i * 0.01,
                memory_usage=512.0 + i * 64,
                configuration={"size": 100 * (i + 1)}
            )
            comparator.add_result(result)
        
        # Compare on accuracy
        acc_comparison = comparator.compare_performance("accuracy")
        assert acc_comparison["best_model"] == "Model2"
        
        # Compare on F1
        f1_comparison = comparator.compare_performance("f1")
        assert f1_comparison["best_model"] == "Model2"
        
        # Benchmark
        benchmark = comparator.benchmark_models()
        assert benchmark["fastest_training"] == "Model2"
        
        # Generate report
        report = comparator.generate_comparison_report()
        assert len(report) > 0

    def test_ablation_workflow(self):
        """Test complete ablation workflow."""
        baseline_config = {
            "feature_a": True,
            "feature_b": True,
            "feature_c": True,
            "learning_rate": 0.01
        }
        
        study = AblationStudy(baseline_config)
        baseline_perf = 0.90
        
        # Test each feature
        features = [
            ("Feature A", "feature_a", False, 0.85),
            ("Feature B", "feature_b", False, 0.70),
            ("Feature C", "feature_c", False, 0.88)
        ]
        
        for name, key, value, ablated_perf in features:
            ablated_config = study.ablate_component(name, key, value)
            study.add_ablation_result(name, baseline_perf, ablated_perf, "accuracy")
        
        # Rank importance
        ranking = study.rank_component_importance("accuracy")
        
        # Feature B has biggest impact (0.90 - 0.70 = 0.20)
        assert ranking[0]["component"] == "Feature B"
        assert abs(ranking[0]["impact"] - 0.20) < 0.001
        
        # Generate report
        report = study.generate_ablation_report()
        assert "Feature B" in report
