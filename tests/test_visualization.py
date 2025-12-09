"""Tests for visualization module."""

import numpy as np
import pytest

from src.visualization import (
    plot_performance_comparison,
    plot_learning_curves,
    create_confusion_matrix,
    plot_confusion_matrix,
    visualize_activity_patterns,
    plot_spike_rate_histogram,
    create_comparison_table,
    plot_network_statistics,
    calculate_accuracy_from_confusion,
    calculate_class_metrics,
    plot_raster,
    plot_psth,
    plot_spike_train_correlation,
)


class TestPerformanceComparison:
    """Tests for performance comparison plotting."""
    
    def test_plot_performance_comparison_basic(self):
        """Test basic performance comparison plot."""
        results = [
            {"name": "Config1", "accuracy": 0.85},
            {"name": "Config2", "accuracy": 0.90},
        ]
        plot_data = plot_performance_comparison(results, metric_name="accuracy")
        
        assert plot_data["type"] == "bar"
        assert len(plot_data["names"]) == 2
        assert len(plot_data["values"]) == 2
        assert plot_data["values"] == [0.85, 0.90]
        assert plot_data["metric_name"] == "accuracy"
    
    def test_plot_performance_comparison_empty(self):
        """Test with empty results."""
        plot_data = plot_performance_comparison([])
        assert "error" in plot_data
    
    def test_plot_performance_comparison_missing_metric(self):
        """Test with missing metric."""
        results = [{"name": "Config1"}]
        plot_data = plot_performance_comparison(results, metric_name="accuracy")
        assert plot_data["values"] == [0.0]
    
    def test_plot_performance_comparison_with_save_path(self):
        """Test with save path."""
        results = [{"name": "Config1", "accuracy": 0.85}]
        plot_data = plot_performance_comparison(results, save_path="test.png")
        assert plot_data["save_path"] == "test.png"


class TestLearningCurves:
    """Tests for learning curve visualization."""
    
    def test_plot_learning_curves_single(self):
        """Test single learning curve."""
        history = [0.1, 0.3, 0.5, 0.7, 0.9]
        plot_data = plot_learning_curves(history)
        
        assert plot_data["type"] == "line"
        assert len(plot_data["curves"]) == 1
        assert plot_data["curves"][0]["y"] == history
        assert len(plot_data["curves"][0]["x"]) == 5
    
    def test_plot_learning_curves_multiple(self):
        """Test multiple learning curves."""
        histories = [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]
        labels = ["Model A", "Model B"]
        plot_data = plot_learning_curves(histories, labels=labels)
        
        assert len(plot_data["curves"]) == 2
        assert plot_data["curves"][0]["label"] == "Model A"
        assert plot_data["curves"][1]["label"] == "Model B"
    
    def test_plot_learning_curves_auto_labels(self):
        """Test automatic label generation."""
        histories = [[0.1, 0.3], [0.2, 0.4]]
        plot_data = plot_learning_curves(histories)
        
        assert plot_data["curves"][0]["label"] == "Curve 1"
        assert plot_data["curves"][1]["label"] == "Curve 2"
    
    def test_plot_learning_curves_with_save_path(self):
        """Test with save path."""
        history = [0.1, 0.3, 0.5]
        plot_data = plot_learning_curves(history, save_path="curve.png")
        assert plot_data["save_path"] == "curve.png"


class TestConfusionMatrix:
    """Tests for confusion matrix creation and visualization."""
    
    def test_create_confusion_matrix_basic(self):
        """Test basic confusion matrix creation."""
        predictions = [0, 1, 2, 0, 1, 2]
        targets = [0, 1, 2, 1, 2, 0]
        
        cm = create_confusion_matrix(predictions, targets, num_classes=3)
        
        assert cm.shape == (3, 3)
        assert cm[0, 0] == 1  # True class 0, predicted 0
        assert cm[1, 0] == 1  # True class 1, predicted 0
        assert cm[0, 2] == 1  # True class 0, predicted 2
        assert cm[2, 1] == 1  # True class 2, predicted 1
    
    def test_create_confusion_matrix_auto_classes(self):
        """Test automatic class count detection."""
        predictions = [0, 1, 1]
        targets = [0, 0, 1]
        
        cm = create_confusion_matrix(predictions, targets)
        
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 1
        assert cm[0, 1] == 1
        assert cm[1, 1] == 1
    
    def test_create_confusion_matrix_mismatched_lengths(self):
        """Test with mismatched prediction and target lengths."""
        with pytest.raises(ValueError, match="same length"):
            create_confusion_matrix([0, 1], [0])
    
    def test_plot_confusion_matrix_basic(self):
        """Test confusion matrix plotting."""
        cm = np.array([[2, 1], [1, 3]])
        plot_data = plot_confusion_matrix(cm)
        
        assert plot_data["type"] == "heatmap"
        assert len(plot_data["matrix"]) == 2
        assert plot_data["class_names"] == ["Class 0", "Class 1"]
    
    def test_plot_confusion_matrix_normalized(self):
        """Test normalized confusion matrix."""
        cm = np.array([[2, 1], [1, 3]])
        plot_data = plot_confusion_matrix(cm, normalize=True)
        
        matrix = np.array(plot_data["matrix"])
        # First row: [2, 1] normalized to [2/3, 1/3]
        assert abs(matrix[0, 0] - 2/3) < 0.01
        assert abs(matrix[0, 1] - 1/3) < 0.01
        assert "Normalized" in plot_data["title"]
    
    def test_plot_confusion_matrix_custom_names(self):
        """Test with custom class names."""
        cm = np.array([[2, 1], [1, 3]])
        class_names = ["Cat", "Dog"]
        plot_data = plot_confusion_matrix(cm, class_names=class_names)
        
        assert plot_data["class_names"] == ["Cat", "Dog"]
    
    def test_plot_confusion_matrix_zero_division(self):
        """Test handling of zero row sums in normalization."""
        cm = np.array([[0, 0], [1, 3]])
        plot_data = plot_confusion_matrix(cm, normalize=True)
        
        # Should not crash, first row stays zero
        matrix = np.array(plot_data["matrix"])
        assert matrix[0, 0] == 0
        assert matrix[0, 1] == 0


class TestActivityPatterns:
    """Tests for activity pattern visualization."""
    
    def test_visualize_activity_patterns_basic(self):
        """Test basic activity pattern visualization."""
        spike_history = [[1, 2], [2, 3], [1, 3]]
        plot_data = visualize_activity_patterns(spike_history)
        
        assert plot_data["type"] == "raster"
        assert len(plot_data["events"]) == 6  # Total spikes
    
    def test_visualize_activity_patterns_time_window(self):
        """Test with time window."""
        spike_history = [[1], [2], [3], [4]]
        plot_data = visualize_activity_patterns(spike_history, time_window=(1, 2))
        
        # Should only include steps 1 and 2
        times = [e[0] for e in plot_data["events"]]
        assert all(1 <= t <= 2 for t in times)
    
    def test_visualize_activity_patterns_neuron_filter(self):
        """Test with neuron ID filter."""
        spike_history = [[1, 2, 3], [1, 2, 3]]
        plot_data = visualize_activity_patterns(spike_history, neuron_ids=[1, 3])
        
        neuron_ids = [e[1] for e in plot_data["events"]]
        assert all(nid in [1, 3] for nid in neuron_ids)
        assert 2 not in neuron_ids
    
    def test_visualize_activity_patterns_empty(self):
        """Test with empty spike history."""
        plot_data = visualize_activity_patterns([])
        assert "error" in plot_data
    
    def test_visualize_activity_patterns_no_spikes_in_window(self):
        """Test with no spikes in specified range."""
        spike_history = [[1], [2]]
        plot_data = visualize_activity_patterns(spike_history, neuron_ids=[99])
        assert "error" in plot_data


class TestSpikeRateHistogram:
    """Tests for spike rate histogram."""
    
    def test_plot_spike_rate_histogram_basic(self):
        """Test basic spike rate histogram."""
        spike_counts = {1: 10, 2: 20, 3: 15, 4: 25}
        plot_data = plot_spike_rate_histogram(spike_counts, bins=2)
        
        assert plot_data["type"] == "histogram"
        assert len(plot_data["counts"]) == 2
        assert len(plot_data["bin_edges"]) == 3
    
    def test_plot_spike_rate_histogram_empty(self):
        """Test with empty spike counts."""
        plot_data = plot_spike_rate_histogram({})
        assert "error" in plot_data
    
    def test_plot_spike_rate_histogram_single_value(self):
        """Test with single spike count."""
        spike_counts = {1: 10}
        plot_data = plot_spike_rate_histogram(spike_counts)
        
        assert plot_data["type"] == "histogram"


class TestComparisonTable:
    """Tests for comparison table creation."""
    
    def test_create_comparison_table_basic(self):
        """Test basic comparison table."""
        results = [
            {"name": "Config1", "accuracy": 0.85, "loss": 0.15},
            {"name": "Config2", "accuracy": 0.90, "loss": 0.10},
        ]
        table_data = create_comparison_table(results)
        
        assert table_data["type"] == "table"
        assert len(table_data["rows"]) == 2
        assert "accuracy" in table_data["columns"]
        assert "loss" in table_data["columns"]
    
    def test_create_comparison_table_specific_metrics(self):
        """Test with specific metrics."""
        results = [
            {"name": "Config1", "accuracy": 0.85, "loss": 0.15, "f1": 0.80},
        ]
        table_data = create_comparison_table(results, metrics=["accuracy", "f1"])
        
        assert table_data["columns"] == ["name", "accuracy", "f1"]
    
    def test_create_comparison_table_missing_metric(self):
        """Test with missing metric."""
        results = [
            {"name": "Config1", "accuracy": 0.85},
        ]
        table_data = create_comparison_table(results, metrics=["accuracy", "loss"])
        
        assert table_data["rows"][0]["loss"] == "N/A"
    
    def test_create_comparison_table_empty(self):
        """Test with empty results."""
        table_data = create_comparison_table([])
        assert "error" in table_data


class TestNetworkStatistics:
    """Tests for network statistics plotting."""
    
    def test_plot_network_statistics_basic(self):
        """Test basic network statistics plot."""
        stats_history = [
            {"mean_rate": 10.0, "std_rate": 2.0},
            {"mean_rate": 12.0, "std_rate": 2.5},
            {"mean_rate": 15.0, "std_rate": 3.0},
        ]
        plot_data = plot_network_statistics(stats_history)
        
        assert plot_data["type"] == "line"
        assert len(plot_data["curves"]) == 2
    
    def test_plot_network_statistics_specific_stats(self):
        """Test with specific statistics."""
        stats_history = [
            {"mean_rate": 10.0, "std_rate": 2.0, "max_rate": 20.0},
            {"mean_rate": 12.0, "std_rate": 2.5, "max_rate": 22.0},
        ]
        plot_data = plot_network_statistics(stats_history, stat_names=["mean_rate"])
        
        assert len(plot_data["curves"]) == 1
        assert plot_data["curves"][0]["label"] == "mean_rate"
    
    def test_plot_network_statistics_empty(self):
        """Test with empty statistics history."""
        plot_data = plot_network_statistics([])
        assert "error" in plot_data
    
    def test_plot_network_statistics_non_numeric(self):
        """Test filtering of non-numeric values."""
        stats_history = [
            {"mean_rate": 10.0, "name": "test"},
            {"mean_rate": 12.0, "name": "test2"},
        ]
        plot_data = plot_network_statistics(stats_history)
        
        # Only numeric stats should be plotted
        labels = [c["label"] for c in plot_data["curves"]]
        assert "mean_rate" in labels
        assert "name" not in labels


class TestAccuracyCalculation:
    """Tests for accuracy calculation from confusion matrix."""
    
    def test_calculate_accuracy_basic(self):
        """Test basic accuracy calculation."""
        cm = np.array([[2, 1], [1, 3]])
        accuracy = calculate_accuracy_from_confusion(cm)
        
        # 2 + 3 correct out of 7 total
        assert abs(accuracy - 5/7) < 0.01
    
    def test_calculate_accuracy_perfect(self):
        """Test with perfect classification."""
        cm = np.array([[3, 0], [0, 4]])
        accuracy = calculate_accuracy_from_confusion(cm)
        
        assert accuracy == 1.0
    
    def test_calculate_accuracy_empty(self):
        """Test with empty matrix."""
        cm = np.array([])
        accuracy = calculate_accuracy_from_confusion(cm)
        
        assert accuracy == 0.0
    
    def test_calculate_accuracy_all_zero(self):
        """Test with all-zero matrix."""
        cm = np.zeros((2, 2))
        accuracy = calculate_accuracy_from_confusion(cm)
        
        assert accuracy == 0.0


class TestClassMetrics:
    """Tests for per-class metrics calculation."""
    
    def test_calculate_class_metrics_basic(self):
        """Test basic class metrics calculation."""
        cm = np.array([[2, 1], [1, 3]])
        metrics = calculate_class_metrics(cm)
        
        assert 0 in metrics
        assert 1 in metrics
        assert "precision" in metrics[0]
        assert "recall" in metrics[0]
        assert "f1_score" in metrics[0]
    
    def test_calculate_class_metrics_precision(self):
        """Test precision calculation."""
        # Class 0: 2 TP, 1 FP -> precision = 2/3
        cm = np.array([[2, 1], [1, 3]])
        metrics = calculate_class_metrics(cm)
        
        assert abs(metrics[0]["precision"] - 2/3) < 0.01
    
    def test_calculate_class_metrics_recall(self):
        """Test recall calculation."""
        # Class 0: 2 TP, 1 FN -> recall = 2/3
        cm = np.array([[2, 1], [1, 3]])
        metrics = calculate_class_metrics(cm)
        
        assert abs(metrics[0]["recall"] - 2/3) < 0.01
    
    def test_calculate_class_metrics_f1(self):
        """Test F1 score calculation."""
        cm = np.array([[2, 1], [1, 3]])
        metrics = calculate_class_metrics(cm)
        
        precision = 2/3
        recall = 2/3
        expected_f1 = 2 * (precision * recall) / (precision + recall)
        assert abs(metrics[0]["f1_score"] - expected_f1) < 0.01
    
    def test_calculate_class_metrics_zero_division(self):
        """Test handling of zero division."""
        cm = np.array([[0, 0], [0, 3]])
        metrics = calculate_class_metrics(cm)
        
        # Class 0 has no predictions or targets
        assert metrics[0]["precision"] == 0.0
        assert metrics[0]["recall"] == 0.0
        assert metrics[0]["f1_score"] == 0.0


class TestRasterPlot:
    """Tests for raster plot visualization."""
    
    def test_plot_raster_basic(self):
        """Test basic raster plot."""
        spike_times = {
            1: [10, 20, 30],
            2: [15, 25, 35],
        }
        plot_data = plot_raster(spike_times)
        
        assert plot_data["type"] == "raster"
        assert len(plot_data["events"]) == 6
        assert plot_data["num_neurons"] == 2
    
    def test_plot_raster_time_window(self):
        """Test raster plot with time window."""
        spike_times = {
            1: [10, 20, 30, 40],
        }
        plot_data = plot_raster(spike_times, time_window=(15, 35))
        
        # Only spikes at 20 and 30 should be included
        assert len(plot_data["events"]) == 2
        times = [e[0] for e in plot_data["events"]]
        assert 20 in times and 30 in times
    
    def test_plot_raster_neuron_filter(self):
        """Test raster plot with neuron filter."""
        spike_times = {
            1: [10, 20],
            2: [15, 25],
            3: [12, 22],
        }
        plot_data = plot_raster(spike_times, neuron_ids=[1, 3])
        
        neuron_ids = [e[1] for e in plot_data["events"]]
        assert 2 not in neuron_ids
        assert all(nid in [1, 3] for nid in neuron_ids)
    
    def test_plot_raster_empty(self):
        """Test with empty spike times."""
        plot_data = plot_raster({})
        assert "error" in plot_data
    
    def test_plot_raster_no_spikes_in_window(self):
        """Test with no spikes in time window."""
        spike_times = {1: [10, 20]}
        plot_data = plot_raster(spike_times, time_window=(50, 100))
        assert "error" in plot_data


class TestPSTH:
    """Tests for PSTH visualization."""
    
    def test_plot_psth_basic(self):
        """Test basic PSTH plot."""
        spike_times = {
            1: [105, 115, 205, 215],  # Spikes around stimulus times
        }
        stimulus_times = [100, 200]
        
        plot_data = plot_psth(spike_times, stimulus_times, pre_window=10, post_window=20, bin_size=10)
        
        assert plot_data["type"] == "psth"
        assert "bin_centers" in plot_data
        assert "firing_rates" in plot_data
        assert plot_data["num_neurons"] == 1
        assert plot_data["num_trials"] == 2
    
    def test_plot_psth_empty_spikes(self):
        """Test PSTH with no spikes."""
        plot_data = plot_psth({}, [100, 200])
        assert "error" in plot_data
    
    def test_plot_psth_empty_stimuli(self):
        """Test PSTH with no stimuli."""
        spike_times = {1: [10, 20]}
        plot_data = plot_psth(spike_times, [])
        assert "error" in plot_data
    
    def test_plot_psth_neuron_filter(self):
        """Test PSTH with neuron filter."""
        spike_times = {
            1: [105],
            2: [115],
        }
        stimulus_times = [100]
        
        plot_data = plot_psth(spike_times, stimulus_times, neuron_ids=[1])
        assert plot_data["num_neurons"] == 1
    
    def test_plot_psth_no_spikes_in_window(self):
        """Test PSTH with no spikes in window."""
        spike_times = {1: [10, 20]}
        stimulus_times = [100, 200]
        
        plot_data = plot_psth(spike_times, stimulus_times, pre_window=10, post_window=20)
        assert "error" in plot_data


class TestCrossCorrelation:
    """Tests for spike train cross-correlation."""
    
    def test_plot_spike_train_correlation_basic(self):
        """Test basic cross-correlation."""
        spike_times_1 = [10, 20, 30, 40]
        spike_times_2 = [15, 25, 35, 45]
        
        plot_data = plot_spike_train_correlation(spike_times_1, spike_times_2, max_lag=50)
        
        assert plot_data["type"] == "cross_correlation"
        assert "lags" in plot_data
        assert "correlation" in plot_data
        assert len(plot_data["lags"]) == len(plot_data["correlation"])
    
    def test_plot_spike_train_correlation_empty(self):
        """Test with empty spike trains."""
        plot_data = plot_spike_train_correlation([], [10, 20])
        assert "error" in plot_data
        
        plot_data = plot_spike_train_correlation([10, 20], [])
        assert "error" in plot_data
    
    def test_plot_spike_train_correlation_identical(self):
        """Test autocorrelation (identical trains)."""
        spike_times = [10, 20, 30, 40, 50]
        
        plot_data = plot_spike_train_correlation(spike_times, spike_times, max_lag=20)
        
        # Should have a peak at lag 0
        lags = plot_data["lags"]
        correlation = plot_data["correlation"]
        zero_lag_index = lags.index(0)
        
        # Correlation at zero lag should be highest (or close to it)
        assert correlation[zero_lag_index] >= max(correlation) * 0.9
    
    def test_plot_spike_train_correlation_custom_bin_size(self):
        """Test with custom bin size."""
        spike_times_1 = [10, 20, 30]
        spike_times_2 = [15, 25, 35]
        
        plot_data = plot_spike_train_correlation(spike_times_1, spike_times_2, max_lag=20, bin_size=5)
        
        assert plot_data["bin_size"] == 5
        assert plot_data["max_lag"] == 20
