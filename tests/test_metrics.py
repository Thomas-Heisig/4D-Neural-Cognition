"""Unit tests for advanced metrics module."""

import pytest
import numpy as np
from src.metrics import (
    calculate_entropy,
    calculate_mutual_information,
    calculate_spike_rate_entropy,
    calculate_network_stability,
    calculate_burst_metrics,
    calculate_population_synchrony,
    calculate_learning_curve_metrics,
    calculate_generalization_metrics
)


class TestEntropyCalculations:
    """Tests for entropy calculations."""
    
    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution has maximum entropy."""
        spike_counts = [10, 10, 10, 10]
        entropy = calculate_entropy(spike_counts)
        assert entropy == pytest.approx(2.0, abs=0.01)  # log2(4) = 2
    
    def test_single_value_zero_entropy(self):
        """Test that single value has zero entropy."""
        spike_counts = [40, 0, 0, 0]
        entropy = calculate_entropy(spike_counts)
        assert entropy == 0.0
    
    def test_empty_list_zero_entropy(self):
        """Test that empty list returns zero entropy."""
        spike_counts = []
        entropy = calculate_entropy(spike_counts)
        assert entropy == 0.0
    
    def test_all_zeros_zero_entropy(self):
        """Test that all zeros returns zero entropy."""
        spike_counts = [0, 0, 0, 0]
        entropy = calculate_entropy(spike_counts)
        assert entropy == 0.0
    
    def test_binary_distribution(self):
        """Test entropy of binary distribution."""
        spike_counts = [50, 50]
        entropy = calculate_entropy(spike_counts)
        assert entropy == pytest.approx(1.0, abs=0.01)  # log2(2) = 1


class TestMutualInformation:
    """Tests for mutual information calculations."""
    
    def test_identical_variables_max_mi(self):
        """Test that identical variables have maximum mutual information."""
        x = [1, 2, 3, 4, 5] * 10
        y = x.copy()
        mi = calculate_mutual_information(x, y)
        # MI should equal entropy of x
        assert mi > 0
    
    def test_independent_variables_zero_mi(self):
        """Test that independent variables have near-zero mutual information."""
        x = [0, 0, 0, 0, 1, 1, 1, 1]
        y = [0, 1, 0, 1, 0, 1, 0, 1]
        mi = calculate_mutual_information(x, y)
        assert mi == pytest.approx(0.0, abs=0.01)
    
    def test_different_length_raises_error(self):
        """Test that different lengths raise ValueError."""
        x = [1, 2, 3]
        y = [1, 2]
        with pytest.raises(ValueError):
            calculate_mutual_information(x, y)
    
    def test_empty_lists(self):
        """Test mutual information with empty lists."""
        x = []
        y = []
        mi = calculate_mutual_information(x, y)
        assert mi == 0.0
    
    def test_correlated_variables(self):
        """Test MI for correlated variables."""
        x = [0, 0, 1, 1, 0, 0, 1, 1]
        y = [0, 0, 1, 1, 0, 0, 1, 1]  # Perfectly correlated
        mi = calculate_mutual_information(x, y)
        assert mi > 0.5  # Should have high MI


class TestSpikeRateEntropy:
    """Tests for spike rate entropy."""
    
    def test_empty_history(self):
        """Test with empty spike history."""
        entropy = calculate_spike_rate_entropy([])
        assert entropy == 0.0
    
    def test_uniform_spiking(self):
        """Test with uniform spiking across neurons."""
        spike_history = [
            [0, 1, 2, 3],
            [0, 1, 2, 3],
            [0, 1, 2, 3]
        ]
        entropy = calculate_spike_rate_entropy(spike_history)
        assert entropy == pytest.approx(2.0, abs=0.01)
    
    def test_single_neuron_dominates(self):
        """Test when single neuron dominates."""
        spike_history = [
            [0],
            [0],
            [0],
            [1]
        ]
        entropy = calculate_spike_rate_entropy(spike_history)
        # Low entropy because neuron 0 dominates
        assert entropy < 1.0


class TestNetworkStability:
    """Tests for network stability metrics."""
    
    def test_constant_activity_stable(self):
        """Test that constant activity is perfectly stable."""
        activity = [10.0] * 50
        metrics = calculate_network_stability(activity)
        
        assert metrics['variance'] == 0.0
        assert metrics['cv'] == 0.0
        assert metrics['local_stability'] > 0.9
        assert abs(metrics['trend']) < 0.01
    
    def test_increasing_trend(self):
        """Test detection of increasing trend."""
        activity = list(range(50))
        metrics = calculate_network_stability(activity)
        
        assert metrics['trend'] > 0.5  # Positive slope
    
    def test_decreasing_trend(self):
        """Test detection of decreasing trend."""
        activity = list(range(50, 0, -1))
        metrics = calculate_network_stability(activity)
        
        assert metrics['trend'] < -0.5  # Negative slope
    
    def test_high_variance_low_stability(self):
        """Test that high variance means low stability."""
        activity = [i % 2 * 10 for i in range(50)]  # Alternating 0, 10, 0, 10...
        metrics = calculate_network_stability(activity)
        
        assert metrics['variance'] > 10.0
        assert metrics['cv'] > 0.5
    
    def test_single_value(self):
        """Test with single value."""
        activity = [5.0]
        metrics = calculate_network_stability(activity)
        
        # Should handle gracefully
        assert metrics['variance'] == 0.0


class TestBurstMetrics:
    """Tests for burst detection metrics."""
    
    def test_no_bursts(self):
        """Test when there are no bursts."""
        spike_times = [0, 10, 20, 30, 40]  # Well-spaced spikes
        metrics = calculate_burst_metrics(spike_times)
        
        assert metrics['num_bursts'] == 0
        assert metrics['burst_rate'] == 0.0
    
    def test_single_burst(self):
        """Test detection of a single burst."""
        spike_times = [10, 11, 12, 13]  # Burst of 4 spikes
        metrics = calculate_burst_metrics(spike_times)
        
        assert metrics['num_bursts'] == 1
        assert metrics['avg_burst_size'] == 4.0
        assert metrics['burst_fraction'] == 1.0
    
    def test_multiple_bursts(self):
        """Test detection of multiple bursts."""
        spike_times = [0, 1, 2, 10, 11, 12, 20, 21, 22]
        metrics = calculate_burst_metrics(spike_times)
        
        assert metrics['num_bursts'] == 3
        assert metrics['avg_burst_size'] == 3.0
    
    def test_too_few_spikes(self):
        """Test with fewer spikes than threshold."""
        spike_times = [0, 1]
        metrics = calculate_burst_metrics(spike_times, burst_threshold=3)
        
        assert metrics['num_bursts'] == 0


class TestPopulationSynchrony:
    """Tests for population synchrony."""
    
    def test_perfect_synchrony(self):
        """Test when all neurons spike together."""
        # All neurons spike at same times
        spike_matrix = np.array([
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 1, 0]
        ])
        synchrony = calculate_population_synchrony(spike_matrix)
        
        # Should have high synchrony
        assert synchrony > 0.1
    
    def test_no_synchrony(self):
        """Test when neurons spike independently."""
        # Each neuron spikes alone
        spike_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])
        synchrony = calculate_population_synchrony(spike_matrix)
        
        # Should have lower synchrony than perfect case
        assert synchrony >= 0.0
    
    def test_empty_matrix(self):
        """Test with empty spike matrix."""
        spike_matrix = np.array([])
        synchrony = calculate_population_synchrony(spike_matrix)
        
        assert synchrony == 0.0


class TestLearningCurveMetrics:
    """Tests for learning curve metrics."""
    
    def test_improving_performance(self):
        """Test with improving performance curve."""
        performance = [0.1, 0.3, 0.5, 0.7, 0.9]
        metrics = calculate_learning_curve_metrics(performance)
        
        assert metrics['initial_performance'] == 0.1
        assert metrics['final_performance'] == 0.9
        assert metrics['improvement'] == 0.8
        assert metrics['learning_rate'] > 0
    
    def test_plateaued_learning(self):
        """Test detection of learning plateau."""
        performance = [0.1, 0.3, 0.5, 0.7, 0.8] + [0.8] * 20
        metrics = calculate_learning_curve_metrics(performance)
        
        assert metrics['plateau_reached'] is True
    
    def test_convergence_detection(self):
        """Test detection of convergence."""
        performance = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.5] * 15
        metrics = calculate_learning_curve_metrics(performance)
        
        # Should detect convergence
        assert metrics['convergence_step'] is not None
    
    def test_empty_history(self):
        """Test with empty performance history."""
        performance = []
        metrics = calculate_learning_curve_metrics(performance)
        
        assert metrics['improvement'] == 0.0
        assert metrics['convergence_step'] is None


class TestGeneralizationMetrics:
    """Tests for generalization metrics."""
    
    def test_good_generalization(self):
        """Test when train and test performance are similar."""
        train = [0.5, 0.6, 0.7, 0.8, 0.9]
        test = [0.4, 0.5, 0.6, 0.7, 0.85]
        metrics = calculate_generalization_metrics(train, test)
        
        assert metrics['generalization_gap'] < 0.1
        assert metrics['overfitting_score'] < 0.2
    
    def test_overfitting_detected(self):
        """Test detection of overfitting."""
        train = [0.5, 0.7, 0.9, 0.95, 0.99]
        test = [0.5, 0.6, 0.6, 0.55, 0.50]  # Test performance decreases
        metrics = calculate_generalization_metrics(train, test)
        
        assert metrics['generalization_gap'] > 0.4
        assert metrics['overfitting_score'] > 0.3
    
    def test_empty_data(self):
        """Test with empty data."""
        train = []
        test = []
        metrics = calculate_generalization_metrics(train, test)
        
        assert metrics['generalization_gap'] == 0.0
    
    def test_test_better_than_train(self):
        """Test when test performance exceeds training."""
        train = [0.5, 0.6, 0.7]
        test = [0.6, 0.7, 0.8]
        metrics = calculate_generalization_metrics(train, test)
        
        # Negative gap is fine (good generalization)
        assert metrics['generalization_gap'] < 0
        assert metrics['overfitting_score'] == 0.0


class TestMetricsIntegration:
    """Integration tests combining multiple metrics."""
    
    def test_realistic_spike_analysis(self):
        """Test analyzing realistic spike data."""
        # Simulate spike history
        spike_history = []
        for t in range(100):
            spikes = []
            # Some neurons spike randomly
            for n in range(20):
                if np.random.rand() < 0.1:
                    spikes.append(n)
            spike_history.append(spikes)
        
        # Calculate entropy
        entropy = calculate_spike_rate_entropy(spike_history)
        assert entropy >= 0.0
        
        # Calculate activity stability
        activity = [len(spikes) for spikes in spike_history]
        stability = calculate_network_stability(activity)
        
        assert 'variance' in stability
        assert 'cv' in stability
    
    def test_learning_analysis_pipeline(self):
        """Test complete learning analysis pipeline."""
        # Simulated learning curves
        train_perf = [0.2 + 0.007 * i + 0.01 * np.random.randn() for i in range(100)]
        test_perf = [0.2 + 0.006 * i + 0.02 * np.random.randn() for i in range(100)]
        
        # Clip to [0, 1]
        train_perf = np.clip(train_perf, 0, 1).tolist()
        test_perf = np.clip(test_perf, 0, 1).tolist()
        
        # Calculate learning curve metrics
        learning_metrics = calculate_learning_curve_metrics(train_perf)
        assert learning_metrics['improvement'] > 0
        
        # Calculate generalization metrics
        gen_metrics = calculate_generalization_metrics(train_perf, test_perf)
        assert 'generalization_gap' in gen_metrics
