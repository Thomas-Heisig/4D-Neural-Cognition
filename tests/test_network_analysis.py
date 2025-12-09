"""Tests for network_analysis module."""

import pytest
import numpy as np
from src.network_analysis import (
    ConnectivityAnalyzer,
    FiringPatternAnalyzer,
    PopulationDynamicsAnalyzer,
)
from src.brain_model import BrainModel
from src.simulation import Simulation


class TestConnectivityAnalyzer:
    """Test network connectivity analysis."""

    @pytest.fixture
    def analyzer(self, populated_model):
        """Create connectivity analyzer with populated model."""
        return ConnectivityAnalyzer(populated_model)

    def test_analyzer_initialization(self, populated_model):
        """Test analyzer initialization."""
        analyzer = ConnectivityAnalyzer(populated_model)
        assert analyzer.model == populated_model
        assert analyzer._adjacency_cache is None

    def test_compute_degree_distribution(self, analyzer):
        """Test degree distribution computation."""
        degrees = analyzer.compute_degree_distribution()
        assert 'in_degree' in degrees
        assert 'out_degree' in degrees
        assert 'total_degree' in degrees
        assert isinstance(degrees['in_degree'], np.ndarray)
        assert isinstance(degrees['out_degree'], np.ndarray)

    def test_compute_degree_distribution_empty(self, brain_model):
        """Test degree distribution with no synapses."""
        analyzer = ConnectivityAnalyzer(brain_model)
        degrees = analyzer.compute_degree_distribution()
        # Should handle empty network gracefully
        assert 'in_degree' in degrees

    def test_compute_clustering_coefficient(self, analyzer):
        """Test clustering coefficient computation."""
        clustering = analyzer.compute_clustering_coefficient()
        assert isinstance(clustering, float)
        assert 0.0 <= clustering <= 1.0

    def test_compute_clustering_coefficient_empty(self, brain_model):
        """Test clustering with no connections."""
        analyzer = ConnectivityAnalyzer(brain_model)
        clustering = analyzer.compute_clustering_coefficient()
        assert clustering == 0.0

    def test_compute_path_lengths(self, analyzer):
        """Test path length computation."""
        path_lengths = analyzer.compute_path_lengths(sample_size=10)
        assert 'mean' in path_lengths
        assert 'max' in path_lengths
        assert 'std' in path_lengths

    def test_compute_path_lengths_empty(self, brain_model):
        """Test path lengths with empty network."""
        analyzer = ConnectivityAnalyzer(brain_model)
        path_lengths = analyzer.compute_path_lengths(sample_size=10)
        assert path_lengths['mean'] == 0.0
        assert path_lengths['max'] == 0.0

    def test_identify_hubs(self, analyzer):
        """Test hub identification."""
        hubs = analyzer.identify_hubs(top_k=5)
        assert isinstance(hubs, list)
        assert len(hubs) <= 5
        # Each hub should be (neuron_id, degree) tuple
        if hubs:
            assert isinstance(hubs[0], tuple)
            assert len(hubs[0]) == 2

    def test_identify_hubs_sorted(self, analyzer):
        """Test that hubs are sorted by degree."""
        hubs = analyzer.identify_hubs(top_k=10)
        if len(hubs) > 1:
            degrees = [h[1] for h in hubs]
            assert degrees == sorted(degrees, reverse=True)

    def test_compute_modularity(self, analyzer):
        """Test modularity computation."""
        # Create simple community structure
        neuron_ids = list(analyzer.model.neurons.keys())
        communities = {nid: i % 2 for i, nid in enumerate(neuron_ids)}
        
        modularity = analyzer.compute_modularity(communities)
        assert isinstance(modularity, float)
        assert -1.0 <= modularity <= 1.0

    def test_compute_modularity_empty(self, brain_model):
        """Test modularity with no synapses."""
        analyzer = ConnectivityAnalyzer(brain_model)
        modularity = analyzer.compute_modularity({})
        assert modularity == 0.0

    def test_clear_cache(self, analyzer):
        """Test cache clearing."""
        # Build cache
        _ = analyzer._get_adjacency_dict()
        assert analyzer._adjacency_cache is not None
        
        # Clear cache
        analyzer.clear_cache()
        assert analyzer._adjacency_cache is None


class TestFiringPatternAnalyzer:
    """Test firing pattern analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create firing pattern analyzer."""
        return FiringPatternAnalyzer()

    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert len(analyzer.spike_history) == 0

    def test_record_spikes(self, analyzer):
        """Test spike recording."""
        analyzer.record_spikes(neuron_id=1, time=0.0)
        analyzer.record_spikes(neuron_id=1, time=0.1)
        analyzer.record_spikes(neuron_id=2, time=0.05)
        
        assert len(analyzer.spike_history[1]) == 2
        assert len(analyzer.spike_history[2]) == 1

    def test_compute_firing_rates(self, analyzer):
        """Test firing rate computation."""
        # Record spikes for two neurons
        for t in [0.0, 0.1, 0.2]:
            analyzer.record_spikes(1, t)
        for t in [0.0, 0.05]:
            analyzer.record_spikes(2, t)
        
        rates = analyzer.compute_firing_rates(time_window=1.0)
        assert 1 in rates
        assert 2 in rates
        assert rates[1] == 3.0  # 3 spikes in 1 second
        assert rates[2] == 2.0  # 2 spikes in 1 second

    def test_compute_firing_rates_no_spikes(self, analyzer):
        """Test firing rates with no spikes."""
        rates = analyzer.compute_firing_rates(time_window=1.0)
        assert len(rates) == 0

    def test_compute_interspike_intervals(self, analyzer):
        """Test ISI computation."""
        analyzer.record_spikes(1, 0.0)
        analyzer.record_spikes(1, 0.1)
        analyzer.record_spikes(1, 0.25)
        
        isis = analyzer.compute_interspike_intervals(1)
        assert len(isis) == 2
        assert np.isclose(isis[0], 0.1)
        assert np.isclose(isis[1], 0.15)

    def test_compute_interspike_intervals_insufficient(self, analyzer):
        """Test ISI with insufficient spikes."""
        analyzer.record_spikes(1, 0.0)
        isis = analyzer.compute_interspike_intervals(1)
        assert len(isis) == 0

    def test_compute_cv(self, analyzer):
        """Test coefficient of variation."""
        # Regular spiking
        for t in [0.0, 0.1, 0.2, 0.3]:
            analyzer.record_spikes(1, t)
        
        cv = analyzer.compute_cv(1)
        assert cv >= 0.0
        # Regular spiking should have low CV
        assert cv < 0.5

    def test_compute_cv_no_spikes(self, analyzer):
        """Test CV with no spikes."""
        cv = analyzer.compute_cv(1)
        assert cv == 0.0

    def test_detect_bursts(self, analyzer):
        """Test burst detection."""
        # Create burst pattern: burst + pause + burst
        times = [0.0, 0.005, 0.010, 0.200, 0.205, 0.210, 0.215]
        for t in times:
            analyzer.record_spikes(1, t)
        
        bursts = analyzer.detect_bursts(1, max_isi=0.02, min_spikes=3)
        assert len(bursts) == 2  # Should detect 2 bursts

    def test_detect_bursts_no_bursts(self, analyzer):
        """Test burst detection with isolated spikes."""
        times = [0.0, 0.1, 0.2, 0.3]
        for t in times:
            analyzer.record_spikes(1, t)
        
        bursts = analyzer.detect_bursts(1, max_isi=0.01, min_spikes=2)
        assert len(bursts) == 0

    def test_compute_synchrony(self, analyzer):
        """Test synchrony computation."""
        # Synchronous spikes
        for t in [0.0, 0.1, 0.2]:
            analyzer.record_spikes(1, t)
            analyzer.record_spikes(2, t + 0.001)  # Nearly synchronous
        
        synchrony = analyzer.compute_synchrony([1, 2], time_window=0.005)
        assert 0.0 <= synchrony <= 1.0
        assert synchrony > 0.0  # Should detect some synchrony

    def test_compute_synchrony_asynchronous(self, analyzer):
        """Test synchrony with asynchronous spikes."""
        analyzer.record_spikes(1, 0.0)
        analyzer.record_spikes(2, 0.1)
        analyzer.record_spikes(1, 0.2)
        analyzer.record_spikes(2, 0.3)
        
        synchrony = analyzer.compute_synchrony([1, 2], time_window=0.005)
        assert synchrony == 0.0  # No synchrony

    def test_reset(self, analyzer):
        """Test analyzer reset."""
        analyzer.record_spikes(1, 0.0)
        analyzer.record_spikes(2, 0.1)
        
        analyzer.reset()
        assert len(analyzer.spike_history) == 0


class TestPopulationDynamicsAnalyzer:
    """Test population dynamics analysis."""

    @pytest.fixture
    def analyzer(self, populated_model):
        """Create population dynamics analyzer."""
        return PopulationDynamicsAnalyzer(populated_model)

    def test_analyzer_initialization(self, populated_model):
        """Test analyzer initialization."""
        analyzer = PopulationDynamicsAnalyzer(populated_model)
        assert analyzer.model == populated_model
        assert len(analyzer.activity_history) == 0

    def test_record_population_activity(self, analyzer):
        """Test activity recording."""
        analyzer.record_population_activity()
        assert len(analyzer.activity_history) == 1
        assert isinstance(analyzer.activity_history[0], np.ndarray)

    def test_record_population_activity_limit(self, analyzer):
        """Test that history is limited."""
        analyzer.max_history = 10
        
        # Record more than max
        for _ in range(15):
            analyzer.record_population_activity()
        
        assert len(analyzer.activity_history) == 10

    def test_compute_population_rate(self, analyzer):
        """Test population rate computation."""
        analyzer.record_population_activity()
        rate = analyzer.compute_population_rate(threshold=0.0)
        assert 0.0 <= rate <= 1.0

    def test_compute_population_rate_empty(self, brain_model):
        """Test population rate with no history."""
        analyzer = PopulationDynamicsAnalyzer(brain_model)
        rate = analyzer.compute_population_rate()
        assert rate == 0.0

    def test_compute_mean_field(self, analyzer):
        """Test mean field computation."""
        # Record some activity
        for _ in range(5):
            analyzer.record_population_activity()
        
        mean_field = analyzer.compute_mean_field()
        assert isinstance(mean_field, np.ndarray)
        assert len(mean_field) > 0

    def test_compute_mean_field_empty(self, brain_model):
        """Test mean field with no history."""
        analyzer = PopulationDynamicsAnalyzer(brain_model)
        mean_field = analyzer.compute_mean_field()
        assert len(mean_field) == 0

    def test_compute_variance(self, analyzer):
        """Test variance computation."""
        analyzer.record_population_activity()
        variance = analyzer.compute_variance()
        assert variance >= 0.0

    def test_compute_variance_empty(self, brain_model):
        """Test variance with no history."""
        analyzer = PopulationDynamicsAnalyzer(brain_model)
        variance = analyzer.compute_variance()
        assert variance == 0.0

    def test_detect_oscillations(self, analyzer):
        """Test oscillation detection."""
        # Record enough history for oscillation detection
        for _ in range(20):
            analyzer.record_population_activity()
        
        oscillations = analyzer.detect_oscillations()
        assert 'detected' in oscillations
        assert isinstance(oscillations['detected'], bool)

    def test_detect_oscillations_insufficient_data(self, analyzer):
        """Test oscillation detection with insufficient data."""
        analyzer.record_population_activity()
        oscillations = analyzer.detect_oscillations()
        assert oscillations['detected'] == False

    def test_compute_dimensionality(self, analyzer):
        """Test dimensionality estimation."""
        # Record enough history
        for _ in range(15):
            analyzer.record_population_activity()
        
        dimensionality = analyzer.compute_dimensionality()
        assert isinstance(dimensionality, int)
        assert dimensionality >= 0

    def test_compute_dimensionality_insufficient_data(self, analyzer):
        """Test dimensionality with insufficient data."""
        analyzer.record_population_activity()
        dimensionality = analyzer.compute_dimensionality()
        assert dimensionality == 0

    def test_reset(self, analyzer):
        """Test analyzer reset."""
        analyzer.record_population_activity()
        analyzer.record_population_activity()
        
        analyzer.reset()
        assert len(analyzer.activity_history) == 0
