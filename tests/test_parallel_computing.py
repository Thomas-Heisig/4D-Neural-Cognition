"""Tests for parallel computing module."""

import pytest
import numpy as np
from src.parallel_computing import (
    SpatialPartition,
    ParallelSimulationEngine,
    ParallelSimulator,
    benchmark_parallel_scaling,
    _process_partition_neurons
)
from src.brain_model import BrainModel, Neuron


class TestSpatialPartition:
    """Test spatial partition functionality."""
    
    def test_partition_creation(self):
        """Test creating a spatial partition."""
        partition = SpatialPartition(
            partition_id=0,
            x_range=(0, 10),
            y_range=(0, 10),
            z_range=(0, 5),
            w_range=(0, 2)
        )
        
        assert partition.partition_id == 0
        assert partition.x_range == (0, 10)
        assert partition.neuron_ids == []
    
    def test_contains_position(self):
        """Test position containment check."""
        partition = SpatialPartition(
            partition_id=0,
            x_range=(0, 10),
            y_range=(0, 10),
            z_range=(0, 5),
            w_range=(0, 2)
        )
        
        # Position inside
        assert partition.contains_position(5, 5, 2, 1)
        
        # Position on boundary
        assert partition.contains_position(0, 0, 0, 0)
        assert partition.contains_position(10, 10, 5, 2)
        
        # Position outside
        assert not partition.contains_position(-1, 5, 2, 1)
        assert not partition.contains_position(11, 5, 2, 1)
        assert not partition.contains_position(5, 5, 6, 1)
    
    def test_volume(self):
        """Test volume calculation."""
        partition = SpatialPartition(
            partition_id=0,
            x_range=(0, 9),  # 10 positions
            y_range=(0, 9),  # 10 positions
            z_range=(0, 4),  # 5 positions
            w_range=(0, 1)   # 2 positions
        )
        
        expected_volume = 10 * 10 * 5 * 2
        assert partition.volume() == expected_volume


class TestParallelSimulationEngine:
    """Test parallel simulation engine."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ParallelSimulationEngine(n_processes=4)
        assert engine.n_processes == 4
        assert engine.partition_strategy == 'spatial'
        assert engine.partitions == []
    
    def test_create_spatial_partitions_x_split(self):
        """Test creating partitions split along x dimension."""
        engine = ParallelSimulationEngine(n_processes=4)
        lattice_shape = [40, 10, 10, 5]  # x is largest
        
        partitions = engine.create_spatial_partitions(lattice_shape)
        
        assert len(partitions) == 4
        
        # Check each partition covers part of x range
        for i, partition in enumerate(partitions):
            assert partition.partition_id == i
            assert partition.y_range == (0, 9)
            assert partition.z_range == (0, 9)
            assert partition.w_range == (0, 4)
    
    def test_create_spatial_partitions_y_split(self):
        """Test creating partitions split along y dimension."""
        engine = ParallelSimulationEngine(n_processes=2)
        lattice_shape = [10, 40, 10, 5]  # y is largest
        
        partitions = engine.create_spatial_partitions(lattice_shape)
        
        assert len(partitions) == 2
        
        # Check partitions cover y range
        for partition in partitions:
            assert partition.x_range == (0, 9)
            assert partition.z_range == (0, 9)
            assert partition.w_range == (0, 4)
    
    def test_create_spatial_partitions_coverage(self):
        """Test that partitions cover entire lattice."""
        engine = ParallelSimulationEngine(n_processes=4)
        lattice_shape = [20, 20, 10, 5]
        
        partitions = engine.create_spatial_partitions(lattice_shape)
        
        # Check that every position is covered by exactly one partition
        covered_positions = set()
        for partition in partitions:
            for x in range(partition.x_range[0], partition.x_range[1] + 1):
                for y in range(partition.y_range[0], partition.y_range[1] + 1):
                    for z in range(partition.z_range[0], partition.z_range[1] + 1):
                        for w in range(partition.w_range[0], partition.w_range[1] + 1):
                            pos = (x, y, z, w)
                            assert pos not in covered_positions, f"Position {pos} covered twice"
                            covered_positions.add(pos)
        
        # Total positions should match lattice size
        expected_positions = 20 * 20 * 10 * 5
        assert len(covered_positions) == expected_positions
    
    def test_assign_neurons_to_partitions(self):
        """Test assigning neurons to partitions."""
        engine = ParallelSimulationEngine(n_processes=2)
        lattice_shape = [20, 20, 10, 5]
        partitions = engine.create_spatial_partitions(lattice_shape)
        
        # Create some test neurons
        neurons = {
            0: Neuron(id=0, x=5, y=5, z=5, w=2),
            1: Neuron(id=1, x=15, y=5, z=5, w=2),
            2: Neuron(id=2, x=8, y=8, z=3, w=1)
        }
        
        engine.assign_neurons_to_partitions(neurons, partitions)
        
        # Check neurons were assigned
        total_assigned = sum(len(p.neuron_ids) for p in partitions)
        assert total_assigned == len(neurons)
        
        # Each neuron should be in exactly one partition
        all_assigned = []
        for p in partitions:
            all_assigned.extend(p.neuron_ids)
        assert len(set(all_assigned)) == len(neurons)
    
    def test_get_load_balance_stats(self):
        """Test load balance statistics."""
        engine = ParallelSimulationEngine(n_processes=3)
        
        # Create partitions with known neuron counts
        partition1 = SpatialPartition(0, (0, 10), (0, 10), (0, 5), (0, 2))
        partition1.neuron_ids = list(range(100))  # 100 neurons
        
        partition2 = SpatialPartition(1, (11, 20), (0, 10), (0, 5), (0, 2))
        partition2.neuron_ids = list(range(100, 150))  # 50 neurons
        
        partition3 = SpatialPartition(2, (21, 30), (0, 10), (0, 5), (0, 2))
        partition3.neuron_ids = list(range(150, 200))  # 50 neurons
        
        engine.partitions = [partition1, partition2, partition3]
        
        stats = engine.get_load_balance_stats()
        
        assert stats['neurons_per_partition'] == [100, 50, 50]
        assert stats['mean'] == pytest.approx(66.67, rel=0.01)
        assert stats['min'] == 50
        assert stats['max'] == 100
        assert stats['imbalance'] > 0  # Should have some imbalance


class TestProcessPartitionNeurons:
    """Test the worker function for processing neurons."""
    
    def test_process_partition_neurons_basic(self):
        """Test basic neuron processing."""
        neuron_ids = [0, 1, 2]
        neuron_data = {
            0: {'v_membrane': -60.0, 'last_spike_time': -100, 'external_input': 1.0},
            1: {'v_membrane': -65.0, 'last_spike_time': -100, 'external_input': 0.0},
            2: {'v_membrane': -55.0, 'last_spike_time': -100, 'external_input': 2.0}
        }
        synaptic_inputs = {0: 0.5, 1: 0.0, 2: 1.0}
        lif_params = {
            'tau_m': 20.0,
            'v_rest': -65.0,
            'v_reset': -70.0,
            'v_threshold': -50.0,
            'refractory_period': 5.0
        }
        
        results = _process_partition_neurons(
            neuron_ids, neuron_data, synaptic_inputs, lif_params, 100, 1.0
        )
        
        assert len(results) == 3
        assert all(nid in results for nid in neuron_ids)
        
        # Each result should be (v_membrane, spiked)
        for nid in neuron_ids:
            v_mem, spiked = results[nid]
            assert isinstance(v_mem, (float, np.floating))
            assert isinstance(spiked, (bool, np.bool_))
    
    def test_process_partition_neurons_spike(self):
        """Test neurons that should spike."""
        neuron_ids = [0]
        neuron_data = {
            0: {'v_membrane': -45.0, 'last_spike_time': -100, 'external_input': 0.0}
        }
        synaptic_inputs = {0: 0.0}
        lif_params = {
            'tau_m': 20.0,
            'v_rest': -65.0,
            'v_reset': -70.0,
            'v_threshold': -50.0,
            'refractory_period': 5.0
        }
        
        results = _process_partition_neurons(
            neuron_ids, neuron_data, synaptic_inputs, lif_params, 100, 1.0
        )
        
        v_mem, spiked = results[0]
        assert spiked
        assert v_mem == lif_params['v_reset']
    
    def test_process_partition_neurons_refractory(self):
        """Test neurons in refractory period."""
        neuron_ids = [0]
        current_step = 100
        neuron_data = {
            0: {'v_membrane': -60.0, 'last_spike_time': 98, 'external_input': 5.0}
        }
        synaptic_inputs = {0: 5.0}
        lif_params = {
            'tau_m': 20.0,
            'v_rest': -65.0,
            'v_reset': -70.0,
            'v_threshold': -50.0,
            'refractory_period': 5.0
        }
        
        results = _process_partition_neurons(
            neuron_ids, neuron_data, synaptic_inputs, lif_params, current_step, 1.0
        )
        
        v_mem, spiked = results[0]
        # Should not spike and membrane should not change much
        assert not spiked
        assert v_mem == pytest.approx(-60.0, abs=0.1)


class TestParallelSimulator:
    """Test high-level parallel simulator."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        config = {
            "lattice_shape": [20, 20, 10, 5],
            "neuron_model": {
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {
                "enabled": False
            },
            "plasticity": {
                "enabled": False
            },
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        # Add some neurons
        for i in range(20):
            model.add_neuron(i % 20, i % 20, i % 10, i % 5)
        
        return model
    
    def test_parallel_simulator_initialization(self, simple_model):
        """Test parallel simulator initialization."""
        sim = ParallelSimulator(simple_model, n_processes=2)
        
        assert sim.model == simple_model
        assert sim.engine is not None
        assert len(sim.engine.partitions) == 2
    
    def test_parallel_neuron_update(self, simple_model):
        """Test parallel neuron update."""
        sim = ParallelSimulator(simple_model, n_processes=2)
        
        synaptic_inputs = {nid: 0.0 for nid in simple_model.neurons.keys()}
        
        results = sim.parallel_neuron_update(synaptic_inputs, 0)
        
        assert isinstance(results, dict)
        assert len(results) == len(simple_model.neurons)
        
        # Check each result
        for nid, (v_mem, spiked) in results.items():
            assert isinstance(v_mem, (float, np.floating))
            assert isinstance(spiked, (bool, np.bool_))
    
    def test_parallel_neuron_update_deterministic(self, simple_model):
        """Test that parallel update is deterministic."""
        sim = ParallelSimulator(simple_model, n_processes=2)
        
        synaptic_inputs = {nid: 1.0 for nid in simple_model.neurons.keys()}
        
        # Run twice with same inputs
        results1 = sim.parallel_neuron_update(synaptic_inputs, 0)
        results2 = sim.parallel_neuron_update(synaptic_inputs, 0)
        
        # Results should be identical
        for nid in simple_model.neurons.keys():
            v1, s1 = results1[nid]
            v2, s2 = results2[nid]
            assert v1 == pytest.approx(v2)
            assert s1 == s2


class TestBenchmarkParallelScaling:
    """Test parallel scaling benchmarks."""
    
    @pytest.fixture
    def benchmark_model(self):
        """Create a model for benchmarking."""
        config = {
            "lattice_shape": [20, 20, 10, 5],
            "neuron_model": {
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {"enabled": False},
            "plasticity": {"enabled": False},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        # Add neurons
        for i in range(50):
            model.add_neuron(i % 20, i % 20, i % 10, i % 5)
        
        return model
    
    def test_benchmark_parallel_scaling_basic(self, benchmark_model):
        """Test basic parallel scaling benchmark."""
        results = benchmark_parallel_scaling(
            benchmark_model,
            n_steps=10,
            process_counts=[1, 2]
        )
        
        assert 'process_counts' in results
        assert 'times' in results
        assert 'speedups' in results
        assert 'efficiencies' in results
        
        assert len(results['times']) == 2
        assert len(results['speedups']) == 2
        assert len(results['efficiencies']) == 2
        
        # Times should be positive
        assert all(t > 0 for t in results['times'])
        
        # First speedup should be 1.0 (baseline)
        assert results['speedups'][0] == pytest.approx(1.0)
        
        # Efficiencies should be positive (may exceed 1.0 due to overhead variations)
        assert all(e > 0 for e in results['efficiencies'])
