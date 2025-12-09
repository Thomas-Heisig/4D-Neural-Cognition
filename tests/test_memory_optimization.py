"""Tests for memory optimization module."""

import pytest
import numpy as np
import tempfile
import os
from src.memory_optimization import (
    MemoryStats,
    MemoryMappedModel,
    InactiveNeuronCompressor,
    CacheOptimizer,
    MemoryProfiler,
    create_memory_mapped_model,
    optimize_model_memory
)
from src.brain_model import BrainModel, Neuron


class TestMemoryStats:
    """Test memory statistics."""
    
    def test_memory_stats_creation(self):
        """Test creating memory stats."""
        stats = MemoryStats(
            total_bytes=1000000,
            neurons_bytes=500000,
            synapses_bytes=300000,
            spike_history_bytes=100000,
            other_bytes=100000
        )
        
        assert stats.total_bytes == 1000000
        assert stats.neurons_bytes == 500000
    
    def test_memory_stats_to_mb(self):
        """Test converting to megabytes."""
        stats = MemoryStats(
            total_bytes=1048576,  # 1 MB
            neurons_bytes=524288,  # 0.5 MB
            synapses_bytes=262144,  # 0.25 MB
            spike_history_bytes=131072,  # 0.125 MB
            other_bytes=131072  # 0.125 MB
        )
        
        mb = stats.to_mb()
        
        assert mb['total_mb'] == pytest.approx(1.0, rel=0.01)
        assert mb['neurons_mb'] == pytest.approx(0.5, rel=0.01)
        assert mb['synapses_mb'] == pytest.approx(0.25, rel=0.01)


class TestMemoryMappedModel:
    """Test memory-mapped model storage."""
    
    def test_create_memory_mapped_model(self):
        """Test creating a memory-mapped model."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            model = MemoryMappedModel(filepath, mode='w')
            assert model.filepath == filepath
            assert model.h5file is not None
            model.close()
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_store_and_load_neurons(self):
        """Test storing and loading neurons."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            # Create and store neurons
            neurons = {
                0: Neuron(id=0, x=5, y=5, z=2, w=1, v_membrane=-60.0, health=0.9, age=10),
                1: Neuron(id=1, x=10, y=8, z=3, w=2, v_membrane=-65.0, health=1.0, age=5),
                2: Neuron(id=2, x=2, y=3, z=1, w=0, v_membrane=-55.0, health=0.8, age=20)
            }
            
            model = MemoryMappedModel(filepath, mode='w')
            model.store_neurons(neurons)
            model.close()
            
            # Load neurons back
            model = MemoryMappedModel(filepath, mode='r')
            loaded = model.load_neurons()
            model.close()
            
            assert len(loaded) == 3
            assert 0 in loaded
            assert loaded[0]['x'] == 5
            assert loaded[0]['v_membrane'] == pytest.approx(-60.0)
            assert loaded[1]['health'] == pytest.approx(1.0)
            assert loaded[2]['age'] == 20
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_store_neurons_with_compression(self):
        """Test storing neurons with compression."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            neurons = {i: Neuron(id=i, x=i, y=i, z=i % 5, w=i % 2) for i in range(100)}
            
            model = MemoryMappedModel(filepath, mode='w')
            model.store_neurons(neurons, compression='gzip')
            model.close()
            
            # Check file exists and is not empty
            assert os.path.exists(filepath)
            assert os.path.getsize(filepath) > 0
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_load_neurons_partial(self):
        """Test loading subset of neurons."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            neurons = {i: Neuron(id=i, x=i, y=i, z=i % 5, w=i % 2) for i in range(10)}
            
            model = MemoryMappedModel(filepath, mode='w')
            model.store_neurons(neurons)
            model.close()
            
            # Load only first 3
            model = MemoryMappedModel(filepath, mode='r')
            loaded = model.load_neurons(indices=[0, 1, 2])
            model.close()
            
            assert len(loaded) == 3
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_store_neurons_invalid_compression(self):
        """Test storing neurons with invalid compression."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            neurons = {0: Neuron(id=0, x=5, y=5, z=2, w=1)}
            
            model = MemoryMappedModel(filepath, mode='w')
            
            with pytest.raises(ValueError):
                model.store_neurons(neurons, compression='invalid')
            
            model.close()
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
    
    def test_factory_function(self):
        """Test factory function."""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            filepath = tmp.name
        
        try:
            model = create_memory_mapped_model(filepath, mode='w')
            assert isinstance(model, MemoryMappedModel)
            model.close()
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


class TestInactiveNeuronCompressor:
    """Test inactive neuron compression."""
    
    def test_compressor_initialization(self):
        """Test compressor initialization."""
        compressor = InactiveNeuronCompressor(inactivity_threshold=100)
        assert compressor.inactivity_threshold == 100
        assert compressor.compressed_neurons == {}
    
    def test_compress_inactive_neurons(self):
        """Test compressing inactive neurons."""
        compressor = InactiveNeuronCompressor(inactivity_threshold=50)
        
        neurons = {
            0: Neuron(id=0, x=5, y=5, z=2, w=1),
            1: Neuron(id=1, x=10, y=8, z=3, w=2),
            2: Neuron(id=2, x=2, y=3, z=1, w=0)
        }
        
        spike_history = {
            0: [100, 150, 200],  # Recent activity
            1: [10, 20],  # Inactive
            2: []  # No spikes
        }
        
        current_step = 200
        
        active, bytes_saved = compressor.compress_inactive_neurons(
            neurons, spike_history, current_step
        )
        
        # Neuron 0 should be active (last spike at 200)
        assert 0 in active
        
        # Neuron 1 should be compressed (last spike at 20, >50 steps ago)
        assert 1 not in active
        assert 1 in compressor.compressed_neurons
        
        # Neuron 2 should be compressed (no spikes)
        assert 2 not in active
        
        assert bytes_saved > 0
    
    def test_decompress_neuron(self):
        """Test decompressing a neuron."""
        compressor = InactiveNeuronCompressor(inactivity_threshold=50)
        
        neurons = {
            0: Neuron(id=0, x=5, y=5, z=2, w=1, health=0.9)
        }
        
        spike_history = {0: [10]}
        
        # Compress (neuron last spiked at 10, now at 100, so 90 steps ago > 50 threshold)
        active, _ = compressor.compress_inactive_neurons(neurons, spike_history, 100)
        
        # Decompress
        decompressed = compressor.decompress_neuron(0)
        
        assert decompressed is not None
        assert decompressed.id == 0
        assert decompressed.x == 5
        assert decompressed.health == pytest.approx(0.9)
    
    def test_decompress_nonexistent_neuron(self):
        """Test decompressing non-existent neuron."""
        compressor = InactiveNeuronCompressor()
        result = compressor.decompress_neuron(999)
        assert result is None
    
    def test_get_compression_stats(self):
        """Test getting compression statistics."""
        compressor = InactiveNeuronCompressor(inactivity_threshold=50)
        
        neurons = {i: Neuron(id=i, x=i, y=i, z=i % 5, w=i % 2) for i in range(10)}
        spike_history = {i: [10] for i in range(10)}
        
        # Compress all neurons (last spike at 10, now at 100, so 90 steps > 50 threshold)
        active, _ = compressor.compress_inactive_neurons(neurons, spike_history, 100)
        
        stats = compressor.get_compression_stats()
        
        assert stats['n_compressed'] == 10
        assert stats['total_bytes'] > 0
        assert stats['avg_bytes_per_neuron'] > 0


class TestCacheOptimizer:
    """Test cache optimization."""
    
    def test_cache_optimizer_initialization(self):
        """Test cache optimizer initialization."""
        optimizer = CacheOptimizer()
        assert optimizer.access_patterns == {}
    
    def test_reorder_neurons_spatial_locality(self):
        """Test reordering neurons for spatial locality."""
        optimizer = CacheOptimizer()
        
        # Create neurons with random ordering
        neurons = {
            0: Neuron(id=0, x=10, y=10, z=5, w=2),
            1: Neuron(id=1, x=0, y=0, z=0, w=0),
            2: Neuron(id=2, x=5, y=5, z=2, w=1),
            3: Neuron(id=3, x=15, y=15, z=7, w=3)
        }
        
        reordered = optimizer.reorder_neurons_spatial_locality(neurons)
        
        # Should still have all neurons
        assert len(reordered) == 4
        assert set(reordered.keys()) == set(neurons.keys())
        
        # Order should be different (based on Morton codes)
        # Neuron at (0,0,0,0) should come first
        first_neuron_id = list(reordered.keys())[0]
        assert reordered[first_neuron_id].x == 0
    
    def test_analyze_access_pattern(self):
        """Test analyzing access patterns."""
        optimizer = CacheOptimizer()
        
        # Sequential access pattern (good locality)
        sequential = list(range(100))
        stats_seq = optimizer.analyze_access_pattern(sequential)
        
        assert stats_seq['n_accesses'] == 100
        assert stats_seq['unique_ids'] == 100
        assert stats_seq['avg_spatial_jump'] == pytest.approx(1.0)
        assert stats_seq['locality_score'] > 0.4
        
        # Random access pattern (poor locality)
        random = [0, 50, 10, 99, 5, 75, 20]
        stats_rand = optimizer.analyze_access_pattern(random)
        
        assert stats_rand['n_accesses'] == 7
        assert stats_rand['avg_spatial_jump'] > 10
        assert stats_rand['locality_score'] < stats_seq['locality_score']
    
    def test_analyze_empty_access_pattern(self):
        """Test analyzing empty access pattern."""
        optimizer = CacheOptimizer()
        stats = optimizer.analyze_access_pattern([])
        
        assert stats['n_accesses'] == 0
        assert stats['avg_spatial_jump'] == 0
        assert stats['locality_score'] == 1.0


class TestMemoryProfiler:
    """Test memory profiler."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        config = {
            "lattice_shape": [10, 10, 5, 2],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {"enabled": False},
            "plasticity": {"enabled": False},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        # Add some neurons
        for i in range(20):
            model.add_neuron(i % 10, i % 10, i % 5, i % 2)
        
        # Add some synapses
        for i in range(10):
            model.add_synapse(i, (i + 1) % 20, weight=0.1)
        
        return model
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        profiler = MemoryProfiler()
        assert profiler.snapshots == []
    
    def test_measure_model_memory(self, simple_model):
        """Test measuring model memory."""
        profiler = MemoryProfiler()
        stats = profiler.measure_model_memory(simple_model)
        
        assert isinstance(stats, MemoryStats)
        assert stats.total_bytes > 0
        assert stats.neurons_bytes > 0
        assert stats.synapses_bytes > 0
    
    def test_take_snapshot(self, simple_model):
        """Test taking memory snapshots."""
        profiler = MemoryProfiler()
        profiler.take_snapshot('initial', simple_model)
        
        assert len(profiler.snapshots) == 1
        assert profiler.snapshots[0][0] == 'initial'
        assert isinstance(profiler.snapshots[0][1], MemoryStats)
    
    def test_take_multiple_snapshots(self, simple_model):
        """Test taking multiple snapshots."""
        profiler = MemoryProfiler()
        
        profiler.take_snapshot('start', simple_model)
        
        # Add more neurons
        for i in range(20, 40):
            simple_model.add_neuron(i % 10, i % 10, i % 5, i % 2)
        
        profiler.take_snapshot('after_add', simple_model)
        
        assert len(profiler.snapshots) == 2
    
    def test_compare_snapshots(self, simple_model):
        """Test comparing snapshots."""
        profiler = MemoryProfiler()
        
        profiler.take_snapshot('start', simple_model)
        
        # Add more neurons
        for i in range(20, 60):
            simple_model.add_neuron(i % 10, i % 10, i % 5, i % 2)
        
        profiler.take_snapshot('after_add', simple_model)
        
        diff = profiler.compare_snapshots('start', 'after_add')
        
        assert diff is not None
        assert 'total_diff_mb' in diff
        assert 'neurons_diff_mb' in diff
        assert diff['neurons_diff_mb'] > 0  # Should increase
    
    def test_compare_nonexistent_snapshots(self):
        """Test comparing non-existent snapshots."""
        profiler = MemoryProfiler()
        diff = profiler.compare_snapshots('fake1', 'fake2')
        assert diff is None
    
    def test_print_report(self, simple_model, capsys):
        """Test printing memory report."""
        profiler = MemoryProfiler()
        profiler.take_snapshot('test', simple_model)
        
        profiler.print_report()
        
        captured = capsys.readouterr()
        assert 'Memory Usage Report' in captured.out
        assert 'test' in captured.out
    
    def test_get_report(self, simple_model):
        """Test getting memory report as string."""
        profiler = MemoryProfiler()
        profiler.take_snapshot('test', simple_model)
        
        report = profiler.get_report()
        
        assert isinstance(report, str)
        assert 'Memory Usage Report' in report
        assert 'test' in report


class TestOptimizeModelMemory:
    """Test high-level memory optimization function."""
    
    @pytest.fixture
    def test_model(self):
        """Create a test model."""
        config = {
            "lattice_shape": [10, 10, 5, 2],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {"enabled": False},
            "plasticity": {"enabled": False},
            "senses": {},
            "areas": []
        }
        model = BrainModel(config=config)
        
        for i in range(50):
            model.add_neuron(i % 10, i % 10, i % 5, i % 2)
        
        return model
    
    def test_optimize_model_memory_basic(self, test_model):
        """Test basic memory optimization."""
        results = optimize_model_memory(
            test_model,
            enable_compression=False,
            enable_cache_optimization=True
        )
        
        assert 'initial_memory_mb' in results
        assert 'final_memory_mb' in results
        assert 'bytes_saved' in results
        assert 'optimizations_applied' in results
        
        assert results['initial_memory_mb'] > 0
        assert 'spatial_locality_reordering' in results['optimizations_applied']
    
    def test_optimize_model_memory_all_features(self, test_model):
        """Test optimization with all features."""
        results = optimize_model_memory(
            test_model,
            enable_compression=True,
            enable_cache_optimization=True
        )
        
        assert isinstance(results, dict)
        assert results['initial_memory_mb'] > 0
