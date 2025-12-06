"""Performance benchmarks for 4D Neural Cognition.

These tests measure performance characteristics of key operations.
They use pytest-benchmark if available, but work without it too.
"""

import pytest
import numpy as np
import time
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.senses import feed_sense_input
from src.storage import save_to_json, load_from_json, save_to_hdf5, load_from_hdf5
import tempfile
import os


@pytest.fixture
def benchmark_config():
    """Create a configuration for benchmark tests."""
    return {
        "lattice_shape": [20, 20, 20, 10],
        "neuron_model": {
            "params_default": {
                "v_rest": -65.0,
                "v_threshold": -50.0,
                "v_reset": -70.0,
                "tau_membrane": 20.0,
                "refractory_period": 5
            }
        },
        "cell_lifecycle": {
            "enabled": True,
            "enable_death": True,
            "enable_reproduction": True,
            "max_age": 100000,
            "health_decay_per_step": 0.0001,
            "aging_rate": 0.001,
            "death_threshold": 0.1,
            "reproduction_threshold": 0.9,
            "reproduction_probability": 0.1,
            "reproduction_radius": 2,
            "mutation_rate": 0.01,
            "mutation_std_params": 0.05,
            "mutation_std_weights": 0.02
        },
        "plasticity": {
            "enabled": True,
            "learning_rate": 0.01,
            "weight_min": 0.0,
            "weight_max": 1.0,
            "weight_decay": 0.001
        },
        "senses": {
            "vision": {
                "areal": "V1_like",
                "enabled": True
            }
        },
        "areas": [
            {
                "name": "V1_like",
                "coord_ranges": {
                    "x": [0, 19],
                    "y": [0, 19],
                    "z": [0, 4],
                    "w": [0, 0]
                }
            }
        ]
    }


class TestNeuronCreationPerformance:
    """Benchmark neuron creation operations."""
    
    def test_add_single_neuron_performance(self, benchmark_config):
        """Measure time to add a single neuron."""
        model = BrainModel(config=benchmark_config)
        
        start_time = time.perf_counter()
        model.add_neuron(5, 5, 5, 0)
        elapsed = time.perf_counter() - start_time
        
        # Should be very fast (< 1ms)
        assert elapsed < 0.001
        assert len(model.neurons) == 1
    
    def test_add_100_neurons_performance(self, benchmark_config):
        """Measure time to add 100 neurons."""
        model = BrainModel(config=benchmark_config)
        
        start_time = time.perf_counter()
        for i in range(100):
            model.add_neuron(i % 20, i % 20, i % 20, 0)
        elapsed = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 100ms)
        assert elapsed < 0.1
        assert len(model.neurons) == 100
        print(f"\nAdded 100 neurons in {elapsed*1000:.2f}ms ({elapsed*10:.2f}ms/neuron)")
    
    def test_add_1000_neurons_performance(self, benchmark_config):
        """Measure time to add 1000 neurons."""
        model = BrainModel(config=benchmark_config)
        
        start_time = time.perf_counter()
        for i in range(1000):
            model.add_neuron(i % 20, (i // 20) % 20, i % 20, 0)
        elapsed = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 1s)
        assert elapsed < 1.0
        assert len(model.neurons) == 1000
        print(f"\nAdded 1000 neurons in {elapsed*1000:.2f}ms ({elapsed:.3f}ms/neuron)")


class TestSynapseCreationPerformance:
    """Benchmark synapse creation operations."""
    
    def test_add_single_synapse_performance(self, benchmark_config):
        """Measure time to add a single synapse."""
        model = BrainModel(config=benchmark_config)
        model.add_neuron(0, 0, 0, 0)
        model.add_neuron(1, 1, 1, 0)
        
        start_time = time.perf_counter()
        model.add_synapse(0, 1, weight=0.5)
        elapsed = time.perf_counter() - start_time
        
        # Should be very fast (< 1ms)
        assert elapsed < 0.001
        assert len(model.synapses) == 1
    
    def test_add_100_synapses_performance(self, benchmark_config):
        """Measure time to add 100 synapses."""
        model = BrainModel(config=benchmark_config)
        
        # Create 50 neurons
        for i in range(50):
            model.add_neuron(i % 20, i % 20, i % 20, 0)
        
        start_time = time.perf_counter()
        for i in range(100):
            pre_id = i % 50
            post_id = (i + 1) % 50
            model.add_synapse(pre_id, post_id, weight=0.5)
        elapsed = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 100ms)
        assert elapsed < 0.1
        assert len(model.synapses) == 100
        print(f"\nAdded 100 synapses in {elapsed*1000:.2f}ms ({elapsed*10:.2f}ms/synapse)")


class TestSimulationStepPerformance:
    """Benchmark simulation step operations."""
    
    def test_single_step_small_network(self, benchmark_config):
        """Measure time for a single simulation step (small network)."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        # Create small network (50 neurons, 100 synapses)
        sim.initialize_neurons(area_names=["V1_like"], density=0.01)
        sim.initialize_random_synapses(connection_probability=0.01)
        
        # Warm-up
        sim.step()
        
        # Measure
        start_time = time.perf_counter()
        sim.step()
        elapsed = time.perf_counter() - start_time
        
        # Should be fast (< 50ms)
        assert elapsed < 0.05
        print(f"\nSimulation step (small network): {elapsed*1000:.2f}ms")
    
    def test_single_step_medium_network(self, benchmark_config):
        """Measure time for a single simulation step (medium network)."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        # Create medium network
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        sim.initialize_random_synapses(connection_probability=0.02)
        
        neuron_count = len(model.neurons)
        synapse_count = len(model.synapses)
        
        # Warm-up
        sim.step()
        
        # Measure
        start_time = time.perf_counter()
        sim.step()
        elapsed = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 200ms)
        assert elapsed < 0.2
        print(f"\nSimulation step (medium network, {neuron_count} neurons, {synapse_count} synapses): {elapsed*1000:.2f}ms")
    
    def test_100_steps_performance(self, benchmark_config):
        """Measure time for 100 simulation steps."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.03)
        sim.initialize_random_synapses(connection_probability=0.01)
        
        # Warm-up
        sim.step()
        
        # Measure
        start_time = time.perf_counter()
        for _ in range(100):
            sim.step()
        elapsed = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 10s)
        assert elapsed < 10.0
        avg_time = elapsed / 100
        print(f"\n100 simulation steps in {elapsed:.2f}s (avg {avg_time*1000:.2f}ms/step)")


class TestSensoryInputPerformance:
    """Benchmark sensory input operations."""
    
    def test_feed_input_performance(self, benchmark_config):
        """Measure time to feed sensory input."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        # Create neurons in vision area
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        
        # Create input
        input_matrix = np.random.rand(20, 20)
        
        # Measure
        start_time = time.perf_counter()
        feed_sense_input(model, "vision", input_matrix)
        elapsed = time.perf_counter() - start_time
        
        # Should be fast (< 10ms)
        assert elapsed < 0.01
        print(f"\nFeed sensory input: {elapsed*1000:.2f}ms")
    
    def test_feed_input_100_times(self, benchmark_config):
        """Measure time to feed input 100 times."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        
        input_matrix = np.random.rand(20, 20)
        
        # Measure
        start_time = time.perf_counter()
        for _ in range(100):
            # Reset external input
            for neuron in model.neurons.values():
                neuron.external_input = 0.0
            feed_sense_input(model, "vision", input_matrix)
        elapsed = time.perf_counter() - start_time
        
        # Should complete in reasonable time (< 1s)
        assert elapsed < 1.0
        avg_time = elapsed / 100
        print(f"\n100 input feeds in {elapsed*1000:.2f}ms (avg {avg_time*1000:.2f}ms/feed)")


class TestStoragePerformance:
    """Benchmark save/load operations."""
    
    def test_save_json_performance(self, benchmark_config):
        """Measure time to save model to JSON."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        # Create medium-sized network
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        sim.initialize_random_synapses(connection_probability=0.02)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            
            # Measure
            start_time = time.perf_counter()
            save_to_json(model, filepath)
            elapsed = time.perf_counter() - start_time
            
            # Should complete in reasonable time (< 1s)
            assert elapsed < 1.0
            
            # Check file size
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"\nSaved to JSON in {elapsed*1000:.2f}ms (file size: {file_size:.1f}KB)")
    
    def test_load_json_performance(self, benchmark_config):
        """Measure time to load model from JSON."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        sim.initialize_random_synapses(connection_probability=0.02)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.json")
            save_to_json(model, filepath)
            
            # Measure
            start_time = time.perf_counter()
            loaded_model = load_from_json(filepath)
            elapsed = time.perf_counter() - start_time
            
            # Should complete in reasonable time (< 1s)
            assert elapsed < 1.0
            print(f"\nLoaded from JSON in {elapsed*1000:.2f}ms")
    
    def test_save_hdf5_performance(self, benchmark_config):
        """Measure time to save model to HDF5."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        sim.initialize_random_synapses(connection_probability=0.02)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            
            # Measure
            start_time = time.perf_counter()
            save_to_hdf5(model, filepath)
            elapsed = time.perf_counter() - start_time
            
            # Should complete in reasonable time (< 1s)
            assert elapsed < 1.0
            
            # Check file size
            file_size = os.path.getsize(filepath) / 1024  # KB
            print(f"\nSaved to HDF5 in {elapsed*1000:.2f}ms (file size: {file_size:.1f}KB)")
    
    def test_load_hdf5_performance(self, benchmark_config):
        """Measure time to load model from HDF5."""
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        sim.initialize_random_synapses(connection_probability=0.02)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.h5")
            save_to_hdf5(model, filepath)
            
            # Measure
            start_time = time.perf_counter()
            loaded_model = load_from_hdf5(filepath)
            elapsed = time.perf_counter() - start_time
            
            # Should complete in reasonable time (< 1s)
            assert elapsed < 1.0
            print(f"\nLoaded from HDF5 in {elapsed*1000:.2f}ms")


class TestScalabilityBenchmarks:
    """Benchmark scalability with different network sizes."""
    
    def test_scaling_with_neuron_count(self, benchmark_config):
        """Measure how performance scales with neuron count."""
        results = []
        
        for density in [0.01, 0.02, 0.05, 0.1]:
            model = BrainModel(config=benchmark_config)
            sim = Simulation(model, seed=42)
            
            sim.initialize_neurons(area_names=["V1_like"], density=density)
            sim.initialize_random_synapses(connection_probability=0.01)
            
            neuron_count = len(model.neurons)
            
            # Measure step time
            sim.step()  # Warm-up
            start_time = time.perf_counter()
            sim.step()
            elapsed = time.perf_counter() - start_time
            
            results.append((neuron_count, elapsed))
            print(f"\nNeurons: {neuron_count}, Step time: {elapsed*1000:.2f}ms")
        
        # Check that performance degrades gracefully
        # Time should increase roughly linearly (or worse) with neuron count
        for i in range(len(results) - 1):
            neurons1, time1 = results[i]
            neurons2, time2 = results[i + 1]
            
            # More neurons should take more time
            assert time2 >= time1 * 0.5  # Allow some variance
    
    def test_scaling_with_synapse_count(self, benchmark_config):
        """Measure how performance scales with synapse count."""
        results = []
        
        model = BrainModel(config=benchmark_config)
        sim = Simulation(model, seed=42)
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        
        for probability in [0.01, 0.02, 0.05]:
            # Create fresh model for each test
            model = BrainModel(config=benchmark_config)
            sim = Simulation(model, seed=42)
            sim.initialize_neurons(area_names=["V1_like"], density=0.05)
            sim.initialize_random_synapses(connection_probability=probability)
            
            synapse_count = len(model.synapses)
            
            # Measure step time
            sim.step()  # Warm-up
            start_time = time.perf_counter()
            sim.step()
            elapsed = time.perf_counter() - start_time
            
            results.append((synapse_count, elapsed))
            print(f"\nSynapses: {synapse_count}, Step time: {elapsed*1000:.2f}ms")
        
        # More synapses should generally take more time
        for i in range(len(results) - 1):
            synapses1, time1 = results[i]
            synapses2, time2 = results[i + 1]
            
            # More synapses should take more time (with some tolerance)
            assert time2 >= time1 * 0.5
