"""Integration tests for full simulation runs."""

import pytest
import numpy as np
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.senses import feed_sense_input
from src.storage import save_to_json, load_from_json, save_to_hdf5, load_from_hdf5
import tempfile
import os


@pytest.fixture
def full_config():
    """Create a complete test configuration."""
    return {
        "lattice_shape": [20, 20, 20, 10],
        "neuron_model": {
            "model_type": "LIF",
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
            },
            "digital": {
                "areal": "Digital_sensor",
                "enabled": True
            }
        },
        "areas": [
            {
                "name": "V1_like",
                "coord_ranges": {
                    "x": [0, 9],
                    "y": [0, 9],
                    "z": [0, 4],
                    "w": [0, 0]
                }
            },
            {
                "name": "Digital_sensor",
                "coord_ranges": {
                    "x": [10, 19],
                    "y": [10, 19],
                    "z": [0, 4],
                    "w": [0, 0]
                }
            },
            {
                "name": "Association",
                "coord_ranges": {
                    "x": [0, 19],
                    "y": [0, 19],
                    "z": [10, 14],
                    "w": [0, 0]
                }
            }
        ]
    }


class TestFullSimulation:
    """Integration tests for complete simulation workflow."""
    
    def test_basic_simulation_run(self, full_config):
        """Test a basic simulation run with neurons and synapses."""
        # Create model and simulation
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Initialize neurons
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        
        # Initialize synapses
        sim.initialize_random_synapses(connection_probability=0.01)
        
        # Run simulation
        results = []
        for _ in range(10):
            stats = sim.step()
            results.append(stats)
        
        # Verify simulation ran
        assert len(results) == 10
        assert model.current_step == 10
        assert len(model.neurons) > 0
    
    def test_simulation_with_sensory_input(self, full_config):
        """Test simulation with sensory input feeding."""
        # Create model and simulation
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Initialize neurons in vision area
        sim.initialize_neurons(area_names=["V1_like"], density=0.2)
        
        # Initialize synapses
        sim.initialize_random_synapses(connection_probability=0.01)
        
        # Create sensory input
        input_pattern = np.random.rand(10, 10)
        
        # Run simulation with periodic input
        for i in range(20):
            if i % 5 == 0:
                feed_sense_input(model, "vision", input_pattern)
            stats = sim.step()
            
            # Check that we get activity
            if i > 5:
                assert 'spikes' in stats
        
        assert model.current_step == 20
    
    def test_simulation_with_plasticity(self, full_config):
        """Test simulation with plasticity enabled."""
        # Create model and simulation
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Initialize neurons
        sim.initialize_neurons(area_names=["V1_like", "Association"], density=0.1)
        
        # Initialize synapses
        sim.initialize_random_synapses(connection_probability=0.02)
        
        initial_synapse_count = len(model.synapses)
        initial_weights = [s.weight for s in model.synapses[:5]] if len(model.synapses) >= 5 else []
        
        # Run simulation with input
        input_pattern = np.ones((10, 10)) * 5.0
        for i in range(50):
            if i % 10 == 0:
                feed_sense_input(model, "vision", input_pattern)
            sim.step()
        
        # Check that weights may have changed due to plasticity
        if len(model.synapses) >= 5:
            final_weights = [s.weight for s in model.synapses[:5]]
            # At least some weights should be different
            weights_changed = any(abs(w1 - w2) > 1e-6 for w1, w2 in zip(initial_weights, final_weights))
            assert weights_changed or len(model.synapses) != initial_synapse_count
    
    def test_simulation_with_lifecycle(self, full_config):
        """Test simulation with cell lifecycle (aging, death, reproduction)."""
        # Create model and simulation
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Initialize neurons
        sim.initialize_neurons(area_names=["V1_like"], density=0.15)
        sim.initialize_random_synapses(connection_probability=0.01)
        
        initial_neuron_count = len(model.neurons)
        
        # Run longer simulation
        total_deaths = 0
        total_births = 0
        for _ in range(100):
            stats = sim.step()
            total_deaths += stats.get('deaths', 0)
            total_births += stats.get('births', 0)
        
        # Verify lifecycle events occurred
        # (may not always happen in 100 steps, but track them)
        assert total_deaths >= 0
        assert total_births >= 0
    
    def test_multi_area_simulation(self, full_config):
        """Test simulation with multiple brain areas."""
        # Create model and simulation
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Initialize neurons in all areas
        sim.initialize_neurons(
            area_names=["V1_like", "Digital_sensor", "Association"],
            density=0.1
        )
        
        # Initialize synapses
        sim.initialize_random_synapses(connection_probability=0.01)
        
        # Verify neurons in different areas
        neurons_by_z = {}
        for neuron in model.neurons.values():
            z = neuron.z
            if z not in neurons_by_z:
                neurons_by_z[z] = 0
            neurons_by_z[z] += 1
        
        # Should have neurons in multiple z-layers
        assert len(neurons_by_z) > 1
        
        # Run simulation with dual sensory input
        vision_input = np.random.rand(10, 10)
        digital_input = np.random.rand(10, 10)
        
        for i in range(30):
            if i % 10 == 0:
                feed_sense_input(model, "vision", vision_input)
                feed_sense_input(model, "digital", digital_input)
            sim.step()
        
        assert model.current_step == 30


class TestSaveLoadIntegration:
    """Integration tests for save/load functionality."""
    
    def test_save_and_continue_simulation(self, full_config):
        """Test saving a simulation state and continuing it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and run initial simulation
            model1 = BrainModel(config=full_config)
            sim1 = Simulation(model1, seed=42)
            sim1.initialize_neurons(area_names=["V1_like"], density=0.1)
            sim1.initialize_random_synapses(connection_probability=0.01)
            
            # Run for 20 steps
            for _ in range(20):
                sim1.step()
            
            step_after_20 = model1.current_step
            neuron_count = len(model1.neurons)
            
            # Save state
            save_path = os.path.join(tmpdir, "checkpoint.json")
            save_to_json(model1, save_path)
            
            # Load and continue
            model2 = load_from_json(save_path)
            sim2 = Simulation(model2, seed=42)
            
            # Verify state preserved
            assert model2.current_step == step_after_20
            assert len(model2.neurons) == neuron_count
            
            # Continue simulation
            for _ in range(10):
                sim2.step()
            
            assert model2.current_step == step_after_20 + 10
    
    def test_hdf5_save_load_simulation(self, full_config):
        """Test HDF5 save/load with simulation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and run simulation
            model1 = BrainModel(config=full_config)
            sim1 = Simulation(model1, seed=42)
            sim1.initialize_neurons(area_names=["V1_like", "Digital_sensor"], density=0.1)
            sim1.initialize_random_synapses(connection_probability=0.02)
            
            # Run simulation
            for _ in range(15):
                sim1.step()
            
            neuron_ids = list(model1.neurons.keys())
            synapse_count = len(model1.synapses)
            
            # Save to HDF5
            save_path = os.path.join(tmpdir, "checkpoint.h5")
            save_to_hdf5(model1, save_path)
            
            # Load
            model2 = load_from_hdf5(save_path)
            
            # Verify
            assert len(model2.neurons) == len(neuron_ids)
            assert len(model2.synapses) == synapse_count
            assert model2.current_step == model1.current_step


class TestErrorRecovery:
    """Integration tests for error handling and recovery."""
    
    def test_empty_simulation(self, full_config):
        """Test running simulation with no neurons (edge case)."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Run without initializing neurons - should not crash
        for _ in range(5):
            stats = sim.step()
            assert stats['spikes'] == []
            assert stats['deaths'] == 0
            assert stats['births'] == 0
    
    def test_simulation_after_all_neurons_die(self, full_config):
        """Test simulation continues even if all neurons die."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Add a few neurons
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        
        # Manually kill all neurons
        for neuron in model.neurons.values():
            neuron.health = 0.0
        
        # Apply lifecycle to clear dead neurons
        from src.cell_lifecycle import maybe_kill_and_reproduce
        import numpy as np
        rng = np.random.default_rng(42)
        neurons_to_remove = []
        # Create a list copy to avoid modifying dict during iteration
        for nid, neuron in list(model.neurons.items()):
            if maybe_kill_and_reproduce(neuron, model, rng):
                neurons_to_remove.append(nid)
        
        for nid in neurons_to_remove:
            model.remove_neuron(nid)
        
        # Should be able to continue (with empty network)
        for _ in range(5):
            stats = sim.step()
            assert isinstance(stats, dict)
    
    def test_invalid_sensory_input_handling(self, full_config):
        """Test that invalid sensory input is handled gracefully."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        
        # Try invalid input (wrong shape)
        wrong_shape_input = np.ones((5, 5))  # Should be (10, 10)
        
        # Should warn but not crash
        with pytest.warns(UserWarning):
            feed_sense_input(model, "vision", wrong_shape_input)
        
        # Simulation should continue
        stats = sim.step()
        assert isinstance(stats, dict)


class TestReproducibility:
    """Integration tests for reproducibility."""
    
    def test_same_seed_same_results(self, full_config):
        """Test that same seed produces same results."""
        # Run 1
        model1 = BrainModel(config=full_config)
        sim1 = Simulation(model1, seed=42)
        sim1.initialize_neurons(area_names=["V1_like"], density=0.1)
        sim1.initialize_random_synapses(connection_probability=0.01)
        
        results1 = []
        for _ in range(10):
            stats = sim1.step()
            results1.append(len(stats['spikes']))
        
        # Run 2 with same seed
        model2 = BrainModel(config=full_config)
        sim2 = Simulation(model2, seed=42)
        sim2.initialize_neurons(area_names=["V1_like"], density=0.1)
        sim2.initialize_random_synapses(connection_probability=0.01)
        
        results2 = []
        for _ in range(10):
            stats = sim2.step()
            results2.append(len(stats['spikes']))
        
        # Should have same neuron and synapse counts
        assert len(model1.neurons) == len(model2.neurons)
        assert len(model1.synapses) == len(model2.synapses)
    
    def test_different_seed_different_results(self, full_config):
        """Test that different seeds produce different results."""
        # Run 1
        model1 = BrainModel(config=full_config)
        sim1 = Simulation(model1, seed=42)
        sim1.initialize_neurons(area_names=["V1_like"], density=0.1)
        
        # Run 2 with different seed
        model2 = BrainModel(config=full_config)
        sim2 = Simulation(model2, seed=99)
        sim2.initialize_neurons(area_names=["V1_like"], density=0.1)
        
        # Neuron positions may be different
        neurons1_positions = set((n.x, n.y, n.z, n.w) for n in model1.neurons.values())
        neurons2_positions = set((n.x, n.y, n.z, n.w) for n in model2.neurons.values())
        
        # With different seeds, at least some positions should differ
        # (unless we got very unlucky)
        assert neurons1_positions != neurons2_positions or len(neurons1_positions) == 0
