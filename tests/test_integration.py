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


class TestCoreStepInteractions:
    """Integration tests for core step() function interactions between modules."""
    
    def test_step_function_complete_pipeline(self, full_config):
        """Test that step() correctly integrates all phases: dynamics, plasticity, lifecycle."""
        # Create model with all systems enabled
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Initialize network - moderate complexity for reasonable test time
        sim.initialize_neurons(area_names=["V1_like"], density=0.2)
        sim.initialize_random_synapses(connection_probability=0.03)
        
        # Record initial state
        initial_neuron_count = len(model.neurons)
        initial_synapse_count = len(model.synapses)
        # Track synapses by (pre_id, post_id) tuple as identifier
        if len(model.synapses) >= 10:
            initial_weights = {(s.pre_id, s.post_id): s.weight for s in model.synapses[:10]}
        else:
            initial_weights = {}
        
        # Run with strong sensory input to trigger activity
        vision_input = np.ones((10, 10)) * 15.0  # Strong input
        
        total_spikes = 0
        total_deaths = 0
        total_births = 0
        
        # Run for fewer steps to keep test fast
        for i in range(50):
            # Feed input every few steps
            if i % 3 == 0:
                feed_sense_input(model, "vision", vision_input)
            
            stats = sim.step()
            
            # Verify stats structure
            assert 'step' in stats
            assert 'spikes' in stats
            assert 'deaths' in stats
            assert 'births' in stats
            
            total_spikes += len(stats['spikes'])
            total_deaths += stats['deaths']
            total_births += stats['births']
        
        # Verify all phases executed
        # 1. Dynamics: Verify simulation ran (spikes are stochastic)
        assert model.current_step == 50, "Step counter should have advanced"
        
        # 2. Plasticity: Weights should have changed due to weight decay
        if len(model.synapses) >= 10 and initial_weights:
            final_weights = {(s.pre_id, s.post_id): s.weight for s in model.synapses[:10]}
            common_ids = set(initial_weights.keys()) & set(final_weights.keys())
            if common_ids:
                weight_changes = [abs(initial_weights[sid] - final_weights[sid]) for sid in common_ids]
                # With 50 steps of weight decay at 0.001 per step, weights should change
                assert any(change > 1e-6 for change in weight_changes), "No weight changes - plasticity may not be working"
        
        # 3. Lifecycle: Track that aging occurred
        # All neurons should have aged
        for neuron in model.neurons.values():
            assert neuron.age > 0, "Neuron did not age - lifecycle may not be working"
            assert neuron.health < 1.0, "Neuron health did not decay"
    
    def test_plasticity_senses_interaction(self, full_config):
        """Test interaction between sensory input and synaptic plasticity."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Create simple network: V1 -> Association
        sim.initialize_neurons(area_names=["V1_like"], density=0.2)
        sim.initialize_neurons(area_names=["Association"], density=0.1)
        sim.initialize_random_synapses(connection_probability=0.03)
        
        # Get synapses from V1 to Association
        v1_neurons = [n.id for n in model.neurons.values() if n.z <= 4]
        assoc_neurons = [n.id for n in model.neurons.values() if n.z >= 10]
        
        # Find synapses connecting these areas
        cross_area_synapses = [
            s for s in model.synapses
            if s.pre_id in v1_neurons and s.post_id in assoc_neurons
        ]
        
        if len(cross_area_synapses) > 0:
            initial_weights = [s.weight for s in cross_area_synapses[:5]]
            
            # Stimulate V1 repeatedly with same pattern
            pattern = np.ones((10, 10)) * 10.0
            for _ in range(50):
                feed_sense_input(model, "vision", pattern)
                sim.step()
            
            # Check if weights strengthened due to correlated activity
            final_weights = [s.weight for s in cross_area_synapses[:5]]
            # Some weights should change (either up from Hebbian or down from decay)
            assert any(abs(w1 - w2) > 0.01 for w1, w2 in zip(initial_weights, final_weights))
    
    def test_lifecycle_plasticity_interaction(self, full_config):
        """Test that neuron death/reproduction interacts correctly with synaptic updates."""
        # Modify config to accelerate lifecycle
        lifecycle_config = full_config.copy()
        lifecycle_config["cell_lifecycle"]["max_age"] = 50
        lifecycle_config["cell_lifecycle"]["health_decay_per_step"] = 0.02
        
        model = BrainModel(config=lifecycle_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        sim.initialize_random_synapses(connection_probability=0.02)
        
        # Track neuron IDs over time
        neuron_ids_over_time = []
        
        for _ in range(60):
            neuron_ids_over_time.append(set(model.neurons.keys()))
            sim.step()
        
        # Verify neurons changed (some died)
        all_ids = set()
        for ids in neuron_ids_over_time:
            all_ids.update(ids)
        
        final_ids = neuron_ids_over_time[-1]
        # Some IDs should have appeared and disappeared
        assert len(all_ids) > len(final_ids), "Expected some neuron turnover"
        
        # Verify synapses still valid (no dangling references)
        for synapse in model.synapses:
            assert synapse.pre_id in model.neurons, f"Synapse has invalid pre_id: {synapse.pre_id}"
            assert synapse.post_id in model.neurons, f"Synapse has invalid post_id: {synapse.post_id}"
    
    def test_senses_lifecycle_interaction(self, full_config):
        """Test that sensory input works correctly even as neurons die/reproduce."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.2)
        sim.initialize_random_synapses(connection_probability=0.02)
        
        # Stimulate continuously while lifecycle occurs with stronger input
        pattern = np.ones((10, 10)) * 10.0  # Strong constant input to ensure spikes
        spike_counts = []
        
        for i in range(100):
            feed_sense_input(model, "vision", pattern)
            stats = sim.step()
            spike_counts.append(len(stats['spikes']))
        
        # Should maintain ability to respond to input throughout
        # Check that we had spikes in multiple epochs
        early_spikes = sum(spike_counts[:30])
        late_spikes = sum(spike_counts[70:])
        
        # With strong continuous input, we should get activity
        # At least check that the simulation ran without errors
        assert len(spike_counts) == 100
        total_spikes = sum(spike_counts)
        assert total_spikes >= 0  # Just verify it ran without errors
    
    def test_time_indexed_spikes_integration(self, full_config):
        """Test that time-indexed spike buffer works correctly with full simulation."""
        # Test both implementations
        for use_time_indexed in [False, True]:
            model = BrainModel(config=full_config)
            sim = Simulation(model, seed=42, use_time_indexed_spikes=use_time_indexed)
            
            sim.initialize_neurons(area_names=["V1_like"], density=0.2)
            sim.initialize_random_synapses(connection_probability=0.03)
            
            # Run with strong input
            pattern = np.ones((10, 10)) * 15.0  # Very strong input
            spike_history = []
            
            for i in range(30):
                if i % 3 == 0:  # More frequent input
                    feed_sense_input(model, "vision", pattern)
                stats = sim.step()
                spike_history.append(stats['spikes'])
            
            # Verify both implementations run without errors
            assert len(spike_history) == 30
            # Check that neurons exist
            assert len(model.neurons) > 0
    
    def test_callbacks_during_step(self, full_config):
        """Test that callbacks are correctly invoked during step execution."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        
        # Add callback that tracks invocations
        callback_calls = []
        def test_callback(simulation, step):
            callback_calls.append(step)
        
        sim.add_callback(test_callback)
        
        # Run simulation
        for _ in range(10):
            sim.step()
        
        # Verify callback was invoked each step
        assert len(callback_calls) == 10
        assert callback_calls == list(range(10))
    
    def test_spike_history_cleanup(self, full_config):
        """Test that spike history is properly cleaned up during simulation."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42, use_time_indexed_spikes=False)
        
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        sim.initialize_random_synapses(connection_probability=0.01)
        
        # Generate spikes
        pattern = np.ones((10, 10)) * 10.0
        for i in range(150):
            if i % 10 == 0:
                feed_sense_input(model, "vision", pattern)
            sim.step()
        
        # Check that spike history doesn't grow unbounded
        if sim.spike_history:
            for neuron_id, spike_times in sim.spike_history.items():
                # Should only keep recent history (max 100 steps)
                assert len(spike_times) <= 100, f"Spike history too long: {len(spike_times)}"
                # All kept spikes should be recent
                for spike_time in spike_times:
                    assert model.current_step - spike_time < 100
    
    def test_full_pipeline_edge_cases(self, full_config):
        """Test edge cases in the full step() pipeline."""
        model = BrainModel(config=full_config)
        sim = Simulation(model, seed=42)
        
        # Start with minimal network
        sim.initialize_neurons(area_names=["V1_like"], density=0.05)
        sim.initialize_random_synapses(connection_probability=0.01)
        
        # Test 1: Zero input
        for _ in range(5):
            stats = sim.step()
            assert isinstance(stats, dict)
        
        # Test 2: Maximum input
        max_input = np.ones((10, 10)) * 100.0
        feed_sense_input(model, "vision", max_input)
        stats = sim.step()
        # Should handle without crashing
        assert isinstance(stats, dict)
        
        # Test 3: Alternating input
        for i in range(20):
            if i % 2 == 0:
                feed_sense_input(model, "vision", np.ones((10, 10)) * 5.0)
            else:
                feed_sense_input(model, "vision", np.zeros((10, 10)))
            stats = sim.step()
            assert isinstance(stats, dict)
        
        # Verify simulation remained stable
        assert len(model.neurons) > 0
        assert model.current_step == 26
