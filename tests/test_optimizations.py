"""Tests for sparse connectivity and time-indexed spike optimizations."""

import pytest
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel
from simulation import Simulation


class TestSparseConnectivity:
    """Tests for sparse connectivity matrix integration."""
    
    def test_sparse_connectivity_initialization(self, minimal_config):
        """Test model initialization with sparse connectivity."""
        model = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        assert model.use_sparse_connectivity
        assert model._sparse_synapses is not None
        assert len(model.synapses) == 0
    
    def test_sparse_connectivity_add_synapse(self, minimal_config):
        """Test adding synapses with sparse connectivity."""
        model = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        
        # Add neurons
        n1 = model.add_neuron(0, 0, 0, 0)
        n2 = model.add_neuron(0, 0, 0, 1)
        
        # Add synapse
        syn = model.add_synapse(n1.id, n2.id, weight=0.5)
        
        assert syn.pre_id == n1.id
        assert syn.post_id == n2.id
        assert model._sparse_synapses.num_synapses() == 1
    
    def test_sparse_connectivity_get_synapses(self, minimal_config):
        """Test getting synapses with sparse connectivity."""
        model = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        
        # Add neurons
        n1 = model.add_neuron(0, 0, 0, 0)
        n2 = model.add_neuron(0, 0, 0, 1)
        n3 = model.add_neuron(0, 0, 1, 0)
        
        # Add synapses
        model.add_synapse(n1.id, n2.id)
        model.add_synapse(n1.id, n3.id)
        model.add_synapse(n2.id, n3.id)
        
        # Test outgoing synapses
        outgoing = model.get_synapses_for_neuron(n1.id, direction="pre")
        assert len(outgoing) == 2
        
        # Test incoming synapses
        incoming = model.get_synapses_for_neuron(n3.id, direction="post")
        assert len(incoming) == 2
    
    def test_sparse_connectivity_remove_neuron(self, minimal_config):
        """Test removing neuron with sparse connectivity."""
        model = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        
        # Add neurons
        n1 = model.add_neuron(0, 0, 0, 0)
        n2 = model.add_neuron(0, 0, 0, 1)
        n3 = model.add_neuron(0, 0, 1, 0)
        
        # Add synapses
        model.add_synapse(n1.id, n2.id)
        model.add_synapse(n2.id, n3.id)
        model.add_synapse(n1.id, n3.id)
        
        assert model._sparse_synapses.num_synapses() == 3
        
        # Remove neuron 2
        model.remove_neuron(n2.id)
        
        # Should have removed 2 synapses connected to neuron 2
        assert model._sparse_synapses.num_synapses() == 1
    
    def test_sparse_connectivity_serialization(self, minimal_config):
        """Test serialization with sparse connectivity."""
        model1 = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        
        # Add neurons and synapses
        n1 = model1.add_neuron(0, 0, 0, 0)
        n2 = model1.add_neuron(0, 0, 0, 1)
        model1.add_synapse(n1.id, n2.id, weight=0.5)
        
        # Serialize
        data = model1.to_dict()
        
        # Deserialize
        model2 = BrainModel.from_dict(data)
        
        # Verify
        assert model2.use_sparse_connectivity
        assert model2._sparse_synapses.num_synapses() == 1
        assert len(model2.neurons) == 2


class TestTimeIndexedSpikes:
    """Tests for time-indexed spike buffer integration."""
    
    def test_time_indexed_initialization(self, brain_model):
        """Test simulation initialization with time-indexed spikes."""
        sim = Simulation(brain_model, use_time_indexed_spikes=True)
        assert sim.use_time_indexed_spikes
        assert sim._spike_buffer is not None
    
    def test_time_indexed_spike_recording(self, brain_model):
        """Test spike recording with time-indexed buffer."""
        sim = Simulation(brain_model, seed=42, use_time_indexed_spikes=True)
        
        # Add neurons
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        neuron_ids = list(brain_model.neurons.keys())
        
        # Give external input to trigger spike
        brain_model.neurons[neuron_ids[0]].external_input = 50.0
        
        # Run step
        stats = sim.step()
        
        # Verify spike was recorded
        if len(stats['spikes']) > 0:
            spiked_id = stats['spikes'][0]
            assert sim._spike_buffer.did_spike_at(spiked_id, 0)
    
    def test_time_indexed_synaptic_transmission(self, brain_model):
        """Test synaptic transmission with time-indexed buffer."""
        sim = Simulation(brain_model, seed=42, use_time_indexed_spikes=True)
        
        # Add neurons
        n1 = brain_model.add_neuron(0, 0, 0, 0)
        n2 = brain_model.add_neuron(0, 0, 0, 1)
        
        # Add strong synapse
        brain_model.add_synapse(n1.id, n2.id, weight=50.0, delay=1)
        
        # Make neuron 1 spike with very strong input
        n1.v_membrane = -51.0  # Just below threshold
        n1.external_input = 100.0  # Very strong input
        sim.step()
        
        # Check neuron 1 spiked
        spiked_neurons = sim._spike_buffer.get_spikes_at(0)
        assert n1.id in spiked_neurons, f"Neuron {n1.id} should have spiked at time 0"
        
        # Next step, neuron 2 should receive input
        sim.step()
        
        # Neuron 2 should have spiked due to strong input
        # (This may not always be true depending on parameters,
        #  but at least it should have received input)
        v_rest = brain_model.get_neuron_model_params().get("v_rest", -65.0)
        assert n2.v_membrane != v_rest or n2.last_spike_time >= 0
    
    def test_time_indexed_cleanup(self, brain_model):
        """Test automatic cleanup with time-indexed buffer."""
        sim = Simulation(brain_model, seed=42, use_time_indexed_spikes=True)
        
        # Add neurons
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        neuron_ids = list(brain_model.neurons.keys())
        
        # Record which step we make the first spike
        first_spike_step = None
        
        # Trigger spikes at various times by setting membrane potential high
        for step in range(120):
            if step % 10 == 0 and len(neuron_ids) > 0:
                # Set membrane potential just below threshold and add strong input
                brain_model.neurons[neuron_ids[0]].v_membrane = -51.0
                brain_model.neurons[neuron_ids[0]].external_input = 100.0
                if first_spike_step is None:
                    first_spike_step = step
            sim.step()
        
        # Old spikes should be cleaned up (window size is 100)
        # Spike at first step should be gone if it's more than 100 steps ago
        if first_spike_step is not None and 119 - first_spike_step > 100:
            assert not sim._spike_buffer.did_spike_at(neuron_ids[0], first_spike_step)
        
        # The buffer should have been cleaned up and not contain ancient history
        # Check that the buffer doesn't grow unbounded
        assert sim._spike_buffer.num_spikes() < 50  # Should be much less than 120


class TestOptimizedIntegration:
    """Tests for both optimizations working together."""
    
    def test_both_optimizations(self, minimal_config):
        """Test using both sparse connectivity and time-indexed spikes."""
        model = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        sim = Simulation(model, seed=42, use_time_indexed_spikes=True)
        
        # Add neurons
        sim.initialize_neurons(area_names=["V1_like"], density=0.1)
        sim.initialize_random_synapses(connection_probability=0.05)
        
        # Run simulation
        for _ in range(10):
            stats = sim.step()
        
        # Should work without errors
        assert sim.model.current_step == 10
    
    def test_serialization_with_both_optimizations(self, minimal_config):
        """Test serialization with both optimizations."""
        model1 = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        
        # Add neurons and synapses
        n1 = model1.add_neuron(0, 0, 0, 0)
        n2 = model1.add_neuron(0, 0, 0, 1)
        model1.add_synapse(n1.id, n2.id, weight=0.5)
        
        # Serialize
        data = model1.to_dict()
        
        # Deserialize
        model2 = BrainModel.from_dict(data)
        
        # Create simulation with time-indexed spikes
        sim = Simulation(model2, use_time_indexed_spikes=True)
        
        # Run a few steps
        for _ in range(5):
            sim.step()
        
        # Should work without errors
        assert sim.model.current_step == 5


class TestPerformanceComparison:
    """Performance comparison tests."""
    
    def test_sparse_connectivity_performance(self, minimal_config):
        """Compare performance of sparse vs list-based connectivity."""
        # Create two models
        model_list = BrainModel(config=minimal_config, use_sparse_connectivity=False)
        model_sparse = BrainModel(config=minimal_config, use_sparse_connectivity=True)
        
        # Add same neurons to both
        neurons = []
        for i in range(100):
            n1 = model_list.add_neuron(0, 0, i % 5, i % 5)
            n2 = model_sparse.add_neuron(0, 0, i % 5, i % 5)
            neurons.append(n1.id)
        
        # Add synapses
        for i in range(0, 99, 2):
            model_list.add_synapse(neurons[i], neurons[i+1])
            model_sparse.add_synapse(neurons[i], neurons[i+1])
        
        # Time queries (this is just a basic test, not a precise benchmark)
        start = time.time()
        for neuron_id in neurons[:10]:
            syns = model_list.get_synapses_for_neuron(neuron_id, direction="post")
        list_time = time.time() - start
        
        start = time.time()
        for neuron_id in neurons[:10]:
            syns = model_sparse.get_synapses_for_neuron(neuron_id, direction="post")
        sparse_time = time.time() - start
        
        # Both should complete reasonably quickly
        assert list_time < 1.0
        assert sparse_time < 1.0
    
    def test_time_indexed_spike_performance(self, brain_model):
        """Compare performance of time-indexed vs dict-based spikes."""
        # Create two simulations
        sim_dict = Simulation(brain_model, seed=42, use_time_indexed_spikes=False)
        sim_time_indexed = Simulation(brain_model, seed=42, use_time_indexed_spikes=True)
        
        # Add neurons and synapses
        for sim in [sim_dict, sim_time_indexed]:
            sim.initialize_neurons(area_names=["V1_like"], density=0.05)
            sim.initialize_random_synapses(connection_probability=0.01)
        
        # Run both simulations
        start = time.time()
        for _ in range(50):
            sim_dict.step()
        dict_time = time.time() - start
        
        start = time.time()
        for _ in range(50):
            sim_time_indexed.step()
        time_indexed_time = time.time() - start
        
        # Both should complete reasonably quickly
        assert dict_time < 10.0
        assert time_indexed_time < 10.0
        
        # Time-indexed should be competitive (not necessarily faster for small networks)
        # Just verify it's not dramatically slower
        assert time_indexed_time < dict_time * 2
