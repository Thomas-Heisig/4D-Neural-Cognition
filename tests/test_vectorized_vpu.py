"""Tests for Vectorized Virtual Processing Unit."""

import pytest
import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brain_model import BrainModel
from simulation import Simulation
from hardware_abstraction.vectorized_vpu import VectorizedVPU


class TestVectorizedVPU:
    """Tests for VectorizedVPU class."""
    
    def test_vectorized_vpu_initialization(self):
        """Test that vectorized VPU initializes correctly."""
        vpu = VectorizedVPU(vpu_id=0, clock_speed_hz=20e6, buffer_size=10)
        
        assert vpu.vpu_id == 0
        assert vpu.clock_speed == 20e6
        assert vpu.buffer_size == 10
        assert vpu.neuron_batch_data is None
        assert vpu.input_buffer is None
        assert len(vpu.output_buffer) == 0
    
    def test_vectorized_batch_initialization_empty(self):
        """Test initialization with no neurons in slice."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))  # Slice with no neurons
        vpu.initialize_batch_vectorized(model, sim)
        
        assert len(vpu.neuron_batch_data) == 0
        assert vpu.input_buffer.shape == (10, 0)
    
    def test_vectorized_batch_initialization_with_neurons(self):
        """Test initialization with neurons in slice."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add neurons
        n1 = model.add_neuron(2, 2, 1, 0)
        n2 = model.add_neuron(3, 3, 1, 0)
        n3 = model.add_neuron(4, 4, 1, 0)
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        assert len(vpu.neuron_batch_data) == 3
        assert vpu.input_buffer.shape == (10, 3)
        
        # Check that neuron IDs are correct
        neuron_ids = set(vpu.neuron_batch_data['id'])
        assert n1.id in neuron_ids
        assert n2.id in neuron_ids
        assert n3.id in neuron_ids
        
        # Check that parameters are initialized correctly
        assert np.all(vpu.neuron_batch_data['v_rest'] == -65.0)
        assert np.all(vpu.neuron_batch_data['v_threshold'] == -50.0)
        assert np.all(vpu.neuron_batch_data['tau_m'] == 20.0)
    
    def test_gather_inputs_vectorized(self):
        """Test vectorized input gathering."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add neurons
        n1 = model.add_neuron(2, 2, 1, 0)
        n2 = model.add_neuron(3, 3, 1, 0)
        
        # Set external inputs
        model.neurons[n1.id].external_input = 5.0
        model.neurons[n2.id].external_input = 3.0
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        inputs = vpu.gather_inputs_vectorized(global_clock_cycle=0)
        
        assert len(inputs) == 2
        assert 5.0 in inputs or 3.0 in inputs  # Order may vary
    
    def test_process_cycle_vectorized_no_spikes(self):
        """Test vectorized processing without spikes."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add neurons with low membrane potential (won't spike)
        n1 = model.add_neuron(2, 2, 1, 0)
        model.neurons[n1.id].v_membrane = -70.0
        model.neurons[n1.id].external_input = 0.0
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        result = vpu.process_cycle_vectorized(global_clock_cycle=0)
        
        assert result["vpu_id"] == 0
        assert result["cycle"] == 0
        assert result["neurons_processed"] == 1
        assert result["spikes"] == 0
        assert result["vectorized"] is True
        assert len(result["spike_neuron_ids"]) == 0
    
    def test_process_cycle_vectorized_with_spikes(self):
        """Test vectorized processing with spikes."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add neuron with high membrane potential (will spike)
        n1 = model.add_neuron(2, 2, 1, 0)
        model.neurons[n1.id].v_membrane = -49.0  # Above threshold
        model.neurons[n1.id].external_input = 0.0
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        result = vpu.process_cycle_vectorized(global_clock_cycle=0)
        
        assert result["spikes"] == 1
        assert n1.id in result["spike_neuron_ids"]
        
        # Check that membrane potential was reset
        assert model.neurons[n1.id].v_membrane == -70.0  # v_reset
        assert model.neurons[n1.id].last_spike_time == 0
    
    def test_vectorized_membrane_potential_update(self):
        """Test that membrane potential updates correctly with LIF dynamics."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add neuron
        n1 = model.add_neuron(2, 2, 1, 0)
        model.neurons[n1.id].v_membrane = -70.0
        model.neurons[n1.id].external_input = 2.0  # Positive input
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        initial_v = model.neurons[n1.id].v_membrane
        
        # Process multiple cycles
        for cycle in range(10):
            vpu.process_cycle_vectorized(global_clock_cycle=cycle)
        
        final_v = model.neurons[n1.id].v_membrane
        
        # Membrane potential should increase due to positive input
        assert final_v > initial_v
    
    def test_vectorized_refractory_period(self):
        """Test that refractory period prevents immediate re-spiking."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add neuron that will spike
        n1 = model.add_neuron(2, 2, 1, 0)
        model.neurons[n1.id].v_membrane = -49.0  # Above threshold
        model.neurons[n1.id].external_input = 10.0  # Strong input
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        # First cycle should spike
        result1 = vpu.process_cycle_vectorized(global_clock_cycle=0)
        assert result1["spikes"] == 1
        
        # During refractory period, should not spike even with strong input
        result2 = vpu.process_cycle_vectorized(global_clock_cycle=1)
        result3 = vpu.process_cycle_vectorized(global_clock_cycle=2)
        result4 = vpu.process_cycle_vectorized(global_clock_cycle=3)
        
        # Should not spike during refractory period
        assert result2["spikes"] == 0
        assert result3["spikes"] == 0
        assert result4["spikes"] == 0
    
    def test_get_neuron_states(self):
        """Test getting neuron states as array."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        n1 = model.add_neuron(2, 2, 1, 0)
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        states = vpu.get_neuron_states()
        
        assert len(states) == 1
        assert states[0]['id'] == n1.id
        assert 'v_membrane' in states.dtype.names
        assert 'v_threshold' in states.dtype.names
    
    def test_get_spike_mask(self):
        """Test getting spike mask."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        
        # Add two neurons, one will spike
        n1 = model.add_neuron(2, 2, 1, 0)
        n2 = model.add_neuron(3, 3, 1, 0)
        model.neurons[n1.id].v_membrane = -49.0  # Will spike
        model.neurons[n2.id].v_membrane = -70.0  # Won't spike
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        vpu.process_cycle_vectorized(global_clock_cycle=0)
        spike_mask = vpu.get_spike_mask()
        
        assert len(spike_mask) == 2
        assert np.sum(spike_mask) == 1  # One neuron spiked
    
    def test_vectorized_statistics(self):
        """Test that statistics are updated correctly."""
        config = {
            "lattice_shape": [10, 10, 5, 4],
            "neuron_model": {
                "type": "LIF",
                "params_default": {
                    "tau_m": 20.0,
                    "v_rest": -65.0,
                    "v_reset": -70.0,
                    "v_threshold": -50.0,
                    "refractory_period": 5.0,
                }
            },
            "cell_lifecycle": {
                "enable_death": False,
                "enable_reproduction": False,
                "max_age": 1000,
            },
            "plasticity": {"learning_rate": 0.01},
            "senses": {},
            "areas": []
        }
        
        model = BrainModel(config=config)
        n1 = model.add_neuron(2, 2, 1, 0)
        n2 = model.add_neuron(3, 3, 1, 0)
        
        sim = Simulation(model, use_vnc=False)
        
        vpu = VectorizedVPU(vpu_id=0)
        vpu.assign_slice((0, 5, 0, 5, 0, 2, 0, 0))
        vpu.initialize_batch_vectorized(model, sim)
        
        # Process 5 cycles
        for cycle in range(5):
            vpu.process_cycle_vectorized(global_clock_cycle=cycle)
        
        stats = vpu.get_statistics()
        
        assert stats["cycles_executed"] == 5
        assert stats["neurons_processed"] == 10  # 2 neurons * 5 cycles
        assert "avg_processing_time_ms" in stats
        assert "neurons_per_second" in stats
