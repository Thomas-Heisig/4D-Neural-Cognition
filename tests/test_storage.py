"""Unit tests for storage module."""

import pytest
import os
import tempfile
import json
from src.brain_model import BrainModel
from src.storage import save_to_hdf5, load_from_hdf5, save_to_json, load_from_json


@pytest.fixture
def model():
    """Create a test brain model."""
    config = {
        "lattice_shape": [10, 10, 10, 10],
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
            "aging_rate": 0.001,
            "death_threshold": 0.1,
            "reproduction_threshold": 0.9,
            "reproduction_probability": 0.1
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
                    "x": [0, 9],
                    "y": [0, 9],
                    "z": [0, 4],
                    "w": [0, 0]
                }
            }
        ]
    }
    model = BrainModel(config=config)
    
    # Add some neurons
    for i in range(5):
        model.add_neuron(i, i, i % 10, 0, health=0.9)
    
    # Add some synapses
    model.add_synapse(0, 1, weight=0.5, delay=1)
    model.add_synapse(1, 2, weight=0.3, delay=2)
    
    return model


class TestJSONStorage:
    """Tests for JSON storage functions."""
    
    def test_save_to_json(self, model):
        """Test saving model to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.json")
            save_to_json(model, filepath)
            
            assert os.path.exists(filepath)
            
            # Verify file is valid JSON
            with open(filepath, 'r') as f:
                data = json.load(f)
                assert isinstance(data, dict)
    
    def test_load_from_json(self, model):
        """Test loading model from JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.json")
            
            # Save then load
            save_to_json(model, filepath)
            loaded_model = load_from_json(filepath)
            
            assert len(loaded_model.neurons) == len(model.neurons)
            assert len(loaded_model.synapses) == len(model.synapses)
            assert loaded_model.lattice_shape == model.lattice_shape
    
    def test_json_roundtrip(self, model):
        """Test that save/load preserves model data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.json")
            
            # Record original state
            original_neuron_count = len(model.neurons)
            original_synapse_count = len(model.synapses)
            original_step = model.current_step
            
            # Save and load
            save_to_json(model, filepath)
            loaded_model = load_from_json(filepath)
            
            # Verify data preserved
            assert len(loaded_model.neurons) == original_neuron_count
            assert len(loaded_model.synapses) == original_synapse_count
            assert loaded_model.current_step == original_step
    
    def test_json_neuron_properties(self, model):
        """Test that neuron properties are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.json")
            
            # Get a neuron's properties
            original_neuron = list(model.neurons.values())[0]
            original_id = original_neuron.id
            original_health = original_neuron.health
            original_pos = (original_neuron.x, original_neuron.y, 
                          original_neuron.z, original_neuron.w)
            
            # Save and load
            save_to_json(model, filepath)
            loaded_model = load_from_json(filepath)
            
            # Verify properties
            loaded_neuron = loaded_model.neurons[original_id]
            assert loaded_neuron.health == original_health
            assert (loaded_neuron.x, loaded_neuron.y, 
                   loaded_neuron.z, loaded_neuron.w) == original_pos
    
    def test_json_synapse_properties(self, model):
        """Test that synapse properties are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.json")
            
            # Get a synapse's properties
            original_synapse = model.synapses[0]
            original_pre = original_synapse.pre_id
            original_post = original_synapse.post_id
            original_weight = original_synapse.weight
            original_delay = original_synapse.delay
            
            # Save and load
            save_to_json(model, filepath)
            loaded_model = load_from_json(filepath)
            
            # Verify properties
            loaded_synapse = loaded_model.synapses[0]
            assert loaded_synapse.pre_id == original_pre
            assert loaded_synapse.post_id == original_post
            assert loaded_synapse.weight == original_weight
            assert loaded_synapse.delay == original_delay
    
    def test_json_directory_creation(self, model):
        """Test that save creates directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "test_model.json")
            
            # Directory doesn't exist yet
            assert not os.path.exists(os.path.dirname(filepath))
            
            # Save should create it
            save_to_json(model, filepath)
            
            assert os.path.exists(filepath)


class TestHDF5Storage:
    """Tests for HDF5 storage functions."""
    
    def test_save_to_hdf5(self, model):
        """Test saving model to HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            save_to_hdf5(model, filepath)
            
            assert os.path.exists(filepath)
    
    def test_load_from_hdf5(self, model):
        """Test loading model from HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            # Save then load
            save_to_hdf5(model, filepath)
            loaded_model = load_from_hdf5(filepath)
            
            assert len(loaded_model.neurons) == len(model.neurons)
            assert len(loaded_model.synapses) == len(model.synapses)
            assert loaded_model.lattice_shape == model.lattice_shape
    
    def test_hdf5_roundtrip(self, model):
        """Test that save/load preserves model data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            # Record original state
            original_neuron_count = len(model.neurons)
            original_synapse_count = len(model.synapses)
            original_step = model.current_step
            
            # Save and load
            save_to_hdf5(model, filepath)
            loaded_model = load_from_hdf5(filepath)
            
            # Verify data preserved
            assert len(loaded_model.neurons) == original_neuron_count
            assert len(loaded_model.synapses) == original_synapse_count
            assert loaded_model.current_step == original_step
    
    def test_hdf5_neuron_properties(self, model):
        """Test that neuron properties are preserved in HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            # Get a neuron's properties
            original_neuron = list(model.neurons.values())[0]
            original_id = original_neuron.id
            original_health = original_neuron.health
            original_pos = (original_neuron.x, original_neuron.y, 
                          original_neuron.z, original_neuron.w)
            
            # Save and load
            save_to_hdf5(model, filepath)
            loaded_model = load_from_hdf5(filepath)
            
            # Verify properties (with some tolerance for float conversion)
            loaded_neuron = loaded_model.neurons[original_id]
            assert abs(loaded_neuron.health - original_health) < 0.01
            assert (loaded_neuron.x, loaded_neuron.y, 
                   loaded_neuron.z, loaded_neuron.w) == original_pos
    
    def test_hdf5_synapse_properties(self, model):
        """Test that synapse properties are preserved in HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model.h5")
            
            # Get a synapse's properties
            original_synapse = model.synapses[0]
            original_pre = original_synapse.pre_id
            original_post = original_synapse.post_id
            original_weight = original_synapse.weight
            original_delay = original_synapse.delay
            
            # Save and load
            save_to_hdf5(model, filepath)
            loaded_model = load_from_hdf5(filepath)
            
            # Verify properties (with some tolerance for float conversion)
            loaded_synapse = loaded_model.synapses[0]
            assert loaded_synapse.pre_id == original_pre
            assert loaded_synapse.post_id == original_post
            assert abs(loaded_synapse.weight - original_weight) < 0.01
            assert loaded_synapse.delay == original_delay
    
    def test_hdf5_directory_creation(self, model):
        """Test that save creates directories if they don't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "subdir", "test_model.h5")
            
            # Directory doesn't exist yet
            assert not os.path.exists(os.path.dirname(filepath))
            
            # Save should create it
            save_to_hdf5(model, filepath)
            
            assert os.path.exists(filepath)
    
    def test_hdf5_empty_model(self):
        """Test saving and loading a model with no neurons or synapses."""
        config = {
            "lattice_shape": [10, 10, 10, 10],
            "neuron_model": {"params_default": {}},
            "cell_lifecycle": {},
            "plasticity": {},
            "senses": {},
            "areas": []
        }
        empty_model = BrainModel(config=config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "empty_model.h5")
            
            save_to_hdf5(empty_model, filepath)
            loaded_model = load_from_hdf5(filepath)
            
            assert len(loaded_model.neurons) == 0
            assert len(loaded_model.synapses) == 0


class TestStorageComparison:
    """Tests comparing JSON and HDF5 storage."""
    
    def test_format_interoperability(self, model):
        """Test that data is consistent between JSON and HDF5."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "model.json")
            hdf5_path = os.path.join(tmpdir, "model.h5")
            
            # Save in both formats
            save_to_json(model, json_path)
            save_to_hdf5(model, hdf5_path)
            
            # Load both
            json_model = load_from_json(json_path)
            hdf5_model = load_from_hdf5(hdf5_path)
            
            # Compare key metrics
            assert len(json_model.neurons) == len(hdf5_model.neurons)
            assert len(json_model.synapses) == len(hdf5_model.synapses)
            assert json_model.lattice_shape == hdf5_model.lattice_shape
