"""Unit tests for brain_model.py."""

import pytest
import json
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel, Neuron, Synapse


class TestNeuron:
    """Tests for Neuron dataclass."""
    
    def test_neuron_creation(self):
        """Test creating a neuron with basic parameters."""
        neuron = Neuron(
            id=0, x=1, y=2, z=3, w=4,
            generation=0, parent_id=-1,
            health=1.0, age=0, v_membrane=-65.0
        )
        assert neuron.id == 0
        assert neuron.x == 1
        assert neuron.y == 2
        assert neuron.z == 3
        assert neuron.w == 4
        assert neuron.generation == 0
        assert neuron.health == 1.0
        
    def test_neuron_position(self, sample_neuron):
        """Test neuron position method."""
        pos = sample_neuron.position()
        assert pos == (5, 5, 2, 1)
        assert len(pos) == 4


class TestSynapse:
    """Tests for Synapse dataclass."""
    
    def test_synapse_creation(self):
        """Test creating a synapse."""
        synapse = Synapse(pre_id=0, post_id=1, weight=0.5, delay=1)
        assert synapse.pre_id == 0
        assert synapse.post_id == 1
        assert synapse.weight == 0.5
        assert synapse.delay == 1
        assert synapse.plasticity_tag == 0.0


class TestBrainModel:
    """Tests for BrainModel class."""
    
    def test_init_with_config_dict(self, minimal_config):
        """Test initialization with config dictionary."""
        model = BrainModel(config=minimal_config)
        assert model.lattice_shape == (10, 10, 5, 2)
        assert len(model.neurons) == 0
        assert len(model.synapses) == 0
        assert model.current_step == 0
        
    def test_init_with_config_file(self, minimal_config, temp_dir):
        """Test initialization with config file."""
        config_path = Path(temp_dir) / "config.json"
        with open(config_path, 'w') as f:
            json.dump(minimal_config, f)
        
        model = BrainModel(config_path=str(config_path))
        assert model.lattice_shape == (10, 10, 5, 2)
        
    def test_init_without_config_raises_error(self):
        """Test that initialization without config raises ValueError."""
        with pytest.raises(ValueError, match="Either config_path or config must be provided"):
            BrainModel()
    
    def test_get_neuron_model_params(self, brain_model):
        """Test getting neuron model parameters."""
        params = brain_model.get_neuron_model_params()
        assert "tau_m" in params
        assert "v_rest" in params
        assert params["tau_m"] == 20.0
        assert params["v_rest"] == -65.0
        
    def test_get_lifecycle_config(self, brain_model):
        """Test getting lifecycle configuration."""
        lifecycle = brain_model.get_lifecycle_config()
        assert "enable_death" in lifecycle
        assert "max_age" in lifecycle
        assert lifecycle["max_age"] == 1000
        
    def test_get_plasticity_config(self, brain_model):
        """Test getting plasticity configuration."""
        plasticity = brain_model.get_plasticity_config()
        assert "learning_rate" in plasticity
        assert "weight_min" in plasticity
        assert plasticity["learning_rate"] == 0.01
        
    def test_get_senses(self, brain_model):
        """Test getting senses configuration."""
        senses = brain_model.get_senses()
        assert "vision" in senses
        assert "digital" in senses
        assert senses["vision"]["areal"] == "V1_like"
        
    def test_get_areas(self, brain_model):
        """Test getting areas configuration."""
        areas = brain_model.get_areas()
        assert len(areas) == 2
        assert areas[0]["name"] in ["V1_like", "Digital_sensor"]
    
    def test_add_neuron(self, brain_model):
        """Test adding a neuron to the model."""
        neuron = brain_model.add_neuron(x=1, y=2, z=3, w=4)
        
        assert neuron.id == 0
        assert neuron.x == 1
        assert neuron.y == 2
        assert neuron.z == 3
        assert neuron.w == 4
        assert neuron.id in brain_model.neurons
        assert brain_model._next_neuron_id == 1
        
    def test_add_multiple_neurons(self, brain_model):
        """Test adding multiple neurons."""
        n1 = brain_model.add_neuron(1, 1, 1, 1)
        n2 = brain_model.add_neuron(2, 2, 2, 2)
        n3 = brain_model.add_neuron(3, 3, 3, 3)
        
        assert len(brain_model.neurons) == 3
        assert n1.id == 0
        assert n2.id == 1
        assert n3.id == 2
        
    def test_add_neuron_with_custom_params(self, brain_model):
        """Test adding neuron with custom parameters."""
        custom_params = {"tau_m": 30.0, "v_rest": -70.0}
        neuron = brain_model.add_neuron(1, 1, 1, 1, params=custom_params)
        
        assert neuron.params["tau_m"] == 30.0
        assert neuron.params["v_rest"] == -70.0
        
    def test_remove_neuron(self, brain_model):
        """Test removing a neuron."""
        neuron = brain_model.add_neuron(1, 1, 1, 1)
        neuron_id = neuron.id
        
        brain_model.remove_neuron(neuron_id)
        assert neuron_id not in brain_model.neurons
        
    def test_remove_neuron_removes_synapses(self, brain_model):
        """Test that removing a neuron also removes its synapses."""
        n1 = brain_model.add_neuron(1, 1, 1, 1)
        n2 = brain_model.add_neuron(2, 2, 2, 2)
        n3 = brain_model.add_neuron(3, 3, 3, 3)
        
        brain_model.add_synapse(n1.id, n2.id)
        brain_model.add_synapse(n2.id, n3.id)
        brain_model.add_synapse(n3.id, n1.id)
        
        assert len(brain_model.synapses) == 3
        
        brain_model.remove_neuron(n2.id)
        
        # Synapses involving n2 should be removed
        assert len(brain_model.synapses) == 1
        remaining_synapse = brain_model.synapses[0]
        assert remaining_synapse.pre_id == n3.id
        assert remaining_synapse.post_id == n1.id
        
    def test_add_synapse(self, brain_model):
        """Test adding a synapse."""
        n1 = brain_model.add_neuron(1, 1, 1, 1)
        n2 = brain_model.add_neuron(2, 2, 2, 2)
        
        synapse = brain_model.add_synapse(n1.id, n2.id, weight=0.5, delay=2)
        
        assert synapse.pre_id == n1.id
        assert synapse.post_id == n2.id
        assert synapse.weight == 0.5
        assert synapse.delay == 2
        assert synapse in brain_model.synapses
        
    def test_get_synapses_for_neuron_pre(self, brain_model):
        """Test getting outgoing synapses for a neuron."""
        n1 = brain_model.add_neuron(1, 1, 1, 1)
        n2 = brain_model.add_neuron(2, 2, 2, 2)
        n3 = brain_model.add_neuron(3, 3, 3, 3)
        
        s1 = brain_model.add_synapse(n1.id, n2.id)
        s2 = brain_model.add_synapse(n1.id, n3.id)
        s3 = brain_model.add_synapse(n2.id, n3.id)
        
        pre_synapses = brain_model.get_synapses_for_neuron(n1.id, direction="pre")
        assert len(pre_synapses) == 2
        assert s1 in pre_synapses
        assert s2 in pre_synapses
        
    def test_get_synapses_for_neuron_post(self, brain_model):
        """Test getting incoming synapses for a neuron."""
        n1 = brain_model.add_neuron(1, 1, 1, 1)
        n2 = brain_model.add_neuron(2, 2, 2, 2)
        n3 = brain_model.add_neuron(3, 3, 3, 3)
        
        s1 = brain_model.add_synapse(n1.id, n3.id)
        s2 = brain_model.add_synapse(n2.id, n3.id)
        s3 = brain_model.add_synapse(n1.id, n2.id)
        
        post_synapses = brain_model.get_synapses_for_neuron(n3.id, direction="post")
        assert len(post_synapses) == 2
        assert s1 in post_synapses
        assert s2 in post_synapses
        
    def test_get_synapses_for_neuron_both(self, brain_model):
        """Test getting all synapses for a neuron."""
        n1 = brain_model.add_neuron(1, 1, 1, 1)
        n2 = brain_model.add_neuron(2, 2, 2, 2)
        n3 = brain_model.add_neuron(3, 3, 3, 3)
        
        s1 = brain_model.add_synapse(n1.id, n2.id)
        s2 = brain_model.add_synapse(n2.id, n3.id)
        s3 = brain_model.add_synapse(n3.id, n2.id)
        
        both_synapses = brain_model.get_synapses_for_neuron(n2.id, direction="both")
        assert len(both_synapses) == 3
        
    def test_coord_to_id_map(self, brain_model):
        """Test coordinate to ID mapping."""
        n1 = brain_model.add_neuron(1, 2, 3, 4)
        n2 = brain_model.add_neuron(5, 6, 7, 8)
        
        coord_map = brain_model.coord_to_id_map()
        
        assert coord_map[(1, 2, 3, 4)] == n1.id
        assert coord_map[(5, 6, 7, 8)] == n2.id
        assert len(coord_map) == 2
        
    def test_get_area_neurons(self, brain_model):
        """Test getting neurons in a specific area."""
        # Add neurons in V1_like area (x:[0,4], y:[0,4], z:[0,2], w:[0,0])
        n1 = brain_model.add_neuron(1, 1, 1, 0)  # Inside V1_like
        n2 = brain_model.add_neuron(3, 3, 2, 0)  # Inside V1_like
        n3 = brain_model.add_neuron(7, 2, 1, 0)  # Inside Digital_sensor
        
        v1_neurons = brain_model.get_area_neurons("V1_like")
        assert len(v1_neurons) == 2
        assert n1 in v1_neurons
        assert n2 in v1_neurons
        assert n3 not in v1_neurons
        
    def test_get_area_neurons_nonexistent(self, brain_model):
        """Test getting neurons from nonexistent area."""
        neurons = brain_model.get_area_neurons("NonExistent")
        assert len(neurons) == 0
        
    def test_to_dict(self, brain_model):
        """Test converting model to dictionary."""
        brain_model.add_neuron(1, 2, 3, 4)
        brain_model.add_neuron(5, 6, 7, 8)
        brain_model.add_synapse(0, 1)
        
        data = brain_model.to_dict()
        
        assert "config" in data
        assert "neurons" in data
        assert "synapses" in data
        assert "current_step" in data
        assert len(data["neurons"]) == 2
        assert len(data["synapses"]) == 1
        
    def test_from_dict(self, brain_model):
        """Test creating model from dictionary."""
        brain_model.add_neuron(1, 2, 3, 4)
        brain_model.add_neuron(5, 6, 7, 8)
        brain_model.add_synapse(0, 1, weight=0.7)
        brain_model.current_step = 42
        
        data = brain_model.to_dict()
        restored_model = BrainModel.from_dict(data)
        
        assert len(restored_model.neurons) == 2
        assert len(restored_model.synapses) == 1
        assert restored_model.current_step == 42
        assert restored_model.synapses[0].weight == 0.7
        
    def test_serialization_roundtrip(self, brain_model):
        """Test full serialization and deserialization."""
        # Populate model
        n1 = brain_model.add_neuron(1, 2, 3, 4, generation=1, health=0.8)
        n2 = brain_model.add_neuron(5, 6, 7, 8, generation=2, health=0.9)
        brain_model.add_synapse(n1.id, n2.id, weight=0.5, delay=2)
        brain_model.current_step = 100
        
        # Serialize and restore
        data = brain_model.to_dict()
        restored = BrainModel.from_dict(data)
        
        # Verify all details
        assert len(restored.neurons) == 2
        assert len(restored.synapses) == 1
        assert restored.current_step == 100
        
        restored_n1 = restored.neurons[n1.id]
        assert restored_n1.x == 1
        assert restored_n1.generation == 1
        assert restored_n1.health == 0.8
        
        restored_synapse = restored.synapses[0]
        assert restored_synapse.weight == 0.5
        assert restored_synapse.delay == 2
