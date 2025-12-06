"""Unit tests for senses module."""

import pytest
import numpy as np
from src.brain_model import BrainModel
from src.senses import (
    coord_to_id, 
    id_to_coord, 
    feed_sense_input, 
    create_digital_sense_input
)


@pytest.fixture
def model():
    """Create a test brain model with senses."""
    config = {
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
            }
        ]
    }
    return BrainModel(config=config)


class TestCoordinateConversion:
    """Tests for coordinate conversion functions."""
    
    def test_coord_to_id_basic(self):
        """Test basic coordinate to ID conversion."""
        lattice_shape = (10, 10, 10, 10)
        neuron_id = coord_to_id(0, 0, 0, 0, lattice_shape)
        assert neuron_id == 0
        
    def test_coord_to_id_nonzero(self):
        """Test coordinate to ID conversion with non-zero values."""
        lattice_shape = (10, 10, 10, 10)
        neuron_id = coord_to_id(1, 0, 0, 0, lattice_shape)
        assert neuron_id == 1
        
    def test_id_to_coord_basic(self):
        """Test basic ID to coordinate conversion."""
        lattice_shape = (10, 10, 10, 10)
        coords = id_to_coord(0, lattice_shape)
        assert coords == (0, 0, 0, 0)
        
    def test_id_to_coord_nonzero(self):
        """Test ID to coordinate conversion with non-zero values."""
        lattice_shape = (10, 10, 10, 10)
        coords = id_to_coord(1, lattice_shape)
        assert coords == (1, 0, 0, 0)
        
    def test_coord_conversion_roundtrip(self):
        """Test that conversion is reversible."""
        lattice_shape = (10, 10, 10, 10)
        original_coords = (3, 4, 5, 2)
        neuron_id = coord_to_id(*original_coords, lattice_shape)
        recovered_coords = id_to_coord(neuron_id, lattice_shape)
        assert recovered_coords == original_coords
        
    def test_coord_conversion_multiple(self):
        """Test multiple coordinate conversions."""
        lattice_shape = (10, 10, 10, 10)
        coords_list = [(0, 0, 0, 0), (1, 2, 3, 4), (9, 9, 9, 9)]
        
        for coords in coords_list:
            neuron_id = coord_to_id(*coords, lattice_shape)
            recovered_coords = id_to_coord(neuron_id, lattice_shape)
            assert recovered_coords == coords


class TestFeedSenseInput:
    """Tests for feed_sense_input function."""
    
    def test_feed_vision_input(self, model):
        """Test feeding input to vision sense."""
        # Create neurons in the vision area
        for x in range(10):
            for y in range(10):
                model.add_neuron(x, y, 0, 0)
        
        # Create input matrix
        input_matrix = np.ones((10, 10)) * 5.0
        
        # Feed input
        feed_sense_input(model, "vision", input_matrix)
        
        # Check that neurons received input
        for x in range(10):
            for y in range(10):
                coord = (x, y, 0, 0)
                coord_map = model.coord_to_id_map()
                if coord in coord_map:
                    neuron_id = coord_map[coord]
                    assert model.neurons[neuron_id].external_input == 5.0
    
    def test_feed_digital_input(self, model):
        """Test feeding input to digital sense."""
        # Create neurons in the digital area
        for x in range(10, 20):
            for y in range(10, 20):
                model.add_neuron(x, y, 0, 0)
        
        # Create input matrix
        input_matrix = np.ones((10, 10)) * 3.0
        
        # Feed input
        feed_sense_input(model, "digital", input_matrix)
        
        # Check that neurons received input
        for x in range(10, 20):
            for y in range(10, 20):
                coord = (x, y, 0, 0)
                coord_map = model.coord_to_id_map()
                if coord in coord_map:
                    neuron_id = coord_map[coord]
                    assert model.neurons[neuron_id].external_input == 3.0
    
    def test_unknown_sense_raises_error(self, model):
        """Test that unknown sense name raises ValueError."""
        input_matrix = np.ones((10, 10))
        
        with pytest.raises(ValueError, match="Unknown sense"):
            feed_sense_input(model, "unknown_sense", input_matrix)
    
    def test_invalid_input_type_raises_error(self, model):
        """Test that non-numpy array input raises TypeError."""
        input_list = [[1, 2], [3, 4]]
        
        with pytest.raises(TypeError, match="must be a numpy array"):
            feed_sense_input(model, "vision", input_list)
    
    def test_wrong_dimensions_raises_error(self, model):
        """Test that non-2D input raises ValueError."""
        input_matrix = np.ones((10, 10, 10))  # 3D instead of 2D
        
        with pytest.raises(ValueError, match="must be 2D"):
            feed_sense_input(model, "vision", input_matrix)
    
    def test_dimension_mismatch_warning(self, model):
        """Test that dimension mismatch produces a warning."""
        # Create neurons in the vision area
        for x in range(10):
            for y in range(10):
                model.add_neuron(x, y, 0, 0)
        
        # Create input matrix with wrong size
        input_matrix = np.ones((5, 5))
        
        # Should produce a warning but still work
        with pytest.warns(UserWarning, match="Input dimension mismatch"):
            feed_sense_input(model, "vision", input_matrix)
    
    def test_z_layer_parameter(self, model):
        """Test feeding input to different z layers."""
        # Create neurons in multiple z layers
        for x in range(10):
            for y in range(10):
                model.add_neuron(x, y, 0, 0)
                model.add_neuron(x, y, 1, 0)
        
        # Feed input to z_layer=1
        input_matrix = np.ones((10, 10)) * 7.0
        feed_sense_input(model, "vision", input_matrix, z_layer=1)
        
        # Check that only z=1 neurons received input
        coord_map = model.coord_to_id_map()
        for x in range(10):
            for y in range(10):
                # z=0 should have no external input
                coord_z0 = (x, y, 0, 0)
                if coord_z0 in coord_map:
                    neuron_id = coord_map[coord_z0]
                    assert model.neurons[neuron_id].external_input == 0.0
                
                # z=1 should have external input
                coord_z1 = (x, y, 1, 0)
                if coord_z1 in coord_map:
                    neuron_id = coord_map[coord_z1]
                    assert model.neurons[neuron_id].external_input == 7.0
    
    def test_invalid_z_layer_raises_error(self, model):
        """Test that invalid z_layer raises ValueError."""
        input_matrix = np.ones((10, 10))
        
        # z_layer out of range for vision area (z: 0-4)
        with pytest.raises(ValueError, match="z_layer.*out of range"):
            feed_sense_input(model, "vision", input_matrix, z_layer=10)
    
    def test_accumulate_input(self, model):
        """Test that multiple inputs accumulate."""
        # Create neurons in the vision area
        for x in range(10):
            for y in range(10):
                model.add_neuron(x, y, 0, 0)
        
        # Feed input twice
        input_matrix = np.ones((10, 10)) * 2.0
        feed_sense_input(model, "vision", input_matrix)
        feed_sense_input(model, "vision", input_matrix)
        
        # Check that inputs accumulated
        coord_map = model.coord_to_id_map()
        for x in range(10):
            for y in range(10):
                coord = (x, y, 0, 0)
                if coord in coord_map:
                    neuron_id = coord_map[coord]
                    assert model.neurons[neuron_id].external_input == 4.0


class TestDigitalSenseInput:
    """Tests for digital sense input creation."""
    
    def test_create_digital_sense_input_basic(self):
        """Test basic digital sense input creation."""
        text = "hello"
        input_matrix = create_digital_sense_input(text, target_shape=(10, 10))
        
        assert isinstance(input_matrix, np.ndarray)
        assert input_matrix.shape == (10, 10)
        assert input_matrix.min() >= 0
        assert input_matrix.max() <= 1
    
    def test_create_digital_sense_input_empty(self):
        """Test digital sense input with empty string."""
        text = ""
        input_matrix = create_digital_sense_input(text, target_shape=(10, 10))
        
        assert isinstance(input_matrix, np.ndarray)
        assert input_matrix.shape == (10, 10)
    
    def test_create_digital_sense_input_different_sizes(self):
        """Test digital sense input with different target sizes."""
        text = "test"
        
        for size in [(5, 5), (10, 10), (20, 20)]:
            input_matrix = create_digital_sense_input(text, target_shape=size)
            assert input_matrix.shape == size
