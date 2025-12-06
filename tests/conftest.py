"""Pytest configuration and shared fixtures for test suite."""

import pytest
import numpy as np
import json
import tempfile
import os
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel, Neuron, Synapse
from simulation import Simulation


@pytest.fixture
def minimal_config():
    """Minimal valid brain model configuration for testing."""
    return {
        "lattice_shape": [10, 10, 10, 10],
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
            "enable_death": True,
            "enable_reproduction": True,
            "max_age": 1000,
            "health_decay_per_step": 0.001,
            "mutation_std_params": 0.05,
            "mutation_std_weights": 0.02,
        },
        "plasticity": {
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "weight_min": -1.0,
            "weight_max": 1.0,
        },
        "senses": {
            "vision": {
                "areal": "V1_like",
                "input_size": [5, 5],
            },
            "digital": {
                "areal": "Digital_sensor",
                "input_size": [5, 5],
            }
        },
        "areas": [
            {
                "name": "V1_like",
                "coord_ranges": {
                    "x": [0, 4],
                    "y": [0, 4],
                    "z": [0, 2],
                    "w": [0, 0],
                }
            },
            {
                "name": "Digital_sensor",
                "coord_ranges": {
                    "x": [5, 9],
                    "y": [0, 4],
                    "z": [0, 2],
                    "w": [0, 0],
                }
            }
        ]
    }


@pytest.fixture
def brain_model(minimal_config):
    """Create a basic brain model for testing."""
    return BrainModel(config=minimal_config)


@pytest.fixture
def populated_model(brain_model):
    """Create a brain model with some neurons and synapses."""
    sim = Simulation(brain_model, seed=42)
    sim.initialize_neurons(area_names=["V1_like"], density=0.3)
    sim.initialize_random_synapses(connection_probability=0.1)
    return brain_model


@pytest.fixture
def simulation(brain_model):
    """Create a simulation with a brain model."""
    return Simulation(brain_model, seed=42)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for file operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_neuron():
    """Create a sample neuron for testing."""
    return Neuron(
        id=0,
        x=5,
        y=5,
        z=2,
        w=1,
        generation=0,
        parent_id=-1,
        health=1.0,
        age=0,
        v_membrane=-65.0,
        params={
            "tau_m": 20.0,
            "v_rest": -65.0,
            "v_reset": -70.0,
            "v_threshold": -50.0,
            "refractory_period": 5.0,
        }
    )


@pytest.fixture
def sample_synapse():
    """Create a sample synapse for testing."""
    return Synapse(
        pre_id=0,
        post_id=1,
        weight=0.5,
        delay=1,
        plasticity_tag=0.0,
    )


@pytest.fixture
def rng():
    """Create a seeded random number generator."""
    return np.random.default_rng(42)
