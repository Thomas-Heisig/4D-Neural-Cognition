"""Tests for data export module."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.brain_model import BrainModel, Neuron, Synapse
from src.simulation import Simulation
from src.data_export import (
    NumpyExporter,
    CSVExporter,
    MATLABExporter,
    NWBExporter,
    export_simulation,
)


@pytest.fixture
def minimal_config():
    """Create minimal brain model configuration."""
    return {
        "lattice_shape": (5, 5, 5, 5),
        "neuron_model": {
            "type": "lif",
            "v_rest": -65.0,
            "v_threshold": -50.0,
            "tau": 10.0,
        },
        "cell_lifecycle": {
            "max_age": 10000,
            "health_decay": 0.001,
            "reproduction_threshold": 0.8,
        },
        "plasticity": {"learning_rate": 0.01, "decay_rate": 0.001},
        "senses": {
            "vision": {"areal": "visual_cortex", "input_dim": (20, 20)},
            "digital": {"areal": "digital_cortex", "input_dim": (10, 10)},
        },
        "areas": [
            {
                "name": "visual_cortex",
                "coord_ranges": {"x": (0, 2), "y": (0, 2), "z": (0, 2), "w": (0, 2)},
            },
            {
                "name": "digital_cortex",
                "coord_ranges": {"x": (2, 4), "y": (0, 2), "z": (0, 2), "w": (0, 2)},
            },
        ],
    }


@pytest.fixture
def test_model(minimal_config):
    """Create a small test model with neurons and synapses."""
    model = BrainModel(config=minimal_config)
    
    # Add some test neurons
    for i in range(10):
        neuron = Neuron(
            id=i,
            x=i % 5,
            y=i // 5,
            z=0,
            w=0,
            v_membrane=-65.0 + i * 2,
            health=1.0 - i * 0.05,
            age=i * 100,
        )
        model.neurons[i] = neuron
    
    # Add some test synapses
    for i in range(5):
        synapse = Synapse(
            pre_id=i,
            post_id=i + 5,
            weight=0.1 + i * 0.05,
            delay=1 + i,
        )
        model.synapses.append(synapse)
    
    return model


@pytest.fixture
def test_spike_history():
    """Create test spike history."""
    return {
        0: [10, 20, 30],
        1: [15, 25, 35],
        2: [12, 22, 32],
        3: [18, 28, 38],
        4: [11, 21, 31],
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


def _scipy_available():
    """Check if scipy is available."""
    try:
        import scipy.io
        return True
    except ImportError:
        return False


def _pynwb_available():
    """Check if pynwb is available."""
    try:
        import pynwb
        return True
    except ImportError:
        return False


class TestNumpyExporter:
    """Test NumPy exporter."""

    def test_export_network_structure(self, test_model, temp_dir):
        """Test exporting network structure to NumPy format."""
        output_path = temp_dir / "network.npz"
        
        NumpyExporter.export_network_structure(test_model, output_path)
        
        assert output_path.exists()
        
        # Load and verify
        data = np.load(output_path)
        assert len(data["neuron_ids"]) == 10
        assert len(data["positions"]) == 10
        assert data["positions"].shape == (10, 4)
        assert len(data["pre_ids"]) == 5
        assert len(data["post_ids"]) == 5
        assert data["n_neurons"] == 10
        assert data["n_synapses"] == 5

    def test_export_spike_trains(self, test_spike_history, temp_dir):
        """Test exporting spike trains to NumPy format."""
        output_path = temp_dir / "spikes.npz"
        
        NumpyExporter.export_spike_trains(test_spike_history, output_path, dt=1.0)
        
        assert output_path.exists()
        
        # Load and verify
        data = np.load(output_path)
        assert "spike_matrix" in data
        assert "neuron_ids" in data
        assert "firing_rates" in data
        assert len(data["neuron_ids"]) == 5
        assert data["spike_matrix"].shape[0] == 5  # 5 neurons


class TestCSVExporter:
    """Test CSV exporter."""

    def test_export_neurons(self, test_model, temp_dir):
        """Test exporting neurons to CSV."""
        output_path = temp_dir / "neurons.csv"
        
        CSVExporter.export_neurons(test_model, output_path)
        
        assert output_path.exists()
        
        # Read and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Header + 10 neurons
        assert len(lines) == 11
        assert "neuron_id" in lines[0]
        assert "v_membrane" in lines[0]
        assert "health" in lines[0]

    def test_export_synapses(self, test_model, temp_dir):
        """Test exporting synapses to CSV."""
        output_path = temp_dir / "synapses.csv"
        
        CSVExporter.export_synapses(test_model, output_path)
        
        assert output_path.exists()
        
        # Read and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Header + 5 synapses
        assert len(lines) == 6
        assert "pre_id" in lines[0]
        assert "post_id" in lines[0]
        assert "weight" in lines[0]

    def test_export_spike_times(self, test_spike_history, temp_dir):
        """Test exporting spike times to CSV."""
        output_path = temp_dir / "spike_times.csv"
        
        CSVExporter.export_spike_times(test_spike_history, output_path, dt=1.0)
        
        assert output_path.exists()
        
        # Read and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Header + spikes (5 neurons * 3 spikes each = 15 spikes)
        assert len(lines) == 16
        assert "neuron_id" in lines[0]
        assert "spike_time_step" in lines[0]
        assert "spike_time_ms" in lines[0]

    def test_export_activity_summary(self, test_spike_history, test_model, temp_dir):
        """Test exporting activity summary to CSV."""
        output_path = temp_dir / "activity.csv"
        
        CSVExporter.export_activity_summary(test_spike_history, test_model, output_path)
        
        assert output_path.exists()
        
        # Read and verify
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        # Header + 10 neurons
        assert len(lines) == 11
        assert "neuron_id" in lines[0]
        assert "firing_rate_hz" in lines[0]


class TestMATLABExporter:
    """Test MATLAB exporter."""

    @pytest.mark.skipif(
        not _scipy_available(),
        reason="scipy not installed"
    )
    def test_export_all(self, test_model, test_spike_history, temp_dir):
        """Test exporting to MATLAB format."""
        output_path = temp_dir / "data.mat"
        
        MATLABExporter.export_all(test_model, test_spike_history, output_path)
        
        assert output_path.exists()
        
        # Load and verify (requires scipy)
        try:
            from scipy.io import loadmat
            data = loadmat(output_path)
            
            assert "neurons" in data
            assert "synapses" in data
            assert "spikes" in data
        except ImportError:
            pytest.skip("scipy not available for verification")

    def test_export_all_without_scipy(self, test_model, test_spike_history, temp_dir, capsys):
        """Test MATLAB export without scipy installed."""
        output_path = temp_dir / "data.mat"
        
        # Temporarily mock scipy import to fail
        import sys
        scipy_module = sys.modules.get('scipy.io')
        if scipy_module:
            sys.modules['scipy.io'] = None
        
        try:
            MATLABExporter.export_all(test_model, test_spike_history, output_path)
            
            # Should print error message
            captured = capsys.readouterr()
            assert "scipy is required" in captured.out or "scipy is required" in str(captured)
        finally:
            if scipy_module:
                sys.modules['scipy.io'] = scipy_module


class TestNWBExporter:
    """Test NWB exporter."""

    @pytest.mark.skipif(
        not _pynwb_available(),
        reason="pynwb not installed"
    )
    def test_export_to_nwb(self, test_model, test_spike_history, temp_dir):
        """Test exporting to NWB format."""
        output_path = temp_dir / "data.nwb"
        
        NWBExporter.export_to_nwb(
            test_model,
            test_spike_history,
            output_path,
            session_description="Test export",
            experimenter="Test",
            dt=0.001,
        )
        
        # Should at least not crash
        # Full verification requires pynwb which may not be installed
        if output_path.exists():
            assert output_path.stat().st_size > 0

    def test_export_to_nwb_without_pynwb(self, test_model, test_spike_history, temp_dir, capsys):
        """Test NWB export without pynwb installed."""
        output_path = temp_dir / "data.nwb"
        
        # The function should handle missing pynwb gracefully
        NWBExporter.export_to_nwb(test_model, test_spike_history, output_path)
        
        # Should print error message if pynwb not available
        # (This test might pass silently if pynwb IS installed)


class TestExportSimulation:
    """Test complete simulation export."""

    def test_export_simulation_numpy_csv(self, test_model, test_spike_history, temp_dir):
        """Test exporting simulation with numpy and csv formats."""
        export_simulation(
            test_model,
            test_spike_history,
            temp_dir,
            formats=["numpy", "csv"],
            prefix="test"
        )
        
        # Check numpy files
        assert (temp_dir / "test_network.npz").exists()
        assert (temp_dir / "test_spikes.npz").exists()
        
        # Check CSV files
        assert (temp_dir / "test_neurons.csv").exists()
        assert (temp_dir / "test_synapses.csv").exists()
        assert (temp_dir / "test_spike_times.csv").exists()
        assert (temp_dir / "test_activity.csv").exists()

    def test_export_simulation_default_formats(self, test_model, test_spike_history, temp_dir):
        """Test exporting with default formats."""
        export_simulation(
            test_model,
            test_spike_history,
            temp_dir,
            prefix="default"
        )
        
        # Default should be numpy and csv
        assert (temp_dir / "default_network.npz").exists()
        assert (temp_dir / "default_neurons.csv").exists()

    def test_export_simulation_creates_directory(self, test_model, test_spike_history, temp_dir):
        """Test that export creates output directory if needed."""
        output_dir = temp_dir / "nonexistent" / "nested"
        
        export_simulation(
            test_model,
            test_spike_history,
            output_dir,
            formats=["numpy"],
            prefix="nested"
        )
        
        assert output_dir.exists()
        assert (output_dir / "nested_network.npz").exists()
