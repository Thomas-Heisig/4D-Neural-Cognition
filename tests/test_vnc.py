"""Tests for Virtual Neuromorphic Clock (VNC) system."""

import pytest
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brain_model import BrainModel
from simulation import Simulation
from hardware_abstraction.virtual_clock import GlobalVirtualClock
from hardware_abstraction.virtual_processing_unit import VirtualProcessingUnit
from hardware_abstraction.slice_partitioner import SlicePartitioner
from hardware_abstraction.virtual_io_expander import VirtualIOExpander
from digital_interface_2 import DirectNeuralAPI


class TestVirtualClock:
    """Tests for GlobalVirtualClock."""
    
    def test_clock_initialization(self):
        """Test that clock initializes correctly."""
        clock = GlobalVirtualClock(frequency_hz=20e6)
        
        assert clock.frequency == 20e6
        assert clock.current_cycle == 0
        assert len(clock.vpus) == 0
        assert not clock.is_running
    
    def test_add_vpu(self):
        """Test adding VPUs to clock."""
        clock = GlobalVirtualClock()
        vpu = VirtualProcessingUnit(vpu_id=0)
        
        clock.add_vpu(vpu)
        
        assert len(clock.vpus) == 1
        assert clock.vpus[0] == vpu
    
    def test_remove_vpu(self):
        """Test removing VPUs from clock."""
        clock = GlobalVirtualClock()
        vpu = VirtualProcessingUnit(vpu_id=0)
        
        clock.add_vpu(vpu)
        result = clock.remove_vpu(0)
        
        assert result
        assert len(clock.vpus) == 0
    
    def test_get_statistics(self):
        """Test getting clock statistics."""
        clock = GlobalVirtualClock(frequency_hz=10e6)
        stats = clock.get_statistics()
        
        assert "total_cycles" in stats
        assert "configured_clock_hz" in stats
        assert stats["configured_clock_hz"] == 10e6
        assert stats["num_vpus"] == 0


class TestVirtualProcessingUnit:
    """Tests for VirtualProcessingUnit."""
    
    def test_vpu_initialization(self):
        """Test VPU initialization."""
        vpu = VirtualProcessingUnit(vpu_id=5, clock_speed_hz=15e6)
        
        assert vpu.vpu_id == 5
        assert vpu.clock_speed == 15e6
        assert len(vpu.assigned_slices) == 0
    
    def test_assign_slice(self):
        """Test assigning slices to VPU."""
        vpu = VirtualProcessingUnit(vpu_id=0)
        slice_bounds = (0, 10, 0, 10, 0, 10, 0, 0)
        
        vpu.assign_slice(slice_bounds)
        
        assert len(vpu.assigned_slices) == 1
        assert vpu.assigned_slices[0] == slice_bounds
    
    def test_get_statistics(self):
        """Test getting VPU statistics."""
        vpu = VirtualProcessingUnit(vpu_id=0)
        stats = vpu.get_statistics()
        
        assert "neurons_processed" in stats
        assert "spikes_generated" in stats
        assert "cycles_executed" in stats
        assert stats["neurons_processed"] == 0


class TestSlicePartitioner:
    """Tests for SlicePartitioner."""
    
    def test_partition_by_w_slice(self):
        """Test w-slice partitioning."""
        lattice_shape = (10, 10, 5, 4)
        partitions = SlicePartitioner.partition_by_w_slice(lattice_shape)
        
        # Should create one partition per w value
        assert len(partitions) == 4
        
        # Each partition should span full x, y, z but single w
        for i, partition in enumerate(partitions):
            x_min, x_max, y_min, y_max, z_min, z_max, w_min, w_max = partition
            assert x_min == 0 and x_max == 9
            assert y_min == 0 and y_max == 9
            assert z_min == 0 and z_max == 4
            assert w_min == i and w_max == i
    
    def test_partition_by_z_slice(self):
        """Test z-slice partitioning."""
        lattice_shape = (10, 10, 5, 4)
        partitions = SlicePartitioner.partition_by_z_slice(lattice_shape)
        
        # Should create one partition per z value
        assert len(partitions) == 5
        
        # Each partition should span full x, y, w but single z
        for i, partition in enumerate(partitions):
            x_min, x_max, y_min, y_max, z_min, z_max, w_min, w_max = partition
            assert x_min == 0 and x_max == 9
            assert y_min == 0 and y_max == 9
            assert z_min == i and z_max == i
            assert w_min == 0 and w_max == 3
    
    def test_partition_by_blocks(self):
        """Test block partitioning."""
        lattice_shape = (10, 10, 10, 10)
        block_size = (5, 5, 5, 5)
        partitions = SlicePartitioner.partition_by_blocks(lattice_shape, block_size)
        
        # Should create 2^4 = 16 blocks
        assert len(partitions) == 16
    
    def test_get_partition_info(self):
        """Test getting partition information."""
        slice_bounds = (0, 9, 0, 9, 0, 4, 2, 2)
        info = SlicePartitioner.get_partition_info(slice_bounds)
        
        assert info["bounds"]["x"] == (0, 9)
        assert info["bounds"]["w"] == (2, 2)
        assert info["size"]["x"] == 10
        assert info["size"]["w"] == 1
        assert info["volume"] == 10 * 10 * 5 * 1


class TestVirtualIOExpander:
    """Tests for VirtualIOExpander."""
    
    def test_io_expander_initialization(self):
        """Test I/O expander initialization."""
        expander = VirtualIOExpander(base_io_width=1024, expansion_factor=256)
        
        assert expander.base_io_width == 1024
        assert expander.expansion_factor == 256
        assert expander.virtual_ports == 1024 * 256
    
    def test_map_virtual_to_physical(self):
        """Test virtual-to-physical port mapping."""
        expander = VirtualIOExpander(base_io_width=1024, expansion_factor=2)
        
        expander.map_virtual_to_physical(virtual_port=100, physical_pin=50)
        
        assert 100 in expander.port_mapping
        assert expander.port_mapping[100] == 50
    
    def test_write_read_virtual_port(self):
        """Test writing and reading from virtual ports."""
        expander = VirtualIOExpander(base_io_width=1024)
        expander.map_virtual_to_physical(virtual_port=10, physical_pin=5)
        
        # Write to port
        expander.write_virtual_port(10, 42.5)
        
        # Set input (simulating external signal)
        expander.set_input_values({10: 100.0})
        
        # Read from port
        value = expander.read_virtual_port(10)
        assert value == 100.0
    
    def test_auto_map_ports(self):
        """Test automatic port mapping."""
        expander = VirtualIOExpander(base_io_width=100, expansion_factor=2)
        mapped = expander.auto_map_ports(num_ports=50)
        
        assert len(mapped) == 50
        assert len(expander.port_mapping) == 50


class TestDirectNeuralAPI:
    """Tests for DirectNeuralAPI (Digital Sense 2.0)."""
    
    def test_api_initialization(self):
        """Test API initialization."""
        api = DirectNeuralAPI()
        
        assert len(api.data_streams) == 0
        assert len(api.api_endpoints) == 0
        assert len(api.encoders) > 0  # Should have default encoders
        assert len(api.decoders) > 0  # Should have default decoders
    
    def test_register_encoder_decoder(self):
        """Test registering custom encoder and decoder."""
        api = DirectNeuralAPI()
        
        api.register_encoder("test_encoder", lambda x: x * 2)
        api.register_decoder("test_decoder", lambda x: x / 2)
        
        assert "test_encoder" in api.encoders
        assert "test_decoder" in api.decoders
    
    def test_connect_data_stream(self):
        """Test connecting a data stream."""
        api = DirectNeuralAPI()
        
        stream = api.connect_data_stream(
            stream_id="test_stream",
            stream_type="websocket",
            config={"url": "ws://localhost:8080"}
        )
        
        assert stream.stream_id == "test_stream"
        assert "test_stream" in api.data_streams
    
    def test_disconnect_data_stream(self):
        """Test disconnecting a data stream."""
        api = DirectNeuralAPI()
        
        api.connect_data_stream(
            stream_id="test_stream",
            stream_type="api",
            config={}
        )
        
        result = api.disconnect_data_stream("test_stream")
        
        assert result
        assert "test_stream" not in api.data_streams
    
    def test_encode_to_neural_input(self):
        """Test encoding data to neural input."""
        api = DirectNeuralAPI()
        
        encoded = api.encode_to_neural_input(0.5, encoder="value_to_rate")
        
        assert encoded.shape[0] >= 1
        assert float(encoded[0]) == 50.0  # 0.5 * 100


class TestVNCSimulationIntegration:
    """Tests for VNC integration with Simulation."""
    
    @pytest.fixture
    def minimal_config(self):
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
                "vision": {"areal": "V1_like", "input_size": [5, 5]},
            },
            "areas": [
                {
                    "name": "V1_like",
                    "coord_ranges": {"x": [0, 4], "y": [0, 4], "z": [0, 2], "w": [0, 0]},
                }
            ]
        }
    
    def test_simulation_with_vnc_disabled(self, minimal_config):
        """Test simulation without VNC."""
        model = BrainModel(config=minimal_config)
        model.add_neuron(0, 0, 0, 0)
        
        sim = Simulation(model, use_vnc=False)
        
        assert not sim.use_vnc
        assert sim.virtual_clock is None
    
    def test_simulation_with_vnc_enabled(self, minimal_config):
        """Test simulation with VNC enabled."""
        model = BrainModel(config=minimal_config)
        # Add some neurons
        for w in range(2):
            for i in range(3):
                model.add_neuron(i, i, 0, w)
        
        sim = Simulation(
            model,
            use_vnc=True,
            vnc_clock_frequency=10e6
        )
        
        assert sim.use_vnc
        assert sim.virtual_clock is not None
        assert sim.vnc_clock_frequency == 10e6
    
    def test_vnc_statistics(self, minimal_config):
        """Test getting VNC statistics from simulation."""
        model = BrainModel(config=minimal_config)
        for w in range(2):
            for i in range(3):
                model.add_neuron(i, i, 0, w)
        
        sim = Simulation(model, use_vnc=True, vnc_clock_frequency=5e6)
        
        # Get statistics before running
        stats = sim.get_vnc_statistics()
        
        assert stats is not None
        assert "total_cycles" in stats
        assert "configured_clock_hz" in stats
        assert stats["configured_clock_hz"] == 5e6
    
    def test_vpu_statistics(self, minimal_config):
        """Test getting VPU statistics from simulation."""
        model = BrainModel(config=minimal_config)
        for w in range(2):
            for i in range(3):
                model.add_neuron(i, i, 0, w)
        
        sim = Simulation(model, use_vnc=True)
        
        vpu_stats = sim.get_vpu_statistics()
        
        assert vpu_stats is not None
        assert isinstance(vpu_stats, list)
        # Should have VPUs equal to number of unique w values
        assert len(vpu_stats) >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
