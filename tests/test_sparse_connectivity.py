"""Unit tests for sparse_connectivity.py."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sparse_connectivity import SparseConnectivityMatrix


class TestSparseConnectivityMatrix:
    """Tests for SparseConnectivityMatrix class."""
    
    def test_init(self):
        """Test initialization."""
        matrix = SparseConnectivityMatrix(max_neuron_id=100)
        assert matrix.num_synapses() == 0
        assert len(matrix._post_index) == 0
        assert len(matrix._pre_index) == 0
    
    def test_add_synapse(self):
        """Test adding a single synapse."""
        matrix = SparseConnectivityMatrix()
        idx = matrix.add_synapse(pre_id=1, post_id=2, weight=0.5, delay=2)
        
        assert idx == 0
        assert matrix.num_synapses() == 1
        
        synapse = matrix.get_synapse(idx)
        assert synapse[0] == 1  # pre_id
        assert synapse[1] == 2  # post_id
        assert synapse[2] == 0.5  # weight
        assert synapse[3] == 2  # delay
    
    def test_add_multiple_synapses(self):
        """Test adding multiple synapses."""
        matrix = SparseConnectivityMatrix()
        
        matrix.add_synapse(1, 2, weight=0.1)
        matrix.add_synapse(1, 3, weight=0.2)
        matrix.add_synapse(2, 3, weight=0.3)
        
        assert matrix.num_synapses() == 3
    
    def test_get_incoming_synapses(self):
        """Test getting incoming synapses."""
        matrix = SparseConnectivityMatrix()
        
        matrix.add_synapse(1, 3, weight=0.1)
        matrix.add_synapse(2, 3, weight=0.2)
        matrix.add_synapse(1, 2, weight=0.15)
        
        # Neuron 3 should have 2 incoming synapses
        incoming = matrix.get_incoming_synapses(3)
        assert len(incoming) == 2
        
        # Check pre-synaptic neuron IDs
        pre_ids = [syn[1] for syn in incoming]
        assert 1 in pre_ids
        assert 2 in pre_ids
    
    def test_get_outgoing_synapses(self):
        """Test getting outgoing synapses."""
        matrix = SparseConnectivityMatrix()
        
        matrix.add_synapse(1, 2, weight=0.1)
        matrix.add_synapse(1, 3, weight=0.2)
        matrix.add_synapse(2, 3, weight=0.3)
        
        # Neuron 1 should have 2 outgoing synapses
        outgoing = matrix.get_outgoing_synapses(1)
        assert len(outgoing) == 2
        
        # Check post-synaptic neuron IDs
        post_ids = [syn[1] for syn in outgoing]
        assert 2 in post_ids
        assert 3 in post_ids
    
    def test_get_synapse_nonexistent(self):
        """Test getting non-existent synapse."""
        matrix = SparseConnectivityMatrix()
        matrix.add_synapse(1, 2)
        
        # Invalid indices should return None
        assert matrix.get_synapse(-1) is None
        assert matrix.get_synapse(100) is None
    
    def test_update_weight(self):
        """Test updating synaptic weight."""
        matrix = SparseConnectivityMatrix()
        idx = matrix.add_synapse(1, 2, weight=0.1)
        
        matrix.update_weight(idx, 0.5)
        
        synapse = matrix.get_synapse(idx)
        assert synapse[2] == 0.5
    
    def test_update_plasticity_tag(self):
        """Test updating plasticity tag."""
        matrix = SparseConnectivityMatrix()
        idx = matrix.add_synapse(1, 2, plasticity_tag=0.0)
        
        matrix.update_plasticity_tag(idx, 1.5)
        
        synapse = matrix.get_synapse(idx)
        assert synapse[4] == 1.5
    
    def test_remove_synapses_for_neuron(self):
        """Test removing synapses connected to a neuron."""
        matrix = SparseConnectivityMatrix()
        
        matrix.add_synapse(1, 2, weight=0.1)
        matrix.add_synapse(1, 3, weight=0.2)
        matrix.add_synapse(2, 3, weight=0.3)
        matrix.add_synapse(3, 4, weight=0.4)
        
        assert matrix.num_synapses() == 4
        
        # Remove all synapses connected to neuron 3
        matrix.remove_synapses_for_neuron(3)
        
        # Should have 1 synapse left (1->2)
        assert matrix.num_synapses() == 1
        
        synapse = matrix.get_synapse(0)
        assert synapse[0] == 1  # pre_id
        assert synapse[1] == 2  # post_id
    
    def test_to_list(self):
        """Test converting to list format."""
        matrix = SparseConnectivityMatrix()
        
        matrix.add_synapse(1, 2, weight=0.1, delay=1)
        matrix.add_synapse(2, 3, weight=0.2, delay=2)
        
        synapse_list = matrix.to_list()
        
        assert len(synapse_list) == 2
        assert synapse_list[0]["pre_id"] == 1
        assert synapse_list[0]["post_id"] == 2
        assert synapse_list[0]["weight"] == 0.1
        assert synapse_list[1]["pre_id"] == 2
        assert synapse_list[1]["post_id"] == 3
    
    def test_from_list(self):
        """Test creating from list format."""
        synapse_list = [
            {"pre_id": 1, "post_id": 2, "weight": 0.1, "delay": 1},
            {"pre_id": 2, "post_id": 3, "weight": 0.2, "delay": 2},
        ]
        
        matrix = SparseConnectivityMatrix.from_list(synapse_list)
        
        assert matrix.num_synapses() == 2
        
        synapse = matrix.get_synapse(0)
        assert synapse[0] == 1
        assert synapse[1] == 2
        assert synapse[2] == 0.1
    
    def test_roundtrip_conversion(self):
        """Test converting to list and back."""
        matrix1 = SparseConnectivityMatrix()
        matrix1.add_synapse(1, 2, weight=0.1, delay=1)
        matrix1.add_synapse(2, 3, weight=0.2, delay=2)
        matrix1.add_synapse(1, 3, weight=0.15, delay=1)
        
        synapse_list = matrix1.to_list()
        matrix2 = SparseConnectivityMatrix.from_list(synapse_list)
        
        assert matrix1.num_synapses() == matrix2.num_synapses()
        
        for i in range(matrix1.num_synapses()):
            syn1 = matrix1.get_synapse(i)
            syn2 = matrix2.get_synapse(i)
            assert syn1 == syn2
    
    def test_empty_incoming_synapses(self):
        """Test getting incoming synapses for neuron with none."""
        matrix = SparseConnectivityMatrix()
        matrix.add_synapse(1, 2)
        
        incoming = matrix.get_incoming_synapses(99)
        assert len(incoming) == 0
    
    def test_empty_outgoing_synapses(self):
        """Test getting outgoing synapses for neuron with none."""
        matrix = SparseConnectivityMatrix()
        matrix.add_synapse(1, 2)
        
        outgoing = matrix.get_outgoing_synapses(99)
        assert len(outgoing) == 0
    
    def test_synapse_type(self):
        """Test synapse types (excitatory/inhibitory)."""
        matrix = SparseConnectivityMatrix()
        
        idx1 = matrix.add_synapse(1, 2, synapse_type="excitatory")
        idx2 = matrix.add_synapse(2, 3, synapse_type="inhibitory")
        
        syn1 = matrix.get_synapse(idx1)
        syn2 = matrix.get_synapse(idx2)
        
        assert syn1[5] == "excitatory"
        assert syn2[5] == "inhibitory"
    
    def test_large_network(self):
        """Test with larger network to verify efficiency."""
        matrix = SparseConnectivityMatrix(max_neuron_id=1000)
        
        # Add 1000 synapses
        for i in range(1000):
            matrix.add_synapse(i % 100, (i + 1) % 100, weight=0.1)
        
        assert matrix.num_synapses() == 1000
        
        # Verify we can query efficiently
        incoming = matrix.get_incoming_synapses(50)
        assert len(incoming) > 0
    
    def test_index_consistency_after_removal(self):
        """Test that indices remain consistent after neuron removal."""
        matrix = SparseConnectivityMatrix()
        
        matrix.add_synapse(1, 2)
        matrix.add_synapse(2, 3)
        matrix.add_synapse(3, 4)
        matrix.add_synapse(4, 5)
        
        # Remove synapses for neuron 3
        matrix.remove_synapses_for_neuron(3)
        
        # Verify remaining synapses are accessible
        assert matrix.num_synapses() == 2
        
        # Check that indices work
        syn0 = matrix.get_synapse(0)
        syn1 = matrix.get_synapse(1)
        
        assert syn0 is not None
        assert syn1 is not None
