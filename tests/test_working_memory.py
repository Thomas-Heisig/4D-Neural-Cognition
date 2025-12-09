"""Tests for working memory module."""

import numpy as np
import pytest
from unittest.mock import Mock, MagicMock

from src.working_memory import (
    PersistentActivityManager,
    AttractorNetwork,
    MemoryGate,
    WorkingMemoryBuffer,
)


class MockNeuron:
    """Mock neuron for testing."""
    
    def __init__(self, neuron_id, x=0, y=0, z=0, w=0):
        self.id = neuron_id
        self.x = x
        self.y = y
        self.z = z
        self.w = w
        self.v = 0.0
        self.external_input = 0.0


class TestPersistentActivityManager:
    """Tests for PersistentActivityManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.model = Mock()
        self.model.neurons = {
            1: MockNeuron(1, x=0, y=0, z=0, w=0),
            2: MockNeuron(2, x=1, y=0, z=0, w=0),
            3: MockNeuron(3, x=2, y=0, z=0, w=0),
        }
        self.model.get_areas = Mock(return_value=[
            {
                "name": "prefrontal_cortex",
                "coord_ranges": {
                    "x": [0, 2],
                    "y": [0, 2],
                    "z": [0, 2],
                    "w": [0, 2],
                }
            }
        ])
    
    def test_init(self):
        """Test initialization."""
        manager = PersistentActivityManager(self.model)
        
        assert manager.model == self.model
        assert manager.memory_area == "prefrontal_cortex"
        assert manager.maintenance_current == 0.5
        assert manager.decay_rate == 0.01
    
    def test_encode_pattern(self):
        """Test pattern encoding."""
        manager = PersistentActivityManager(self.model)
        pattern = np.array([1.0, 2.0, 3.0])
        
        manager.encode_pattern("test_pattern", pattern)
        
        assert "test_pattern" in manager.active_patterns
        assert np.array_equal(manager.active_patterns["test_pattern"], pattern)
        assert len(manager.pattern_neurons["test_pattern"]) == 3
    
    def test_maintain_activity(self):
        """Test activity maintenance."""
        manager = PersistentActivityManager(self.model, maintenance_current=1.0)
        pattern = np.array([1.0, 0.5])
        
        manager.encode_pattern("test", pattern)
        initial_inputs = [n.external_input for n in self.model.neurons.values()]
        
        manager.maintain_activity("test")
        
        # External inputs should increase
        final_inputs = [n.external_input for n in self.model.neurons.values()]
        assert any(f > i for f, i in zip(final_inputs, initial_inputs))
    
    def test_decay_activity(self):
        """Test activity decay."""
        manager = PersistentActivityManager(self.model, decay_rate=0.1)
        pattern = np.array([1.0, 2.0])
        
        manager.encode_pattern("test", pattern)
        initial_pattern = manager.active_patterns["test"].copy()
        
        manager.decay_activity("test")
        
        # Pattern should decay
        assert np.all(manager.active_patterns["test"] < initial_pattern)
    
    def test_retrieve_pattern(self):
        """Test pattern retrieval."""
        manager = PersistentActivityManager(self.model)
        pattern = np.array([1.0, 2.0])
        
        manager.encode_pattern("test", pattern)
        retrieved = manager.retrieve_pattern("test")
        
        assert retrieved is not None
        assert len(retrieved) == len(manager.pattern_neurons["test"])
    
    def test_retrieve_nonexistent_pattern(self):
        """Test retrieving nonexistent pattern."""
        manager = PersistentActivityManager(self.model)
        
        retrieved = manager.retrieve_pattern("nonexistent")
        
        assert retrieved is None
    
    def test_clear_pattern(self):
        """Test pattern clearing."""
        manager = PersistentActivityManager(self.model)
        pattern = np.array([1.0, 2.0])
        
        manager.encode_pattern("test", pattern)
        manager.clear_pattern("test")
        
        assert "test" not in manager.active_patterns
        assert "test" not in manager.pattern_neurons
    
    def test_clear_all(self):
        """Test clearing all patterns."""
        manager = PersistentActivityManager(self.model)
        
        manager.encode_pattern("test1", np.array([1.0]))
        manager.encode_pattern("test2", np.array([2.0]))
        
        manager.clear_all()
        
        assert len(manager.active_patterns) == 0
        assert len(manager.pattern_neurons) == 0
    
    def test_get_memory_neurons_no_area(self):
        """Test getting neurons when area doesn't exist."""
        self.model.get_areas = Mock(return_value=[])
        manager = PersistentActivityManager(self.model)
        
        neurons = manager._get_memory_neurons()
        
        # Should return some neurons even if area not found
        assert len(neurons) > 0


class TestAttractorNetwork:
    """Tests for AttractorNetwork."""
    
    def test_init(self):
        """Test initialization."""
        network = AttractorNetwork(size=10, num_attractors=3, learning_rate=0.1)
        
        assert network.size == 10
        assert network.num_attractors == 3
        assert network.learning_rate == 0.1
        assert network.weights.shape == (10, 10)
    
    def test_store_pattern(self):
        """Test pattern storage."""
        network = AttractorNetwork(size=5)
        pattern = np.array([1, -1, 1, -1, 1])
        
        network.store_pattern(pattern)
        
        assert len(network.stored_patterns) == 1
        assert np.array_equal(network.stored_patterns[0], pattern)
        # Diagonal should be zero
        assert np.all(np.diag(network.weights) == 0)
    
    def test_store_pattern_wrong_size(self):
        """Test storing pattern with wrong size."""
        network = AttractorNetwork(size=5)
        pattern = np.array([1, -1, 1])
        
        with pytest.raises(ValueError, match="does not match"):
            network.store_pattern(pattern)
    
    def test_store_multiple_patterns(self):
        """Test storing multiple patterns."""
        network = AttractorNetwork(size=5, num_attractors=3)
        
        for i in range(3):
            pattern = np.random.choice([1, -1], size=5)
            network.store_pattern(pattern)
        
        assert len(network.stored_patterns) == 3
    
    def test_store_pattern_limit(self):
        """Test storage limit."""
        network = AttractorNetwork(size=5, num_attractors=2)
        
        for i in range(3):
            pattern = np.random.choice([1, -1], size=5)
            network.store_pattern(pattern)
        
        # Should only store up to num_attractors
        assert len(network.stored_patterns) == 2
    
    def test_set_state(self):
        """Test setting network state."""
        network = AttractorNetwork(size=5)
        state = np.array([1, -1, 1, -1, 1])
        
        network.set_state(state)
        
        assert np.array_equal(network.state, state)
    
    def test_set_state_wrong_size(self):
        """Test setting state with wrong size."""
        network = AttractorNetwork(size=5)
        state = np.array([1, -1, 1])
        
        with pytest.raises(ValueError, match="does not match"):
            network.set_state(state)
    
    def test_update_async(self):
        """Test asynchronous update."""
        network = AttractorNetwork(size=5)
        pattern = np.array([1, -1, 1, -1, 1])
        network.store_pattern(pattern)
        network.set_state(np.array([1, -1, 1, 1, 1]))  # Slight noise
        
        updated_state = network.update_async(num_updates=10)
        
        assert updated_state.shape == (5,)
        assert all(abs(v) == 1 for v in updated_state)  # Binary values
    
    def test_update_sync(self):
        """Test synchronous update."""
        network = AttractorNetwork(size=5)
        pattern = np.array([1, -1, 1, -1, 1])
        network.store_pattern(pattern)
        network.set_state(pattern)
        
        updated_state = network.update_sync()
        
        assert updated_state.shape == (5,)
        assert all(abs(v) == 1 for v in updated_state)
    
    def test_recall_perfect_cue(self):
        """Test recall with perfect cue."""
        network = AttractorNetwork(size=10)
        pattern = np.random.choice([1, -1], size=10)
        network.store_pattern(pattern)
        
        recalled, converged = network.recall(pattern, max_iterations=50)
        
        assert recalled.shape == (10,)
        # Should converge quickly with perfect cue
        assert converged
    
    def test_recall_noisy_cue(self):
        """Test recall with noisy cue."""
        network = AttractorNetwork(size=10)
        pattern = np.random.choice([1, -1], size=10)
        network.store_pattern(pattern)
        
        # Add noise (flip 2 bits)
        noisy_cue = pattern.copy()
        noisy_cue[0] *= -1
        noisy_cue[1] *= -1
        
        recalled, converged = network.recall(noisy_cue, max_iterations=100)
        
        assert recalled.shape == (10,)
    
    def test_compute_energy(self):
        """Test energy computation."""
        network = AttractorNetwork(size=5)
        pattern = np.array([1, -1, 1, -1, 1])
        network.store_pattern(pattern)
        network.set_state(pattern)
        
        energy = network.compute_energy()
        
        assert isinstance(energy, float)
        # Energy should be negative for stored patterns
        assert energy < 0
    
    def test_find_nearest_attractor(self):
        """Test finding nearest attractor."""
        network = AttractorNetwork(size=5)
        pattern1 = np.array([1, 1, 1, 1, 1])
        pattern2 = np.array([-1, -1, -1, -1, -1])
        
        network.store_pattern(pattern1)
        network.store_pattern(pattern2)
        
        test_state = np.array([1, 1, 1, -1, -1])
        nearest = network.find_nearest_attractor(test_state)
        
        assert nearest in [0, 1]
    
    def test_find_nearest_attractor_empty(self):
        """Test finding nearest attractor with no stored patterns."""
        network = AttractorNetwork(size=5)
        state = np.array([1, -1, 1, -1, 1])
        
        nearest = network.find_nearest_attractor(state)
        
        assert nearest is None


class TestMemoryGate:
    """Tests for MemoryGate."""
    
    def test_init(self):
        """Test initialization."""
        gate = MemoryGate(gate_threshold=0.5, update_strength=1.0)
        
        assert gate.gate_threshold == 0.5
        assert gate.update_strength == 1.0
        assert gate.gate_state == 0.0
    
    def test_update_gate(self):
        """Test gate update."""
        gate = MemoryGate()
        
        gate.update_gate(0.7)
        
        assert gate.gate_state == 0.7
    
    def test_update_gate_clipping(self):
        """Test gate update with clipping."""
        gate = MemoryGate()
        
        gate.update_gate(1.5)
        assert gate.gate_state == 1.0
        
        gate.update_gate(-0.5)
        assert gate.gate_state == 0.0
    
    def test_is_open_above_threshold(self):
        """Test gate open check above threshold."""
        gate = MemoryGate(gate_threshold=0.5)
        gate.update_gate(0.8)
        
        assert gate.is_open()
    
    def test_is_open_below_threshold(self):
        """Test gate open check below threshold."""
        gate = MemoryGate(gate_threshold=0.5)
        gate.update_gate(0.3)
        
        assert not gate.is_open()
    
    def test_apply_gate_open(self):
        """Test applying gate when open."""
        gate = MemoryGate(gate_threshold=0.5)
        gate.update_gate(0.8)
        
        input_value = np.array([1.0, 2.0, 3.0])
        memory_value = np.array([0.0, 0.0, 0.0])
        
        output = gate.apply_gate(input_value, memory_value)
        
        # Should be blend of input and memory
        assert not np.array_equal(output, memory_value)
        assert not np.array_equal(output, input_value)
    
    def test_apply_gate_closed(self):
        """Test applying gate when closed."""
        gate = MemoryGate(gate_threshold=0.5)
        gate.update_gate(0.3)
        
        input_value = np.array([1.0, 2.0, 3.0])
        memory_value = np.array([0.5, 0.5, 0.5])
        
        output = gate.apply_gate(input_value, memory_value)
        
        # Should maintain memory
        assert np.array_equal(output, memory_value)
    
    def test_reset(self):
        """Test gate reset."""
        gate = MemoryGate()
        gate.update_gate(0.8)
        gate.gated_value = np.array([1.0, 2.0])
        
        gate.reset()
        
        assert gate.gate_state == 0.0
        assert gate.gated_value is None


class TestWorkingMemoryBuffer:
    """Tests for WorkingMemoryBuffer."""
    
    def test_init(self):
        """Test initialization."""
        buffer = WorkingMemoryBuffer(num_slots=7, slot_size=50)
        
        assert buffer.num_slots == 7
        assert buffer.slot_size == 50
        assert len(buffer.slots) == 7
        assert all(slot is None for slot in buffer.slots)
    
    def test_store_item(self):
        """Test storing an item."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        item = np.random.rand(10)
        
        slot_idx = buffer.store(item)
        
        assert 0 <= slot_idx < 3
        assert buffer.slots[slot_idx] is not None
        assert np.array_equal(buffer.slots[slot_idx], item)
    
    def test_store_item_specific_slot(self):
        """Test storing to specific slot."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        item = np.random.rand(10)
        
        slot_idx = buffer.store(item, slot_index=1)
        
        assert slot_idx == 1
        assert buffer.slots[1] is not None
    
    def test_store_item_invalid_slot(self):
        """Test storing to invalid slot."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        item = np.random.rand(10)
        
        with pytest.raises(ValueError, match="Invalid slot"):
            buffer.store(item, slot_index=10)
    
    def test_store_item_resize_pad(self):
        """Test storing item that needs padding."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        item = np.random.rand(5)  # Too small
        
        slot_idx = buffer.store(item)
        
        assert len(buffer.slots[slot_idx]) == 10
    
    def test_store_item_resize_truncate(self):
        """Test storing item that needs truncation."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        item = np.random.rand(15)  # Too large
        
        slot_idx = buffer.store(item)
        
        assert len(buffer.slots[slot_idx]) == 10
    
    def test_store_replaces_oldest(self):
        """Test that storage replaces oldest item when full."""
        buffer = WorkingMemoryBuffer(num_slots=2, slot_size=5)
        
        # Fill buffer
        buffer.store(np.array([1, 2, 3, 4, 5]))
        buffer.store(np.array([6, 7, 8, 9, 10]))
        
        # Age first slot
        buffer.slot_ages[0] = 10
        buffer.slot_ages[1] = 5
        
        # Store new item
        new_item = np.array([11, 12, 13, 14, 15])
        slot_idx = buffer.store(new_item)
        
        # Should replace oldest (slot 0)
        assert slot_idx == 0
    
    def test_retrieve_item(self):
        """Test retrieving an item."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        item = np.random.rand(10)
        
        slot_idx = buffer.store(item)
        retrieved = buffer.retrieve(slot_idx)
        
        assert retrieved is not None
        assert np.array_equal(retrieved, item)
    
    def test_retrieve_empty_slot(self):
        """Test retrieving from empty slot."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        
        retrieved = buffer.retrieve(0)
        
        assert retrieved is None
    
    def test_retrieve_invalid_slot(self):
        """Test retrieving from invalid slot."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        
        retrieved = buffer.retrieve(10)
        
        assert retrieved is None
    
    def test_update_with_gating(self):
        """Test updating slot with gating."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=5)
        
        old_value = np.array([1, 1, 1, 1, 1])
        new_value = np.array([2, 2, 2, 2, 2])
        
        slot_idx = buffer.store(old_value)
        buffer.update(slot_idx, new_value, control_signal=0.5)
        
        retrieved = buffer.retrieve(slot_idx)
        # Should be blend of old and new
        assert not np.array_equal(retrieved, old_value)
        assert not np.array_equal(retrieved, new_value)
    
    def test_update_empty_slot(self):
        """Test updating empty slot."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=5)
        new_value = np.array([1, 2, 3, 4, 5])
        
        buffer.update(0, new_value, control_signal=1.0)
        
        assert buffer.slots[0] is not None
    
    def test_clear_slot(self):
        """Test clearing a slot."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        item = np.random.rand(10)
        
        slot_idx = buffer.store(item)
        buffer.clear_slot(slot_idx)
        
        assert buffer.slots[slot_idx] is None
        assert buffer.slot_ages[slot_idx] == 0
    
    def test_clear_all(self):
        """Test clearing all slots."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        
        for i in range(3):
            buffer.store(np.random.rand(10))
        
        buffer.clear_all()
        
        assert all(slot is None for slot in buffer.slots)
    
    def test_age_memory(self):
        """Test aging memory slots."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=10)
        
        buffer.store(np.random.rand(10))
        buffer.store(np.random.rand(10))
        
        initial_ages = buffer.slot_ages.copy()
        buffer.age_memory()
        
        # Ages should increase for occupied slots
        assert buffer.slot_ages[0] == initial_ages[0] + 1
        assert buffer.slot_ages[1] == initial_ages[1] + 1
        # Empty slot age should stay 0
        assert buffer.slot_ages[2] == 0
    
    def test_get_occupancy(self):
        """Test getting occupancy ratio."""
        buffer = WorkingMemoryBuffer(num_slots=4, slot_size=10)
        
        assert buffer.get_occupancy() == 0.0
        
        buffer.store(np.random.rand(10))
        buffer.store(np.random.rand(10))
        
        assert buffer.get_occupancy() == 0.5
        
        buffer.store(np.random.rand(10))
        buffer.store(np.random.rand(10))
        
        assert buffer.get_occupancy() == 1.0
    
    def test_search_content(self):
        """Test content-based search."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=5)
        
        item1 = np.array([1, 0, 0, 0, 0])
        item2 = np.array([0, 1, 0, 0, 0])
        item3 = np.array([1, 1, 0, 0, 0])
        
        buffer.store(item1)
        buffer.store(item2)
        buffer.store(item3)
        
        query = np.array([1, 0.5, 0, 0, 0])
        results = buffer.search_content(query, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        # Results should be sorted by similarity
        if len(results) == 2:
            assert results[0][1] >= results[1][1]
    
    def test_search_content_empty_buffer(self):
        """Test searching in empty buffer."""
        buffer = WorkingMemoryBuffer(num_slots=3, slot_size=5)
        query = np.array([1, 0, 0, 0, 0])
        
        results = buffer.search_content(query)
        
        assert len(results) == 0
    
    def test_search_content_resize_query(self):
        """Test content search with mismatched query size."""
        buffer = WorkingMemoryBuffer(num_slots=2, slot_size=10)
        
        buffer.store(np.random.rand(10))
        
        # Query too small
        query = np.array([1, 2, 3])
        results = buffer.search_content(query)
        
        assert len(results) > 0
