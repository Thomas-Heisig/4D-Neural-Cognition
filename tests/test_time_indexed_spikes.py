"""Unit tests for time_indexed_spikes.py."""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from time_indexed_spikes import TimeIndexedSpikeBuffer, SpikeHistoryAdapter


class TestTimeIndexedSpikeBuffer:
    """Tests for TimeIndexedSpikeBuffer class."""
    
    def test_init(self):
        """Test initialization."""
        buffer = TimeIndexedSpikeBuffer(window_size=50)
        assert buffer.get_window_size() == 50
        assert buffer.get_current_time() == 0
        assert buffer.num_spikes() == 0
    
    def test_add_spike(self):
        """Test adding a single spike."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(neuron_id=1, time=10)
        
        assert buffer.did_spike_at(1, 10)
        assert not buffer.did_spike_at(1, 11)
        assert buffer.num_spikes() == 1
    
    def test_add_multiple_spikes_same_time(self):
        """Test adding multiple spikes at the same time."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 10)
        buffer.add_spike(2, 10)
        buffer.add_spike(3, 10)
        
        assert buffer.did_spike_at(1, 10)
        assert buffer.did_spike_at(2, 10)
        assert buffer.did_spike_at(3, 10)
        assert buffer.num_spikes() == 3
    
    def test_add_multiple_spikes_different_times(self):
        """Test adding spikes at different times."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(1, 10)
        buffer.add_spike(1, 15)
        
        assert buffer.did_spike_at(1, 5)
        assert buffer.did_spike_at(1, 10)
        assert buffer.did_spike_at(1, 15)
        assert not buffer.did_spike_at(1, 12)
    
    def test_get_spikes_at(self):
        """Test getting all spikes at a specific time."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 10)
        buffer.add_spike(2, 10)
        buffer.add_spike(3, 11)
        
        spikes_at_10 = buffer.get_spikes_at(10)
        assert len(spikes_at_10) == 2
        assert 1 in spikes_at_10
        assert 2 in spikes_at_10
        
        spikes_at_11 = buffer.get_spikes_at(11)
        assert len(spikes_at_11) == 1
        assert 3 in spikes_at_11
    
    def test_get_spike_times(self):
        """Test getting all spike times for a neuron."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(1, 10)
        buffer.add_spike(1, 15)
        buffer.add_spike(2, 10)
        
        spike_times = buffer.get_spike_times(1)
        assert len(spike_times) == 3
        assert 5 in spike_times
        assert 10 in spike_times
        assert 15 in spike_times
    
    def test_get_spike_times_with_range(self):
        """Test getting spike times within a range."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(1, 10)
        buffer.add_spike(1, 15)
        buffer.add_spike(1, 20)
        
        spike_times = buffer.get_spike_times(1, start_time=8, end_time=17)
        assert len(spike_times) == 2
        assert 10 in spike_times
        assert 15 in spike_times
        assert 5 not in spike_times
        assert 20 not in spike_times
    
    def test_cleanup_old_spikes(self):
        """Test that old spikes are removed after window size."""
        buffer = TimeIndexedSpikeBuffer(window_size=10)
        
        # Add spikes at times 0-20
        for t in range(21):
            buffer.add_spike(1, t)
        
        # At time 20, spikes before time 10 should be removed
        buffer.advance_time(20)
        
        # Spike at time 5 should be gone
        assert not buffer.did_spike_at(1, 5)
        # Spike at time 15 should still exist
        assert buffer.did_spike_at(1, 15)
    
    def test_advance_time(self):
        """Test advancing time."""
        buffer = TimeIndexedSpikeBuffer(window_size=10)
        buffer.add_spike(1, 5)
        
        buffer.advance_time(20)
        assert buffer.get_current_time() == 20
        
        # Old spike should be cleaned up
        assert not buffer.did_spike_at(1, 5)
    
    def test_clear(self):
        """Test clearing all spikes."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(2, 10)
        
        buffer.clear()
        
        assert buffer.num_spikes() == 0
        assert buffer.get_current_time() == 0
        assert not buffer.did_spike_at(1, 5)
    
    def test_did_spike_at_nonexistent(self):
        """Test querying non-existent spike."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 10)
        
        # Different neuron
        assert not buffer.did_spike_at(2, 10)
        # Different time
        assert not buffer.did_spike_at(1, 11)
        # Both different
        assert not buffer.did_spike_at(3, 5)
    
    def test_window_size_boundary(self):
        """Test behavior at window size boundary."""
        buffer = TimeIndexedSpikeBuffer(window_size=5)
        
        buffer.add_spike(1, 0)
        buffer.add_spike(1, 1)
        buffer.add_spike(1, 2)
        buffer.add_spike(1, 3)
        buffer.add_spike(1, 4)
        
        # All should be present
        assert buffer.num_spikes() == 5
        
        # Add spike at time 5 and advance
        buffer.add_spike(1, 5)
        buffer.advance_time(5)
        
        # Spike at time 0 should be removed (5 - 5 = 0 cutoff)
        assert not buffer.did_spike_at(1, 0)
        # Spike at time 1 should still exist
        assert buffer.did_spike_at(1, 1)
    
    def test_multiple_neurons_same_pattern(self):
        """Test multiple neurons with overlapping spike patterns."""
        buffer = TimeIndexedSpikeBuffer()
        
        # Add spikes for 3 neurons at various times
        for neuron_id in [1, 2, 3]:
            for t in [5, 10, 15]:
                buffer.add_spike(neuron_id, t)
        
        # Verify all neurons spiked at all times
        for neuron_id in [1, 2, 3]:
            for t in [5, 10, 15]:
                assert buffer.did_spike_at(neuron_id, t)
        
        assert buffer.num_spikes() == 9
    
    def test_large_time_range(self):
        """Test with large time values."""
        buffer = TimeIndexedSpikeBuffer(window_size=100)
        
        buffer.add_spike(1, 10000)
        buffer.add_spike(1, 10050)
        buffer.add_spike(1, 10100)
        
        assert buffer.did_spike_at(1, 10000)
        assert buffer.did_spike_at(1, 10050)
        
        # Advance to time 10150
        buffer.advance_time(10150)
        
        # Spike at 10000 should be removed (> 100 steps ago)
        assert not buffer.did_spike_at(1, 10000)
        # Spike at 10100 should still exist
        assert buffer.did_spike_at(1, 10100)


class TestSpikeHistoryAdapter:
    """Tests for SpikeHistoryAdapter class."""
    
    def test_to_dict(self):
        """Test converting buffer to dict format."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(1, 10)
        buffer.add_spike(2, 10)
        
        adapter = SpikeHistoryAdapter(buffer)
        spike_dict = adapter.to_dict()
        
        assert 1 in spike_dict
        assert 2 in spike_dict
        assert 5 in spike_dict[1]
        assert 10 in spike_dict[1]
        assert 10 in spike_dict[2]
    
    def test_from_dict(self):
        """Test creating buffer from dict format."""
        spike_dict = {
            1: [5, 10, 15],
            2: [10, 20],
        }
        
        buffer = SpikeHistoryAdapter.from_dict(spike_dict)
        
        assert buffer.did_spike_at(1, 5)
        assert buffer.did_spike_at(1, 10)
        assert buffer.did_spike_at(2, 20)
    
    def test_roundtrip_conversion(self):
        """Test converting to dict and back."""
        buffer1 = TimeIndexedSpikeBuffer()
        buffer1.add_spike(1, 5)
        buffer1.add_spike(1, 10)
        buffer1.add_spike(2, 10)
        
        adapter = SpikeHistoryAdapter(buffer1)
        spike_dict = adapter.to_dict()
        buffer2 = SpikeHistoryAdapter.from_dict(spike_dict)
        
        # Verify all spikes are preserved
        assert buffer2.did_spike_at(1, 5)
        assert buffer2.did_spike_at(1, 10)
        assert buffer2.did_spike_at(2, 10)
    
    def test_dict_like_interface_get(self):
        """Test dict-like get method."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(1, 10)
        
        adapter = SpikeHistoryAdapter(buffer)
        
        spike_times = adapter.get(1)
        assert 5 in spike_times
        assert 10 in spike_times
        
        # Non-existent neuron with default
        empty = adapter.get(99, default=[])
        assert empty == []
    
    def test_dict_like_interface_getitem(self):
        """Test dict-like __getitem__ method."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(1, 10)
        
        adapter = SpikeHistoryAdapter(buffer)
        
        spike_times = adapter[1]
        assert 5 in spike_times
        assert 10 in spike_times
    
    def test_dict_like_interface_contains(self):
        """Test dict-like __contains__ method."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        
        adapter = SpikeHistoryAdapter(buffer)
        
        assert 1 in adapter
        assert 99 not in adapter
    
    def test_dict_like_interface_keys(self):
        """Test dict-like keys method."""
        buffer = TimeIndexedSpikeBuffer()
        buffer.add_spike(1, 5)
        buffer.add_spike(2, 10)
        buffer.add_spike(3, 15)
        
        adapter = SpikeHistoryAdapter(buffer)
        keys = adapter.keys()
        
        assert len(keys) == 3
        assert 1 in keys
        assert 2 in keys
        assert 3 in keys


class TestPerformance:
    """Performance tests to verify O(1) lookup."""
    
    def test_lookup_performance_constant_time(self):
        """Test that lookup time is constant regardless of history size."""
        import time
        
        # Create buffer with many spikes
        buffer = TimeIndexedSpikeBuffer(window_size=100)
        
        # Add 10,000 spikes spread across time
        for i in range(10000):
            buffer.add_spike(neuron_id=i % 100, time=i % 100)
        
        # Measure lookup time (should be O(1))
        start = time.time()
        for _ in range(1000):
            buffer.did_spike_at(50, 50)
        elapsed = time.time() - start
        
        # Should complete very quickly (< 0.1 seconds for 1000 lookups)
        assert elapsed < 0.1
    
    def test_memory_efficiency(self):
        """Test memory efficiency of circular buffer."""
        buffer = TimeIndexedSpikeBuffer(window_size=10)
        
        # Add many spikes over time
        for t in range(1000):
            buffer.add_spike(1, t)
            buffer.advance_time(t)
        
        # Buffer should only keep spikes within window
        # So num_spikes should be around window_size
        assert buffer.num_spikes() <= 15  # Some buffer for cleanup timing
