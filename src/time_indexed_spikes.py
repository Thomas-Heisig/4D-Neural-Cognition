"""Time-indexed spike buffer for O(1) spike lookup.

This module provides an efficient data structure for storing and querying
spike times. It uses a circular buffer combined with hash tables for O(1)
access to spikes at specific time points.
"""

from typing import Set, Dict, List
from collections import defaultdict


class TimeIndexedSpikeBuffer:
    """Circular buffer for spike times with O(1) lookup.
    
    This data structure is optimized for the common operation in neural
    network simulations: "Did neuron X spike at time T - delay?"
    
    Instead of iterating through spike history (O(n)), this provides O(1)
    lookup using time-indexed hash tables.
    
    Memory complexity: O(window_size * avg_spikes_per_step)
    Lookup complexity: O(1) average case
    """
    
    def __init__(self, window_size: int = 100):
        """Initialize time-indexed spike buffer.
        
        Args:
            window_size: Number of time steps to retain (older spikes are discarded)
        """
        self.window_size = window_size
        
        # Current simulation time
        self._current_time = 0
        
        # Maps time -> set of neuron IDs that spiked at that time
        # Using modulo arithmetic for circular buffer behavior
        self._spikes_by_time: Dict[int, Set[int]] = defaultdict(set)
        
        # Track oldest time we have data for (for cleanup)
        self._oldest_time = 0
    
    def add_spike(self, neuron_id: int, time: int) -> None:
        """Record a spike at a specific time.
        
        Args:
            neuron_id: ID of the neuron that spiked
            time: Simulation time step when spike occurred
        """
        self._spikes_by_time[time].add(neuron_id)
        
        # Update current time if this is the latest
        if time > self._current_time:
            self._current_time = time
    
    def did_spike_at(self, neuron_id: int, time: int) -> bool:
        """Check if a neuron spiked at a specific time.
        
        This is the key optimization: O(1) lookup instead of O(n) iteration.
        
        Args:
            neuron_id: ID of the neuron to check
            time: Time step to check
            
        Returns:
            True if the neuron spiked at that time, False otherwise
        """
        return neuron_id in self._spikes_by_time[time]
    
    def get_spikes_at(self, time: int) -> Set[int]:
        """Get all neurons that spiked at a specific time.
        
        Args:
            time: Time step to query
            
        Returns:
            Set of neuron IDs that spiked at that time
        """
        return self._spikes_by_time[time].copy()
    
    def get_spike_times(self, neuron_id: int, start_time: int = None, end_time: int = None) -> List[int]:
        """Get all times when a neuron spiked within a time range.
        
        Note: This is O(window_size) and should be used sparingly.
        Use did_spike_at() for single time queries.
        
        Args:
            neuron_id: ID of the neuron
            start_time: Start of time range (inclusive), defaults to oldest_time
            end_time: End of time range (inclusive), defaults to current_time
            
        Returns:
            List of time steps when the neuron spiked
        """
        if start_time is None:
            start_time = self._oldest_time
        if end_time is None:
            end_time = self._current_time
            
        spike_times = []
        for t in range(start_time, end_time + 1):
            if neuron_id in self._spikes_by_time[t]:
                spike_times.append(t)
        return spike_times
    
    def _cleanup_old_spikes(self, current_time: int) -> None:
        """Remove spikes older than the window size.
        
        Args:
            current_time: Current simulation time
        """
        # Keep spikes within window_size steps from current time
        # If current_time is 20 and window_size is 10, keep times >= 11 (20 - 10 + 1)
        cutoff_time = current_time - self.window_size
        
        # Remove all times at or before cutoff
        times_to_remove = [t for t in self._spikes_by_time.keys() if t <= cutoff_time]
        for t in times_to_remove:
            del self._spikes_by_time[t]
        
        # Update oldest time to first time we still have data for
        self._oldest_time = cutoff_time + 1 if times_to_remove else self._oldest_time
    
    def advance_time(self, new_time: int) -> None:
        """Advance the current time and cleanup old spikes.
        
        This should be called once per simulation step.
        
        Args:
            new_time: New simulation time step
        """
        if new_time > self._current_time:
            self._current_time = new_time
        # Always cleanup, even if time hasn't changed (to clean up after batch adds)
        self._cleanup_old_spikes(new_time)
    
    def clear(self) -> None:
        """Clear all spike data."""
        self._spikes_by_time.clear()
        self._current_time = 0
        self._oldest_time = 0
    
    def num_spikes(self) -> int:
        """Get total number of spikes in buffer."""
        total = 0
        for spike_set in self._spikes_by_time.values():
            total += len(spike_set)
        return total
    
    def get_current_time(self) -> int:
        """Get current simulation time."""
        return self._current_time
    
    def get_window_size(self) -> int:
        """Get the window size."""
        return self.window_size


class SpikeHistoryAdapter:
    """Adapter to convert between old dict-based spike history and new buffer.
    
    This provides backward compatibility while using the new efficient structure.
    """
    
    def __init__(self, spike_buffer: TimeIndexedSpikeBuffer):
        """Initialize adapter.
        
        Args:
            spike_buffer: The time-indexed spike buffer to wrap
        """
        self._buffer = spike_buffer
    
    def to_dict(self) -> Dict[int, List[int]]:
        """Convert to old dict format: {neuron_id: [spike_times]}.
        
        Note: This is O(window_size * num_neurons) and should only be used
        for compatibility with legacy code.
        
        Returns:
            Dictionary mapping neuron IDs to lists of spike times
        """
        result: Dict[int, List[int]] = defaultdict(list)
        
        # Iterate through all time steps in the buffer
        for time in range(self._buffer._oldest_time, self._buffer._current_time + 1):
            for neuron_id in self._buffer.get_spikes_at(time):
                result[neuron_id].append(time)
        
        return dict(result)
    
    @staticmethod
    def from_dict(spike_dict: Dict[int, List[int]], window_size: int = 100) -> TimeIndexedSpikeBuffer:
        """Create buffer from old dict format.
        
        Args:
            spike_dict: Dictionary mapping neuron IDs to lists of spike times
            window_size: Window size for the buffer
            
        Returns:
            New TimeIndexedSpikeBuffer instance
        """
        buffer = TimeIndexedSpikeBuffer(window_size)
        
        for neuron_id, spike_times in spike_dict.items():
            for time in spike_times:
                buffer.add_spike(neuron_id, time)
        
        return buffer
    
    def get(self, neuron_id: int, default: List[int] = None) -> List[int]:
        """Get spike times for a neuron (dict-like interface).
        
        Args:
            neuron_id: ID of the neuron
            default: Default value if neuron has no spikes
            
        Returns:
            List of spike times
        """
        spike_times = self._buffer.get_spike_times(neuron_id)
        if not spike_times and default is not None:
            return default
        return spike_times
    
    def __getitem__(self, neuron_id: int) -> List[int]:
        """Get spike times for a neuron (dict-like interface).
        
        Args:
            neuron_id: ID of the neuron
            
        Returns:
            List of spike times
        """
        return self._buffer.get_spike_times(neuron_id)
    
    def __contains__(self, neuron_id: int) -> bool:
        """Check if neuron has any spikes (dict-like interface).
        
        Args:
            neuron_id: ID of the neuron
            
        Returns:
            True if neuron has spikes, False otherwise
        """
        return len(self._buffer.get_spike_times(neuron_id)) > 0
    
    def keys(self):
        """Get all neuron IDs that have spikes (dict-like interface).
        
        Returns:
            List of neuron IDs
        """
        neuron_ids = set()
        for time in range(self._buffer._oldest_time, self._buffer._current_time + 1):
            neuron_ids.update(self._buffer.get_spikes_at(time))
        return list(neuron_ids)
