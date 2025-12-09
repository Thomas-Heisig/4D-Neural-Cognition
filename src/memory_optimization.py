"""Memory optimization module for 4D Neural Cognition.

This module provides memory optimization features including:
- Memory-mapped files for large models
- Compression for inactive neurons
- Cache optimization strategies
- Memory profiling and monitoring
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import h5py
import os
import warnings
from dataclasses import dataclass
import mmap
import pickle

try:
    from .brain_model import BrainModel, Neuron, Synapse
except ImportError:
    from brain_model import BrainModel, Neuron, Synapse


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    
    total_bytes: int
    neurons_bytes: int
    synapses_bytes: int
    spike_history_bytes: int
    other_bytes: int
    
    def to_mb(self) -> Dict[str, float]:
        """Convert to megabytes."""
        return {
            'total_mb': self.total_bytes / (1024 ** 2),
            'neurons_mb': self.neurons_bytes / (1024 ** 2),
            'synapses_mb': self.synapses_bytes / (1024 ** 2),
            'spike_history_mb': self.spike_history_bytes / (1024 ** 2),
            'other_mb': self.other_bytes / (1024 ** 2)
        }


class MemoryMappedModel:
    """Memory-mapped storage for large brain models.
    
    Uses memory-mapped files to handle models larger than available RAM.
    Data is stored on disk but accessed as if in memory.
    """
    
    def __init__(self, filepath: str, mode: str = 'r+'):
        """Initialize memory-mapped model.
        
        Args:
            filepath: Path to HDF5 file for storage
            mode: File access mode ('r' read, 'r+' read/write, 'w' write)
        """
        self.filepath = filepath
        self.mode = mode
        self.h5file: Optional[h5py.File] = None
        
        # Create/open file
        if mode == 'w' or not os.path.exists(filepath):
            self.h5file = h5py.File(filepath, 'w')
            self._initialize_datasets()
        else:
            self.h5file = h5py.File(filepath, mode)
    
    def _initialize_datasets(self) -> None:
        """Initialize HDF5 datasets with appropriate chunking for memory mapping."""
        if self.h5file is None:
            return
        
        # Create groups
        self.h5file.create_group('neurons')
        self.h5file.create_group('synapses')
        self.h5file.create_group('metadata')
    
    def store_neurons(
        self,
        neurons: Dict[int, Neuron],
        chunk_size: int = 1000,
        compression: Optional[str] = 'gzip'
    ) -> None:
        """Store neurons in memory-mapped file.
        
        Args:
            neurons: Dictionary of neurons
            chunk_size: HDF5 chunk size for optimal access
            compression: Compression algorithm ('gzip', 'lzf', None)
        
        Raises:
            ValueError: If compression algorithm is not supported
        """
        # Validate compression parameter
        valid_compression = ['gzip', 'lzf', None]
        if compression not in valid_compression:
            raise ValueError(f"Compression must be one of {valid_compression}, got {compression}")
        if self.h5file is None:
            raise RuntimeError("File not opened")
        
        n_neurons = len(neurons)
        
        # Create datasets with chunking
        neuron_group = self.h5file['neurons']
        
        # Store neuron IDs
        neuron_group.create_dataset(
            'ids',
            data=list(neurons.keys()),
            chunks=(min(chunk_size, n_neurons),),
            compression=compression
        )
        
        # Store positions
        positions = np.array([
            [n.x, n.y, n.z, n.w] for n in neurons.values()
        ], dtype=np.int32)
        neuron_group.create_dataset(
            'positions',
            data=positions,
            chunks=(min(chunk_size, n_neurons), 4),
            compression=compression
        )
        
        # Store states
        v_membranes = np.array([n.v_membrane for n in neurons.values()], dtype=np.float32)
        neuron_group.create_dataset(
            'v_membranes',
            data=v_membranes,
            chunks=(min(chunk_size, n_neurons),),
            compression=compression
        )
        
        # Store other attributes
        healths = np.array([n.health for n in neurons.values()], dtype=np.float32)
        neuron_group.create_dataset(
            'healths',
            data=healths,
            chunks=(min(chunk_size, n_neurons),),
            compression=compression
        )
        
        ages = np.array([n.age for n in neurons.values()], dtype=np.int32)
        neuron_group.create_dataset(
            'ages',
            data=ages,
            chunks=(min(chunk_size, n_neurons),),
            compression=compression
        )
    
    def load_neurons(self, indices: Optional[List[int]] = None) -> Dict[int, Dict]:
        """Load neurons from memory-mapped file.
        
        Args:
            indices: Optional list of indices to load (for partial loading)
            
        Returns:
            Dictionary of neuron data
        """
        if self.h5file is None:
            raise RuntimeError("File not opened")
        
        neuron_group = self.h5file['neurons']
        
        if indices is None:
            # Load all
            neuron_ids = neuron_group['ids'][:]
            positions = neuron_group['positions'][:]
            v_membranes = neuron_group['v_membranes'][:]
            healths = neuron_group['healths'][:]
            ages = neuron_group['ages'][:]
        else:
            # Load specific indices
            neuron_ids = neuron_group['ids'][indices]
            positions = neuron_group['positions'][indices]
            v_membranes = neuron_group['v_membranes'][indices]
            healths = neuron_group['healths'][indices]
            ages = neuron_group['ages'][indices]
        
        # Reconstruct neuron data
        neurons = {}
        for i, nid in enumerate(neuron_ids):
            neurons[int(nid)] = {
                'x': int(positions[i, 0]),
                'y': int(positions[i, 1]),
                'z': int(positions[i, 2]),
                'w': int(positions[i, 3]),
                'v_membrane': float(v_membranes[i]),
                'health': float(healths[i]),
                'age': int(ages[i])
            }
        
        return neurons
    
    def close(self) -> None:
        """Close the memory-mapped file."""
        if self.h5file is not None:
            self.h5file.close()
            self.h5file = None


class InactiveNeuronCompressor:
    """Compress state of inactive neurons to save memory.
    
    Inactive neurons (those that haven't spiked recently) can have their
    detailed state compressed or archived.
    """
    
    def __init__(self, inactivity_threshold: int = 1000):
        """Initialize compressor.
        
        Args:
            inactivity_threshold: Steps without spiking to consider inactive
        """
        self.inactivity_threshold = inactivity_threshold
        self.compressed_neurons: Dict[int, bytes] = {}
    
    def compress_inactive_neurons(
        self,
        neurons: Dict[int, Neuron],
        spike_history: Dict[int, List[int]],
        current_step: int
    ) -> Tuple[Dict[int, Neuron], int]:
        """Compress inactive neurons to save memory.
        
        Args:
            neurons: Dictionary of all neurons
            spike_history: Spike history for each neuron
            current_step: Current simulation step
            
        Returns:
            Tuple of (active_neurons_dict, bytes_saved)
        """
        active_neurons = {}
        bytes_saved = 0
        
        for neuron_id, neuron in neurons.items():
            # Check if neuron is active
            spikes = spike_history.get(neuron_id, [])
            if spikes:
                last_spike = spikes[-1]
            else:
                last_spike = -float('inf')
            
            time_since_spike = current_step - last_spike
            
            if time_since_spike > self.inactivity_threshold:
                # Compress this neuron
                neuron_bytes = pickle.dumps(neuron)
                self.compressed_neurons[neuron_id] = neuron_bytes
                bytes_saved += len(neuron_bytes)
            else:
                # Keep active
                active_neurons[neuron_id] = neuron
        
        return active_neurons, bytes_saved
    
    def decompress_neuron(self, neuron_id: int) -> Optional[Neuron]:
        """Decompress a neuron if it was compressed.
        
        Args:
            neuron_id: ID of neuron to decompress
            
        Returns:
            Decompressed Neuron or None if not found
        """
        if neuron_id in self.compressed_neurons:
            return pickle.loads(self.compressed_neurons[neuron_id])
        return None
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression statistics.
        
        Returns:
            Dictionary with compression metrics
        """
        total_bytes = sum(len(b) for b in self.compressed_neurons.values())
        return {
            'n_compressed': len(self.compressed_neurons),
            'total_bytes': total_bytes,
            'avg_bytes_per_neuron': total_bytes / len(self.compressed_neurons) if self.compressed_neurons else 0
        }


class CacheOptimizer:
    """Optimize data access patterns for better CPU cache utilization."""
    
    def __init__(self):
        """Initialize cache optimizer."""
        self.access_patterns: Dict[str, int] = {}
    
    def reorder_neurons_spatial_locality(
        self,
        neurons: Dict[int, Neuron]
    ) -> Dict[int, Neuron]:
        """Reorder neurons to improve spatial locality in memory.
        
        Uses Z-order curve (Morton order) to map 4D positions to 1D,
        improving cache locality for spatially local operations.
        
        Args:
            neurons: Dictionary of neurons
            
        Returns:
            Reordered dictionary with better spatial locality
        """
        # Compute Morton codes for each neuron
        def morton_encode_4d(x: int, y: int, z: int, w: int) -> int:
            """Compute Morton code (Z-order) for 4D coordinates."""
            result = 0
            for i in range(16):  # 16 bits per dimension max
                result |= ((x >> i) & 1) << (4 * i)
                result |= ((y >> i) & 1) << (4 * i + 1)
                result |= ((z >> i) & 1) << (4 * i + 2)
                result |= ((w >> i) & 1) << (4 * i + 3)
            return result
        
        # Sort neurons by Morton code
        neuron_list = list(neurons.items())
        neuron_list.sort(key=lambda item: morton_encode_4d(
            item[1].x, item[1].y, item[1].z, item[1].w
        ))
        
        # Return as ordered dictionary
        return dict(neuron_list)
    
    def prefetch_neuron_data(
        self,
        neuron_ids: List[int],
        data_source: Any
    ) -> None:
        """Prefetch neuron data to improve cache hit rate.
        
        Args:
            neuron_ids: List of neuron IDs to prefetch
            data_source: Source to prefetch from (e.g., memory-mapped file)
        """
        # This is a placeholder - actual implementation would depend on
        # the data source type and system capabilities
        pass
    
    def analyze_access_pattern(
        self,
        access_sequence: List[int]
    ) -> Dict[str, Any]:
        """Analyze memory access patterns to identify optimization opportunities.
        
        Args:
            access_sequence: Sequence of neuron IDs accessed
            
        Returns:
            Dictionary with access pattern statistics
        """
        # Calculate spatial locality metric
        spatial_jumps = []
        for i in range(len(access_sequence) - 1):
            jump = abs(access_sequence[i+1] - access_sequence[i])
            spatial_jumps.append(jump)
        
        return {
            'n_accesses': len(access_sequence),
            'unique_ids': len(set(access_sequence)),
            'avg_spatial_jump': np.mean(spatial_jumps) if spatial_jumps else 0,
            'max_spatial_jump': max(spatial_jumps) if spatial_jumps else 0,
            'locality_score': 1.0 / (1.0 + np.mean(spatial_jumps)) if spatial_jumps else 1.0
        }


class MemoryProfiler:
    """Profile and monitor memory usage of the simulation."""
    
    def __init__(self):
        """Initialize memory profiler."""
        self.snapshots: List[Tuple[str, MemoryStats]] = []
    
    def measure_model_memory(self, model: BrainModel) -> MemoryStats:
        """Measure memory usage of a brain model.
        
        Args:
            model: Brain model to measure
            
        Returns:
            MemoryStats object with detailed breakdown
        """
        import sys
        
        # Measure neurons
        neurons_bytes = sum(sys.getsizeof(n) for n in model.neurons.values())
        
        # Measure synapses
        synapses_bytes = sum(sys.getsizeof(s) for s in model.synapses)
        
        # Estimate spike history (if accessible)
        spike_history_bytes = 0
        
        # Other data structures
        other_bytes = sys.getsizeof(model.config)
        
        total_bytes = neurons_bytes + synapses_bytes + spike_history_bytes + other_bytes
        
        return MemoryStats(
            total_bytes=total_bytes,
            neurons_bytes=neurons_bytes,
            synapses_bytes=synapses_bytes,
            spike_history_bytes=spike_history_bytes,
            other_bytes=other_bytes
        )
    
    def take_snapshot(self, label: str, model: BrainModel) -> None:
        """Take a memory usage snapshot.
        
        Args:
            label: Label for this snapshot
            model: Brain model to measure
        """
        stats = self.measure_model_memory(model)
        self.snapshots.append((label, stats))
    
    def compare_snapshots(
        self,
        label1: str,
        label2: str
    ) -> Optional[Dict[str, float]]:
        """Compare two memory snapshots.
        
        Args:
            label1: First snapshot label
            label2: Second snapshot label
            
        Returns:
            Dictionary with memory differences in MB
        """
        snap1 = None
        snap2 = None
        
        for label, stats in self.snapshots:
            if label == label1:
                snap1 = stats
            if label == label2:
                snap2 = stats
        
        if snap1 is None or snap2 is None:
            return None
        
        return {
            'total_diff_mb': (snap2.total_bytes - snap1.total_bytes) / (1024 ** 2),
            'neurons_diff_mb': (snap2.neurons_bytes - snap1.neurons_bytes) / (1024 ** 2),
            'synapses_diff_mb': (snap2.synapses_bytes - snap1.synapses_bytes) / (1024 ** 2)
        }
    
    def get_report(self) -> str:
        """Get a memory usage report as a formatted string.
        
        Returns:
            Formatted memory usage report
        """
        lines = []
        lines.append("\nMemory Usage Report")
        lines.append("=" * 60)
        
        for label, stats in self.snapshots:
            mb = stats.to_mb()
            lines.append(f"\n{label}:")
            lines.append(f"  Total:        {mb['total_mb']:>10.2f} MB")
            lines.append(f"  Neurons:      {mb['neurons_mb']:>10.2f} MB")
            lines.append(f"  Synapses:     {mb['synapses_mb']:>10.2f} MB")
            lines.append(f"  Spike History:{mb['spike_history_mb']:>10.2f} MB")
            lines.append(f"  Other:        {mb['other_mb']:>10.2f} MB")
        
        return '\n'.join(lines)
    
    def print_report(self) -> None:
        """Print a memory usage report."""
        print(self.get_report())


def create_memory_mapped_model(filepath: str, mode: str = 'r+') -> MemoryMappedModel:
    """Factory function to create a memory-mapped model.
    
    Args:
        filepath: Path to storage file
        mode: File access mode
        
    Returns:
        MemoryMappedModel instance
    """
    return MemoryMappedModel(filepath, mode)


def optimize_model_memory(
    model: BrainModel,
    enable_compression: bool = True,
    enable_cache_optimization: bool = True
) -> Dict[str, Any]:
    """Apply memory optimizations to a model.
    
    Args:
        model: Brain model to optimize
        enable_compression: Enable inactive neuron compression
        enable_cache_optimization: Enable cache optimization
        
    Returns:
        Dictionary with optimization results
    """
    results = {
        'initial_memory_mb': 0.0,
        'final_memory_mb': 0.0,
        'bytes_saved': 0,
        'optimizations_applied': []
    }
    
    # Measure initial memory
    profiler = MemoryProfiler()
    initial_stats = profiler.measure_model_memory(model)
    results['initial_memory_mb'] = initial_stats.total_bytes / (1024 ** 2)
    
    # Apply cache optimization
    if enable_cache_optimization:
        optimizer = CacheOptimizer()
        model.neurons = optimizer.reorder_neurons_spatial_locality(model.neurons)
        results['optimizations_applied'].append('spatial_locality_reordering')
    
    # Note: Compression would require integration with simulation loop
    # to maintain compressed/active neuron tracking
    
    # Measure final memory
    final_stats = profiler.measure_model_memory(model)
    results['final_memory_mb'] = final_stats.total_bytes / (1024 ** 2)
    results['bytes_saved'] = initial_stats.total_bytes - final_stats.total_bytes
    
    return results
