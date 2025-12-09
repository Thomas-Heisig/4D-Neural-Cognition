"""Parallel computing module for 4D Neural Cognition.

This module provides multi-core CPU parallelization capabilities including:
- Spatial partitioning for parallel neuron updates
- Load balancing across cores
- Parallel synapse computation
- Benchmark tools for scaling characteristics
"""

from typing import List, Dict, Tuple, Optional, Callable, Any
import numpy as np
from multiprocessing import Pool, cpu_count, Manager
from functools import partial
import warnings

try:
    from .brain_model import BrainModel, Neuron, Synapse
except ImportError:
    from brain_model import BrainModel, Neuron, Synapse


class SpatialPartition:
    """Represents a spatial partition of the 4D lattice for parallel processing."""
    
    def __init__(
        self,
        partition_id: int,
        x_range: Tuple[int, int],
        y_range: Tuple[int, int],
        z_range: Tuple[int, int],
        w_range: Tuple[int, int]
    ):
        """Initialize spatial partition.
        
        Args:
            partition_id: Unique identifier for this partition
            x_range: (min, max) range in x dimension
            y_range: (min, max) range in y dimension
            z_range: (min, max) range in z dimension
            w_range: (min, max) range in w dimension
        """
        self.partition_id = partition_id
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.w_range = w_range
        self.neuron_ids: List[int] = []
        self.boundary_neurons: List[int] = []  # Neurons near partition boundaries
    
    def contains_position(self, x: int, y: int, z: int, w: int) -> bool:
        """Check if a position is within this partition.
        
        Args:
            x, y, z, w: 4D coordinates
            
        Returns:
            True if position is in partition
        """
        return (self.x_range[0] <= x <= self.x_range[1] and
                self.y_range[0] <= y <= self.y_range[1] and
                self.z_range[0] <= z <= self.z_range[1] and
                self.w_range[0] <= w <= self.w_range[1])
    
    def volume(self) -> int:
        """Calculate the volume (number of positions) in this partition."""
        return ((self.x_range[1] - self.x_range[0] + 1) *
                (self.y_range[1] - self.y_range[0] + 1) *
                (self.z_range[1] - self.z_range[0] + 1) *
                (self.w_range[1] - self.w_range[0] + 1))


class ParallelSimulationEngine:
    """Engine for parallel simulation of neural network dynamics."""
    
    def __init__(
        self,
        n_processes: Optional[int] = None,
        partition_strategy: str = 'spatial',
        load_balancing: bool = True
    ):
        """Initialize parallel simulation engine.
        
        Args:
            n_processes: Number of parallel processes (default: number of CPU cores)
            partition_strategy: How to partition neurons ('spatial', 'random', 'area-based')
            load_balancing: Whether to enable dynamic load balancing
        """
        self.n_processes = n_processes or cpu_count()
        self.partition_strategy = partition_strategy
        self.load_balancing = load_balancing
        self.partitions: List[SpatialPartition] = []
        
        print(f"Parallel computing initialized with {self.n_processes} processes")
    
    def create_spatial_partitions(
        self,
        lattice_shape: List[int],
        n_partitions: Optional[int] = None
    ) -> List[SpatialPartition]:
        """Create spatial partitions of the 4D lattice.
        
        Divides the lattice into roughly equal-sized regions for parallel processing.
        
        Args:
            lattice_shape: [x_size, y_size, z_size, w_size] dimensions
            n_partitions: Number of partitions (default: n_processes)
            
        Returns:
            List of SpatialPartition objects
        """
        n_partitions = n_partitions or self.n_processes
        partitions = []
        
        x_size, y_size, z_size, w_size = lattice_shape
        
        # Simple strategy: partition along the largest dimension
        total_volume = x_size * y_size * z_size * w_size
        target_volume_per_partition = total_volume // n_partitions
        
        # Find best dimension to split
        dims = [('x', x_size), ('y', y_size), ('z', z_size), ('w', w_size)]
        dims.sort(key=lambda x: x[1], reverse=True)
        split_dim = dims[0][0]
        
        if split_dim == 'x':
            split_size = x_size // n_partitions
            for i in range(n_partitions):
                x_start = i * split_size
                x_end = (i + 1) * split_size - 1 if i < n_partitions - 1 else x_size - 1
                partition = SpatialPartition(
                    partition_id=i,
                    x_range=(x_start, x_end),
                    y_range=(0, y_size - 1),
                    z_range=(0, z_size - 1),
                    w_range=(0, w_size - 1)
                )
                partitions.append(partition)
        elif split_dim == 'y':
            split_size = y_size // n_partitions
            for i in range(n_partitions):
                y_start = i * split_size
                y_end = (i + 1) * split_size - 1 if i < n_partitions - 1 else y_size - 1
                partition = SpatialPartition(
                    partition_id=i,
                    x_range=(0, x_size - 1),
                    y_range=(y_start, y_end),
                    z_range=(0, z_size - 1),
                    w_range=(0, w_size - 1)
                )
                partitions.append(partition)
        elif split_dim == 'z':
            split_size = z_size // n_partitions
            for i in range(n_partitions):
                z_start = i * split_size
                z_end = (i + 1) * split_size - 1 if i < n_partitions - 1 else z_size - 1
                partition = SpatialPartition(
                    partition_id=i,
                    x_range=(0, x_size - 1),
                    y_range=(0, y_size - 1),
                    z_range=(z_start, z_end),
                    w_range=(0, w_size - 1)
                )
                partitions.append(partition)
        else:  # w
            split_size = w_size // n_partitions
            for i in range(n_partitions):
                w_start = i * split_size
                w_end = (i + 1) * split_size - 1 if i < n_partitions - 1 else w_size - 1
                partition = SpatialPartition(
                    partition_id=i,
                    x_range=(0, x_size - 1),
                    y_range=(0, y_size - 1),
                    z_range=(0, z_size - 1),
                    w_range=(w_start, w_end)
                )
                partitions.append(partition)
        
        self.partitions = partitions
        return partitions
    
    def assign_neurons_to_partitions(
        self,
        neurons: Dict[int, Neuron],
        partitions: List[SpatialPartition]
    ) -> None:
        """Assign neurons to spatial partitions based on their positions.
        
        Args:
            neurons: Dictionary of neuron_id -> Neuron
            partitions: List of spatial partitions
        """
        for neuron_id, neuron in neurons.items():
            assigned = False
            for partition in partitions:
                if partition.contains_position(neuron.x, neuron.y, neuron.z, neuron.w):
                    partition.neuron_ids.append(neuron_id)
                    assigned = True
                    break
            
            if not assigned:
                warnings.warn(f"Neuron {neuron_id} not assigned to any partition")
    
    def get_load_balance_stats(self) -> Dict[str, Any]:
        """Get statistics about load balancing across partitions.
        
        Returns:
            Dictionary with load balance metrics:
                - neurons_per_partition: List of neuron counts
                - mean: Average neurons per partition
                - std: Standard deviation
                - min: Minimum neurons in a partition
                - max: Maximum neurons in a partition
                - imbalance: (max - min) / mean
        """
        counts = [len(p.neuron_ids) for p in self.partitions]
        mean_count = np.mean(counts) if counts else 0
        
        return {
            'neurons_per_partition': counts,
            'mean': mean_count,
            'std': np.std(counts) if counts else 0,
            'min': min(counts) if counts else 0,
            'max': max(counts) if counts else 0,
            'imbalance': (max(counts) - min(counts)) / mean_count if mean_count > 0 else 0
        }


def _process_partition_neurons(
    partition_neuron_ids: List[int],
    neuron_data: Dict[int, Dict],
    synaptic_inputs: Dict[int, float],
    lif_params: Dict[str, float],
    current_step: int,
    dt: float
) -> Dict[int, Tuple[float, bool]]:
    """Worker function to process neurons in a partition.
    
    This function runs in a separate process.
    
    Args:
        partition_neuron_ids: List of neuron IDs in this partition
        neuron_data: Dictionary of neuron states (serializable)
        synaptic_inputs: Pre-computed synaptic inputs for each neuron
        lif_params: LIF model parameters
        current_step: Current simulation step
        dt: Time step
        
    Returns:
        Dictionary mapping neuron_id -> (new_v_membrane, spiked)
    """
    results = {}
    
    tau_m = lif_params.get('tau_m', 20.0)
    v_rest = lif_params.get('v_rest', -65.0)
    v_reset = lif_params.get('v_reset', -70.0)
    v_threshold = lif_params.get('v_threshold', -50.0)
    refractory_period = lif_params.get('refractory_period', 5.0)
    
    for neuron_id in partition_neuron_ids:
        if neuron_id not in neuron_data:
            continue
        
        neuron = neuron_data[neuron_id]
        v_membrane = neuron['v_membrane']
        last_spike_time = neuron['last_spike_time']
        external_input = neuron['external_input']
        
        # Check refractory period
        time_since_spike = current_step - last_spike_time
        if time_since_spike < refractory_period:
            results[neuron_id] = (v_membrane, False)
            continue
        
        # Get synaptic input
        synaptic_input = synaptic_inputs.get(neuron_id, 0.0)
        
        # Total input
        total_input = synaptic_input + external_input
        
        # Leaky integration
        dv = (-(v_membrane - v_rest) + total_input) / tau_m * dt
        v_membrane += dv
        
        # Check for NaN/Inf
        if np.isnan(v_membrane) or np.isinf(v_membrane):
            v_membrane = v_rest
        
        # Check for spike
        spiked = False
        if v_membrane >= v_threshold:
            v_membrane = v_reset
            spiked = True
        
        results[neuron_id] = (v_membrane, spiked)
    
    return results


class ParallelSimulator:
    """High-level interface for parallel simulation."""
    
    def __init__(
        self,
        model: BrainModel,
        n_processes: Optional[int] = None,
        use_spatial_partitioning: bool = True
    ):
        """Initialize parallel simulator.
        
        Args:
            model: Brain model to simulate
            n_processes: Number of processes (default: CPU count)
            use_spatial_partitioning: Use spatial partitioning strategy
        """
        self.model = model
        self.engine = ParallelSimulationEngine(
            n_processes=n_processes,
            partition_strategy='spatial' if use_spatial_partitioning else 'random'
        )
        
        # Create partitions
        lattice_shape = model.config['lattice_shape']
        self.engine.create_spatial_partitions(lattice_shape)
        
        # Assign neurons to partitions
        self.engine.assign_neurons_to_partitions(model.neurons, self.engine.partitions)
        
        # Print load balance stats
        stats = self.engine.get_load_balance_stats()
        print(f"Load balance: {stats['mean']:.1f} ± {stats['std']:.1f} neurons/partition "
              f"(range: {stats['min']}-{stats['max']}, imbalance: {stats['imbalance']:.2%})")
    
    def parallel_neuron_update(
        self,
        synaptic_inputs: Dict[int, float],
        current_step: int,
        dt: float = 1.0
    ) -> Dict[int, Tuple[float, bool]]:
        """Update all neurons in parallel across partitions.
        
        Args:
            synaptic_inputs: Pre-computed synaptic inputs
            current_step: Current simulation step
            dt: Time step
            
        Returns:
            Dictionary mapping neuron_id -> (new_v_membrane, spiked)
        """
        # Prepare neuron data for serialization
        neuron_data = {
            nid: {
                'v_membrane': n.v_membrane,
                'last_spike_time': n.last_spike_time,
                'external_input': n.external_input
            }
            for nid, n in self.model.neurons.items()
        }
        
        # Get LIF parameters
        lif_params = self.model.get_neuron_model_params()
        
        # Create tasks for each partition
        tasks = []
        for partition in self.engine.partitions:
            tasks.append((
                partition.neuron_ids,
                neuron_data,
                synaptic_inputs,
                lif_params,
                current_step,
                dt
            ))
        
        # Execute in parallel
        with Pool(processes=self.engine.n_processes) as pool:
            partition_results = pool.starmap(_process_partition_neurons, tasks)
        
        # Merge results
        all_results = {}
        for result_dict in partition_results:
            all_results.update(result_dict)
        
        return all_results


def benchmark_parallel_scaling(
    model: BrainModel,
    n_steps: int = 100,
    process_counts: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Benchmark parallel scaling characteristics.
    
    Measures speedup and efficiency as a function of process count.
    
    Args:
        model: Brain model to benchmark
        n_steps: Number of simulation steps
        process_counts: List of process counts to test (default: [1, 2, 4, 8, ...])
        
    Returns:
        Dictionary with benchmark results:
            - process_counts: List of process counts tested
            - times: Execution times for each count
            - speedups: Speedup relative to single process
            - efficiencies: Parallel efficiency (speedup / n_processes)
    """
    import time
    
    if process_counts is None:
        max_processes = cpu_count()
        process_counts = [1]
        p = 2
        while p <= max_processes:
            process_counts.append(p)
            p *= 2
    
    results = {
        'process_counts': process_counts,
        'times': [],
        'speedups': [],
        'efficiencies': []
    }
    
    # Dummy synaptic inputs for testing
    synaptic_inputs = {nid: 0.0 for nid in model.neurons.keys()}
    
    baseline_time = None
    
    for n_proc in process_counts:
        print(f"Benchmarking with {n_proc} processes...")
        
        # Create parallel simulator
        sim = ParallelSimulator(model, n_processes=n_proc)
        
        # Time the execution
        start = time.perf_counter()
        for step in range(n_steps):
            sim.parallel_neuron_update(synaptic_inputs, step)
        elapsed = time.perf_counter() - start
        
        results['times'].append(elapsed)
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        efficiency = speedup / n_proc if n_proc > 0 else 0.0
        
        results['speedups'].append(speedup)
        results['efficiencies'].append(efficiency)
        
        print(f"  Time: {elapsed:.2f}s, Speedup: {speedup:.2f}x, Efficiency: {efficiency:.1%}")
    
    return results
