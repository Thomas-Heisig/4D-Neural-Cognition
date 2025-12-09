"""Performance Optimization Demo for 4D Neural Cognition.

This example demonstrates GPU acceleration, parallel computing, and memory
optimization features for large-scale neural network simulations.
"""

import time
import sys
import os
import numpy as np

# Add parent directory to path to import src modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.brain_model import BrainModel
from src.simulation import Simulation
from src.gpu_acceleration import GPUAccelerator, CUPY_AVAILABLE
from src.parallel_computing import ParallelSimulator, benchmark_parallel_scaling
from src.memory_optimization import (
    MemoryProfiler,
    InactiveNeuronCompressor,
    CacheOptimizer,
    optimize_model_memory
)


def create_test_model(n_neurons: int = 1000):
    """Create a test model with specified number of neurons."""
    config = {
        "lattice_shape": [50, 50, 20, 10],
        "neuron_model": {
            "params_default": {
                "v_rest": -65.0,
                "v_threshold": -50.0,
                "v_reset": -70.0,
                "tau_m": 20.0,
                "refractory_period": 5
            }
        },
        "cell_lifecycle": {
            "enabled": False,
            "enable_death": False,
            "enable_reproduction": False,
            "max_age": 100000,
            "health_decay_per_step": 0.0001,
            "aging_rate": 0.001,
            "death_threshold": 0.1,
            "reproduction_threshold": 0.9,
            "reproduction_probability": 0.1,
            "reproduction_radius": 2,
            "mutation_rate": 0.01,
            "mutation_std_params": 0.05,
            "mutation_std_weights": 0.02
        },
        "plasticity": {
            "enabled": False
        },
        "senses": {},
        "areas": []
    }
    
    # Create model with sparse connectivity
    print(f"Creating model with {n_neurons} neurons...")
    model = BrainModel(config=config, use_sparse_connectivity=True)
    
    # Add neurons
    for i in range(n_neurons):
        x = i % 50
        y = (i // 50) % 50
        z = (i // 2500) % 20
        w = (i // 50000) % 10
        model.add_neuron(x, y, z, w)
    
    # Add synapses with 1% connectivity
    connection_prob = 0.01
    neuron_ids = list(model.neurons.keys())
    n_connections = int(n_neurons * n_neurons * connection_prob)
    print(f"Adding ~{n_connections} synapses...")
    
    for _ in range(n_connections):
        pre_id = np.random.choice(neuron_ids)
        post_id = np.random.choice(neuron_ids)
        if pre_id != post_id:
            weight = np.random.normal(0.1, 0.05)
            model.add_synapse(pre_id, post_id, weight)
    
    return model


def demo_gpu_acceleration():
    """Demonstrate GPU acceleration features."""
    print("\n" + "="*60)
    print("GPU ACCELERATION DEMO")
    print("="*60)
    
    # Initialize GPU accelerator
    accelerator = GPUAccelerator(use_gpu=True)
    
    if not accelerator.is_gpu_available():
        print("⚠️  GPU not available. Install CuPy for GPU support:")
        print("   pip install cupy-cuda12x")
        print("\nFalling back to CPU demonstrations...")
    else:
        print(f"✓ GPU acceleration enabled")
        
        # Show GPU memory
        memory = accelerator.get_memory_usage()
        print(f"  GPU Memory: {memory['used']:.1f} / {memory['total']:.1f} MB")
    
    # Benchmark vectorized LIF update
    print("\nBenchmarking vectorized LIF neuron update:")
    print("  Testing with different neuron counts...")
    
    results = accelerator.benchmark_vs_cpu(
        operation='lif_update',
        array_sizes=[100, 500, 1000, 5000],
        n_iterations=50
    )
    
    print("\n  Results:")
    print("  Neurons | CPU Time | GPU Time | Speedup")
    print("  --------|----------|----------|--------")
    for size, cpu_time, gpu_time, speedup in zip(
        results['array_sizes'],
        results['cpu_times'],
        results['gpu_times'],
        results['speedups']
    ):
        gpu_str = f"{gpu_time*1000:.2f}ms" if gpu_time != float('inf') else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        print(f"  {size:7d} | {cpu_time*1000:8.2f}ms | {gpu_str:>8} | {speedup_str:>7}")
    
    # Benchmark matrix multiplication
    print("\nBenchmarking matrix multiplication:")
    results_matmul = accelerator.benchmark_vs_cpu(
        operation='matmul',
        array_sizes=[100, 500, 1000],
        n_iterations=30
    )
    
    print("\n  Results:")
    print("  Size    | CPU Time | GPU Time | Speedup")
    print("  --------|----------|----------|--------")
    for size, cpu_time, gpu_time, speedup in zip(
        results_matmul['array_sizes'],
        results_matmul['cpu_times'],
        results_matmul['gpu_times'],
        results_matmul['speedups']
    ):
        gpu_str = f"{gpu_time*1000:.2f}ms" if gpu_time != float('inf') else "N/A"
        speedup_str = f"{speedup:.2f}x" if speedup > 0 else "N/A"
        print(f"  {size:3d}x{size:3d} | {cpu_time*1000:8.2f}ms | {gpu_str:>8} | {speedup_str:>7}")


def demo_parallel_computing(model):
    """Demonstrate parallel computing features."""
    print("\n" + "="*60)
    print("PARALLEL COMPUTING DEMO")
    print("="*60)
    
    n_neurons = len(model.neurons)
    print(f"Model size: {n_neurons} neurons, {len(model.synapses)} synapses")
    
    # Initialize parallel simulator
    print("\nInitializing parallel simulator with 4 processes...")
    sim = ParallelSimulator(model, n_processes=4, use_spatial_partitioning=True)
    
    # Show load balance
    stats = sim.engine.get_load_balance_stats()
    print(f"\nLoad balance statistics:")
    print(f"  Neurons per partition: {stats['neurons_per_partition']}")
    print(f"  Mean: {stats['mean']:.1f}, Std Dev: {stats['std']:.1f}")
    print(f"  Range: {stats['min']} - {stats['max']}")
    print(f"  Imbalance: {stats['imbalance']:.2%}")
    
    # Benchmark parallel scaling
    print("\nBenchmarking parallel scaling...")
    print("  (This may take a minute...)")
    
    scaling_results = benchmark_parallel_scaling(
        model,
        n_steps=50,
        process_counts=[1, 2, 4]
    )
    
    print("\n  Results:")
    print("  Processes | Time      | Speedup | Efficiency")
    print("  ----------|-----------|---------|------------")
    for n_proc, time_val, speedup, efficiency in zip(
        scaling_results['process_counts'],
        scaling_results['times'],
        scaling_results['speedups'],
        scaling_results['efficiencies']
    ):
        print(f"  {n_proc:9d} | {time_val:8.2f}s | {speedup:6.2f}x | {efficiency:9.1%}")


def demo_memory_optimization(model):
    """Demonstrate memory optimization features."""
    print("\n" + "="*60)
    print("MEMORY OPTIMIZATION DEMO")
    print("="*60)
    
    # Memory profiling
    print("\n1. Memory Profiling:")
    profiler = MemoryProfiler()
    
    profiler.take_snapshot('initial', model)
    initial_stats = profiler.snapshots[0][1]
    mb = initial_stats.to_mb()
    
    print(f"  Initial memory usage:")
    print(f"    Total:        {mb['total_mb']:>10.2f} MB")
    print(f"    Neurons:      {mb['neurons_mb']:>10.2f} MB")
    print(f"    Synapses:     {mb['synapses_mb']:>10.2f} MB")
    print(f"    Other:        {mb['other_mb']:>10.2f} MB")
    
    # Cache optimization
    print("\n2. Cache Optimization:")
    optimizer = CacheOptimizer()
    
    # Analyze random access pattern
    random_access = np.random.choice(list(model.neurons.keys()), size=100)
    stats_random = optimizer.analyze_access_pattern(list(random_access))
    
    # Analyze sequential access pattern
    sequential_access = sorted(list(model.neurons.keys()))[:100]
    stats_seq = optimizer.analyze_access_pattern(sequential_access)
    
    print(f"  Random access pattern:")
    print(f"    Locality score: {stats_random['locality_score']:.3f}")
    print(f"    Avg spatial jump: {stats_random['avg_spatial_jump']:.1f}")
    
    print(f"  Sequential access pattern:")
    print(f"    Locality score: {stats_seq['locality_score']:.3f}")
    print(f"    Avg spatial jump: {stats_seq['avg_spatial_jump']:.1f}")
    
    print(f"\n  Reordering neurons for spatial locality...")
    model.neurons = optimizer.reorder_neurons_spatial_locality(model.neurons)
    print(f"  ✓ Neurons reordered using Z-order curve")
    
    # High-level optimization
    print("\n3. Applying combined optimizations:")
    opt_results = optimize_model_memory(
        model,
        enable_compression=False,  # Skip compression for this demo
        enable_cache_optimization=True
    )
    
    print(f"  Optimizations applied: {opt_results['optimizations_applied']}")
    print(f"  Memory before: {opt_results['initial_memory_mb']:.2f} MB")
    print(f"  Memory after:  {opt_results['final_memory_mb']:.2f} MB")
    
    # Inactive neuron compression demo
    print("\n4. Inactive Neuron Compression:")
    compressor = InactiveNeuronCompressor(inactivity_threshold=50)
    
    # Simulate some spike history
    spike_history = {}
    for neuron_id in model.neurons.keys():
        # Some neurons are active, some inactive
        if np.random.rand() < 0.3:
            spike_history[neuron_id] = [10, 20, 30, 40, 95]  # Active
        else:
            spike_history[neuron_id] = [10]  # Inactive
    
    active_neurons, bytes_saved = compressor.compress_inactive_neurons(
        model.neurons,
        spike_history,
        current_step=100
    )
    
    n_compressed = len(model.neurons) - len(active_neurons)
    print(f"  Compressed {n_compressed} inactive neurons")
    print(f"  Memory saved: {bytes_saved / 1024:.2f} KB")
    
    comp_stats = compressor.get_compression_stats()
    print(f"  Avg bytes per compressed neuron: {comp_stats['avg_bytes_per_neuron']:.1f}")


def demo_combined_optimizations():
    """Demonstrate combining all optimization strategies."""
    print("\n" + "="*60)
    print("COMBINED OPTIMIZATIONS DEMO")
    print("="*60)
    
    # Create a medium-sized model
    print("\nCreating medium-sized model (2000 neurons)...")
    model = create_test_model(n_neurons=2000)
    
    # Profile baseline
    print("\nBaseline performance (no optimizations):")
    sim_baseline = Simulation(model, seed=42)
    
    start = time.perf_counter()
    for _ in range(100):
        sim_baseline.step()
    baseline_time = time.perf_counter() - start
    print(f"  Time for 100 steps: {baseline_time:.2f}s")
    
    # With optimizations
    print("\nOptimized performance:")
    
    # 1. Enable sparse connectivity (already enabled in create_test_model)
    print("  ✓ Sparse connectivity enabled")
    
    # 2. Enable time-indexed spikes
    sim_optimized = Simulation(model, seed=42, use_time_indexed_spikes=True)
    print("  ✓ Time-indexed spike lookup enabled")
    
    # 3. Optimize memory
    optimize_model_memory(model, enable_cache_optimization=True)
    print("  ✓ Cache optimization applied")
    
    start = time.perf_counter()
    for _ in range(100):
        sim_optimized.step()
    optimized_time = time.perf_counter() - start
    
    speedup = baseline_time / optimized_time if optimized_time > 0 else 0
    print(f"\n  Time for 100 steps: {optimized_time:.2f}s")
    print(f"  Speedup: {speedup:.2f}x")


def main():
    """Run all demonstrations."""
    print("\n" + "#"*60)
    print("# 4D NEURAL COGNITION - PERFORMANCE OPTIMIZATION DEMO")
    print("#"*60)
    
    # Check for GPU support
    if CUPY_AVAILABLE:
        print("\n✓ CuPy available - GPU acceleration supported")
    else:
        print("\n⚠️  CuPy not available - GPU demos will show CPU fallback")
    
    # Demo 1: GPU Acceleration
    demo_gpu_acceleration()
    
    # Create model for other demos
    model = create_test_model(n_neurons=1000)
    
    # Demo 2: Parallel Computing
    demo_parallel_computing(model)
    
    # Demo 3: Memory Optimization
    demo_memory_optimization(model)
    
    # Demo 4: Combined Optimizations
    demo_combined_optimizations()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nFor more information, see docs/PERFORMANCE_OPTIMIZATION.md")


if __name__ == "__main__":
    main()
