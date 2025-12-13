#!/usr/bin/env python3
"""Demo of Virtual Neuromorphic Clock (VNC) system.

This example demonstrates how to use the VNC system for massively parallel
neuron processing with configurable clock frequencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel
from simulation import Simulation
import time


def create_demo_model():
    """Create a demo brain model with multiple w-slices."""
    config = {
        "lattice_shape": [20, 20, 5, 8],  # 8 w-slices
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
            "enable_death": False,  # Disable for stable demo
            "enable_reproduction": False,
            "max_age": 10000,
            "health_decay_per_step": 0.0,
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
            "vision": {"areal": "V1", "input_size": [10, 10]},
        },
        "areas": [
            {
                "name": "V1",
                "coord_ranges": {"x": [0, 19], "y": [0, 19], "z": [0, 4], "w": [0, 7]},
            }
        ]
    }
    
    model = BrainModel(config=config)
    
    # Add neurons distributed across w-slices
    print("Creating neurons across w-slices...")
    neuron_count = 0
    for w in range(8):
        for i in range(50):  # 50 neurons per w-slice
            x = (i * 3) % 20
            y = (i * 5) % 20
            z = i % 5
            model.add_neuron(x, y, z, w)
            neuron_count += 1
    
    print(f"Created {neuron_count} neurons across 8 w-slices")
    return model


def demo_standard_mode(model):
    """Run simulation in standard mode (no VNC)."""
    print("\n" + "="*60)
    print("STANDARD MODE (No VNC)")
    print("="*60)
    
    sim = Simulation(model, use_vnc=False, seed=42)
    
    # Initialize some synapses
    sim.initialize_random_synapses(connection_probability=0.05)
    
    print(f"Neurons: {len(model.neurons)}")
    print(f"Synapses: {len(model.synapses)}")
    
    # Run simulation and time it
    start_time = time.time()
    results = sim.run(n_steps=100, verbose=False)
    elapsed_time = time.time() - start_time
    
    total_spikes = sum(len(r["spikes"]) for r in results)
    
    print(f"\nResults:")
    print(f"  Time: {elapsed_time:.3f} seconds")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Spikes/second: {total_spikes/elapsed_time:.0f}")
    print(f"  Neurons processed: {len(model.neurons) * 100}")
    print(f"  Neurons/second: {(len(model.neurons) * 100)/elapsed_time:.0f}")
    
    return elapsed_time


def demo_vnc_mode(model, clock_frequency=20e6):
    """Run simulation in VNC mode."""
    print("\n" + "="*60)
    print(f"VNC MODE (Clock: {clock_frequency/1e6:.1f} MHz)")
    print("="*60)
    
    sim = Simulation(
        model,
        use_vnc=True,
        vnc_clock_frequency=clock_frequency,
        seed=42
    )
    
    # Initialize some synapses
    sim.initialize_random_synapses(connection_probability=0.05)
    
    print(f"Neurons: {len(model.neurons)}")
    print(f"Synapses: {len(model.synapses)}")
    
    # Get VNC configuration
    vnc_stats = sim.get_vnc_statistics()
    print(f"VPUs: {vnc_stats['num_vpus']}")
    print(f"Configured clock: {vnc_stats['configured_clock_hz']/1e6:.1f} MHz")
    
    # Run simulation and time it
    start_time = time.time()
    results = sim.run(n_steps=100, verbose=False)
    elapsed_time = time.time() - start_time
    
    total_spikes = sum(len(r["spikes"]) for r in results)
    
    # Get final VNC statistics
    vnc_stats = sim.get_vnc_statistics()
    
    print(f"\nResults:")
    print(f"  Time: {elapsed_time:.3f} seconds")
    print(f"  Total spikes: {total_spikes}")
    print(f"  Spikes/second: {total_spikes/elapsed_time:.0f}")
    print(f"  Neurons processed: {vnc_stats['total_neurons_processed']}")
    print(f"  Neurons/second: {vnc_stats.get('neurons_per_second', 0):.0f}")
    print(f"  Effective clock: {vnc_stats.get('effective_clock_hz', 0):.0f} Hz")
    print(f"  Total cycles: {vnc_stats['total_cycles']}")
    
    # Show VPU statistics
    vpu_stats = sim.get_vpu_statistics()
    print(f"\nVPU Statistics:")
    for i, vpu in enumerate(vpu_stats[:3]):  # Show first 3 VPUs
        print(f"  VPU {i}: {vpu['neurons_processed']} neurons, "
              f"{vpu['cycles_executed']} cycles, "
              f"avg {vpu.get('avg_processing_time_ms', 0):.3f} ms/cycle")
    if len(vpu_stats) > 3:
        print(f"  ... and {len(vpu_stats) - 3} more VPUs")
    
    return elapsed_time


def compare_modes():
    """Compare standard and VNC modes."""
    print("\n" + "="*60)
    print("VNC SYSTEM DEMONSTRATION")
    print("="*60)
    print("\nThis demo compares standard simulation mode with the")
    print("Virtual Neuromorphic Clock (VNC) parallel processing mode.")
    print()
    
    # Create model
    model = create_demo_model()
    
    # Run standard mode
    std_time = demo_standard_mode(model)
    
    # Recreate model for fair comparison
    model = create_demo_model()
    
    # Run VNC mode with 20 MHz clock
    vnc_time = demo_vnc_mode(model, clock_frequency=20e6)
    
    # Compare
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    print(f"Standard mode: {std_time:.3f} seconds")
    print(f"VNC mode:      {vnc_time:.3f} seconds")
    
    if vnc_time < std_time:
        speedup = std_time / vnc_time
        print(f"\n✅ VNC mode is {speedup:.2f}x faster!")
    else:
        slowdown = vnc_time / std_time
        print(f"\n⚠️  VNC mode is {slowdown:.2f}x slower")
        print("Note: For small networks, VNC overhead may exceed benefits.")
        print("VNC shows advantages with larger networks (10k+ neurons).")
    
    print("\n" + "="*60)
    print("Key VNC Features:")
    print("  • Parallel processing across multiple VPUs")
    print("  • Configurable clock frequency (0.1 MHz - 1 GHz)")
    print("  • Automatic 4D slice partitioning")
    print("  • Adaptive load balancing")
    print("  • Virtual I/O expansion (262k+ ports)")
    print("  • Dashboard monitoring and control")
    print("="*60)


if __name__ == "__main__":
    try:
        compare_modes()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
