#!/usr/bin/env python3
"""Example script demonstrating the 4D Neural Cognition system.

This script:
1. Initializes the 4D brain model from configuration
2. Creates neurons in sensory areas
3. Runs a simulation with sensory input (including digital sense)
4. Saves the model state to JSON and optionally HDF4
"""

import sys
import os
import json
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input, create_digital_sense_input
from storage import save_to_json, load_from_json


def main():
    """Run the 4D Neural Cognition example."""
    print("=" * 60)
    print("4D Neural Cognition - Example Simulation")
    print("=" * 60)

    # Load configuration
    config_path = os.path.join(
        os.path.dirname(__file__), "brain_base_model.json"
    )

    print(f"\n1. Loading configuration from: {config_path}")
    model = BrainModel(config_path=config_path)

    print(f"   Lattice shape: {model.lattice_shape}")
    print(f"   Total possible neurons: {np.prod(model.lattice_shape)}")
    print(f"   Senses: {list(model.get_senses().keys())}")
    print(f"   Areas: {[a['name'] for a in model.get_areas()]}")

    # Initialize simulation
    print("\n2. Initializing simulation...")
    sim = Simulation(model, seed=42)

    # Initialize neurons in selected areas (using lower density for demo)
    # We'll use just vision and digital areas for this example
    areas_to_init = ["V1_like", "Digital_sensor"]
    density = 0.1  # 10% of positions filled

    print(f"   Initializing neurons in areas: {areas_to_init}")
    print(f"   Density: {density * 100}%")

    sim.initialize_neurons(area_names=areas_to_init, density=density)
    print(f"   Created {len(model.neurons)} neurons")

    # Create some random synapses
    print("\n3. Creating synaptic connections...")
    sim.initialize_random_synapses(
        connection_probability=0.001,  # 0.1% connection probability (reduced for faster demo)
        weight_mean=0.1,
        weight_std=0.05,
    )
    print(f"   Created {len(model.synapses)} synapses")

    # Prepare sensory inputs
    print("\n4. Preparing sensory inputs...")

    # Vision input: simple pattern (gradient)
    vision_input = np.zeros((20, 20))
    for i in range(20):
        for j in range(20):
            vision_input[i, j] = (i + j) / 40.0 * 10.0  # Gradient pattern
    print(f"   Vision input shape: {vision_input.shape}")

    # Digital sense input: binary data from a string
    digital_data = "Hello, 4D Neural Cognition!"
    digital_input = create_digital_sense_input(digital_data, target_shape=(20, 20))
    print(f"   Digital input from: '{digital_data}'")
    print(f"   Digital input shape: {digital_input.shape}")

    # Run simulation
    print("\n5. Running simulation...")
    n_steps = 100
    print(f"   Running {n_steps} steps...")

    total_spikes = 0
    total_deaths = 0
    total_births = 0

    # Find neurons at z=0 (input layer) to inject strong input
    z0_neurons = [n for n in model.neurons.values() if n.z == 0]
    print(f"   Neurons at input layer (z=0): {len(z0_neurons)}")

    for step in range(n_steps):
        # Apply sensory input every 10 steps
        if step % 10 == 0:
            feed_sense_input(model, "vision", vision_input * 2.0)  # Stronger input
            feed_sense_input(model, "digital", digital_input * 20.0)  # Stronger input

            # Also inject strong input directly to some neurons for demonstration
            for n in z0_neurons[:10]:
                n.external_input += 500.0  # Very strong direct stimulation (needs to exceed threshold)

        stats = sim.step()
        total_spikes += len(stats["spikes"])
        total_deaths += stats["deaths"]
        total_births += stats["births"]

        if (step + 1) % 20 == 0:
            print(
                f"   Step {step + 1}/{n_steps}: "
                f"{len(stats['spikes'])} spikes, "
                f"{len(model.neurons)} neurons, "
                f"{len(model.synapses)} synapses"
            )

    print(f"\n   Simulation complete!")
    print(f"   Total spikes: {total_spikes}")
    print(f"   Total deaths: {total_deaths}")
    print(f"   Total births: {total_births}")

    # Show sample neuron state
    print("\n6. Sample neuron states:")
    sample_neurons = list(model.neurons.values())[:3]
    for n in sample_neurons:
        print(
            f"   Neuron {n.id}: pos={n.position()}, "
            f"gen={n.generation}, health={n.health:.4f}, age={n.age}"
        )

    # Save model state
    print("\n7. Saving model state...")

    # Save to JSON
    json_path = os.path.join(os.path.dirname(__file__), "brain_state.json")
    save_to_json(model, json_path)
    print(f"   Saved to JSON: {json_path}")

    # Try to save to HDF5 (if h5py is available)
    try:
        from storage import save_to_hdf5

        hdf_path = os.path.join(os.path.dirname(__file__), "brain_state.h5")
        save_to_hdf5(model, hdf_path)
        print(f"   Saved to HDF5: {hdf_path}")
    except ImportError:
        print("   HDF5 save skipped (h5py not installed)")
    except Exception as e:
        print(f"   HDF5 save failed: {e}")

    # Verify JSON save/load
    print("\n8. Verifying JSON save/load...")
    loaded_model = load_from_json(json_path)
    print(f"   Loaded {len(loaded_model.neurons)} neurons")
    print(f"   Loaded {len(loaded_model.synapses)} synapses")
    print(f"   Current step: {loaded_model.current_step}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
