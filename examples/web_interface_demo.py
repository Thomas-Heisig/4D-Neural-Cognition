#!/usr/bin/env python3
"""
Web Interface Demo

This script demonstrates how to programmatically interact with the web interface
features and prepare data for visualization, analytics, and collaboration.
"""

import json
import time
from pathlib import Path

import numpy as np

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_model import BrainModel
from simulation import Simulation


def create_sample_model():
    """Create a sample model for demonstration."""
    print("Creating sample model...")
    
    # Initialize model
    model = BrainModel(config_path="brain_base_model.json")
    sim = Simulation(model, seed=42)
    
    # Initialize neurons
    print("Initializing neurons...")
    sim.initialize_neurons(area_names=["V1_like", "Digital_sensor"], density=0.05)
    print(f"Created {len(model.neurons)} neurons")
    
    # Initialize synapses
    print("Initializing synapses...")
    sim.initialize_random_synapses(connection_probability=0.001)
    print(f"Created {len(model.synapses)} synapses")
    
    return model, sim


def run_simulation_with_metrics(model, sim, steps=50):
    """Run simulation and collect metrics for analytics."""
    print(f"\nRunning simulation for {steps} steps...")
    
    metrics = {
        "spike_counts": [],
        "neuron_counts": [],
        "synapse_counts": [],
        "steps": []
    }
    
    for step in range(steps):
        stats = sim.step()
        
        metrics["spike_counts"].append(len(stats["spikes"]))
        metrics["neuron_counts"].append(len(model.neurons))
        metrics["synapse_counts"].append(len(model.synapses))
        metrics["steps"].append(model.current_step)
        
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{steps}: {len(stats['spikes'])} spikes, "
                  f"{len(model.neurons)} neurons, {len(model.synapses)} synapses")
    
    return metrics


def generate_visualization_data(model):
    """Generate data suitable for 3D/4D visualization."""
    print("\nGenerating visualization data...")
    
    neurons_data = []
    for neuron_id, neuron in list(model.neurons.items())[:100]:  # Limit for demo
        neurons_data.append({
            "id": neuron_id,
            "x": neuron.x,
            "y": neuron.y,
            "z": neuron.z,
            "w": 0,  # 4th dimension
            "v_membrane": neuron.v_membrane,
            "health": neuron.health,
            "age": neuron.age,
            "activity": 1.0 if abs(neuron.v_membrane) > 50 else 0.0
        })
    
    # Generate connection data
    connections_data = []
    for synapse in list(model.synapses)[:50]:  # Limit for demo
        pre_neuron = model.neurons.get(synapse.pre_id)
        post_neuron = model.neurons.get(synapse.post_id)
        
        if pre_neuron and post_neuron:
            connections_data.append({
                "from": {
                    "x": pre_neuron.x,
                    "y": pre_neuron.y,
                    "z": pre_neuron.z,
                    "w": 0
                },
                "to": {
                    "x": post_neuron.x,
                    "y": post_neuron.y,
                    "z": post_neuron.z,
                    "w": 0
                },
                "weight": synapse.weight
            })
    
    print(f"Generated {len(neurons_data)} neurons and {len(connections_data)} connections")
    
    return {
        "neurons": neurons_data,
        "connections": connections_data
    }


def create_experiment_configurations():
    """Create sample experiment configurations for parameter sweeps."""
    print("\nCreating experiment configurations...")
    
    # Base configuration
    base_config = {
        "learning_rate": 0.01,
        "density": 0.05,
        "weight_mean": 0.1,
        "weight_std": 0.05,
        "connection_probability": 0.001
    }
    
    # Create parameter sweep for learning rate
    learning_rates = [0.001, 0.01, 0.1]
    experiments = []
    
    for i, lr in enumerate(learning_rates):
        exp = {
            "id": f"exp_{i}",
            "name": f"Learning Rate {lr}",
            "description": f"Experiment with learning rate {lr}",
            "parameters": {**base_config, "learning_rate": lr},
            "status": "created"
        }
        experiments.append(exp)
    
    print(f"Created {len(experiments)} experiment configurations")
    return experiments


def generate_performance_metrics(metrics):
    """Generate performance metrics from simulation data."""
    print("\nGenerating performance metrics...")
    
    # Calculate statistics
    avg_spikes = np.mean(metrics["spike_counts"])
    std_spikes = np.std(metrics["spike_counts"])
    avg_neurons = np.mean(metrics["neuron_counts"])
    
    # Simulate performance metrics
    performance = {
        "accuracy": min(1.0, avg_spikes / 100.0),
        "precision": 0.75 + np.random.random() * 0.2,
        "recall": 0.70 + np.random.random() * 0.25,
        "f1Score": 0.72 + np.random.random() * 0.23,
        "stability": 1.0 - (std_spikes / (avg_spikes + 1))
    }
    
    print("Performance metrics:")
    for key, value in performance.items():
        print(f"  {key}: {value:.3f}")
    
    return performance


def save_data_for_web_interface(model, metrics, viz_data, experiments, performance):
    """Save all data in formats suitable for the web interface."""
    print("\nSaving data for web interface...")
    
    output_dir = Path("web_demo_output")
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics for analytics
    with open(output_dir / "analytics_data.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved analytics data to {output_dir / 'analytics_data.json'}")
    
    # Save visualization data
    with open(output_dir / "visualization_data.json", "w") as f:
        json.dump(viz_data, f, indent=2)
    print(f"Saved visualization data to {output_dir / 'visualization_data.json'}")
    
    # Save experiment configurations
    with open(output_dir / "experiments.json", "w") as f:
        json.dump(experiments, f, indent=2)
    print(f"Saved experiments to {output_dir / 'experiments.json'}")
    
    # Save performance metrics
    with open(output_dir / "performance_metrics.json", "w") as f:
        json.dump(performance, f, indent=2)
    print(f"Saved performance metrics to {output_dir / 'performance_metrics.json'}")
    
    # Generate README for the output
    readme_content = """# Web Interface Demo Output

This directory contains sample data generated for the web interface demonstration.

## Files

- `analytics_data.json`: Time-series data for spike counts, neuron counts, and synapse counts
- `visualization_data.json`: 3D/4D neuron positions and connections
- `experiments.json`: Sample experiment configurations
- `performance_metrics.json`: Performance evaluation metrics

## Usage

1. Start the web server: `python app.py`
2. Navigate to http://localhost:5000/advanced
3. Use the various tabs to explore the features:
   - **Visualization**: Load the visualization data
   - **Analytics**: Import analytics data to see charts
   - **Experiments**: Import experiment configurations
   - **Collaboration**: Create annotations and versions

## Data Format

### Visualization Data
- Neurons: Array of objects with x, y, z, w coordinates and properties
- Connections: Array of objects with from/to coordinates and weights

### Analytics Data
- Time-series arrays for various metrics
- Compatible with Chart.js visualizations

### Experiments
- Experiment configurations with parameters
- Ready for import into experiment manager

### Performance Metrics
- Accuracy, precision, recall, F1-score, stability
- Format compatible with radar chart visualization
"""
    
    with open(output_dir / "README.md", "w") as f:
        f.write(readme_content)
    print(f"Saved README to {output_dir / 'README.md'}")


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("Web Interface Feature Demonstration")
    print("=" * 60)
    
    # Create model
    model, sim = create_sample_model()
    
    # Run simulation
    metrics = run_simulation_with_metrics(model, sim, steps=50)
    
    # Generate visualization data
    viz_data = generate_visualization_data(model)
    
    # Create experiment configurations
    experiments = create_experiment_configurations()
    
    # Generate performance metrics
    performance = generate_performance_metrics(metrics)
    
    # Save all data
    save_data_for_web_interface(model, metrics, viz_data, experiments, performance)
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the web server: python app.py")
    print("2. Open http://localhost:5000/advanced")
    print("3. Explore the following features:")
    print("   - 3D/4D Visualization: Load neurons and connections")
    print("   - Real-time Analytics: View charts and metrics")
    print("   - Experiment Management: Create and compare experiments")
    print("   - Collaboration: Add annotations and create versions")
    print("\nGenerated data is available in: web_demo_output/")


if __name__ == "__main__":
    main()
