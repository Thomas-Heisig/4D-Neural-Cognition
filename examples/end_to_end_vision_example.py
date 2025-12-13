#!/usr/bin/env python3
"""End-to-end vision example: Data → Training → Evaluation → Visualization

This example demonstrates the complete workflow for using the 4D Neural Cognition
system for a vision task. It includes:

1. Data preprocessing (creating synthetic vision patterns)
2. Model configuration and initialization
3. Training with sensory input
4. Evaluation with metrics
5. Result visualization and export

This serves as a template for real-world applications.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import json
from pathlib import Path
from src.brain_model import BrainModel
from src.simulation import Simulation
from src.senses import feed_sense_input
from src.metrics import calculate_network_stability, calculate_population_synchrony
from src.experiment_management import (
    ExperimentDatabase,
    get_git_status,
    ExperimentConfig
)
import time


class VisionExample:
    """Complete end-to-end vision processing example."""
    
    def __init__(self, output_dir: str = "examples/vision_output"):
        """Initialize the example.
        
        Args:
            output_dir: Directory to save outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize experiment tracking
        self.db = ExperimentDatabase(
            db_path=str(self.output_dir / "experiments.db")
        )
        
        print("=" * 70)
        print("4D NEURAL COGNITION - END-TO-END VISION EXAMPLE")
        print("=" * 70)
        print()
    
    def step1_prepare_data(self):
        """Step 1: Prepare synthetic vision data."""
        print("STEP 1: Data Preparation")
        print("-" * 70)
        
        # Create synthetic patterns (like simplified MNIST)
        # Pattern A: Vertical line
        pattern_a = np.zeros((10, 10))
        pattern_a[:, 4:6] = 10.0
        
        # Pattern B: Horizontal line
        pattern_b = np.zeros((10, 10))
        pattern_b[4:6, :] = 10.0
        
        # Pattern C: Diagonal line
        pattern_c = np.zeros((10, 10))
        np.fill_diagonal(pattern_c, 10.0)
        
        # Pattern D: Cross
        pattern_d = np.zeros((10, 10))
        pattern_d[:, 4:6] = 10.0
        pattern_d[4:6, :] = 10.0
        
        self.patterns = {
            'vertical': pattern_a,
            'horizontal': pattern_b,
            'diagonal': pattern_c,
            'cross': pattern_d
        }
        
        print(f"✓ Created {len(self.patterns)} synthetic patterns")
        print(f"  Pattern shape: {pattern_a.shape}")
        print()
        
        return self.patterns
    
    def step2_configure_model(self):
        """Step 2: Configure the neural model."""
        print("STEP 2: Model Configuration")
        print("-" * 70)
        
        config = {
            "lattice_shape": [20, 20, 20, 5],
            "neuron_model": {
                "model_type": "LIF",
                "params_default": {
                    "v_rest": -65.0,
                    "v_threshold": -50.0,
                    "v_reset": -70.0,
                    "tau_m": 20.0,
                    "refractory_period": 5
                }
            },
            "cell_lifecycle": {
                "enabled": True,
                "enable_death": False,  # Disable for training stability
                "enable_reproduction": False,
                "max_age": 100000,
                "health_decay_per_step": 0.0001,
            },
            "plasticity": {
                "enabled": True,
                "learning_rate": 0.01,
                "weight_min": 0.0,
                "weight_max": 1.0,
                "weight_decay": 0.001
            },
            "senses": {
                "vision": {
                    "areal": "V1",
                    "enabled": True
                }
            },
            "areas": [
                {
                    "name": "V1",
                    "coord_ranges": {
                        "x": [0, 9],
                        "y": [0, 9],
                        "z": [0, 4],
                        "w": [0, 1]
                    }
                },
                {
                    "name": "V2",
                    "coord_ranges": {
                        "x": [10, 19],
                        "y": [0, 9],
                        "z": [5, 9],
                        "w": [2, 3]
                    }
                },
                {
                    "name": "IT",  # Inferotemporal (higher-level)
                    "coord_ranges": {
                        "x": [0, 19],
                        "y": [10, 19],
                        "z": [10, 14],
                        "w": [4, 4]
                    }
                }
            ]
        }
        
        print("✓ Configuration created")
        print(f"  Lattice shape: {config['lattice_shape']}")
        print(f"  Brain areas: {', '.join([a['name'] for a in config['areas']])}")
        print(f"  Plasticity: {config['plasticity']['enabled']}")
        print()
        
        # Save configuration
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✓ Configuration saved to {config_path}")
        print()
        
        return config
    
    def step3_initialize_network(self, config, seed=42):
        """Step 3: Initialize the neural network."""
        print("STEP 3: Network Initialization")
        print("-" * 70)
        
        # Create model
        model = BrainModel(config=config)
        sim = Simulation(model, seed=seed)
        
        # Initialize neurons in all areas
        print("Initializing neurons...")
        sim.initialize_neurons(
            area_names=["V1", "V2", "IT"],
            density=0.15  # 15% of positions filled
        )
        
        print(f"✓ Created {len(model.neurons)} neurons")
        
        # Initialize synapses
        print("Creating synaptic connections...")
        sim.initialize_random_synapses(
            connection_probability=0.02,
            weight_mean=0.1,
            weight_std=0.05
        )
        
        print(f"✓ Created {len(model.synapses)} synapses")
        print(f"  Average connections per neuron: {len(model.synapses) / len(model.neurons):.2f}")
        print()
        
        return model, sim
    
    def step4_train_network(self, model, sim, patterns, n_epochs=5):
        """Step 4: Train the network on patterns."""
        print("STEP 4: Training")
        print("-" * 70)
        
        # Create experiment tracking
        exp_id = f"vision_training_{int(time.time())}"
        run_id = f"{exp_id}_run"
        
        self.db.add_experiment(
            exp_id=exp_id,
            name="Vision Pattern Training",
            config=model.config,
            description="Training on 4 synthetic vision patterns"
        )
        
        self.db.add_run(
            run_id=run_id,
            experiment_id=exp_id,
            config=model.config,
            seed=42
        )
        
        start_time = time.time()
        
        # Training loop
        pattern_names = list(patterns.keys())
        total_steps = n_epochs * len(patterns) * 10
        
        print(f"Training for {n_epochs} epochs ({total_steps} total steps)...")
        print()
        
        step_count = 0
        epoch_stats = []
        
        for epoch in range(n_epochs):
            epoch_spikes = 0
            
            # Present each pattern multiple times
            for pattern_name in pattern_names:
                pattern = patterns[pattern_name]
                
                # Present pattern for 10 steps
                for _ in range(10):
                    feed_sense_input(model, 'vision', pattern)
                    stats = sim.step()
                    epoch_spikes += len(stats['spikes'])
                    
                    # Log metrics periodically
                    if step_count % 20 == 0:
                        self.db.add_metric(
                            run_id=run_id,
                            step=step_count,
                            metric_name='spike_count',
                            metric_value=float(len(stats['spikes']))
                        )
                    
                    step_count += 1
            
            avg_spikes = epoch_spikes / (len(patterns) * 10)
            epoch_stats.append({
                'epoch': epoch,
                'avg_spikes': avg_spikes,
                'total_spikes': epoch_spikes
            })
            
            print(f"  Epoch {epoch + 1}/{n_epochs}: "
                  f"avg_spikes={avg_spikes:.1f}, "
                  f"total={epoch_spikes}")
        
        duration = time.time() - start_time
        
        # Update run status
        self.db.update_run(
            run_id=run_id,
            status='completed',
            duration=duration,
            metrics={
                'total_spikes': sum(e['total_spikes'] for e in epoch_stats),
                'avg_spikes_per_step': sum(e['avg_spikes'] for e in epoch_stats) / n_epochs
            }
        )
        
        print()
        print(f"✓ Training completed in {duration:.2f}s")
        print()
        
        return epoch_stats
    
    def step5_evaluate(self, model, sim, patterns):
        """Step 5: Evaluate network performance."""
        print("STEP 5: Evaluation")
        print("-" * 70)
        
        # Test network response to each pattern
        results = {}
        
        for pattern_name, pattern in patterns.items():
            # Present pattern and measure response
            spike_counts = []
            
            for _ in range(20):  # 20 test steps per pattern
                feed_sense_input(model, 'vision', pattern)
                stats = sim.step()
                spike_counts.append(len(stats['spikes']))
            
            results[pattern_name] = {
                'mean_spikes': float(np.mean(spike_counts)),
                'std_spikes': float(np.std(spike_counts)),
                'max_spikes': int(np.max(spike_counts)),
                'min_spikes': int(np.min(spike_counts))
            }
        
        print("Pattern-specific responses:")
        for pattern_name, result in results.items():
            print(f"  {pattern_name:12s}: "
                  f"mean={result['mean_spikes']:.1f}, "
                  f"std={result['std_spikes']:.1f}")
        
        # Calculate network stability
        try:
            # Collect spike train for stability calculation
            spike_trains = {}
            for neuron_id in list(model.neurons.keys())[:100]:  # Sample of neurons
                if hasattr(sim, 'spike_history') and neuron_id in sim.spike_history:
                    spike_trains[neuron_id] = sim.spike_history[neuron_id]
            
            if spike_trains:
                stability = calculate_network_stability(spike_trains, window_size=10)
                print()
                print("Network stability metrics:")
                print(f"  Stability coefficient: {stability:.4f}")
        except Exception as e:
            pass  # Skip if not enough data
        
        print()
        print("✓ Evaluation completed")
        print()
        
        return results, {}
    
    def step6_visualize_and_export(self, model, results):
        """Step 6: Export results and create visualizations."""
        print("STEP 6: Visualization & Export")
        print("-" * 70)
        
        # Save results
        results_path = self.output_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Results saved to {results_path}")
        
        # Export network state
        from src.storage import save_to_json
        model_path = self.output_dir / "trained_model.json"
        save_to_json(model, str(model_path))
        print(f"✓ Model saved to {model_path}")
        
        # Summary statistics
        print()
        print("Network Summary:")
        print(f"  Total neurons: {len(model.neurons)}")
        print(f"  Total synapses: {len(model.synapses)}")
        print(f"  Simulation steps: {model.current_step}")
        
        # Neuron health distribution
        healths = [n.health for n in model.neurons.values()]
        print(f"  Neuron health: mean={np.mean(healths):.3f}, "
              f"min={np.min(healths):.3f}, max={np.max(healths):.3f}")
        
        # Synaptic weight distribution
        weights = [s.weight for s in model.synapses]
        print(f"  Synapse weights: mean={np.mean(weights):.3f}, "
              f"min={np.min(weights):.3f}, max={np.max(weights):.3f}")
        
        print()
        print("✓ Export completed")
        print()
    
    def run_complete_pipeline(self):
        """Run the complete end-to-end pipeline."""
        # Step 1: Data
        patterns = self.step1_prepare_data()
        
        # Step 2: Configuration
        config = self.step2_configure_model()
        
        # Step 3: Initialize
        model, sim = self.step3_initialize_network(config)
        
        # Step 4: Train
        epoch_stats = self.step4_train_network(model, sim, patterns, n_epochs=3)
        
        # Step 5: Evaluate
        results, performance = self.step5_evaluate(model, sim, patterns)
        
        # Step 6: Export
        self.step6_visualize_and_export(model, results)
        
        print("=" * 70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print()
        print(f"All outputs saved to: {self.output_dir}")
        print()
        print("Next steps:")
        print("  1. Review results.json for detailed metrics")
        print("  2. Load trained_model.json for further experiments")
        print("  3. Query experiments.db for run history")
        print("  4. Modify parameters and re-run for comparison")
        print()


def main():
    """Main entry point."""
    example = VisionExample()
    example.run_complete_pipeline()


if __name__ == "__main__":
    main()
