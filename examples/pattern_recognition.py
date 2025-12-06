#!/usr/bin/env python3
"""Pattern Recognition Example

This example demonstrates how to train a neural network to recognize
different visual patterns using the 4D Neural Cognition system.

The network learns to distinguish between:
- Horizontal lines
- Vertical lines
- Diagonal lines
- Checkerboard patterns
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input


def create_patterns():
    """Create a set of distinct visual patterns."""
    patterns = {}
    
    # Horizontal line
    horizontal = np.zeros((10, 10))
    horizontal[5, :] = 1.0
    patterns['horizontal'] = horizontal
    
    # Vertical line
    vertical = np.zeros((10, 10))
    vertical[:, 5] = 1.0
    patterns['vertical'] = vertical
    
    # Diagonal (top-left to bottom-right)
    diagonal = np.eye(10)
    patterns['diagonal'] = diagonal
    
    # Checkerboard
    x, y = np.meshgrid(range(10), range(10))
    checkerboard = ((x + y) % 2).astype(float)
    patterns['checkerboard'] = checkerboard
    
    return patterns


def visualize_pattern(pattern, name):
    """Print a text visualization of a pattern."""
    print(f"\n{name}:")
    for row in pattern:
        print(''.join(['██' if x > 0.5 else '··' for x in row]))


def train_on_patterns(sim, patterns, epochs=5):
    """Train the network to recognize patterns.
    
    Args:
        sim: Simulation instance
        patterns: Dictionary of pattern_name -> pattern_array
        epochs: Number of training epochs
    """
    print("\n" + "="*60)
    print("TRAINING PHASE")
    print("="*60)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for name, pattern in patterns.items():
            # Show pattern to network
            feed_sense_input(
                sim.model,
                sense_name="vision",
                input_matrix=pattern)
            
            # Let network process
            for step in range(30):
                sim.step()
                
                # Apply learning periodically
                if step % 5 == 0:
                    sim.apply_plasticity()
            
            # Clear input by running without stimulus
            for step in range(10):
                sim.step()
        
        # Check weights after epoch
        if epoch % 2 == 0:
            weights = [s['weight'] for s in sim.model.synapses.values()]
            print(f"  Mean weight: {np.mean(weights):.3f} "
                  f"(std: {np.std(weights):.3f})")


def test_recognition(sim, patterns):
    """Test pattern recognition by measuring neural responses.
    
    Args:
        sim: Simulation instance
        patterns: Dictionary of pattern_name -> pattern_array
    """
    print("\n" + "="*60)
    print("TESTING PHASE")
    print("="*60)
    
    results = {}
    
    for name, pattern in patterns.items():
        # Present pattern
        feed_sense_input(
            sim.model,
            sense_name="vision",
            input_matrix=pattern)
        
        # Measure response
        spike_counts = []
        for step in range(30):
            spiked = sim.step()
            spike_counts.append(len(spiked))
        
        # Calculate metrics
        total_spikes = sum(spike_counts)
        avg_activity = np.mean(spike_counts)
        peak_activity = max(spike_counts)
        
        results[name] = {
            'total_spikes': total_spikes,
            'avg_activity': avg_activity,
            'peak_activity': peak_activity
        }
        
        print(f"\n{name}:")
        print(f"  Total spikes: {total_spikes}")
        print(f"  Avg activity: {avg_activity:.1f} spikes/step")
        print(f"  Peak activity: {peak_activity} spikes")
        
        # Clear for next pattern
        for step in range(10):
            sim.step()
    
    return results


def analyze_results(results):
    """Analyze and compare pattern responses."""
    print("\n" + "="*60)
    print("ANALYSIS")
    print("="*60)
    
    # Find which pattern produced strongest response
    strongest = max(results.items(), key=lambda x: x[1]['total_spikes'])
    weakest = min(results.items(), key=lambda x: x[1]['total_spikes'])
    
    print(f"\nStrongest response: {strongest[0]} ({strongest[1]['total_spikes']} spikes)")
    print(f"Weakest response: {weakest[0]} ({weakest[1]['total_spikes']} spikes)")
    
    # Calculate response variability
    all_spikes = [r['total_spikes'] for r in results.values()]
    variability = np.std(all_spikes) / np.mean(all_spikes) if np.mean(all_spikes) > 0 else 0
    
    print(f"\nResponse variability (CV): {variability:.2f}")
    if variability > 0.3:
        print("  → Good discrimination between patterns!")
    else:
        print("  → Network may need more training or different parameters")


def main():
    """Run the pattern recognition example."""
    print("="*60)
    print("PATTERN RECOGNITION EXAMPLE")
    print("4D Neural Cognition System")
    print("="*60)
    
    # Create patterns
    print("\nCreating patterns...")
    patterns = create_patterns()
    
    # Visualize patterns
    print("\nPattern Library:")
    for name, pattern in patterns.items():
        visualize_pattern(pattern, name)
    
    # Setup neural network
    print("\n" + "="*60)
    print("SETUP")
    print("="*60)
    
    model = BrainModel(config_path=str(Path(__file__).parent.parent / "brain_base_model.json"))
    sim = Simulation(model, seed=42)
    
    # Initialize neurons in visual area
    print("\nInitializing network...")
    sim.initialize_neurons(
        area_names=["V1_like"],
        density=0.1
    )
    
    # Create connections
    sim.initialize_random_synapses(
        connection_probability=0.1,
        weight_mean=0.5,
        weight_std=0.1
    )
    
    print(f"  Neurons: {len(sim.model.neurons)}")
    print(f"  Synapses: {len(sim.model.synapses)}")
    
    # Test before training
    print("\n" + "="*60)
    print("BASELINE (Before Training)")
    print("="*60)
    results_before = test_recognition(sim, patterns)
    
    # Train
    train_on_patterns(sim, patterns, epochs=5)
    
    # Test after training
    results_after = test_recognition(sim, patterns)
    
    # Analyze
    print("\n" + "="*60)
    print("COMPARISON: Before vs After Training")
    print("="*60)
    
    for name in patterns.keys():
        before = results_before[name]['total_spikes']
        after = results_after[name]['total_spikes']
        change = after - before
        print(f"\n{name}:")
        print(f"  Before: {before} spikes")
        print(f"  After:  {after} spikes")
        print(f"  Change: {change:+d} ({change/before*100:+.1f}%)" if before > 0 else "  Change: N/A")
    
    analyze_results(results_after)
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    print("\nKey Takeaways:")
    print("  • Network learns to respond differently to different patterns")
    print("  • Hebbian plasticity strengthens active connections")
    print("  • Response patterns reflect learned associations")
    print("  • More training epochs improve discrimination")
    
    print("\nNext Steps:")
    print("  • Try different learning rates and network sizes")
    print("  • Add more complex patterns")
    print("  • Experiment with multi-area networks")
    print("  • Save and load trained models")


if __name__ == "__main__":
    main()
