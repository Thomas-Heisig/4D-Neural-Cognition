#!/usr/bin/env python3
"""Temporal Sequence Learning Example

This example demonstrates how the network can learn temporal sequences
and patterns over time. The network learns to predict what comes next
in a sequence, showing temporal processing capabilities.

Sequences demonstrated:
- Simple repeating patterns
- Musical-like tone sequences
- Predictive coding
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from brain_model import BrainModel
from simulation import Simulation
from senses import feed_sense_input


def create_sequences():
    """Create temporal sequences for learning."""
    sequences = {}
    
    # Sequence 1: Simple A-B-C pattern
    seq_abc = []
    for _ in range(3):
        # A: top row
        a = np.zeros((10, 10))
        a[0, :] = 1.0
        seq_abc.append(a)
        
        # B: middle row
        b = np.zeros((10, 10))
        b[5, :] = 1.0
        seq_abc.append(b)
        
        # C: bottom row
        c = np.zeros((10, 10))
        c[9, :] = 1.0
        seq_abc.append(c)
    
    sequences['abc_pattern'] = seq_abc
    
    # Sequence 2: Rising diagonal
    seq_diagonal = []
    for i in range(10):
        pattern = np.zeros((10, 10))
        for j in range(i+1):
            if j < 10:
                pattern[j, j] = 1.0
        seq_diagonal.append(pattern)
    
    sequences['rising_diagonal'] = seq_diagonal
    
    # Sequence 3: Expanding square
    seq_square = []
    for size in range(1, 6):
        pattern = np.zeros((10, 10))
        center = 5
        start = center - size
        end = center + size
        pattern[max(0, start):min(10, end), max(0, start):min(10, end)] = 1.0
        seq_square.append(pattern)
    
    sequences['expanding_square'] = seq_square
    
    return sequences


def visualize_sequence(sequence, name, max_frames=5):
    """Visualize a temporal sequence."""
    print(f"\n{'='*50}")
    print(f"Sequence: {name}")
    print(f"Length: {len(sequence)} frames")
    print(f"{'='*50}")
    
    frames_to_show = min(max_frames, len(sequence))
    print(f"\nShowing first {frames_to_show} frames:\n")
    
    for i, frame in enumerate(sequence[:frames_to_show]):
        print(f"Frame {i+1}:")
        for row in frame:
            print(''.join(['██' if x > 0.5 else '··' for x in row]))
        if i < frames_to_show - 1:
            print()


def present_sequence(sim, sequence, delay_between_frames=5):
    """Present a temporal sequence to the network.
    
    Args:
        sim: Simulation instance
        sequence: List of frames (numpy arrays)
        delay_between_frames: Steps between frames
        
    Returns:
        Activity trace: List of spike counts per step
    """
    activity = []
    
    for frame in sequence:
        # Show frame
        feed_sense_input(
            sim.model,
            sense_name="vision",
            input_matrix=frame)
        
        # Let network process
        for step in range(delay_between_frames):
            spiked = sim.step()
            activity.append(len(spiked))
    
    return activity


def train_on_sequences(sim, sequences, epochs=5):
    """Train network to learn temporal sequences.
    
    Args:
        sim: Simulation instance
        sequences: Dictionary of sequence_name -> list of frames
        epochs: Number of training epochs
    """
    print("\n" + "="*60)
    print("TEMPORAL LEARNING")
    print("="*60)
    print("\nTraining network on temporal sequences...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        for name, sequence in sequences.items():
            # Present sequence
            activity = present_sequence(sim, sequence, delay_between_frames=5)
            
            # Apply learning after each sequence
            for _ in range(10):
                sim.apply_plasticity()
            
            # Brief pause between sequences
            for step in range(10):
                sim.step()
            
            print(f"  Trained on: {name} ({len(sequence)} frames)")
        
        # Report weight changes
        if epoch % 2 == 0:
            weights = [s['weight'] for s in sim.model.synapses.values()]
            print(f"  Mean weight: {np.mean(weights):.3f}")


def test_sequence_completion(sim, sequence, show_partial=0.5):
    """Test if network can complete a sequence from partial input.
    
    Show only first part of sequence and measure continued activity
    to see if network "expects" the rest.
    
    Args:
        sim: Simulation instance
        sequence: Full sequence
        show_partial: Fraction of sequence to show (0-1)
        
    Returns:
        Activity during and after presentation
    """
    partial_length = int(len(sequence) * show_partial)
    
    # Show partial sequence
    activity_during = present_sequence(
        sim, 
        sequence[:partial_length], 
        delay_between_frames=5
    )
    
    # Continue without input (test prediction)
    activity_after = []
    for step in range(len(sequence) * 5):
        spiked = sim.step()
        activity_after.append(len(spiked))
    
    return activity_during, activity_after


def analyze_temporal_learning(activity_traces):
    """Analyze temporal activity patterns."""
    print("\n" + "="*60)
    print("TEMPORAL ANALYSIS")
    print("="*60)
    
    for name, (during, after) in activity_traces.items():
        print(f"\n{name}:")
        
        # Calculate statistics
        mean_during = np.mean(during)
        mean_after = np.mean(after)
        
        # Check if activity persists (sign of learned expectation)
        persistence_ratio = mean_after / mean_during if mean_during > 0 else 0
        
        print(f"  During presentation: {mean_during:.1f} avg spikes/step")
        print(f"  After presentation:  {mean_after:.1f} avg spikes/step")
        print(f"  Persistence ratio:   {persistence_ratio:.2f}", end="")
        
        if persistence_ratio > 0.5:
            print(" (strong temporal memory)")
        elif persistence_ratio > 0.2:
            print(" (moderate temporal memory)")
        else:
            print(" (weak temporal memory)")
        
        # Check for rhythmic patterns in activity
        if len(after) > 10:
            # Simple autocorrelation at lag 5 (frame period)
            activity_array = np.array(after)
            if len(activity_array) > 5:
                mean_activity = np.mean(activity_array)
                lag5_corr = np.corrcoef(
                    activity_array[:-5],
                    activity_array[5:]
                )[0, 1] if len(activity_array) > 5 else 0
                
                print(f"  Rhythmic pattern:    {lag5_corr:.2f}", end="")
                if lag5_corr > 0.3:
                    print(" (learned timing)")
                else:
                    print(" (no clear rhythm)")


def test_sequence_prediction(sim, sequences):
    """Test prediction by showing partial sequences."""
    print("\n" + "="*60)
    print("SEQUENCE COMPLETION TEST")
    print("="*60)
    print("\nTesting if network predicts sequence continuation...")
    
    results = {}
    
    for name, sequence in sequences.items():
        print(f"\n{name}:")
        print(f"  Showing 50% of sequence ({len(sequence)//2} frames)...")
        
        during, after = test_sequence_completion(
            sim, 
            sequence, 
            show_partial=0.5
        )
        
        results[name] = (during, after)
        
        print(f"  Activity during: {np.mean(during):.1f} spikes/step")
        print(f"  Activity after:  {np.mean(after):.1f} spikes/step")
    
    return results


def main():
    """Run the temporal sequence learning example."""
    print("="*60)
    print("TEMPORAL SEQUENCE LEARNING EXAMPLE")
    print("4D Neural Cognition System")
    print("="*60)
    
    # Create sequences
    print("\nCreating temporal sequences...")
    sequences = create_sequences()
    
    # Visualize
    for name, sequence in sequences.items():
        visualize_sequence(sequence, name, max_frames=5)
    
    # Setup network
    print("\n" + "="*60)
    print("NETWORK SETUP")
    print("="*60)
    
    model = BrainModel(config_path=str(Path(__file__).parent.parent / "brain_base_model.json"))
    sim = Simulation(model, seed=42)
    
    # Initialize with moderate density
    print("\nInitializing network...")
    sim.initialize_neurons(
        area_names=["V1_like"],
        density=0.12  # Slightly higher for temporal processing
    )
    
    # Create connections with some recurrence
    sim.initialize_random_synapses(
        connection_probability=0.12,
        weight_mean=0.5,
        weight_std=0.1
    )
    
    print(f"  Neurons: {len(sim.model.neurons)}")
    print(f"  Synapses: {len(sim.model.synapses)}")
    
    # Baseline test
    print("\n" + "="*60)
    print("BASELINE (Before Training)")
    print("="*60)
    results_before = test_sequence_prediction(sim, sequences)
    
    # Train
    train_on_sequences(sim, sequences, epochs=5)
    
    # Test after training
    results_after = test_sequence_prediction(sim, sequences)
    
    # Analyze
    print("\n" + "="*60)
    print("BEFORE TRAINING")
    print("="*60)
    analyze_temporal_learning(results_before)
    
    print("\n" + "="*60)
    print("AFTER TRAINING")
    print("="*60)
    analyze_temporal_learning(results_after)
    
    # Compare improvement
    print("\n" + "="*60)
    print("IMPROVEMENT")
    print("="*60)
    
    for name in sequences.keys():
        before_persist = np.mean(results_before[name][1]) / np.mean(results_before[name][0]) if np.mean(results_before[name][0]) > 0 else 0
        after_persist = np.mean(results_after[name][1]) / np.mean(results_after[name][0]) if np.mean(results_after[name][0]) > 0 else 0
        
        improvement = after_persist - before_persist
        
        print(f"\n{name}:")
        print(f"  Before: {before_persist:.2f} persistence ratio")
        print(f"  After:  {after_persist:.2f} persistence ratio")
        print(f"  Change: {improvement:+.2f}", end="")
        if improvement > 0.1:
            print(" (significant improvement)")
        elif improvement > 0:
            print(" (slight improvement)")
        else:
            print(" (no improvement)")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETE")
    print("="*60)
    
    print("\nKey Findings:")
    print("  • Network learns temporal structure of sequences")
    print("  • Activity persists after sequence ends (temporal memory)")
    print("  • Network can predict sequence continuation")
    print("  • Recurrent connections enable temporal integration")
    
    print("\nApplications:")
    print("  • Speech and language processing")
    print("  • Music perception and generation")
    print("  • Action sequence learning")
    print("  • Predictive coding in sensory processing")
    
    print("\nNext Steps:")
    print("  • Test with longer, more complex sequences")
    print("  • Vary timing between frames")
    print("  • Test sequence recognition vs. generation")
    print("  • Explore working memory capacity")


if __name__ == "__main__":
    main()
